# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl

import paddle
from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import (
    SubstituteTemplate,
    build_package,
    extract_triton_kernel,
    find_so_path,
    get_pointer_hint,
    get_value_hint,
    multi_process_do,
    paddle_custom_op_head_part,
    rename_c_to_cu,
    tune_and_invoke_part,
)


def get_wint8_kernel_config():
    configs = []
    for num_stages in [2, 3, 4]:
        for block_m in [16]:
            for block_n in [32, 64, 128, 256]:
                for block_k in [32, 64, 128, 256]:
                    for split_k in [1, 2]:
                        configs.append(
                            triton.Config(
                                {
                                    "SPLIT_K": split_k,
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": 1,
                                },
                                num_stages=num_stages,
                                num_warps=4,
                            )
                        )
    return configs


@triton.autotune(
    configs=get_wint8_kernel_config(),
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr'],
)
@triton.jit
def wint8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """

    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_k = tl.max_contiguous(
        tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K
    )

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    magic_number = 0x00006400
    magic_number = magic_number.to(tl.uint16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # fp_b = b.to(tl.float16)

        fp_b = b | magic_number
        fp_b = fp_b.to(tl.float16, bitcast=True)
        fp_b = fp_b - 1152

        bs_ptrs = bs_ptr + offs_bn[None, :]
        bs = tl.load(bs_ptrs)
        fp_b = fp_b * bs

        accumulator += tl.dot(a, fp_b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    # only let the first block do epilogue
    if bias_ptr is not None and pid_sp_k == 0:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs)
        accumulator += bias[None, :]

    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


triton_wint8_template = (
    paddle_custom_op_head_part
    + """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x,
    const paddle::Tensor& qweight,
    const paddle::Tensor& scales,
    paddle::optional<paddle::Tensor>& bias,
    bool bool_trans_w) {
int m = x.shape()[0];
int k = x.shape()[1];
int n = scales.shape()[0];
auto c_out = paddle::full({m, n}, 0, x.dtype(), x.place());
auto dev_x = get_tensor_ptr(x);
auto dev_weight = get_tensor_ptr(qweight);
auto dev_c = get_tensor_ptr(c_out);
auto dev_scales = get_tensor_ptr(scales);
CUdeviceptr dev_bias = (CUdeviceptr)(nullptr);
if (bias) {
    dev_bias = get_tensor_ptr(*bias);
}
int stride_bk = n;
int stride_bn = 1;
if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = k;
}
auto run_triton_kernel = [&](int algo_id) -> CUresult{
    return ${op_name}_kernel(c_out.stream(),
                            dev_x,
                            dev_weight,
                            dev_c,
                            dev_scales,
                            dev_bias,
                            m,
                            n,
                            k,
                            k,
                            1,
                            stride_bk,
                            stride_bn,
                            n,
                            1,
                            algo_id);
};
std::vector<int> problem_size = {m, n, k};
"""
    + tune_and_invoke_part
    + """
return {c_out};
}
std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& a_shape,
                                                        const std::vector<int64_t>& b_shape,
                                                        const std::vector<int64_t>& c_shape,
                                                        const std::vector<int64_t>& d_shape,
                                                        bool bool_trans_w) {
    if (bool_trans_w) {
        return {{a_shape[0], b_shape[0]}};
    } else {
        return {{a_shape[0], b_shape[1]}};
    }
}
std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype};
}
PD_BUILD_OP(${op_name})
    .Inputs({"x", "qweight", "scales", paddle::Optional("bias")})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"bool_trans_w: bool"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


def weight_only_int8(x, qweight, scales, bias=None, bool_trans_w=True):
    M, K = x.shape
    if bool_trans_w:
        N = qweight.shape[0]
        stride_bk = 1
        stride_bn = K
    else:
        N = qweight.shape[1]
        stride_bk = N
        stride_bn = 1

    if in_dynamic_or_pir_mode():
        assert x.is_contiguous(), ""
        assert qweight.is_contiguous(), ""

        # output = paddle.zeros((M, N), dtype=x.dtype)

        # grid = lambda META: (
        #     triton.cdiv(M, META['BLOCK_SIZE_M'])
        #     * triton.cdiv(N, META['BLOCK_SIZE_N']),
        #     META['SPLIT_K'],
        # )

        # wint8_kernel[grid](
        #     x,
        #     qweight,
        #     output,
        #     scales,
        #     bias,
        #     M,
        #     N,
        #     K,
        #     K,
        #     1,  # A always is rowmajor
        #     stride_bk,
        #     stride_bn,
        #     N,
        #     1,  # C always is rowmajor
        # )
        # return output

    import os
    import sys

    op_name = "triton_wint8"
    if bool_trans_w:
        op_name += "_trans"

    address_hint = get_pointer_hint(x) + ","
    address_hint += get_pointer_hint(qweight) + ","
    # output type is same as x
    address_hint += get_pointer_hint(x) + ","
    address_hint += get_pointer_hint(scales) + ","
    # bias type is same as x
    address_hint += get_pointer_hint(x) + ","

    value_hint = get_value_hint(M) + ","
    value_hint += get_value_hint(N) + ","
    value_hint += get_value_hint(K) + ","
    value_hint += get_value_hint(K) + ","
    value_hint += get_value_hint(1) + ","
    value_hint += get_value_hint(stride_bk) + ","
    value_hint += get_value_hint(stride_bn) + ","
    value_hint += get_value_hint(N) + ","
    value_hint += get_value_hint(1) + ","

    op_name += value_hint.replace(",", "").replace(":", "")

    python_package_name = f"{op_name}_package"

    from paddle.base.framework import OpProtoHolder

    if (
        op_name in OpProtoHolder.instance().op_proto_map.keys()
        and in_dynamic_or_pir_mode()
    ):
        outs = _C_ops._run_custom_op(
            op_name, x, qweight, scales, bias, bool_trans_w
        )
        return outs[0]

    generated_dir = (
        f"/zhoukangkang/2023-06-06minigpt/PaddleNLP/llm/inference/{op_name}"
    )
    os.makedirs(generated_dir, exist_ok=True)

    py_script_file = f"{generated_dir}/triton_kernels.py"
    extract_triton_kernel(wint8_kernel, py_script_file)

    op_dict = {"op_name": op_name, "reset_zero_when_tune": " "}
    # when tunning, we need to reset the out to zero.
    op_dict[
        "reset_zero_when_tune"
    ] = "cudaMemset((void*)dev_c, 0, sizeof(phi::dtype::float16) * m * n);"
    paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
    so_path = find_so_path(generated_dir, python_package_name)

    if so_path is None:
        with open(paddle_custom_op_file_path, "w") as f:
            f.write(SubstituteTemplate(triton_wint8_template, op_dict))
            f.close()

        compile_file = triton.__path__[0] + "/tools/compile.py"
        link_file = triton.__path__[0] + "/tools/link.py"
        python_path = sys.executable

        # ahead of time compile command.
        aot_template = (
            f"""{python_path}   {compile_file} {py_script_file}   -n wint8_kernel   -o {generated_dir}/{op_name}_kernel --out-name {op_name}_kernel """
            + """ -w {num_warps}   -ns {num_stages}  \
        -s   "{address_hint} {value_hint}  {block_m},{block_n},{block_k}, 1, {split_k}"   \
        -g   "((M+{block_m}-1)/{block_m}) * ((N+{block_n}-1)/{block_n}), {split_k}, 1" \
        """
        )

        codegen_commands = []
        for config in get_wint8_kernel_config():
            split_k = config.kwargs["SPLIT_K"]
            block_m = config.kwargs["BLOCK_SIZE_M"]
            block_n = config.kwargs["BLOCK_SIZE_N"]
            block_k = config.kwargs["BLOCK_SIZE_K"]
            num_stages = config.num_stages
            num_warps = config.num_warps

            if K % (split_k * block_k) != 0:
                print("config is not supported:", config)
                continue
            codegen_command = aot_template.format(
                address_hint=address_hint,
                value_hint=value_hint,
                num_warps=num_warps,
                num_stages=num_stages,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                split_k=split_k,
            )
            codegen_commands.append(codegen_command)

        multi_process_do(codegen_commands)

        link_command = f"{python_path}  {link_file}  {generated_dir}/*.h -o {generated_dir}/{op_name}_kernel"
        re = os.system(link_command)
        assert re == 0

        # rename the .c file to .cu
        rename_c_to_cu(generated_dir)
        # build the package to so, not install
        build_package(generated_dir, python_package_name)

    from paddle.base.framework import OpProtoHolder

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        so_path = find_so_path(generated_dir, python_package_name)
        paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name, x, qweight, scales, bias, bool_trans_w
        )
        return outs[0]

    helper = LayerHelper(op_name, **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {
        'x': x,
        'qweight': qweight,
        'scales': scales,
        'bias@OPTIONAL': bias,
    }

    helper.append_op(
        type=op_name,
        inputs=inputs,
        attrs={"bool_trans_w": True},
        outputs={'out': out},
    )
    return out


def get_group_norm_kernel_config():
    configs = []
    for num_stages in [2, 3, 4]:
        for block_m in [32, 64, 128, 256, 512]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                    },
                    num_stages=num_stages,
                    num_warps=4,
                )
            )
    return configs


@triton.autotune(
    configs=get_group_norm_kernel_config(),
    key=['batch_stride', 'channel_stride', 'hw_stride', 'group_stride', 'group_num'],
)
@triton.jit
def group_norm_first_stage(
    sample_ptr,
    output_sum_ptr,
    output_sum_squares_ptr,
    N,
    batch_stride,
    channel_stride,
    hw_stride,
    group_stride,
    group_num,
    group_size, # numbers of channel
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
):
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_id = tl.program_id(2)
    offset_channel = tl.arange(0, BLOCK_SIZE_G)
    offset_block = tl.arange(0, BLOCK_SIZE_M)
    data_start = batch_id * batch_stride + group_id * group_stride
    sample_ptrs = sample_ptr + data_start + offset_channel[:, None] * channel_stride + offset_block[None, :] * hw_stride + block_id * BLOCK_SIZE_M
    # 计算均值
    channel_mask = offset_channel[:,None] < group_size
    offset_block_mask = offset_block[None, :] <  (channel_stride - block_id * BLOCK_SIZE_M)
    sample_ = tl.load(sample_ptrs, mask = channel_mask & offset_block_mask, other=0.0)
    # tl.static_print(" SAMPLE TYPE ", sample)
    sample = sample_.to(tl.float32)
    # sample_fp32 = sample
    _sum = tl.sum(sample)
    
    _sum_squares = tl.sum(sample * sample)
    # tl.static_print(_sum)
    # tl.static_print(_sum_squares)
    # # 直接add
    output_start = batch_id * group_num + group_id + tl.arange(0,1)
    tl.atomic_add(output_sum_ptr + output_start, _sum)
    tl.atomic_add(output_sum_squares_ptr + output_start, _sum_squares)

@triton.autotune(
    configs=get_group_norm_kernel_config(),
    key=['batch_stride', 'channel_stride', 'hw_stride', 'group_stride', 'group_num'],
)
@triton.jit
def group_norm_second_stage(
    sample_ptr,
    output_ptr,
    output_sum_ptr,
    output_sum_squares_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    N,
    batch_stride,
    channel_stride,
    hw_stride,
    group_stride,
    group_num,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
):
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_id = tl.program_id(2)
    offset_channel = (tl.arange(0, BLOCK_SIZE_G)) % group_size
    offset_block = tl.arange(0, BLOCK_SIZE_M)
    data_start = batch_id * batch_stride + group_id * group_stride
    sample_ptrs = sample_ptr + data_start + offset_channel[:, None] * channel_stride + (offset_block[None,:] * hw_stride + block_id * BLOCK_SIZE_M) % channel_stride
    sample_ = tl.load(sample_ptrs)
    sample = sample_.to(tl.float32)
    # 
    start = batch_id * group_num + group_id + tl.arange(0,1)
    # 计算均值
    _sum = tl.load(output_sum_ptr + start)
    # tl.device_print("sum",_sum)
    _mean = _sum / group_stride
    # 计算方差
    _sum_squares = tl.load(output_sum_squares_ptr + start)
    _var = _sum_squares / group_stride - _mean * _mean
    rstd = 1 / tl.sqrt(_var + eps)
    re_ = (sample - _mean) * rstd

    weight_para = tl.zeros((BLOCK_SIZE_G, 1), dtype= tl.float32)
    bias_para = tl.zeros((BLOCK_SIZE_G, 1), dtype= tl.float32)
    if weight_ptr:
        weight_para_temp = tl.load(weight_ptr + group_id * group_size + offset_channel[:,None])
        weight_para = weight_para_temp.to(tl.float32)
        # tl.static_print("weight_para", weight_para)
        # tl.static_print("re_________", re_)
        re_ = re_ * weight_para
    if bias_ptr:
        bias_para_temp = tl.load(bias_ptr + group_id * group_size + offset_channel[:,None])
        bias_para = bias_para_temp.to(tl.float32)
        re_ = re_ + bias_para
    # 这个得修改，以适应各种type
    re = re_.to(tl.float16)
    output_ptrs = output_ptr + data_start + offset_channel[:, None] * channel_stride + (offset_block[None,:] * hw_stride + block_id * BLOCK_SIZE_M) % (channel_stride)
    tl.store(output_ptrs, re)







triton_group_norm_template = (
    paddle_custom_op_head_part
    + """

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& bias,
    int group_num,
    float eps) {
int N = x.shape()[0];
int C = x.shape()[1];
int H = x.shape()[2];
int W = x.shape()[3];

auto sum_out = paddle::full({N, group_num}, 0, paddle::DataType::FLOAT32, x.place());
auto square_sum_out = paddle::full({N, group_num}, 0, paddle::DataType::FLOAT32, x.place());
auto c_out = paddle::full({N, C, H, W}, 0, x.dtype(), x.place());

auto dev_x = get_tensor_ptr(x);
auto dev_weight = get_tensor_ptr(weight);
auto dev_bias = get_tensor_ptr(bias);
auto dev_sum_out = get_tensor_ptr(sum_out);
auto dev_square_sum_out = get_tensor_ptr(square_sum_out);

int batch_stride = C * H * W;
int channel_stride = H * W;
int group_stride = group_num * H * W;
int hw_stride = 1;
int group_size = C / group_num;
auto run_triton_first_kernel = [&](int algo_id) -> CUresult{
    return ${first_kernel_name}_kernel(sum_out.stream(),
                            dev_x,
                            dev_sum_out,
                            dev_square_sum_out,
                            N,
                            batch_stride,
                            channel_stride,
                            hw_stride,
                            group_stride,
                            group_num,
                            group_size,
                            algo_id);
};

auto run_triton_second_kernel = [&](int algo_id) -> CUresult{
    return ${second_kernel_name}_kernel(sum_out.stream(),
                            dev_x,
                            dev_c,
                            dev_sum_out,
                            dev_square_sum_out,
                            dev_weight,
                            dev_bias,
                            eps,
                            N,
                            batch_stride,
                            channel_stride,
                            hw_stride,
                            group_stride,
                            group_num,
                            group_size,
                            algo_id);
};

std::vector<int> problem_size = {N, C, H*W, group_num};
"""
    + tune_and_invoke_part
    + """
return {c_out};
}
std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& weight_shape,
                                                        const std::vector<int64_t>& bias_shape,
                                                        int group_num,
                                                        float eps) {

    return {{a_shape[0], a_shape[1], a_shape[2], a_shape[3]}};
}
std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype};
}
PD_BUILD_OP(${op_name})
    .Inputs({"x", "weight", "bias"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"group_num: int", "eps: float"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)

# weight 和 bias 设成必选


def group_norm(sample, weight = None , bias = None, eps=1e-5, num_group = 1, data_format="NCHW"):


    N,C,H,W = sample.shape
    group_size = C // num_group

    BLOCK_SIZE_G = triton.next_power_of_2(group_size)
    batch_stride, channel_stride, group_stride, hw_stride = C * H * W, H * W, group_size * H * W, 1


    if in_dynamic_or_pir_mode():
        output_sum = paddle.zeros((N, num_group), dtype=paddle.float32)
        output_sum_squares = paddle.zeros((N, num_group), dtype=paddle.float32)
        
        # grid = lambda META: (
        #     N,
        #     num_group,
        #     triton.cdiv(H*W, META['BLOCK_SIZE_M'])
        # )
        # output = paddle.empty((N, C, H, W), dtype=sample.dtype)
        # group_norm_first_stage[grid](
        #     sample,
        #     output_sum,
        #     output_sum_squares,
        #     N,
        #     batch_stride,
        #     channel_stride,
        #     hw_stride,
        #     group_stride,
        #     num_group,
        #     group_size,
        #     BLOCK_SIZE_G=BLOCK_SIZE_G
        # )
        # group_norm_second_stage[grid](
        #     sample,
        #     output,
        #     output_sum,
        #     output_sum_squares,
        #     weight,
        #     bias,
        #     eps,
        #     N,
        #     batch_stride,
        #     channel_stride,
        #     hw_stride,
        #     group_stride,
        #     num_group,
        #     group_size,
        #     BLOCK_SIZE_G=BLOCK_SIZE_G,
        # )
        # return output
    

    import os
    import sys

    op_name = "triton_group_norm_" + str(BLOCK_SIZE_G)
    
    address_hint1 = get_pointer_hint(sample) + ","
    address_hint1 += "*fp32:16" + ","
    address_hint1 += "*fp32:16" + ","

    value_hint1 = get_value_hint(batch_stride) + ","
    value_hint1 += get_value_hint(channel_stride) + ","
    value_hint1 += get_value_hint(hw_stride) + ","
    value_hint1 += get_value_hint(group_stride) + ","
    value_hint1 += get_value_hint(num_group) + ","
    value_hint1 += get_value_hint(group_size) + ","


    address_hint2 = get_pointer_hint(sample) + ","
    address_hint2 += get_pointer_hint(sample) + ","
    address_hint2 += "*fp32:16" + ","
    address_hint2 += "*fp32:16" + ","
    address_hint2 += get_pointer_hint(sample) + ","
    address_hint2 += get_pointer_hint(sample) + ","

    value_hint2 = "fp32,"
    value_hint2 = get_value_hint(batch_stride) + ","
    value_hint2 += get_value_hint(channel_stride) + ","
    value_hint2 += get_value_hint(hw_stride) + ","
    value_hint2 += get_value_hint(group_stride) + ","
    value_hint2 += get_value_hint(num_group) + ","
    value_hint2 += get_value_hint(group_size) + ","

    op_name += (value_hint1.replace(",", "").replace(":", "") + value_hint2.replace(",", "").replace(":", ""))
    
    first_kernel_name = op_name + "_first"
    second_kernel_name = op_name + "_second"

    python_package_name = f"{op_name}_package"

    from paddle.base.framework import OpProtoHolder

    if (
        op_name in OpProtoHolder.instance().op_proto_map.keys()
        and in_dynamic_or_pir_mode()
    ):
        outs = _C_ops._run_custom_op(
            op_name, sample, weight, bias, num_group, eps)
        return outs[0]

    generated_dir = (
        f"/nishirong/Paddle/triton/aot/{op_name}"
    )
    os.makedirs(generated_dir, exist_ok=True)

    py_script_file = f"{generated_dir}/triton_kernels.py"
    extract_triton_kernel(group_norm_first_stage, py_script_file)
    extract_triton_kernel(group_norm_second_stage, py_script_file)

    op_dict = {"op_name": op_name, "reset_zero_when_tune": " "}
    paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
    so_path = find_so_path(generated_dir, python_package_name)

    if so_path is None:
        with open(paddle_custom_op_file_path, "w") as f:
            f.write(SubstituteTemplate(triton_group_norm_template, op_dict))
            f.close()

        compile_file = triton.__path__[0] + "/tools/compile.py"
        link_file = triton.__path__[0] + "/tools/link.py"
        python_path = sys.executable

        
        # ahead of time compile command.
        aot_template1 = (
            f"""{python_path}   {compile_file} {py_script_file}   -n group_norm_first_stage   -o {generated_dir}/{first_kernel_name}_kernel --out-name {first_kernel_name}_kernel """
            + """ -w {num_warps}   -ns {num_stages}  \
        -s   "{address_hint} {value_hint}  {block_m}, {block_g}"   \
        -g   " N , group_num,  channel_stride/{block_m}," \
        """
        )

        aot_template2 = (
            f"""{python_path}   {compile_file} {py_script_file}   -n group_norm_second_stage   -o {generated_dir}/{second_kernel_name}_kernel --out-name {second_kernel_name}_kernel """
            + """ -w {num_warps}   -ns {num_stages}  \
        -s   "{address_hint} {value_hint}  {block_m}, {block_g}"   \
        -g   " N , group_num,  channel_stride/{block_m}" \
        """
        )

        codegen_commands = []
        for config in get_group_norm_kernel_config():
            block_m = config.kwargs["BLOCK_SIZE_M"]
            num_stages = config.num_stages
            num_warps = config.num_warps
            

            codegen_command = aot_template1.format(
                address_hint=address_hint1,
                value_hint=value_hint1,
                num_warps=num_warps,
                num_stages=num_stages,
                block_m=block_m,
                block_g=BLOCK_SIZE_G,
            )
            codegen_commands.append(codegen_command)
            print("codegen_command0", codegen_command)


            codegen_command = aot_template1.format(
                address_hint=address_hint2,
                value_hint=value_hint2,
                num_warps=num_warps,
                num_stages=num_stages,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                split_k=split_k,
            )
            print("codegen_command1", codegen_command)
            codegen_commands.append(codegen_command)


        multi_process_do(codegen_commands)

        link_command0 = f"{python_path}  {link_file}  {generated_dir}/{first_kernel_name}*.h -o {generated_dir}/{first_kernel_name}_kernel"
        link_command1 = f"{python_path}  {link_file}  {generated_dir}/{second_kernel_name}*.h -o {generated_dir}/{second_kernel_name}_kernel"

        re = os.system(link_command0)
        assert re == 0
        re = os.system(link_command1)
        assert re == 0

        # rename the .c file to .cu
        rename_c_to_cu(generated_dir)
        # build the package to so, not install
        build_package(generated_dir, python_package_name)



