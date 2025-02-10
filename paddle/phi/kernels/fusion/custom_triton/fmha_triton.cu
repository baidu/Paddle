#include <vector>
#include "paddle/extension.h"
#include "generated/aot/fmha/fp16/fmha_fp16.h"
#include <cuda_fp16.h>

std::vector<paddle::Tensor> TritonFMHA(const paddle::Tensor& q, 
                                              const paddle::Tensor& k, const paddle::Tensor& v,
                                              const float& scale) {

    int batch = q.shape()[0];
    int head = q.shape()[1];
    int seq = q.shape()[2];
    int hidden_dim = q.shape()[3];

    auto qkv_out = paddle::full({batch, head, seq, hidden_dim}, 0, q.dtype(), q.place());
    auto l_tmp = paddle::full({batch * head, seq}, 0, paddle::DataType::FLOAT32, q.place());
    auto m_tmp = paddle::full({batch * head, seq}, 0, paddle::DataType::FLOAT32, q.place());

    auto dev_q = q.data<phi::dtype::float16>();
    auto dev_k = k.data<phi::dtype::float16>();
    auto dev_v = v.data<phi::dtype::float16>();
    auto dev_out = qkv_out.data<phi::dtype::float16>();
    auto dev_l = l_tmp.data<float>();
    auto dev_m = m_tmp.data<float>();

    auto status = fmha_kernel_fp16(qkv_out.stream(), 
    (CUdeviceptr)(dev_out), (CUdeviceptr)(dev_l), (CUdeviceptr)(dev_m),
    (CUdeviceptr)(dev_q), (CUdeviceptr)(dev_k), (CUdeviceptr)(dev_v),
    scale, batch, head, seq, 0);

    assert(status == CUDA_SUCCESS);

    return {qkv_out};                                                            
}

std::vector<std::vector<int64_t>> TritonFMHAInferShape(const std::vector<int64_t>& q_shape,
                                                              const std::vector<int64_t>& k_shape, 
                                                              const std::vector<int64_t>& v_shape) {
    return {q_shape};
}

std::vector<paddle::DataType> TritonFMHAInferDtype(const paddle::DataType& q_dtype,
                                                        const paddle::DataType& k_dtype,
                                                        const paddle::DataType& v_dtype) {
    return {q_dtype};
}

PD_BUILD_OP(triton_fmha)
    .Inputs({"q", "k", "v"})
    .Outputs({"out"})
    .Attrs({"scale:float"})
    .SetKernelFn(PD_KERNEL(TritonFMHA))
    .SetInferShapeFn(PD_INFER_SHAPE(TritonFMHAInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TritonFMHAInferDtype));
    