#include <vector>
#include "paddle/extension.h"
#include "generated/aot/fmha2/fp16/fmha2_fp16.h"
#include <cuda_fp16.h>
#include <cmath>  
#include <iostream>

std::vector<paddle::Tensor> TritonFMHA2(const paddle::Tensor& q, 
                                              const paddle::Tensor& k, const paddle::Tensor& v,
                                              const float& scale) {

    int batch = q.shape()[0];
    int head = q.shape()[1];
    int seq = q.shape()[2];
    int hidden_dim = q.shape()[3];
    int stride_qz = head * seq * hidden_dim;
    int stride_qh = seq * hidden_dim;
    int stride_qm = hidden_dim;
    int stride_qk = 1;
    int stride_kz, stride_kh, stride_kn, stride_kk;
    int stride_vz, stride_vh, stride_vk, stride_vn;
    int stride_oz, stride_oh, stride_om, stride_on;
    stride_kz = stride_vz = stride_oz = stride_qz;
    stride_kh = stride_vh = stride_oh = stride_qh;
    stride_kn = stride_vk = stride_om = stride_qm;
    stride_kk = stride_vn = stride_on = stride_qk;
    int Z = batch;
    int H = head;
    auto qkv_out = paddle::full({batch, head, seq, hidden_dim}, 0, q.dtype(), q.place());
    // auto l_tmp = paddle::full({batch * head, seq}, 0, paddle::DataType::FLOAT32, q.place());
    auto m_tmp = paddle::full({batch, head, seq}, 0, paddle::DataType::FLOAT32, q.place());

    auto dev_q = q.data<phi::dtype::float16>();
    auto dev_k = k.data<phi::dtype::float16>();
    auto dev_v = v.data<phi::dtype::float16>();
    auto dev_out = qkv_out.data<phi::dtype::float16>();
    // auto dev_l = l_tmp.data<float>();
    auto dev_m = m_tmp.data<float>();
    auto status = fmha2_kernel_fp16_default(
                    qkv_out.stream(), (CUdeviceptr)(dev_q), (CUdeviceptr)(dev_k), (CUdeviceptr)(dev_v), 
                    scale, 
                    (CUdeviceptr)(dev_m), (CUdeviceptr)(dev_out), 
                    stride_qz, stride_qh, stride_qm, stride_qk, 
                    stride_kz, stride_kh, stride_kn, stride_kk, 
                    stride_vz, stride_vh, stride_vk, stride_vn, 
                    stride_oz, stride_oh, stride_om, stride_on, 
                    Z, H);
    
    assert(status == CUDA_SUCCESS);

    return {qkv_out};                                                            
}

std::vector<std::vector<int64_t>> TritonFMHA2InferShape(const std::vector<int64_t>& q_shape,
                                                              const std::vector<int64_t>& k_shape, 
                                                              const std::vector<int64_t>& v_shape) {
    return {q_shape};
}

std::vector<paddle::DataType> TritonFMHA2InferDtype(const paddle::DataType& q_dtype,
                                                        const paddle::DataType& k_dtype,
                                                        const paddle::DataType& v_dtype) {
    return {q_dtype};
}

PD_BUILD_OP(triton_fmha2)
    .Inputs({"q", "k", "v"})
    .Outputs({"out"})
    .Attrs({"scale: float"})
    .SetKernelFn(PD_KERNEL(TritonFMHA2))
    .SetInferShapeFn(PD_INFER_SHAPE(TritonFMHA2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TritonFMHA2InferDtype));
