#include <vector>
#include "paddle/extension.h"
#include "generated/aot/FcRelu/fp16/FcRelu_fp16.h"
#include <cuda_fp16.h>
#include <cmath>  
#include <iostream>

std::vector<paddle::Tensor> TritonFcRelu(const paddle::Tensor& a, 
                                            const paddle::Tensor& b,
                                            const paddle::Tensor& bias) {

    
    // auto a_shape = a.shape();
    auto b_shape = b.shape();
    int n = b_shape[1];
    int k = b_shape[0];
    int m = a.size() / k;
    // for (int i = 0; i < a.shape().size()-1; ++i) {
    //     m *= a.shape()[i];
    // }
    ;
    
    // std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
    auto out_shape = a.shape();
    out_shape.back() = n;
    auto c_out = paddle::full(out_shape, 0, a.dtype(), a.place());

    auto dev_a = a.data<phi::dtype::float16>();
    auto dev_b = b.data<phi::dtype::float16>();
    auto dev_bias = bias.data<phi::dtype::float16>();
    auto dev_c = c_out.data<phi::dtype::float16>();

    if(m % 16 ==0 && n % 16 ==0 && k%16==0) {
        if( m  > 2560){
            auto status = FcRelu_kernel_fp16_normal(c_out.stream(), 
                (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
                (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
                m,n,k,
                k,1,
                n,1,
                n,1,
                2);
            assert(status == CUDA_SUCCESS);
        }else{
            auto status = FcRelu_kernel_fp16_normal(c_out.stream(), 
                (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
                (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
                m,n,k,
                k,1,
                n,1,
                n,1,
                3);
            assert(status == CUDA_SUCCESS);
        }
    }else {
        if( m  > 2560){
            auto status = FcRelu_kernel_fp16_normal(c_out.stream(), 
                (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
                (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
                m,n,k,
                k,1,
                n,1,
                n,1,
                0);
            assert(status == CUDA_SUCCESS);
        }else{
            auto status = FcRelu_kernel_fp16_normal(c_out.stream(), 
                (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
                (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
                m,n,k,
                k,1,
                n,1,
                n,1,
                1);
            assert(status == CUDA_SUCCESS);
        }
    }
    

    return {c_out};                                                            
}
std::vector<std::vector<int64_t>> TritonFcReluInferShape(const std::vector<int64_t>& a_shape,
                                                              const std::vector<int64_t>& b_shape, 
                                                              const std::vector<int64_t>& bias_shape) {
    auto out_shape = a_shape;
    out_shape.back() = b_shape[1];
    // out_shape.pop_back();
    // out_shape.push_back(b_shape[1]);
    // std::cout << "out_shape: " << out_shape[0] << ", " << out_shape[1] << out_shape.size() << std::endl;
    return {out_shape};

}

std::vector<paddle::DataType> TritonFcReluInferDtype(const paddle::DataType& a_dtype,
                                                        const paddle::DataType& b_dtype,
                                                        const paddle::DataType& bias_dtype) {
    return {a_dtype};
}

PD_BUILD_OP(triton_FcRelu)
    .Inputs({"a", "b", "bias"})
    .Outputs({"c"})
    .SetKernelFn(PD_KERNEL(TritonFcRelu))
    .SetInferShapeFn(PD_INFER_SHAPE(TritonFcReluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TritonFcReluInferDtype));
