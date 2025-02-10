#include <vector>
#include "paddle/extension.h"
#include "generated/aot/Fc/fp16/Fc_fp16.h"
#include <cuda_fp16.h>
#include <cmath>  
#include <iostream>

std::vector<paddle::Tensor> TritonFc(const paddle::Tensor& a, 
                                            const paddle::Tensor& b){
                                            // const paddle::Tensor& bias) {

    
    // // auto a_shape = a.shape();
    // auto b_shape = b.shape();
    // int n = b_shape[1];
    // int k = b_shape[0];
    // int m = a.size() / k;
    // // for (int i = 0; i < a.shape().size()-1; ++i) {
    // //     m *= a.shape()[i];
    // // }
    // ;
    
    // // std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
    // auto out_shape = a.shape();
    // out_shape.back() = n;
    // auto c_out = paddle::full(out_shape, 0, a.dtype(), a.place());

    // auto dev_a = a.data<phi::dtype::float16>();
    // auto dev_b = b.data<phi::dtype::float16>();
    // // auto dev_bias = bias.data<phi::dtype::float16>();
    // auto dev_c = c_out.data<phi::dtype::float16>();
    int m = a.shape()[0];
    int n = b.shape()[1];
    int k = a.shape()[1];
    auto c_out = paddle::full({m, n}, 0, a.dtype(), a.place());

    auto dev_a = a.data<phi::dtype::float16>();
    auto dev_b = b.data<phi::dtype::float16>();
    auto dev_c = c_out.data<phi::dtype::float16>();

    auto status = Fc_kernel_fp16_default(c_out.stream(), 
                (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
                // (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
                (CUdeviceptr)(dev_c),
                m,n,k,
                k,1,
                n,1,
                n,1);
    assert(status == CUDA_SUCCESS);
    return {c_out};
    // if(m % 16 ==0 && n % 16 ==0 && k%16==0) {
    //     if( m  > 2560){
    //         auto status = Fc_kernel_fp16_normal(c_out.stream(), 
    //             (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
    //             // (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
    //             (CUdeviceptr)(dev_c),
    //             m,n,k,
    //             k,1,
    //             n,1,
    //             n,1,
    //             2);
    //         assert(status == CUDA_SUCCESS);
    //     }else{
    //         auto status = Fc_kernel_fp16_normal(c_out.stream(), 
    //             (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
    //             // (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
    //             (CUdeviceptr)(dev_c),
    //             m,n,k,
    //             k,1,
    //             n,1,
    //             n,1,
    //             3);
    //         assert(status == CUDA_SUCCESS);
    //     }
    // }else {
    //     if( m  > 2560){
    //         auto status = Fc_kernel_fp16_normal(c_out.stream(), 
    //             (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
    //             // (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
    //             (CUdeviceptr)(dev_c),
    //             m,n,k,
    //             k,1,
    //             n,1,
    //             n,1,
    //             0);
    //         assert(status == CUDA_SUCCESS);
    //     }else{
    //         auto status = Fc_kernel_fp16_normal(c_out.stream(), 
    //             (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
    //             // (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
    //             (CUdeviceptr)(dev_c),
    //             m,n,k,
    //             k,1,
    //             n,1,
    //             n,1,
    //             1);
    //         assert(status == CUDA_SUCCESS);
    //     }
    // }
    

    // return {c_out};                                                            
}
std::vector<std::vector<int64_t>> TritonFcInferShape(const std::vector<int64_t>& a_shape,
                                                              const std::vector<int64_t>& b_shape) {
    auto out_shape = a_shape;
    out_shape.back() = b_shape[1];
    // out_shape.pop_back();
    // out_shape.push_back(b_shape[1]);
    // std::cout << "out_shape: " << out_shape[0] << ", " << out_shape[1] << out_shape.size() << std::endl;
    return {out_shape};

}

std::vector<paddle::DataType> TritonFcInferDtype(const paddle::DataType& a_dtype,
                                                        const paddle::DataType& b_dtype) {
    return {a_dtype};
}

PD_BUILD_OP(triton_Fc)
    .Inputs({"a", "b"})
    .Outputs({"c"})
    .SetKernelFn(PD_KERNEL(TritonFc))
    .SetInferShapeFn(PD_INFER_SHAPE(TritonFcInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TritonFcInferDtype));

// PD_BUILD_OP(triton_Fc)
//     .Inputs({"a", "b", "bias"})
//     .Outputs({"c"})
//     .SetKernelFn(PD_KERNEL(TritonFc))
//     .SetInferShapeFn(PD_INFER_SHAPE(TritonFcInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(TritonFcInferDtype));
