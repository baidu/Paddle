
#include <vector>
#include "paddle/extension.h"
#include "generated/matmul/fp16.h"
#include <cuda_fp16.h>

std::map<std::vector<int>, int> map_problem_triton_matmul;

std::vector<paddle::Tensor> TritonMatMul(const paddle::Tensor& a, 
                                              const paddle::Tensor& b,
                                              paddle::optional<paddle::Tensor>& bias,
                                              bool with_bias,
                                              bool bool_trans_w) {

    int m = a.shape()[0];
    int n = bool_trans_w?b.shape()[0]:b.shape()[1];
    int k = a.shape()[1];
    auto c_out = paddle::full({m, n}, 0, a.dtype(), a.place());

    auto dev_a = a.data<phi::dtype::float16>();
    auto dev_b = b.data<phi::dtype::float16>();
    auto dev_c = c_out.data<phi::dtype::float16>();
    phi::dtype::float16* dev_bias = nullptr;

    if (with_bias) {
        dev_bias = bias->data<phi::dtype::float16>();
    }
    int stride_bk = n;
    int stride_bn = 1;
    if (bool_trans_w) {
        stride_bk = 1;
        stride_bn = k;
    }
    std::vector<int> problem_size = {m, n, k};
    if(map_problem_triton_matmul.count(problem_size)) {
        int algo_id = map_problem_triton_matmul[problem_size];
        auto status = matmul_kernel_fp16(c_out.stream(), 
                    (CUdeviceptr)(dev_a), 
                    (CUdeviceptr)(dev_b), 
                    (CUdeviceptr)(dev_c),
                    (CUdeviceptr)(dev_bias),
                    m,n,k,
                    k,1,
                    stride_bk,stride_bn,
                    n,1,
                    algo_id);
        assert(status == CUDA_SUCCESS);
        return {c_out};
    }
    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;
    for(int algo_id = 0; algo_id < matmul_kernel_fp16_get_num_algos(); algo_id++) {
        cudaEvent_t begin[REPEAT];
        cudaEvent_t end[REPEAT];
        float elapsed_times[REPEAT];
        auto status = CUDA_SUCCESS;
        for(int ii = 0; ii < WARMUP + REPEAT; ii++){
            int repeat_id = ii - WARMUP;
            if(repeat_id >= 0){
                (cudaEventCreate(begin + repeat_id));
                (cudaEventCreate(end + repeat_id));
                (cudaEventRecord(begin[repeat_id]));
            }
            // 为什么不可以是dev_a place呢？
            auto flush_l2_cache = paddle::full(
                {10 * 1024 * 1024}, 0, paddle::DataType::INT32, a.place());
            cudaMemset(dev_c, 0, sizeof(phi::dtype::float16) * m * n);
            // std::cout<<(CUdeviceptr)(dev_bias)<<std::endl;
            status = matmul_kernel_fp16(c_out.stream(), 
                        (CUdeviceptr)(dev_a), 
                        (CUdeviceptr)(dev_b), 
                        (CUdeviceptr)(dev_c),
                        (CUdeviceptr)(dev_bias),
                        m,n,k,
                        k,1,
                        stride_bk,stride_bn,
                        n,1,
                        algo_id);
            // std::cout<<m<<" "<<n<<" "<<k<<" "<<stride_bk<<" "<<stride_bn<<" "<<std::endl;
            // assert(status == CUDA_SUCCESS);
            if(repeat_id >= 0){
                (cudaEventRecord(end[repeat_id]));
                (cudaEventSynchronize(end[repeat_id]));
                (cudaEventElapsedTime(elapsed_times + repeat_id, begin[repeat_id], end[repeat_id]));
            }
        }
        float avg_elapsed_time = 0.f;
        for(int ii = 0; ii < REPEAT; ii++){
            avg_elapsed_time += elapsed_times[ii];
        }
        if(avg_elapsed_time < min_time && status == CUDA_SUCCESS){
            min_time = avg_elapsed_time;
            select_id = algo_id;
        }
    }
    map_problem_triton_matmul[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    return {c_out};                                                            
}

// std::vector<std::vector<int64_t>> TritonMatMulInferShape(const std::vector<int64_t>& a_shape,
//                                                               const std::vector<int64_t>& b_shape) {
//     return {{a_shape[0], b_shape[1]}};
// }

// std::vector<paddle::DataType> TritonMatMulInferDtype(const paddle::DataType& A_dtype,
//                                                         const paddle::DataType& B_dtype) {
//     return {A_dtype};
// }

PD_BUILD_OP(triton_matmul)
    .Inputs({"A", "B", paddle::Optional("Bias")})
    .Outputs({"C"})
    .SetKernelFn(PD_KERNEL(TritonMatMul))
    .Attrs({"with_bias:bool","bool_trans_w:bool"});
    // .SetInferShapeFn(PD_INFER_SHAPE(TritonMatMulInferShape))
    // .SetInferDtypeFn(PD_INFER_DTYPE(TritonMatMulInferDtype));
