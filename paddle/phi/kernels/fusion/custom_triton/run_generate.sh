
# clone triton and reset the specified commit id
# rm -rf generated 
# git clone https://github.com/openai/triton.git
# cd triton
# git reset --hard 2217bd2f5c271009f50ab2d2a639bbbb407a2650
# cd -


# these two files are used to compile and link
# compile_file=triton/python/triton/tools/compile.py
# link_file=triton/python/triton/tools/link.py
compile_file=triton/python/triton/tools/compile.py
link_file=triton/python/triton/tools/link.py

rm -rf generated/matmul
matmul_dir=generated/matmul/fp16
mkdir -p ${matmul_dir}


# -n : the kernel name decorated by @triton.jit in your py file
# -o : the output file name 
# --out-name : the name of the kernel in c++ for your invoke

rm -rf generated/aot/fmha
fmha_dir=generated/aot/fmha/fp16
mkdir -p ${fmha_dir}

python3.8  ${compile_file}     \
fmha_triton.py     \
-n fused_attention_kernel   \
-o ${fmha_dir}/fmha_fp16     \
--out-name fmha_kernel_fp16     \
-w 4  -ns 3     \
-s "*fp16:16, *fp32:16, *fp32:16, *fp16:16, *fp16:16, *fp16:16, fp32, i32, i32, i32, 32, 128, 32" \
-g "(seq_len + 31) / 32, batch_size * num_heads, 1"

python3.8  ${compile_file}     \
 fmha_triton.py     \
 -n fused_attention_kernel   \
 -o ${fmha_dir}/fmha_fp16     \
 --out-name fmha_kernel_fp16     \
 -w 4  -ns 3     \
 -s "*fp16:16, *fp32:16, *fp32:16, *fp16:16, *fp16:16, *fp16:16, fp32, i32, i32, i32, 64, 128, 32" \
 -g "(seq_len + 63) / 64, batch_size * num_heads, 1"

python3.8  ${link_file}  ${fmha_dir}/*.h -o ${fmha_dir}/fmha_fp16
 

rm -rf generated/aot/fmha2
fmha_dir=generated/aot/fmha2/fp16
mkdir -p ${fmha_dir}

python3.8  ${compile_file}     \
fmha2_triton.py     \
-n _attn_fwd   \
-o ${fmha_dir}/fmha2_fp16     \
--out-name fmha2_kernel_fp16     \
-w 4  -ns 3     \
-s "*fp16:16, *fp16:16, *fp16:16, fp32, *fp32:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 128, 32, 3" \
-g "(( stride_qh / stride_qm ) + 63) / 64, Z*H, 1"
-s "*fp16:16, *fp16:16, *fp16:16, fp32, *fp32:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, 42, 42, 128, 42, 3" \

-g "(( stride_qh / stride_qm ) + 127) / 128, Z*H, 1"

python3.8  ${link_file}  ${fmha_dir}/*.h -o ${fmha_dir}/fmha2_fp16

rm -rf generated/aot/fmha3
fmha_dir=generated/aot/fmha3/fp16
mkdir -p ${fmha_dir}

python3.8  ${compile_file}     \
fmha3_triton.py     \
-n _attn_fwd   \
-o ${fmha_dir}/fmha3_fp16     \
--out-name fmha3_kernel_fp16     \
-w 4  -ns 3     \
-s "*fp16:16, *fp16:16, *fp16:16, fp32, *fp32:16, *fp16:16, i32, i32, i32, i32, 42, 64, 128, 32" \
-g "( S + 63) / 64, Z*H, 1"
# -g "(( stride_qh / stride_qm ) + 127) / 128, Z*H, 1"

python3.8  ${link_file}  ${fmha_dir}/*.h -o ${fmha_dir}/fmha3_fp16


# mv the .c file to .cu file
for file in `find ./generated/ -name "*.c"`
do
    mv ${file} ${file}u
done


# 安装triton算子和运行单元测试
python3.8 setup_cuda.py install
#python3.8 test.py










