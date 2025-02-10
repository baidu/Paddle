OpenAI Triton为专用领域的编程语言与编译器，巧妙地在硬件算子性能和开发效率之间达到了均衡。
Paddle提供了简单的示例，展示了如何将用户编写的Trtion自定义算子，接入Paddle。

# 如何利用Triton生成一个 cuda kernel

用户只需要在py文件中编写一个被`@triton.jit`装饰的核函数，即可实现一个Triton算子。
借助Triton的AOT工具，可以编译出cubin kernel。
下面以`matmul_triton.py`为例，展示如何利用Triton生成一个cubin kernel。

1. 用户在`matmul_triton.py`编写一个由`@triton.jit`装饰的核函数。

2. 利用如下命令产出cubin kernel。


# 当前目录下文件说明
1. 包含了**Fc**的triton算子，在`Fc_triton.py`中实现，单测实现在`Fc_test.py`中，生成自定义算子的c++文件在`Fc_triton.cu`中。目前文件中的实现将➕bias的部分注释掉了，如果要添加bias部分可以将对应部分取消注释。目前`Fc_triton.cu`文件中的支持高维矩阵乘法的部分注释掉了，如果需要高维矩阵乘法，可以将对应部分取消注释。除此之外`Fc_triton.cu`中还注释了一部分选择triton aot kernel的部分，可以进行一些参数选择等，但是只是一些尝试。
2. 包含**FcRelu**的Triton算子，在`FcRelu_triton.py`中实现，单测实现在`FcRelu_test.py`中，生成自定算子的c++文件在`FcRelu_triton.cu`中。
3. 包含**fmha**的Triton算子：
（1）基于开源v1版本修改了读写访存越界的问题的triton算子，在`fmha_triton.py`中实现，生成自定算子的c++文件在`fmha_triton.cu`中。
（2）开源v2版本，与`fmha2_triton.py`对应，文件中调用的另一个triton函数在`fmha2_triton_until.py`,生成自定算子的c++文件在`fmha2_triton.cu`. 相较于开源版本的fmha2 triton算子，做了两点变化。替换了N_CTX参数，变成了seqlen，以便于aot生成时，不同shape生成的kernel指令相同。 增加了写回时的boundary check选项，读取时的boundary check选项也添加了，但是注释了，因为没必要。
（3）将`fmha2_triton.py`中的多个参数删除，只传shape参数，对应实现在`fmha3_triton.py`中，文件中调用的另一个triton函数在`fmha3_triton_until.py`,因为大多数都要mask，所以删除了一些mask判断。删除了tl.debug_barrier语句，实验单测结果是正确的，但是个人理解应该是要加上的，因为前后两个循环是依赖的。
（4）将`fmha_test.py`包含了以上所有部分的单测。
4. 包含了**matmul**的triton算子，算子中bias是可选项，功能更全的matmul算子(支持转置，支持bias，支持split-k)在另一个pr中。生成自定义算子的c++文件在`matmul.cu`中，这个文件可支持自动选择最优的超参配置。
5. `implict_gemm_triton.py`是copy的一个naive版本的implict_gemm triton算子，copy下来应该是尝试增加了一个dilation部分。
5. 其他的一些配置文件
`generate_matmul_config.py`文件可以根据模板指令生成多种配置的cubin kernel的，中调用。
`run_generate.sh`包含了一些指令 aot生成 fmha 各个版本的cubin kernel。
`setup_cuda.py` 注册安装算子。












