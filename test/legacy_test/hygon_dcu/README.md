# 海光DCU兼容飞桨大模型
## 适配算子测试
### 测试目的
国产硬件在大模型训练和推理过程中，高性能核心算子的覆盖度和质量确实至关重要，它们直接决定了海光DCU能够适配的模型数量及性能表现。
### 测试内容
在test_hygon_llama_op.sh脚本中，详细测试了adamw、arange、bitwise_and等65个算子。
基于大模型的训练推理场景，大部分测试用例的数据类型均为float16。
适配核心算子的数值精度以及梯度数值精度符合飞桨多硬件接入标准（float16绝对容忍误差atol<=0.1，相对容忍误差rtol<=0.01）。
### 测试方法
- 在飞桨仓库中提交了相关单测的[PR](https://github.com/PaddlePaddle/Paddle/pull/68603)并完成合入，在飞桨仓库的CI中会自动进行以上适配算子的[测试](https://github.com/PaddlePaddle/Paddle/blob/b78ce03485d19edd9afc960fb7145944bb2bf5d2/test/legacy_test/CMakeLists.txt#L103)。

- 也可在准备好的DTK环境中，编译完飞桨框架后，运行以下指令：
```
source /opt/dtk/env.sh
cd build/test/legacy_test/hygon_dcu
bash test_hygon_llama_op.sh
```
## 通信库测试
### 测试目的
大模型训练对计算和网络通信都提出极高的要求，海光DCU的大模型训练不仅依赖核心算子的开发，同时需要依赖集合通信库的适配；通过对海光DCU在集合通信库的测试，确保具备大模型训练的基础条件。
### 测试内容
以飞桨分布式Model Parallelism、Group Sharded等功能单元测试为基础，测试海光DCU分布式通信库是否能正常运行，精度是否符合误差。

具体包含[Layer Sharding](https://github.com/PaddlePaddle/Paddle/blob/develop/test/collective/fleet/test_parallel_dygraph_mp_layers.py), [Model Sharding](https://github.com/PaddlePaddle/Paddle/blob/develop/test/collective/fleet/hybrid_parallel_mp_model.py), [Sharding Stage 2](https://github.com/PaddlePaddle/Paddle/blob/develop/test/collective/fleet/test_dygraph_sharding_stage2.py), [Sharding Stage 3](https://github.com/PaddlePaddle/Paddle/blob/develop/test/collective/fleet/test_dygraph_sharding_stage3_for_eager.py), [Recompute](https://github.com/PaddlePaddle/Paddle/blob/develop/test/collective/fleet/test_dygraph_recompute.py)和[Recompute for Eager](https://github.com/PaddlePaddle/Paddle/blob/develop/test/collective/fleet/test_dygraph_recompute_for_eager.py)几个功能。
### 测试方法
在飞桨仓库中提交了通信库测试的[PR](https://github.com/PaddlePaddle/Paddle/pull/68877)并完成合入，在飞桨仓库的CI中自动上述[通信接口](https://github.com/PaddlePaddle/Paddle/blob/b78ce03485d19edd9afc960fb7145944bb2bf5d2/test/legacy_test/CMakeLists.txt)。
