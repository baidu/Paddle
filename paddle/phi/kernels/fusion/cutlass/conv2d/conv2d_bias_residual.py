# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import enum

from conv2d_common import (
    CommonConvFunction,
    CommonCutlassConvKernelDeclare,
    CommonCutlassConvKernelExecute,
    CommonTail,
    GenerateFunctionForPhi,
)
from util import SubstituteTemplate, TileDesc, parse_args, write_kernel_to_file

# this is a file's header part

cbr_header = '''
// Generated by conv2d_bias_residual.py - Do not edit.

#include <mutex>
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_util.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {
'''

# This is a cutlass kernel, will be many these like kernels

dict_for_declare_part = {
    "conv_kind_name": "DefaultConv2dFpropWithBroadcast",
    "epi_part": "cutlass::epilogue::thread::LinearCombinationResidualBlock< ${element_c}, ${element_accum}, ${element_epilogue}, ${element_residul}, ${epilogue_vector_length}, ${act1}, ${binary}, ${act2}>",
}

cbr_kernel = (
    SubstituteTemplate(CommonCutlassConvKernelDeclare, dict_for_declare_part)
    + '''
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {input, {ic, ic * iw, ic * iw * ih}},
      {weight, {kc, kc * kw, kc * kw * kh}},
      {residual, {oc, oc * ow, oc * ow * oh}},
      {output, {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f},
      cutlass::conv::SplitKMode::kSerial,
      (cutlass::half_t *)(bias), nullptr,
      0, oc};
'''
    + CommonCutlassConvKernelExecute
)


class CbrAct(enum.Enum):
    Identity = 1
    Relu = 2
    Silu = 3


ActTag = {
    CbrAct.Identity: 'cutlass::epilogue::thread::Identity',
    CbrAct.Silu: 'cutlass::epilogue::thread::SiLu',
    CbrAct.Relu: 'cutlass::epilogue::thread::ReLu',
}

# Some global variables used, now we only support these residual blocks.
SupportedEpilogue = [
    (CbrAct.Silu, "cutlass::plus", CbrAct.Identity),
    (CbrAct.Identity, "cutlass::plus", CbrAct.Relu),
    (CbrAct.Identity, "cutlass::plus", CbrAct.Identity),
]

UnderScoreName = {
    SupportedEpilogue[0]: "conv2d_bias_silu_add",
    SupportedEpilogue[1]: "conv2d_bias_add_relu",
    SupportedEpilogue[2]: "conv2d_bias_add",
}

CamelName = {
    SupportedEpilogue[0]: "Conv2dBiasSiluAdd",
    SupportedEpilogue[1]: "Conv2dBiasAddRelu",
    SupportedEpilogue[2]: "Conv2dBiasAdd",
}

# Generate sm75 TensorOp conv code.
# CUTLASS Tensor Core operations are implemented using CUDA's mma instruction.
# Here is mma.m16n8k8.


def generate_sm75_1688():
    kernel_dict = {
        "conv_kind_name": "Fprop",
        "element_a": "cutlass::half_t",
        "layout_a": "cutlass::layout::TensorNHWC",
        "element_b": "cutlass::half_t",
        "layout_b": "cutlass::layout::TensorNHWC",
        "element_c": "cutlass::half_t",
        "layout_c": "cutlass::layout::TensorNHWC",
        "opcode_class": "cutlass::arch::OpClassTensorOp",
        "arch": "cutlass::arch::Sm75",
        "stages": "2",
        "swizzling_functor": "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>",
        # alpha is always float!
        "element_epilogue": "float",
        "math_operator": "cutlass::arch::OpMultiplyAdd",
        "element_residul": "cutlass::half_t",
    }

    kernel_dict["stride_support"] = "cutlass::conv::StrideSupport::kStrided"

    # iterate over this loop
    element_accums = ["cutlass::half_t", "float"]
    iterator_algorithms = [
        "cutlass::conv::IteratorAlgorithm::kOptimized",
        # "cutlass::conv::IteratorAlgorithm::kAnalytic",
    ]

    math_instructions = [
        # (
        #     "16,8,8",
        #     "cutlass::half_t",
        #     "cutlass::half_t",
        #     "cutlass::half_t",
        # ),
        (
            "16,8,8",
            "cutlass::half_t",
            "cutlass::half_t",
            "float",
        ),
    ]

    alignments = [8]

    kernel_dict["align_a"] = "8"
    kernel_dict["align_b"] = "8"
    kernel_dict["epilogue_vector_length"] = "8"
    kernel_dict["split_k_slices"] = "1"

    sm75_code = ""
    for epi_res_block in SupportedEpilogue:
        op_dict = {}
        op_dict["func_name"] = (
            UnderScoreName[epi_res_block].lower() + "_sm75_fp16"
        )
        op_dict["enum_op_name"] = UnderScoreName[epi_res_block].upper()
        # for a op, we record all its kernels into a std::vector in C++ code
        all_kernel_names = ""
        all_kernel_declares = ""
        suffix = 0
        for iterator_algorithm in iterator_algorithms:
            for alignment in alignments:
                for math_inst in math_instructions:
                    tiles = [
                        TileDesc("64, 64, 64", 2, "32, 32, 64", math_inst),
                        TileDesc("64, 32, 64", 2, "32, 32, 64", math_inst),
                        TileDesc("128, 32, 64", 2, "32, 32, 64", math_inst),
                        TileDesc("128, 64, 64", 2, "32, 32, 64", math_inst),
                        TileDesc("64, 64, 32", 2, "32, 32, 32", math_inst),
                        TileDesc("64, 128, 32", 2, "32, 64, 32", math_inst),
                        # diff is too large, so comment it
                        # TileDesc("64, 128, 64", 2, "64, 64, 32", math_inst),
                        TileDesc("64, 256, 32", 2, "64, 64, 32", math_inst),
                        TileDesc("128, 64, 32", 2, "64, 32, 32", math_inst),
                        TileDesc("128, 128, 32", 2, "64, 64, 32", math_inst),
                        TileDesc("128, 256, 32", 2, "64, 64, 32", math_inst),
                        TileDesc("256, 64, 32", 2, "64, 64, 32", math_inst),
                        TileDesc("256, 128, 32", 2, "64, 64, 32", math_inst),
                    ]
                    for tile in tiles:
                        kernel_dict["iterator_algorithm"] = iterator_algorithm
                        kernel_dict["Tshape"] = tile.Tshape
                        kernel_dict["Wshape"] = tile.Wshape
                        kernel_dict["Ishape"] = tile.math_inst[0]
                        kernel_dict["element_accum"] = tile.math_inst[3]
                        kernel_dict["kernel_func_name"] = op_dict[
                            "func_name"
                        ] + str(suffix)
                        kernel_dict["act1"] = ActTag[epi_res_block[0]]
                        kernel_dict["binary"] = epi_res_block[1]
                        kernel_dict["act2"] = ActTag[epi_res_block[2]]
                        suffix += 1

                        # sm75_code += SubstituteTemplate(cbr_kernel, kernel_dict)

                        kernel_str = (
                            cbr_header
                            + SubstituteTemplate(cbr_kernel, kernel_dict)
                            + CommonTail
                        )
                        file_name = (
                            "generated_tmp/"
                            + kernel_dict["kernel_func_name"]
                            + ".cu"
                        )
                        write_kernel_to_file(kernel_str, file_name)

                        all_kernel_names += (
                            kernel_dict["kernel_func_name"] + ", \n"
                        )
                        all_kernel_declares += (
                            "cutlass::Status "
                            + kernel_dict["kernel_func_name"]
                            + "(const ConvAllParams& params);"
                        )

        # Generate op code
        op_dict["kernel_func_declare"] = all_kernel_declares
        op_dict["all_kernel_func_name"] = all_kernel_names
        sm75_code += SubstituteTemplate(CommonConvFunction, op_dict)
    return sm75_code


def generate_sm80_16816(cutlass_dtype="cutlass::half_t"):
    kernel_dict = {
        "conv_kind_name": "Fprop",
        "element_a": cutlass_dtype,
        "layout_a": "cutlass::layout::TensorNHWC",
        "element_b": cutlass_dtype,
        "layout_b": "cutlass::layout::TensorNHWC",
        "element_c": cutlass_dtype,
        "layout_c": "cutlass::layout::TensorNHWC",
        "opcode_class": "cutlass::arch::OpClassTensorOp",
        "arch": "cutlass::arch::Sm80",
        "swizzling_functor": "cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>",
        # alpha is always float!
        "element_epilogue": "float",
        "math_operator": "cutlass::arch::OpMultiplyAdd",
        "element_residul": cutlass_dtype,
    }

    kernel_dict["stride_support"] = "cutlass::conv::StrideSupport::kStrided"

    # iterate over this loop
    iterator_algorithms = [
        "cutlass::conv::IteratorAlgorithm::kOptimized",
    ]

    math_instructions = [
        (
            "16,8,16",
            cutlass_dtype,
            cutlass_dtype,
            "float",
        ),
    ]

    alignments = [8]

    kernel_dict["align_a"] = "8"
    kernel_dict["align_b"] = "8"
    kernel_dict["epilogue_vector_length"] = "8"
    kernel_dict["split_k_slices"] = "1"

    sm80_code = ""
    for epi_res_block in SupportedEpilogue:
        op_dict = {}
        op_dict["func_name"] = (
            UnderScoreName[epi_res_block].lower()
            + "_sm80_"
            + ("fp16" if "half" in cutlass_dtype else "bf16")
        )

        op_dict["enum_op_name"] = UnderScoreName[epi_res_block].upper()
        # for a op, we record all its kernels into a std::vector in C++ code
        all_kernel_names = ""
        all_kernel_declares = ""
        suffix = 0
        for iterator_algorithm in iterator_algorithms:
            for alignment in alignments:
                for math_inst in math_instructions:
                    tiles = [
                        TileDesc("256, 128, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("128, 256, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("256, 64, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("256, 64, 32", 4, "64, 64, 32", math_inst),
                        TileDesc("64, 256, 32", 4, "64, 64, 32", math_inst),
                        TileDesc("128, 128, 32", 3, "64, 64, 32", math_inst),
                        TileDesc("128, 128, 32", 4, "64, 64, 32", math_inst),
                        TileDesc("128, 128, 32", 5, "64, 64, 32", math_inst),
                        TileDesc("128, 64, 32", 6, "64, 32, 32", math_inst),
                        TileDesc("64, 128, 32", 6, "32, 64, 32", math_inst),
                        TileDesc("64, 64, 32", 10, "32, 32, 32", math_inst),
                        TileDesc("256, 128, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("128, 256, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("256, 64, 64", 4, "64, 64, 64", math_inst),
                        TileDesc("64, 256, 64", 4, "64, 64, 64", math_inst),
                        TileDesc("128, 128, 64", 4, "64, 64, 64", math_inst),
                        TileDesc("256, 64, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("64, 256, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("128, 128, 64", 3, "64, 64, 64", math_inst),
                        TileDesc("128, 64, 64", 3, "64, 32, 64", math_inst),
                        TileDesc("64, 128, 64", 3, "32, 64, 64", math_inst),
                        TileDesc("64, 64, 64", 5, "32, 32, 64", math_inst),
                    ]

                    for tile in tiles:
                        kernel_dict["iterator_algorithm"] = iterator_algorithm
                        kernel_dict["Tshape"] = tile.Tshape
                        kernel_dict["Wshape"] = tile.Wshape
                        kernel_dict["Ishape"] = tile.math_inst[0]
                        kernel_dict["stages"] = str(tile.stages)
                        kernel_dict["element_accum"] = tile.math_inst[3]
                        kernel_dict["kernel_func_name"] = op_dict[
                            "func_name"
                        ] + str(suffix)
                        kernel_dict["act1"] = ActTag[epi_res_block[0]]
                        kernel_dict["binary"] = epi_res_block[1]
                        kernel_dict["act2"] = ActTag[epi_res_block[2]]
                        suffix += 1

                        # sm80_code += SubstituteTemplate(cbr_kernel, kernel_dict)
                        kernel_str = (
                            cbr_header
                            + SubstituteTemplate(cbr_kernel, kernel_dict)
                            + CommonTail
                        )
                        file_name = (
                            "generated_tmp/"
                            + kernel_dict["kernel_func_name"]
                            + ".cu"
                        )
                        write_kernel_to_file(kernel_str, file_name)

                        all_kernel_names += (
                            kernel_dict["kernel_func_name"] + ", \n"
                        )
                        all_kernel_declares += (
                            "cutlass::Status "
                            + kernel_dict["kernel_func_name"]
                            + "(const ConvAllParams& params);"
                        )

        # Generate op code
        op_dict["kernel_func_declare"] = all_kernel_declares
        op_dict["all_kernel_func_name"] = all_kernel_names
        sm80_code += SubstituteTemplate(CommonConvFunction, op_dict)
    return sm80_code


if __name__ == "__main__":
    sm_versions_and_types = []
    args = parse_args()

    all_code = cbr_header
    if args.cuda_arch == "75":
        sm_versions_and_types.append(["75", "fp16"])
        all_code += generate_sm75_1688()
    if args.cuda_arch in ["80", "86", "89"]:
        sm_versions_and_types.append(["80", "fp16"])
        sm_versions_and_types.append(["80", "bf16"])
        all_code += generate_sm80_16816()
        all_code += generate_sm80_16816(cutlass_dtype="cutlass::bfloat16_t")

    all_code += GenerateFunctionForPhi(
        sm_versions_and_types, SupportedEpilogue, UnderScoreName, CamelName
    )
    all_code += CommonTail
    with open("generated_tmp/conv2d_bias_residual.cu", "w") as f:
        f.write(all_code)
        f.close()
