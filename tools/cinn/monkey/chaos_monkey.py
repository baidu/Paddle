from dataclasses import dataclass
from collections import namedtuple
from typing import Generator, Union
from defensive_list import DList
from dag_generator import PickWeight
from script import Script
import dag_generator
import dim_eq1_generator as dim_eq1_generator
import dims_eq1_generator as dims_eq1_generator
import op_name_generator as op_name_generator
import tensor_name_generator as tensor_name_generator
from tensor_name_generator import TensorNameGenRequirement
import shape_signature_inferer as shape_signature_inferer
from shape_signature_inferer import StaticDim
import instruction_util as instruction_util
import op_call_code_gen
from paddle_eager_generator import PaddleEagerGenerator
from numpy_generator import NumpyGenerator
from unit_test_case_spec import (
    UnitTestCaseRequirement,
    UnitTestCaseSpec,
    GenerateRandomUnitTestCaseSpec,
    GetAblatedUnitTestCaseSpec
)

def GenerateUnitTestCaseSpec(
    unit_test_case_requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    return GenerateRandomUnitTestCaseSpec(
        requirement=unit_test_case_requirement
    )

if __name__ == '__main__':
    unit_test_case_requirement=UnitTestCaseRequirement(
        dag_gen_requirement=dag_generator.DAGGenRequirement(
            min_num_sources=0,
            max_num_sources=0,
            pick_probability=dag_generator.DAGGenTypePickProbability()
        ),
        dims_eq1_gen_requirement=dims_eq1_generator.DimsEq1GenRequirement(
            dims_eq1_probability=[0.1, 0.15, 0.2]
        ),
        op_name_gen_requirement=op_name_generator.OpNameGenRequirement(),
        tensor_name_gen_requirement=TensorNameGenRequirement(),
        dim_size_requirement=shape_signature_inferer.DimSizeRequirement(
            dim_size=[StaticDim(128), StaticDim(64), StaticDim(32)]
        )
    )
    unit_test_case_spec = GenerateUnitTestCaseSpec(
        unit_test_case_requirement=unit_test_case_requirement
    )

    numpy_gen = NumpyGenerator()

    paddle_eager_gen = PaddleEagerGenerator()
    script = numpy_gen.Generate(unit_test_case_spec.patched_instruction_code_gen_spec)
    print("import numpy")
    print(script.file_content)

#    ablated_unit_test_case_spec = GetAblatedUnitTestCaseSpec(
#        instructions=unit_test_case_spec.instructions,
#        requirement=unit_test_case_requirement,
#        bottom_up_ablation_size=-1,
#        component_ablation_size=-1,
#    )
#    print("#", "*"*80)
#    print("# full ablated")
#    print("#", "*"*80)
#    script = numpy_gen.Generate(ablated_unit_test_case_spec.patched_instruction_code_gen_spec)
#    print("import paddle")
#    print(script.file_content)
#
#
#    ablated_unit_test_case_spec = GetAblatedUnitTestCaseSpec(
#        instructions=unit_test_case_spec.instructions,
#        requirement=unit_test_case_requirement,
#        bottom_up_ablation_size=len(unit_test_case_spec.instructions)/2,
#        component_ablation_size=-1,
#    )
#    print("#", "*"*80)
#    print("# half ablated")
#    print("#", "*"*80)
#    script = numpy_gen.Generate(ablated_unit_test_case_spec.patched_instruction_code_gen_spec)
#    print("import paddle")
#    print(script.file_content)
