from dataclasses import dataclass, field
from collections import namedtuple
from typing import Union, List, Optional
from defensive_list import DList
import dag_generator as dag_generator
import dim_eq1_generator as dim_eq1_generator
import dims_eq1_generator as dims_eq1_generator
import op_name_generator as op_name_generator
import tensor_name_generator as tensor_name_generator
import shape_signature_inferer as shape_signature_inferer
import instruction_util as instruction_util
import op_call_code_gen as op_call_code_gen
from tensor_name_generator import TensorNameGenRequirement

@dataclass
class UnitTestCaseRequirement:
    dag_gen_requirement: dag_generator.DAGGenRequirement
    dims_eq1_gen_requirement: dims_eq1_generator.DimsEq1GenRequirement
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement
    dim_size_requirement: shape_signature_inferer.DimSizeRequirement
    tensor_name_gen_requirement: tensor_name_generator.TensorNameGenRequirement
    patch_tensor_name_gen_requirement: Optional[TensorNameGenRequirement] = None


@dataclass
class UnitTestCaseSpec:
    instructions: List["Instruction"]
    patched_instruction_code_gen_spec: DList["Instruction", "CodeGenSpec"]

def GenerateRandomUnitTestCaseSpec(
    requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    instructions = instruction_util.GenerateInstructions(
        dag_gen_requirement=requirement.dag_gen_requirement,
        dims_eq1_gen_requirement=requirement.dims_eq1_gen_requirement,
        op_name_gen_requirement=requirement.op_name_gen_requirement,
        tensor_name_gen_requirement=requirement.tensor_name_gen_requirement,
    )
    return CompleteToUnitTestCaseSpec(instructions, requirement)

def GetAblatedUnitTestCaseSpec(
    instructions: List["Instruction"],
    requirement: UnitTestCaseRequirement,
    bottom_up_ablation_size: int,
    component_ablation_size: int,
) -> UnitTestCaseSpec:
    ablated_instructions = instruction_util.AblateInstructions(
        instructions=instructions,
        bottom_up_ablation_size=bottom_up_ablation_size,
        component_ablation_size=component_ablation_size
    )
    return CompleteToUnitTestCaseSpec(ablated_instructions, requirement)

def CompleteToUnitTestCaseSpec(
    instructions: List["Instruction"],
    requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    patch_tensor_name_gen_requirement = (
      requirement.patch_tensor_name_gen_requirement
      if requirement.patch_tensor_name_gen_requirement is not None
      else requirement.tensor_name_gen_requirement
    )
    patched_instructions = instruction_util.PatchInstructions(
        instructions=instructions,
        op_name_gen_requirement=requirement.op_name_gen_requirement,
        tensor_name_gen_requirement=patch_tensor_name_gen_requirement
    )
    patched_instruction_code_gen_spec = instruction_util.InferCodeGenSpecs(
        instructions=patched_instructions,
        dim_size_requirement=requirement.dim_size_requirement
    )
    return UnitTestCaseSpec(
        instructions=instructions,
        patched_instruction_code_gen_spec=patched_instruction_code_gen_spec
    )
