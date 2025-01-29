from typing import List
import dag_generator as dag_generator
import dim_eq1_generator as dim_eq1_generator
import dims_eq1_generator as dims_eq1_generator
import op_name_generator as op_name_generator
import tensor_name_generator as tensor_name_generator
import code_gen_spec_inferer as code_gen_spec_inferer
import dims_eq1_signature_inferer as dims_eq1_signature_inferer
from defensive_list import DList
from instruction import IrGenerator, Instruction, InstructionComponents
from instruction_id import MakeUniqueInstructionId,InstructionId
from dag_dims_eq1_patcher import DAGDimsEq1Patcher
from op_name_patcher import OpNamePatcher
from tensor_name_patcher import TensorNamePatcher
from shape_signature_inferer import ShapeSignatureInferer
from tensor_name_signature_inferer import TensorNameSignatureInferer

def GenerateInstructions(
    dag_gen_requirement: dag_generator.DAGGenRequirement,
    dims_eq1_gen_requirement: dims_eq1_generator.DimsEq1GenRequirement,
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement,
    tensor_name_gen_requirement: tensor_name_generator.TensorNameGenRequirement
) -> List["Instruction"]:
    # generators
    dag_gen = dag_generator.DAGGenerator(requirement=dag_gen_requirement)
    dims_eq1_gen = dims_eq1_generator.DimsEq1Generator(
        requirement=dims_eq1_gen_requirement
    )
    op_name_gen = op_name_generator.OpNameGenerator(
        requirement=op_name_gen_requirement
    )
    tensor_name_gen = tensor_name_generator.TensorNameGenerator(
        requirement=tensor_name_gen_requirement
    )
    ir_gen = IrGenerator()
    # inferers
    dims_eq1_sig_inferer = dims_eq1_signature_inferer.DimsEq1SignatureInferer()
    # instructions
    dag_gen_instructions = dag_gen.Generate()
    instruction_ids = [
        MakeUniqueInstructionId() for _ in range(len(dag_gen_instructions))
    ]
    dims_eq1_gen_instructions = dims_eq1_gen.Generate(dag_gen_instructions)
    op_name_gen_instructions = op_name_gen.Generate(dag_gen_instructions)
    tensor_name_gen_instructions = tensor_name_gen.Generate(dag_gen_instructions)
    return ir_gen.Generate(
        dag_gen_instructions=dag_gen_instructions,
        instruction_ids=instruction_ids,
        dims_eq1_gen_instructions=dims_eq1_gen_instructions,
        op_name_gen_instructions=op_name_gen_instructions,
        tensor_name_gen_instructions=tensor_name_gen_instructions
    )


def PatchInstructions(
    instructions: List["Instruction"],
    op_name_gen_requirement: op_name_generator.OpNameGenRequirement,
    tensor_name_gen_requirement: tensor_name_generator.TensorNameGenRequirement
) -> List["Instruction"]:
    def GetComponentList(base_cls):
        return [
            instruction.GetComponent(base_cls)
            for instruction in instructions
        ]
    dag_gen_instructions = GetComponentList(dag_generator.DAGGenInstruction)

    instruction_ids = GetComponentList(InstructionId)
    dims_eq1_instructions = GetComponentList(dims_eq1_generator.DimsEq1GenInstruction)
    op_name_instructions = GetComponentList(op_name_generator.OpNameGenInstruction)
    tensor_name_instructions = GetComponentList(
        tensor_name_generator.TensorNameGenInstruction
    )
    # patch dag and dims_eq1
    PatchDAGDimsEq1 = DAGDimsEq1Patcher().Patch
    new_dag_gen_instrs, new_instr_ids, new_dims_eq1_instrs = PatchDAGDimsEq1(
        dag_gen_instructions=dag_gen_instructions,
        instruction_ids=instruction_ids,
        dims_eq1_gen_instructions=dims_eq1_instructions
    )
    # patch op name
    instruction_id2existed_op_name = {
        instruction_id:existed_op_name_instr
        for instruction_id, existed_op_name_instr in zip(
            instruction_ids, op_name_instructions
        )
    }
    PatchOpName = OpNamePatcher(op_name_gen_requirement).Patch
    new_op_name_instrs = PatchOpName(
        dag_gen_instructions=new_dag_gen_instrs,
        instruction_ids=new_instr_ids,
        instruction_id2existed_op_name=instruction_id2existed_op_name
    )
    # patch tensor name
    instruction_id2existed_tensor_name = {
        instruction_id:existed_tensor_name_instr
        for instruction_id, existed_tensor_name_instr in zip(
            instruction_ids, tensor_name_instructions
        )
    }
    PatchTensorName = TensorNamePatcher(tensor_name_gen_requirement).Patch
    new_tensor_name_instrs = PatchTensorName(
        dag_gen_instructions=new_dag_gen_instrs,
        instruction_ids=new_instr_ids,
        instruction_id2existed_tensor_name=instruction_id2existed_tensor_name
    )
    GenerateInstructions = IrGenerator().Generate
    return GenerateInstructions(
        dag_gen_instructions=new_dag_gen_instrs,
        instruction_ids=new_instr_ids,
        dims_eq1_gen_instructions=new_dims_eq1_instrs,
        op_name_gen_instructions=new_op_name_instrs,
        tensor_name_gen_instructions=new_tensor_name_instrs
    )

def InferCodeGenSpecs(
    instructions: List["Instruction"],
    dim_size_requirement: "DimSizeRequirement",
) -> DList["Instruction", "CodeGenSpec"]:
    def GetComponentList(base_cls):
        return [
            instruction.GetComponent(base_cls)
            for instruction in instructions
        ]
    dag_gen_instructions = GetComponentList(dag_generator.DAGGenInstruction)
    instruction_ids = GetComponentList(InstructionId)
    dims_eq1_instructions = GetComponentList(dims_eq1_generator.DimsEq1GenInstruction)
    tensor_name_gen_instructions = GetComponentList(
        tensor_name_generator.TensorNameGenInstruction
    )
    InferDimsEq1Signature = dims_eq1_signature_inferer.DimsEq1SignatureInferer().Infer
    guarded_dims_eq1_sigs = InferDimsEq1Signature(
        dag_gen_instructions=dag_gen_instructions,
        dims_eq1_gen_instructions=dims_eq1_instructions,
    )
    dims_eq1_signatures = [v for k,v in guarded_dims_eq1_sigs.Unguard()]
    InferShapeSignature = ShapeSignatureInferer(dim_size_requirement).Infer
    guarded_shape_signatures = InferShapeSignature(
        dag_gen_instructions=dag_gen_instructions,
        dims_eq1_signatures=dims_eq1_signatures
    )
    InferTensorNameSignature = TensorNameSignatureInferer().Infer
    guarded_tensor_name_signatures = InferTensorNameSignature(
        dag_gen_instructions=dag_gen_instructions,
        tensor_name_gen_instructions=tensor_name_gen_instructions
    )
    shape_signatures = [v for k,v in guarded_shape_signatures.Unguard()]
    tensor_name_signatures = [v for k,v in guarded_tensor_name_signatures.Unguard()]
    InferCodeGenSpec = code_gen_spec_inferer.CodeGenSpecInferer().Infer
    code_gen_specs = InferCodeGenSpec(
        dag_gen_instructions=dag_gen_instructions,
        dims_eq1_signatures=dims_eq1_signatures,
        shape_signatures=shape_signatures,
        tensor_name_signatures=tensor_name_signatures
    )
    return DList(instructions, code_gen_specs)

def AblateInstructions(
    instructions: List["Instruction"],
    bottom_up_ablation_size: int,
    component_ablation_size: int
) -> List["Instruction"]:
    ComponentClasses = Instruction.GetComponentBaseClasses()
    def IsValidBottomUpAblationSize(size):
        return size >= 0 and size <= len(instructions)
    def IsValidComponentAblationSize(size):
        return size >= 0 and size <= len(ComponentClasses)
    def NeedAblation(instruction_idx, component_id):
        if instruction_idx + 1 + bottom_up_ablation_size > len(instructions):
            return True
        if instruction_idx + 1 + bottom_up_ablation_size < len(instructions):
            return False
        return component_id + 1 + component_ablation_size > len(ComponentClasses)
    def GetComponentList(base_cls):
        return [instruction.GetComponent(base_cls) for instruction in instructions]

    if not IsValidBottomUpAblationSize(bottom_up_ablation_size):
        bottom_up_ablation_size = len(instructions)
    if not IsValidComponentAblationSize(component_ablation_size):
        component_ablation_size = len(ComponentClasses)
    instructions_compoments = tuple(
        GetComponentList(base_cls)
        for base_cls in ComponentClasses
    )
    ablated_instructions_compoments = tuple(
        [
            (
                instruction.AblateToTrivial()
                if NeedAblation(instruction_idx, component_id)
                else instruction
            )
            for instruction_idx, instruction in enumerate(component)
        ]
        for component_id, component in enumerate(instructions_compoments)
    )
    GenerateInstructions = IrGenerator().Generate
    return GenerateInstructions(
        **InstructionComponents(*ablated_instructions_compoments)._asdict()
    )
