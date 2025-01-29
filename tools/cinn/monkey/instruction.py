from dataclasses import dataclass
import dag_generator as dag_generator
import dims_eq1_generator as dims_eq1_generator
import op_name_generator as op_name_generator
import tensor_name_generator as tensor_name_generator
from typing import List, get_type_hints
from hash_combine import HashCombine
from instruction_id import InstructionId, MakeUniqueInstructionId
from collections import namedtuple


@dataclass
class Instruction:

    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]

    @classmethod
    def GetComponentBaseClasses(cls):
        return kComponentClasses


def GetHashValue(self):
    dag_gen_class = self.GetDAGGenClass()
    hash_value = 0
    for base_class in kComponentClasses:
        cls = base_class.GetDerivedClassByDAGGenClass(dag_gen_class)
        hash_value = HashCombine(hash_value, cls.__hash__(self))
    return hash_value

def GetComponent(self, base_class):
    dag_gen_class = self.GetDAGGenClass()
    cls = base_class.GetDerivedClassByDAGGenClass(dag_gen_class)
    params = {
        field_name: getattr(self, field_name)
        for field_name in get_type_hints(cls).keys()
    }
    ret = cls(**params)
    return ret

InstructionComponents = namedtuple(
    'InstructionComponents',
    (
        "dag_gen_instructions",
        "instruction_ids",
        "dims_eq1_gen_instructions",
        "op_name_gen_instructions",
        "tensor_name_gen_instructions",
    )
)

kComponentClasses = InstructionComponents(
    dag_gen_instructions=dag_generator.DAGGenInstruction,
    instruction_ids=InstructionId,
    dims_eq1_gen_instructions=dims_eq1_generator.DimsEq1GenInstruction,
    op_name_gen_instructions=op_name_generator.OpNameGenInstruction,
    tensor_name_gen_instructions=tensor_name_generator.TensorNameGenInstruction
)

def MergeClass(dag_gen_class, component_classes):
    class_name = dag_gen_class.__name__
    base_classes = (Instruction, )
    base_classes += tuple(
        component_class.GetDerivedClassByDAGGenClass(dag_gen_class)
        for component_class in component_classes
    )
    methods = dict(
        __hash__=GetHashValue,
        GetComponent=GetComponent,
        GetDAGGenClass=lambda self:dag_gen_class
    )
    return dataclass(type(class_name, base_classes, methods))

Nope = MergeClass(dag_generator.Nope, kComponentClasses)
AddSourceTensor = MergeClass(dag_generator.AddSourceTensor, kComponentClasses)
AddSinkTensor = MergeClass(dag_generator.AddSinkTensor, kComponentClasses)
AddUnaryOp = MergeClass(dag_generator.AddUnaryOp, kComponentClasses)
AddBinaryOp = MergeClass(dag_generator.AddBinaryOp, kComponentClasses)
AddBinaryClone = MergeClass(dag_generator.AddBinaryClone, kComponentClasses)
AddSourceOp = MergeClass(dag_generator.AddSourceOp, kComponentClasses)


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class IrGenerator:
    def __init__(self):
        pass

    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        instruction_ids: List[InstructionId],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"],
        op_name_gen_instructions: List["OpNameGenInstruction"],
        tensor_name_gen_instructions: List["TensorNameGenInstruction"]
    ) -> List["Instruction"]:
        def CreateIrGenInstruction(dag_gen_instr, *other_instr):
            params = {}
            for instr in [dag_gen_instr, *other_instr]:
                params.update(vars(instr))
            ir_gen_class = kDAGGenClassToDerivedClass[type(dag_gen_instr)]
            return ir_gen_class(**params)
        return [
            CreateIrGenInstruction(*triple)
            for triple in zip(
                dag_gen_instructions,
                instruction_ids,
                dims_eq1_gen_instructions,
                op_name_gen_instructions,
                tensor_name_gen_instructions
            )
        ]
