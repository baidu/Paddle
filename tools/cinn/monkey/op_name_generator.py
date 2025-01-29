from dataclasses import dataclass, field
import dag_generator as dag_generator
from defensive_list import DList
from pick_weight import PickWeight
from typing import List, Set
import random

@dataclass
class OpNameGenRequirement:
    unary_elementwise_op_names: List[str] = field(
        default_factory=lambda: ["Negative"]
    )
    binary_elementwise_op_names: List[str] = field(
        default_factory=lambda: ["Add"]
    )
    broadcast_op_names: List[str] = field(
        default_factory=lambda: ["Expand"]
    )
    reduce_op_names: List[str] = field(
        default_factory=lambda: ["ReduceSum"]
    )
    reshape_op_names: List[str] = field(
        default_factory=lambda: ["Reshape"]
    )
    permute_op_names: List[str] = field(
        default_factory=lambda: ["Transpose"]
    )
    source_op_names: List[str] = field(
        default_factory=lambda: ["Zeros", "Ones"]
    )

@dataclass
class OpNameGenInstruction:
    def AblateToTrivial(self):
        return self

    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(OpNameGenInstruction):

    def __hash__(self):
        return hash(id(Nope))

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        return Nope()


@dataclass
class AddSourceTensor(OpNameGenInstruction):

    def __hash__(self):
        return hash(id(AddSourceTensor))

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        return AddSourceTensor()


@dataclass
class AddSinkTensor(OpNameGenInstruction):

    def __hash__(self):
        return hash(id(AddSinkTensor))

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        return AddSinkTensor()


@dataclass
class AddUnaryOp(OpNameGenInstruction):
    op_name: str

    def __hash__(self):
        return hash(self.op_name)

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        convert_type = dag_gen_instruction.convert_type
        method_name = "MakeRandomInstance_" + type(convert_type).__name__
        return getattr(cls, method_name)(requirement, dag_gen_instruction)

    @classmethod
    def MakeRandomInstance_NoConvertType(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction"
    ) -> "OpNameGenInstruction":
        return AddUnaryOp(
            op_name=_GetRandomUnaryOpName(requirement)
        )

    @classmethod
    def MakeRandomInstance_ReduceConvertType(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction"
    ) -> "OpNameGenInstruction":
        return AddUnaryOp(
            op_name=_GetRandomReduceOpName(requirement)
        )

    @classmethod
    def MakeRandomInstance_BroadcastConvertType(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction"
    ) -> "OpNameGenInstruction":
        return AddUnaryOp(
            op_name=_GetRandomBroadcastOpName(requirement)
        )

    @classmethod
    def MakeRandomInstance_UnclassifiedConvertType(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction"
    ) -> "OpNameGenInstruction":
        raise NotImplementedError("UnclassifiedConvertType not supported.")


@dataclass
class AddBinaryOp(OpNameGenInstruction):
    op_name: str

    def __hash__(self):
        return hash(self.op_name)

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        return AddBinaryOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class AddBinaryClone(OpNameGenInstruction):

    def __hash__(self):
        return hash(id(AddBinaryClone))

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        return AddBinaryClone()


@dataclass
class AddSourceOp(OpNameGenInstruction):
    op_name: str

    def __hash__(self):
        return hash(self.op_name)

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_gen_instruction: "DAGGenInstruction",
    ) -> "OpNameGenInstruction":
        return AddSourceOp(op_name=_GetRandomSourceOpName(requirement))


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

def _GetRandomReduceOpName(requirement: OpNameGenRequirement):
    return _GetRandomOpName(requirement.reduce_op_names)

def _GetRandomBroadcastOpName(requirement: OpNameGenRequirement):
    return _GetRandomOpName(requirement.broadcast_op_names)

def _GetRandomUnaryOpName(requirement: OpNameGenRequirement):
    return _GetRandomOpName(requirement.unary_elementwise_op_names)

def _GetRandomBinaryOpName(requirement: OpNameGenRequirement):
    return _GetRandomOpName(requirement.binary_elementwise_op_names)

def _GetRandomSourceOpName(requirement: OpNameGenRequirement):
    return _GetRandomOpName(requirement.source_op_names)

def _GetRandomOpName(op_names: List[str]):
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randint(0, kRangeMax)
    return op_names[random_int]

class OpNameGenerator:
    def __init__(self, requirement: OpNameGenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"]
    ) -> List["OpNameGenInstruction"]:
        def CreateOpNameGenInstruction(dag_gen_instruction):
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            ret = cls.MakeRandomInstance(
                self.requirement,
                dag_gen_instruction
            )
            return ret
        return [
            CreateOpNameGenInstruction(x)
            for x in dag_gen_instructions
        ]
