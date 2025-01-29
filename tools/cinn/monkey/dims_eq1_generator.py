from dataclasses import dataclass
import dag_generator as dag_generator
import dim_eq1_generator as dim_eq1_generator
from pick_weight import PickWeight
from typing import List
from hash_combine import HashCombine
from defensive_list import DList

@dataclass
class DimsEq1GenRequirement:
    dims_eq1_probability: List[float]

@dataclass
class DimsEq1GenInstruction:

    def AblateToTrivial(self):
        raise NotImplementedError()

    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(DimsEq1GenInstruction):

    def __hash__(self):
        return hash(id(Nope))

    def AblateToTrivial(self):
        return self

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return Nope()

@dataclass
class AddSourceTensor(DimsEq1GenInstruction):

    def AblateToTrivial(self):
        return self

    def __hash__(self):
        return hash(id(AddSourceTensor))
    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return AddSourceTensor()

@dataclass
class AddSinkTensor(DimsEq1GenInstruction):
    sink_tensor_dims_eq1: List[bool]

    def __hash__(self):
        hash_value = 0
        for dim_eq1 in self.sink_tensor_dims_eq1:
            hash_value = HashCombine(hash_value, hash(dim_eq1))
        return hash_value

    def AblateToTrivial(self):
        return AddSinkTensor(
            sink_tensor_dims_eq1=tuple(False for _ in self.sink_tensor_dims_eq1)
        )

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return AddSinkTensor(
            sink_tensor_dims_eq1=tuple(
                dim_eq1_gen_instruction.sink_tensor_dim_eq1
                for dim_eq1_gen_instruction in dim_eq1_gen_instructions
            )
        )


@dataclass
class AddUnaryOp(DimsEq1GenInstruction):
    source_tensor_dims_eq1: List[bool]

    def __hash__(self):
        hash_value = 0
        for dim_eq1 in self.source_tensor_dims_eq1:
            hash_value = HashCombine(hash_value, hash(dim_eq1))
        return hash_value

    def AblateToTrivial(self):
        return AddUnaryOp(
            source_tensor_dims_eq1=tuple(False for _ in self.source_tensor_dims_eq1)
        )

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return AddUnaryOp(
            source_tensor_dims_eq1=tuple(
                dim_eq1_gen_instance.source_tensor_dim_eq1
                for dim_eq1_gen_instance in dim_eq1_gen_instructions
            )
        )


@dataclass
class AddBinaryOp(DimsEq1GenInstruction):
    lhs_source_tensor_dims_eq1: List[bool]
    rhs_source_tensor_dims_eq1: List[bool]

    def __hash__(self):
        hash_value = 0
        for dim_eq1 in self.lhs_source_tensor_dims_eq1:
            hash_value = HashCombine(hash_value, hash(dim_eq1))
        for dim_eq1 in self.rhs_source_tensor_dims_eq1:
            hash_value = HashCombine(hash_value, hash(dim_eq1))
        return hash_value

    def AblateToTrivial(self):
        false_tuple = tuple(False for _ in self.rhs_source_tensor_dims_eq1)
        return AddBinaryOp(
            lhs_source_tensor_dims_eq1=false_tuple,
            rhs_source_tensor_dims_eq1=false_tuple,
        )

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return AddBinaryOp(
            lhs_source_tensor_dims_eq1=tuple(
                dim_eq1_gen_instruction.lhs_source_tensor_dim_eq1
                for dim_eq1_gen_instruction in dim_eq1_gen_instructions
            ),
            rhs_source_tensor_dims_eq1=tuple(
                dim_eq1_gen_instruction.rhs_source_tensor_dim_eq1
                for dim_eq1_gen_instruction in dim_eq1_gen_instructions
            )
        )


@dataclass
class AddBinaryClone(DimsEq1GenInstruction):

    def __hash__(self):
        return hash(id(AddBinaryClone))

    def AblateToTrivial(self):
        return AddBinaryClone()

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return AddBinaryClone()


@dataclass
class AddSourceOp(DimsEq1GenInstruction):

    def __hash__(self):
        return hash(id(AddSourceOp))

    def AblateToTrivial(self):
        return AddSourceOp()

    @classmethod
    def Merge(cls, dim_eq1_gen_instructions: List["DimEq1GenInstruction"]):
        return AddSourceOp()


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimsEq1Generator:
    def __init__(
        self,
        requirement: DimsEq1GenRequirement
    ):
        self.requirement = requirement
        def MakeDimEq1Generator(ratio):
            return type(self)._MakeDimEq1Generator(ratio)
        self.dim_eq1_generators = [
            MakeDimEq1Generator(ratio)
            for ratio in requirement.dims_eq1_probability
        ]

    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"]
    ) -> List["DimsEq1GenInstruction"]:
        dim_eq1_gen_instructions_tuple = [
            generator.Generate(dag_gen_instructions)
            for generator in self.dim_eq1_generators
        ]
        instructions = zip(
            dag_gen_instructions,
            zip(*dim_eq1_gen_instructions_tuple)
        )

        def MakeDimsEq1GenInstruction(dag_gen_instruction, dim_eq1_gen_instructions):
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            return cls.Merge(dim_eq1_gen_instructions)
        return [
            MakeDimsEq1GenInstruction(x, dim_eq1_gen_instructions)
            for x, dim_eq1_gen_instructions in instructions
        ]

    @classmethod
    def _MakeDimEq1Generator(
        cls,
        dim_eq1_probability: float
    ):
        pick_probability = dim_eq1_generator.DimEq1GenTypePickProbability(
            dim_eq1_probability=PickWeight(dim_eq1_probability)
        )
        requirement = dim_eq1_generator.DimEq1GenRequirement(
            pick_probability
        )
        return dim_eq1_generator.DimEq1Generator(requirement)
