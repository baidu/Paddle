from dataclasses import dataclass, field
import dag_generator as dag_generator
from defensive_list import DList
from pick_weight import PickWeight
from typing import List, Set, Dict, Optional
from hash_combine import HashCombine
import random

@dataclass
class TensorNameGenRequirement:
    tensor_name_prefix: str = "tensor"

class TensorSeqCounter:
    def __init__(self):
        self.prefix2counter: Dict[str, int] = {}

    def AutoIncrementalCounter(self, prefix: str):
        if prefix not in self.prefix2counter:
            self.prefix2counter[prefix] = 0
        self.prefix2counter[prefix] += 1
        return self.prefix2counter[prefix]

@dataclass
class SourceTensorNames:
    names: List[str] = field(
        default_factory=list
    )

@dataclass
class TensorNameGenContext:
    requirement: TensorNameGenRequirement
    dag_gen_instruction: "DAGGenInstruction"
    tensor_seq_counter: TensorSeqCounter

    def NewTensorName(self, prefix = None):
        prefix = (
            prefix if prefix is not None else self.requirement.tensor_name_prefix
        )
        seq_no = self.tensor_seq_counter.AutoIncrementalCounter(prefix)
        return "%s%d" % (prefix, seq_no)


@dataclass
class TensorNameGenInstruction:
    def AblateToTrivial(self):
        return self

    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(TensorNameGenInstruction):

    def __hash__(self):
        return hash(id(Nope))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return Nope()

@dataclass
class AddSourceTensor(TensorNameGenInstruction):
    source_tensor_name: str

    def __hash__(self):
        return hash(id(self.source_tensor_name))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddSourceTensor(
            source_tensor_name=ctx.NewTensorName()
        )


@dataclass
class AddSinkTensor(TensorNameGenInstruction):

    def __hash__(self):
        return hash(id(AddSinkTensor))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddSinkTensor()


@dataclass
class AddUnaryOp(TensorNameGenInstruction):
    output_tensor_name: str

    def __hash__(self):
        return hash(self.output_tensor_name)

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        name_prefix = cls.GetNamePrefix(ctx.dag_gen_instruction.convert_type)
        return AddUnaryOp(
            output_tensor_name=ctx.NewTensorName(name_prefix)
        )

    @classmethod
    def GetNamePrefix(cls, convert_type) -> Optional[str]:
        method_name = "GetNamePrefix_%s" % (type(convert_type).__name__)
        return getattr(cls, method_name)()

    @classmethod
    def GetNamePrefix_NoConvertType(cls):
        return None

    @classmethod
    def GetNamePrefix_ReduceConvertType(cls):
        return "reduced"

    @classmethod
    def GetNamePrefix_BroadcastConvertType(cls):
        return "expanded"

    @classmethod
    def GetNamePrefix_UnclassifiedConvertType(cls):
        return None


@dataclass
class AddBinaryOp(TensorNameGenInstruction):
    output_tensor_name: str

    def __hash__(self):
        return hash(self.output_tensor_name)

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddBinaryOp(
            output_tensor_name=ctx.NewTensorName(),
        )


@dataclass
class AddBinaryClone(TensorNameGenInstruction):

    def __hash__(self):
        return hash(id(AddBinaryClone))

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddBinaryClone(
        )


@dataclass
class AddSourceOp(TensorNameGenInstruction):
    output_tensor_name: str

    def __hash__(self):
        return hash(self.output_tensor_name)

    @classmethod
    def GenerateTensorNames(
        cls,
        ctx: TensorNameGenContext
    ) -> TensorNameGenInstruction:
        return AddSourceOp(
            output_tensor_name=ctx.NewTensorName()
        )


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class TensorNameGenerator:
    def __init__(self, requirement: TensorNameGenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_gen_instructions: List["DAGGenInstruction"]
    ) -> List["TensorNameGenInstruction"]:
        def CreateTensorNameGenInstruction(dag_gen_instruction):
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            return cls.GenerateTensorNames(
                TensorNameGenContext(
                    requirement=self.requirement,
                    dag_gen_instruction=dag_gen_instruction,
                    tensor_seq_counter=_global_tensor_seq_counter
                )
            )
        return list(reversed([
            CreateTensorNameGenInstruction(x)
            for x in reversed(dag_gen_instructions)
        ]))

_global_tensor_seq_counter = TensorSeqCounter()