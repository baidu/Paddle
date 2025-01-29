from dataclasses import dataclass, field
from pick_weight import PickWeight
from typing import List
from collections import namedtuple
import functools
import random
from hash_combine import HashCombine

@dataclass
class DAGGenTypePickProbability:
    nope: PickWeight = field(
        default_factory=lambda: PickWeight(0)
    )
    add_sink_tensor: PickWeight = field(
        default_factory=lambda: PickWeight(0.1)
    )
    add_unary_op: PickWeight = field(
        default_factory=lambda: PickWeight(1)
    )
    add_binary_op: PickWeight = field(
        default_factory=lambda: PickWeight(1)
    )
    add_binary_clone: PickWeight = field(
        default_factory=lambda: PickWeight(0.5)
    )
    add_source_op: PickWeight = field(
        default_factory=lambda: PickWeight(0.5)
    )

@dataclass
class DAGGenRequirement:
    min_num_sources: int = 1
    max_num_sources: int = 1
    min_num_sinks: int = 1
    max_num_sinks: int = 1
    min_width: int = 1
    max_width: int = 10
    max_body_instructions: int = 10
    pick_probability: DAGGenTypePickProbability = field(
        default_factory=lambda: DAGGenTypePickProbability()
    )

    def CheckFields(self):
        assert self.min_num_sources >= 0
        assert self.max_num_sources >= self.min_num_sources
        assert self.min_num_sinks > 0
        assert self.max_num_sinks >= self.min_num_sinks
        assert self.min_width > 0
        assert self.max_width >= self.min_width

@dataclass
class DAGGenContext:
    num_source_tensors: int
    num_sink_tensors: int

@dataclass
class DAGGenInstruction:

    def AblateToTrivial(self):
        return self

    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return dag_gen_class

@dataclass
class Nope(DAGGenInstruction):

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        return int(id(Nope))

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return True
    
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 0

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        return Nope()

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.nope.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return True

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return True

@dataclass
class AddSourceTensor(DAGGenInstruction):

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        return int(id(AddSourceTensor))

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 0

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return True
    
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        return AddSourceTensor(
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        raise NotImplementedError("Dead code")

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        raise NotImplementedError("Dead code")

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        raise NotImplementedError("Dead code")

@dataclass
class AddSinkTensor(DAGGenInstruction):

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        return int(id(AddSinkTensor))

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 1

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return True
    
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 1

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        return AddSinkTensor(
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_sink_tensor.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_source_tensors <= requirement.max_width

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_sink_tensors <= requirement.max_num_sinks


@dataclass
class ConvertType:
    pass

@dataclass
class NoConvertType(ConvertType):
    pass

@dataclass
class ReduceConvertType(ConvertType):
    pass

@dataclass
class BroadcastConvertType(ConvertType):
    pass

@dataclass
class UnclassifiedConvertType(ConvertType):
    pass


@dataclass
class AddUnaryOp(DAGGenInstruction):
    source_tensor_index: int
    convert_type: ConvertType

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        hash_value = int(id(AddUnaryOp))
        hash_value = HashCombine(hash_value, hash(self.source_tensor_index))
        return hash_value

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return self.source_tensor_index < ctx.num_source_tensors
        
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 0

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        source_tensor_index = random.randint(0, ctx.num_source_tensors - 1)
        return AddUnaryOp(
            source_tensor_index=source_tensor_index,
            convert_type=NoConvertType()
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_unary_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_source_tensors > 0

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_sink_tensors >= requirement.min_num_sinks


@dataclass
class AddBinaryOp(DAGGenInstruction):
    source_tensor_index: int

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        hash_value = int(id(AddBinaryOp))
        hash_value = HashCombine(hash_value, hash(self.source_tensor_index))
        return hash_value

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return self.source_tensor_index < ctx.num_source_tensors
        
    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return 1

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        source_tensor_index = random.randint(0, ctx.num_source_tensors - 1)
        return AddBinaryOp(
            source_tensor_index=source_tensor_index,
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_binary_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_source_tensors <= requirement.max_width

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_sink_tensors >= requirement.min_num_sinks


@dataclass
class AddBinaryClone(DAGGenInstruction):
    lhs_source_tensor_index: int
    rhs_source_tensor_index: int

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        hash_value = int(id(AddBinaryClone))
        hash_value = HashCombine(hash_value, hash(self.lhs_source_tensor_index))
        hash_value = HashCombine(hash_value, hash(self.rhs_source_tensor_index))
        return hash_value

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return (self.lhs_source_tensor_index >= 0
            and self.lhs_source_tensor_index < ctx.num_source_tensors
            and self.rhs_source_tensor_index >= 0
            and self.rhs_source_tensor_index < ctx.num_source_tensors
            and self.lhs_source_tensor_index != self.rhs_source_tensor_index
        )

    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return -1

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        lhs_random_int = random.randint(
            0, ctx.num_source_tensors - 1
        )
        rhs_random_int = random.randint(
            0,
            ctx.num_source_tensors - 1
        )
        if lhs_random_int > rhs_random_int:
            lhs_random_int, rhs_random_int = (rhs_random_int, lhs_random_int)
        return AddBinaryClone(
            lhs_source_tensor_index=lhs_random_int,
            rhs_source_tensor_index=rhs_random_int,
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_binary_clone.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_source_tensors > 1

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_sink_tensors >= requirement.min_num_sinks


@dataclass
class AddSourceOp(DAGGenInstruction):
    source_tensor_index: int

    def __hash__(self):
        return self.GetHashValue()

    def GetHashValue(self):
        hash_value = int(id(AddBinaryClone))
        hash_value = HashCombine(hash_value, hash(self.source_tensor_index))
        return hash_value

    def IsValidSourceTensorIndex(
        self,
        ctx: DAGGenContext
    ) -> bool:
        return self.source_tensor_index < ctx.num_source_tensors

    @classmethod
    def GetDeltaNumSourceTensors(cls):
        return -1

    @classmethod
    def GetDeltaNumSinkTensors(cls):
        return 0

    @classmethod
    def RandomGenerate(
        cls,
        requirement: DAGGenRequirement,
        ctx: DAGGenContext
    ):
        source_tensor_index = random.randint(
            0,
            ctx.num_source_tensors - 1
        )
        return AddSourceOp(
            source_tensor_index=source_tensor_index,
        )

    @classmethod
    def GetWeight(
        cls,
        pick_probability: DAGGenTypePickProbability
    ) -> float:
        return pick_probability.add_source_op.weight

    @classmethod
    def IsValidNumSourceTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_source_tensors > requirement.min_width

    @classmethod
    def IsValidNumSinkTensors(
        cls,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> bool:
        return ctx.num_sink_tensors >= requirement.min_num_sinks


kDAGRandomReversedGenInstructionClasses = [
    Nope,
    # AddSourceTensor is not rolled in random reversed generating
    AddSinkTensor,
    AddUnaryOp,
    AddBinaryOp,
    AddBinaryClone,
    AddSourceOp,
]

def GetTotalWeight(
    pick_probability: DAGGenTypePickProbability,
    dag_gen_classes: List[type]
) -> float:
    def GetWeight(dag_gen_class):
        return dag_gen_class.GetWeight(pick_probability)
    return functools.reduce(
        lambda a, b: a + b,
        map(GetWeight, dag_gen_classes)
    )

DAGGenClassRollingRange = namedtuple(
    "DAGGenClassRollingRange",
    ["start", "end", "dag_gen_class"]
)

class DAGGenClassGenerator:
    def __init__(
        self,
        pick_probability: DAGGenTypePickProbability
    ):
        self.pick_probability = pick_probability
    
    def GetRandomDAGGenClass(
        self,
        ctx: DAGGenContext,
        requirement: DAGGenRequirement
    ) -> type:
        def IsValidDAGGenClass(dag_gen_class):
            return (
                dag_gen_class.IsValidNumSourceTensors(ctx, requirement)
                and dag_gen_class.IsValidNumSinkTensors(ctx, requirement)
            )
        rolling_ranges = type(self)._MakeRollingRange(
            self.pick_probability,
            [
                x for x in kDAGRandomReversedGenInstructionClasses
                if IsValidDAGGenClass(x)
            ]
        )
        def Roll():
            random_int = random.randint(0, type(self)._RollingLimit())
            for start, end, dag_gen_class in rolling_ranges:
                if random_int >= start and random_int < end:
                    return dag_gen_class
            return None
        kTryCnt = 10
        for i in range(kTryCnt):
            cls = Roll()
            if cls is not None:
                return cls
        return Nope()

    @classmethod
    def _MakeRollingRange(
        cls,
        pick_probability: DAGGenTypePickProbability,
        dag_gen_classes: List[type]
    ) -> List[DAGGenClassRollingRange]:
        total_weight = GetTotalWeight(pick_probability, dag_gen_classes)
        start = 0
        def GetRange(dag_gen_class):
            nonlocal start
            current_start = start
            current_end = current_start
            weight = dag_gen_class.GetWeight(pick_probability)
            current_end += weight * cls._RollingLimit() / total_weight
            start = current_end
            return DAGGenClassRollingRange(
                start=current_start,
                end=current_end,
                dag_gen_class=dag_gen_class,
            )
        return [GetRange(cls) for cls in dag_gen_classes]

    @classmethod
    def _RollingLimit(cls):
        return 10000


class MutDAGGenInstructions:
    def __init__(self):
        self.instructions = []
        self.current_num_source_tensors = 0
        self.current_num_sink_tensors = 0

    def Push(
        self,
        instruction: "DAGGenInstruction"
    ):
        self.instructions.append(instruction)
        top = instruction
        self.current_num_source_tensors += type(top).GetDeltaNumSourceTensors()
        self.current_num_sink_tensors += type(top).GetDeltaNumSinkTensors()

class BodyAndHeaderDAGGenerator:
    def __init__(
        self,
        requirement: DAGGenRequirement
    ):
        self.requirement = requirement
        self.result_dag_gen_instructions = MutDAGGenInstructions()
        self.dag_gen_class_generator = DAGGenClassGenerator(
            requirement.pick_probability
        )

    def GenerateBodyAndHeader(self) -> List["DAGGenInstruction"]:
        self.GenerateBody()
        self.GenerateHeader()
        return self.result_dag_gen_instructions.instructions

    def GenerateBody(self):
        def MakeInstruction(ctx: DAGGenContext):
            return self._MakeRandomInstruction(ctx)
        for i in range(self.requirement.max_body_instructions):
            self._GenerateOneInstruction(MakeInstruction)

    def GenerateHeader(self):
        CheckDeadLoop = DeadLoopChecker()
        def CurrentNumSources():
            return self.result_dag_gen_instructions.current_num_source_tensors
        def IncreaseCurrentNumSourcesByAddBinaryOp(ctx):
            return AddBinaryOp.RandomGenerate(self.requirement, ctx)
        while CurrentNumSources() < self.requirement.min_num_sources:
            self._GenerateOneInstruction(IncreaseCurrentNumSourcesByAddBinaryOp)
            CheckDeadLoop()
        def DecreaseCurrentNumSourcesByAddBinaryClone(ctx):
            if CurrentNumSources() == 1:
                return AddSourceOp.RandomGenerate(self.requirement, ctx)
            return AddBinaryClone.RandomGenerate(self.requirement, ctx)
        while CurrentNumSources() > self.requirement.max_num_sources:
            self._GenerateOneInstruction(DecreaseCurrentNumSourcesByAddBinaryClone)
            CheckDeadLoop()
        def PrependInstructionAddSourceTensor(ctx):
            return AddSourceTensor.RandomGenerate(self.requirement, ctx)
        for i in range(CurrentNumSources()):
            self._GenerateOneInstruction(PrependInstructionAddSourceTensor)


    def _GenerateOneInstruction(self, NewInstruction):
        ctx = DAGGenContext(
            num_source_tensors = (
                self.result_dag_gen_instructions.current_num_source_tensors
            ),
            num_sink_tensors = (
                self.result_dag_gen_instructions.current_num_sink_tensors
            )
        )
        new_instruction = NewInstruction(ctx)
        is_valid = new_instruction.IsValidSourceTensorIndex(ctx)
        if is_valid:
            self.result_dag_gen_instructions.Push(new_instruction)

    def _MakeRandomInstruction(
            self,
            ctx: DAGGenContext
        ):
        dag_gen_class = self.dag_gen_class_generator.GetRandomDAGGenClass(
            ctx,
            self.requirement
        )
        return dag_gen_class.RandomGenerate(self.requirement, ctx)

class DAGGenerator:
    def __init__(self, requirement: DAGGenRequirement):
        requirement.CheckFields()
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on the front of list.
    def Generate(self) -> List["DAGGenInstruction"]:
        body_and_head_generator = BodyAndHeaderDAGGenerator(self.requirement)
        return body_and_head_generator.GenerateBodyAndHeader()


def DeadLoopChecker(limit=10000):
    counter = 0
    def Checker():
        nonlocal counter
        counter += 1
        assert counter < limit, "dead loop detected"
    return Checker