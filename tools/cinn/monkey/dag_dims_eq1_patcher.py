from dataclasses import dataclass
import dag_generator as dag_generator
import dims_eq1_generator as dims_eq1_generator
from dims_eq1_signature_inferer import DimsEq1SignatureInferer
from pick_weight import PickWeight
from typing import List, Iterator, Tuple
from collections import namedtuple
from defensive_list import DList
from instruction_id import InstructionId, MakeUniqueInstructionId
from axis_flag_util import IsLhsGreaterThanRhs

@dataclass
class PatchContext:
    dag_gen_instruction: "DAGGenInstruction"
    instruction_id: InstructionId
    dims_eq1_signature: "DimsEq1Signature"

@dataclass
class DAGDimsEq1Instruction:
    dag_gen_instruction: "DAGGenInstruction"
    instruction_id: InstructionId
    dims_eq1_instruction: "DimsEq1GenInstruction"


@dataclass
class Nope:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        yield from []

@dataclass
class AddSourceTensor:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=ctx.dag_gen_instruction,
            instruction_id=ctx.instruction_id,
            dims_eq1_instruction=dims_eq1_generator.AddSourceTensor()
        )



@dataclass
class AddSinkTensor:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=ctx.dag_gen_instruction,
            instruction_id=ctx.instruction_id,
            dims_eq1_instruction=dims_eq1_generator.AddSinkTensor(
                sink_tensor_dims_eq1=ctx.dims_eq1_signature.sink_tensor_dims_eq1
            )
        )

def AddUnaryOps(
    input_dims_eq1,
    output_dims_eq1,
    source_tensor_index,
    instruction_id
):
    middle_dims_eq1 = tuple(
        x or y
        for x, y in zip(input_dims_eq1, output_dims_eq1)
    )
    if IsLhsGreaterThanRhs(middle_dims_eq1, output_dims_eq1):
        # broadcast
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=dag_generator.AddUnaryOp(
                source_tensor_index=source_tensor_index,
                convert_type=dag_generator.BroadcastConvertType()
            ),
            instruction_id=MakeUniqueInstructionId(),
            dims_eq1_instruction=dims_eq1_generator.AddUnaryOp(
                source_tensor_dims_eq1=middle_dims_eq1
            )
        )
    if IsLhsGreaterThanRhs(middle_dims_eq1, input_dims_eq1):
        # reduce
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=dag_generator.AddUnaryOp(
                source_tensor_index=source_tensor_index,
                convert_type=dag_generator.ReduceConvertType()
            ),
            instruction_id=MakeUniqueInstructionId(),
            dims_eq1_instruction=dims_eq1_generator.AddUnaryOp(
                source_tensor_dims_eq1=input_dims_eq1
            )
        )
    yield DAGDimsEq1Instruction(
        dag_gen_instruction=dag_generator.AddUnaryOp(
            source_tensor_index=source_tensor_index,
            convert_type=dag_generator.NoConvertType()
        ),
        instruction_id=instruction_id,
        dims_eq1_instruction=dims_eq1_generator.AddUnaryOp(
            source_tensor_dims_eq1=input_dims_eq1
        )
    )


@dataclass
class AddUnaryOp:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        input_dims_eq1 = ctx.dims_eq1_signature.input_dims_eq1
        output_dims_eq1 = ctx.dims_eq1_signature.output_dims_eq1
        yield from AddUnaryOps(
            input_dims_eq1=input_dims_eq1,
            output_dims_eq1=output_dims_eq1,
            source_tensor_index=ctx.dag_gen_instruction.source_tensor_index,
            instruction_id=ctx.instruction_id
        )


@dataclass
class AddBinaryOp:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        lhs_input_dims_eq1 = ctx.dims_eq1_signature.lhs_input_dims_eq1
        rhs_input_dims_eq1 = ctx.dims_eq1_signature.rhs_input_dims_eq1
        output_dims_eq1 = ctx.dims_eq1_signature.output_dims_eq1
        output_idx = ctx.dag_gen_instruction.source_tensor_index
        broadcast_dims_eq1 = tuple(
            x and y
            for x, y in zip(lhs_input_dims_eq1, rhs_input_dims_eq1)
        )
        if broadcast_dims_eq1 != output_dims_eq1:
            yield from AddUnaryOps(
                input_dims_eq1=broadcast_dims_eq1,
                output_dims_eq1=output_dims_eq1,
                source_tensor_index=ctx.dag_gen_instruction.source_tensor_index,
                instruction_id=MakeUniqueInstructionId()
            )
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=ctx.dag_gen_instruction,
            instruction_id=ctx.instruction_id,
            dims_eq1_instruction=dims_eq1_generator.AddBinaryOp(
                lhs_source_tensor_dims_eq1=lhs_input_dims_eq1,
                rhs_source_tensor_dims_eq1=rhs_input_dims_eq1
            )
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        lhs_output_dims_eq1 = ctx.dims_eq1_signature.lhs_output_dims_eq1
        rhs_output_dix = ctx.dag_gen_instruction.rhs_source_tensor_index
        rhs_output_dims_eq1 = ctx.dims_eq1_signature.rhs_output_dims_eq1
        if lhs_output_dims_eq1 != rhs_output_dims_eq1:
            yield from AddUnaryOps(
                input_dims_eq1=lhs_output_dims_eq1,
                output_dims_eq1=rhs_output_dims_eq1,
                source_tensor_index=ctx.dag_gen_instruction.rhs_source_tensor_index,
                instruction_id=MakeUniqueInstructionId()
            )
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=ctx.dag_gen_instruction,
            instruction_id=ctx.instruction_id,
            dims_eq1_instruction=dims_eq1_generator.AddBinaryClone()
        )


@dataclass
class AddSourceOp:

    @classmethod
    def Patch(
        cls,
        ctx: PatchContext
    ) -> Iterator[DAGDimsEq1Instruction]:
        yield DAGDimsEq1Instruction(
            dag_gen_instruction=ctx.dag_gen_instruction,
            instruction_id=ctx.instruction_id,
            dims_eq1_instruction=dims_eq1_generator.AddSourceOp()
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


class DAGDimsEq1Patcher:
    def __init__(self):
        pass

    def Patch(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        instruction_ids: List[InstructionId],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"]
    ) -> Tuple[
            List["DAGGenInstruction"],
            List[InstructionId],
            List["DimsEq1GenInstruction"]
        ]:
        # inferer
        Infer = DimsEq1SignatureInferer().Infer
        # patch
        guarded_dims_eq1_sigs = Infer(dag_gen_instructions, dims_eq1_gen_instructions)
        (dag_gen_instrs, instruction_ids, dims_eq1_instrs) = self.PatchOnce(
            instruction_ids, guarded_dims_eq1_sigs
        )
        return dag_gen_instrs, instruction_ids, dims_eq1_instrs

    def PatchOnce(
        self,
        instruction_ids: List[InstructionId],
        guarded_dims_eq1_sigs: DList["DAGGenInstruction", "DimsEq1Signature"]
    ) -> Tuple[List["DAGGenInstruction"], List["DimsEq1GenInstruction"]]:
        def CreateDAGDimsEq1GenInstructions(instruction_id, pair):
            dag_instr, dims_eq1_sig = pair
            cls = kDAGGenClassToDerivedClass[type(dag_instr)]
            flat_mapped = cls.Patch(
                PatchContext(
                    dag_gen_instruction=dag_instr,
                    instruction_id=instruction_id,
                    dims_eq1_signature=dims_eq1_sig
                ),
            )
            for instruction in flat_mapped:
                yield (
                    instruction.dag_gen_instruction,
                    instruction.instruction_id,
                    instruction.dims_eq1_instruction
                )

        pairs = zip(
            instruction_ids,
            guarded_dims_eq1_sigs.Unguard()
        )
        triples = [
            triple
            for x in pairs
            for triple in CreateDAGDimsEq1GenInstructions(*x)
        ]
        return (
            [x for x,y,z in triples],
            [y for x,y,z in triples],
            [z for x,y,z in triples]
        )
