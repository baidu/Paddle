from dataclasses import dataclass
from typing import List
from collections import namedtuple
import dag_generator as dag_generator
import dims_eq1_generator as dims_eq1_generator
from pick_weight import PickWeight
from guarded_box import GuardedBox
from defensive_list import DList
from signature_constructor import NaiveSignatureConstructor
from signature_inferer import SignatureInferer, InputIdx, OutputIdx

DAGDimsEq1GenInstruction = namedtuple("DAGDimsEq1GenInstruction", ["dag", "dims_eq1"])


@dataclass
class DimsEq1Signature:
    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(DimsEq1Signature):

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction
    ):
        return Nope()


@dataclass
class AddSourceTensor(DimsEq1Signature):

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction
    ):
        return AddSourceTensor()


@dataclass
class AddSinkTensor(DimsEq1Signature):
    sink_tensor_dims_eq1: List[bool] = InputIdx(0)

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction
    ):
        sink_tensor_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.sink_tensor_dims_eq1
        )
        return AddSinkTensor(
            sink_tensor_dims_eq1=sink_tensor_dims_eq1
        )


@dataclass
class AddUnaryOp(DimsEq1Signature):
    input_dims_eq1: List[bool] = InputIdx(0)
    output_dims_eq1: List[bool] = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        output_dims_eq1: List[bool]
    ):
        input_dims_eq1 = dag_dims_eq1_gen_instruction.dims_eq1.source_tensor_dims_eq1
        return AddUnaryOp(
            input_dims_eq1=input_dims_eq1,
            output_dims_eq1=output_dims_eq1
        )
       

@dataclass
class AddBinaryOp(DimsEq1Signature):
    lhs_input_dims_eq1: List[bool] = InputIdx(0)
    rhs_input_dims_eq1: List[bool] = InputIdx(1)
    output_dims_eq1: List[bool] = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        output_dims_eq1: List[bool]
    ):
        lhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.lhs_source_tensor_dims_eq1
        )
        rhs_input_dims_eq1 = (
            dag_dims_eq1_gen_instruction.dims_eq1.rhs_source_tensor_dims_eq1
        )
        return AddBinaryOp(
            lhs_input_dims_eq1=lhs_input_dims_eq1,
            rhs_input_dims_eq1=rhs_input_dims_eq1,
            output_dims_eq1=output_dims_eq1
        )


@dataclass
class AddBinaryClone(DimsEq1Signature):
    lhs_output_dims_eq1: List[bool] = OutputIdx(0)
    rhs_output_dims_eq1: List[bool] = OutputIdx(1)

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        lhs_output_dims_eq1: List[bool],
        rhs_output_dims_eq1: List[bool]
    ):
        return AddBinaryClone(
            lhs_output_dims_eq1=lhs_output_dims_eq1,
            rhs_output_dims_eq1=rhs_output_dims_eq1
        )


@dataclass
class AddSourceOp(DimsEq1Signature):
    output_dims_eq1: List[bool] = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        dag_dims_eq1_gen_instruction: DAGDimsEq1GenInstruction,
        output_dims_eq1: List[bool]
    ):
        return AddSourceOp(output_dims_eq1=output_dims_eq1)


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class DimsEq1SignatureInferer:
    def __init__(self):
        pass

    def Infer(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        dims_eq1_gen_instructions: List["DimsEq1GenInstruction"]
    ) -> DList["DAGGenInstruction", "DimsEq1Signature"]:
        assert len(dag_gen_instructions) == len(dims_eq1_gen_instructions)
        signature_inferer = SignatureInferer()
        def MakeDimsEq1Signature(ctx):
            dag_gen_class = type(ctx.dag)
            cls = kDAGGenClassToDerivedClass[dag_gen_class]
            return signature_inferer.Infer(
                NaiveSignatureConstructor(
                    dag_gen_instruction=ctx.dag,
                    constructor=lambda *args: cls.Make(ctx, *args)
                )
            )
        dims_eq1_signatures = [
            MakeDimsEq1Signature(x)
            for x in _ZipDAGDimsInstr(dag_gen_instructions, dims_eq1_gen_instructions)
        ]
        return DList(dag_gen_instructions, dims_eq1_signatures)

def _ZipDAGDimsInstr(dag_gen_instructions, dims_eq1_gen_instructions):
    return [
        DAGDimsEq1GenInstruction(*instruction_tuple)
        for instruction_tuple in zip(dag_gen_instructions, dims_eq1_gen_instructions)
    ]
