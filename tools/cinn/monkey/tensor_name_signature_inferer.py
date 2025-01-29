from dataclasses import dataclass, field
import dag_generator as dag_generator
from defensive_list import DList
from pick_weight import PickWeight
from typing import List, Set
import random
from signature_constructor import NaiveSignatureConstructor
from signature_inferer import TopDownSignatureInferer, InputIdx, OutputIdx

@dataclass
class InferContext:
    dag_gen_instruction: "DAGGenInstruction"
    tensor_name_gen_instruction: "TensorNameGenInstruction"


@dataclass
class TensorNameSignature:
    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(TensorNameSignature):
    
    @classmethod
    def Make(
        cls,
        ctx: InferContext
    ):
        return Nope()


@dataclass
class AddSourceTensor(TensorNameSignature):
    source_tensor_name: str = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        ctx: InferContext
    ):
        return AddSourceTensor(
            source_tensor_name=ctx.tensor_name_gen_instruction.source_tensor_name
        )

@dataclass
class AddSinkTensor(TensorNameSignature):

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        input_tensor_name
    ):
        return AddSinkTensor()


@dataclass
class AddUnaryOp(TensorNameSignature):
    input_tensor_name: str = InputIdx(0)
    output_tensor_name: str = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        input_tensor_name: str
    ):
        output_tensor_name = ctx.tensor_name_gen_instruction.output_tensor_name
        return AddUnaryOp(
            input_tensor_name=input_tensor_name,
            output_tensor_name=output_tensor_name
        )


@dataclass
class AddBinaryOp(TensorNameSignature):
    lhs_input_tensor_name: str = InputIdx(0)
    rhs_input_tensor_name: str = InputIdx(1)
    output_tensor_name: str = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        lhs_input_tensor_name: str,
        rhs_input_tensor_name: str
    ):
        output_tensor_name = ctx.tensor_name_gen_instruction.output_tensor_name
        return AddBinaryOp(
            lhs_input_tensor_name=lhs_input_tensor_name,
            rhs_input_tensor_name=rhs_input_tensor_name,
            output_tensor_name=output_tensor_name
        )


@dataclass
class AddBinaryClone(TensorNameSignature):
    input_tensor_name: str = InputIdx(0)
    lhs_output_tensor_name: str = OutputIdx(0)
    rhs_output_tensor_name: str = OutputIdx(1)

    @classmethod
    def Make(
        cls,
        ctx: InferContext,
        input_tensor_name: str,
    ):
        return AddBinaryClone(
            input_tensor_name=input_tensor_name,
            lhs_output_tensor_name=input_tensor_name,
            rhs_output_tensor_name=input_tensor_name
        )


@dataclass
class AddSourceOp(TensorNameSignature):
    output_tensor_name: str = OutputIdx(0)

    @classmethod
    def Make(
        cls,
        ctx: InferContext
    ):
        return AddSourceOp(
            output_tensor_name=ctx.tensor_name_gen_instruction.output_tensor_name
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

class TensorNameSignatureInferer:
    def __init__(self):
        pass
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Infer(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        tensor_name_gen_instructions: List["TensorNameGenInstruction"]
    ) -> DList["DAGGenInstruction", "TensorNameSignature"]:
        signature_inferer = TopDownSignatureInferer()
        def CreateTensorNameSignature(pair):
            dag_gen_instruction, tensor_name_gen_instruction = pair
            ctx = InferContext(
                dag_gen_instruction=dag_gen_instruction,
                tensor_name_gen_instruction=tensor_name_gen_instruction
            )
            cls = kDAGGenClassToDerivedClass[type(dag_gen_instruction)]
            return signature_inferer.Infer(
                NaiveSignatureConstructor(
                    dag_gen_instruction=dag_gen_instruction,
                    constructor=lambda *args: cls.Make(ctx, *args),
                )
            )
        pairs = list(zip(
            dag_gen_instructions,
            tensor_name_gen_instructions
        ))
        tensor_name_signatures = list(reversed([
            CreateTensorNameSignature(x)
            for x in reversed(pairs)
        ]))
        return DList(dag_gen_instructions, tensor_name_signatures)
