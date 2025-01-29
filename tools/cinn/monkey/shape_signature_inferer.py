from dataclasses import dataclass
from typing import List, Union
from collections import namedtuple
import dag_generator as dag_generator
import dims_eq1_generator as dims_eq1_generator
from pick_weight import PickWeight
from guarded_box import GuardedBox
from defensive_list import DList

@dataclass
class StaticDim:
    size: int


@dataclass
class DynamicDim:
    symbol: str


@dataclass
class DimSizeRequirement:
    dim_size: List[Union[StaticDim, DynamicDim]]

@dataclass
class ShapeInferContext:
    dims_eq1_signature: "DimsEq1Signature"
    dim_size_requirement: DimSizeRequirement

@dataclass
class ShapeSignature:
    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


@dataclass
class Nope(ShapeSignature):
    
    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        return Nope()

@dataclass
class AddSourceTensor(ShapeSignature):

    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        return AddSourceTensor()

@dataclass
class AddSinkTensor(ShapeSignature):
    sink_tensor_shape: List[int]

    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        sink_tensor_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.sink_tensor_dims_eq1
        )
        return AddSinkTensor(
            sink_tensor_shape=sink_tensor_shape
        )

@dataclass
class AddUnaryOp(ShapeSignature):
    input_shape: List[int]
    output_shape: List[int]

    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        input_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.input_dims_eq1
        )
        output_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.output_dims_eq1
        )
        return AddUnaryOp(
            input_shape=input_shape,
            output_shape=output_shape
        )
       

@dataclass
class AddBinaryOp(ShapeSignature):
    lhs_input_shape: List[int]
    rhs_input_shape: List[int]
    output_shape: List[int]

    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        lhs_input_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.lhs_input_dims_eq1
        )
        rhs_input_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.rhs_input_dims_eq1
        )
        output_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.output_dims_eq1
        )
        return AddBinaryOp(
            lhs_input_shape=lhs_input_shape,
            rhs_input_shape=rhs_input_shape,
            output_shape=output_shape
        )


@dataclass
class AddBinaryClone(ShapeSignature):

    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        lhs_output_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.lhs_output_dims_eq1
        )
        rhs_output_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.rhs_output_dims_eq1
        )
        assert lhs_output_shape == rhs_output_shape
        return AddBinaryClone()


@dataclass
class AddSourceOp(ShapeSignature):
    output_shape: List[int]

    @classmethod
    def InferShape(
        cls,
        infer_ctx: ShapeInferContext
    ):
        output_shape = _GetStaticShape(
            total_shape=infer_ctx.dim_size_requirement.dim_size,
            dims_eq1=infer_ctx.dims_eq1_signature.output_dims_eq1
        )
        return AddSourceOp(
            output_shape=output_shape
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

class ShapeSignatureInferer:
    def __init__(self, requirement: DimSizeRequirement):
        self.requirement = requirement

    def Infer(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        dims_eq1_signatures: List["DimsEq1Signature"]
    ) -> DList["DAGGenInstruction", "ShapeSignature"]:
        assert len(dag_gen_instructions) == len(dims_eq1_signatures)
        def MakeShapeSignature(pair):
            dag_gen_instruction, dims_eq1_signature = pair
            dag_gen_class = type(dag_gen_instruction)
            cls = kDAGGenClassToDerivedClass[dag_gen_class]
            return cls.InferShape(
                ShapeInferContext(
                    dims_eq1_signature=dims_eq1_signature,
                    dim_size_requirement=self.requirement
                )
            )
        shape_signatures = [
            MakeShapeSignature(x)
            for x in zip(dag_gen_instructions, dims_eq1_signatures)
        ]
        return DList(dag_gen_instructions, shape_signatures)


def _GetStaticShape(
    total_shape: List[Union[StaticDim, DynamicDim]],
    dims_eq1: List[bool]
) -> List[int]:
    assert len(total_shape) == len(dims_eq1)
    def GetStaticDim(i):
        if dims_eq1[i]:
            return 1
        dim = total_shape[i]
        assert isinstance(dim, StaticDim)
        return dim.size
    return tuple(GetStaticDim(i) for i in range(len(total_shape)))