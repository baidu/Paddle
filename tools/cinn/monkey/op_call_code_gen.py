from dataclasses import dataclass
from typing import List
from axis_flag_util import IsLhsGreaterThanRhs

@dataclass
class OpCallCodeGenRequirement:
    module_name: str


@dataclass
class CodeGenContext:
    requirement: OpCallCodeGenRequirement
    instruction: "Instruction"
    code_gen_spec: "CodeGenSpec"


class Add:
    @classmethod
    def paddle(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)


    @classmethod
    def numpy(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)


    @classmethod
    def Call(cls, ctx: CodeGenContext):
        lhs_input_shape = ctx.code_gen_spec.lhs_input_shape
        rhs_input_shape = ctx.code_gen_spec.rhs_input_shape
        output_shape = ctx.code_gen_spec.output_shape
        lhs_input_tensor = ctx.code_gen_spec.lhs_input_tensor_name
        rhs_input_tensor = ctx.code_gen_spec.rhs_input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        yield ""
        yield f"# {lhs_input_tensor}: {lhs_input_shape}"
        yield f"# {rhs_input_tensor}: {rhs_input_shape}"
        yield f"{output_tensor} = {lhs_input_tensor} + {rhs_input_tensor}"
        yield f"assert {output_tensor}.shape == {output_shape}"


class Negative:
    @classmethod
    def paddle(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)

    @classmethod
    def numpy(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)

    @classmethod
    def Call(cls, ctx: CodeGenContext):
        input_tensor = ctx.code_gen_spec.input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        input_shape = ctx.code_gen_spec.input_shape
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"# {input_tensor}: {input_shape}"
        yield f"{output_tensor} = -{input_tensor}"
        yield f"assert {output_tensor}.shape == {output_shape}"


class ReduceSum:
    @classmethod
    def paddle(cls, ctx: CodeGenContext):
        axes = cls.GetReduceAxes(ctx)
        input_tensor = ctx.code_gen_spec.input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        input_shape = ctx.code_gen_spec.input_shape
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"# {input_tensor}: {input_shape}"
        yield f"{output_tensor} = {input_tensor}.sum(axis={axes}, keepdim=True)"
        yield f"assert {output_tensor}.shape == {output_shape}"

    @classmethod
    def numpy(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        axes = cls.GetReduceAxes(ctx)
        input_tensor = ctx.code_gen_spec.input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        input_shape = ctx.code_gen_spec.input_shape
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"# {input_tensor}: {input_shape}"
        yield f"{output_tensor} = {m}.sum({input_tensor}, axis={axes}, keepdims=True)"
        yield f"assert {output_tensor}.shape == {output_shape}"

    @classmethod
    def GetReduceAxes(cls, ctx: CodeGenContext):
        input_dims_eq1 = ctx.code_gen_spec.input_dims_eq1
        output_dims_eq1 = ctx.code_gen_spec.output_dims_eq1
        is_reduced = [
            (not x) and y
            for x, y in zip(input_dims_eq1, output_dims_eq1)
        ]
        return tuple(
            i
            for i, is_reduced in enumerate(is_reduced)
            if is_reduced
        )

class Expand:
    @classmethod
    def paddle(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        input_tensor = ctx.code_gen_spec.input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        input_shape = ctx.code_gen_spec.input_shape
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"# {input_tensor}: {input_shape}"
        yield f"{output_tensor} = {m}.expand({input_tensor}, shape={output_shape})"
        yield f"assert {output_tensor}.shape == {output_shape}"

    @classmethod
    def numpy(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        input_tensor = ctx.code_gen_spec.input_tensor_name
        output_tensor = ctx.code_gen_spec.output_tensor_name
        input_shape = ctx.code_gen_spec.input_shape
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"# {input_tensor}: {input_shape}"
        for input_dim, output_dim in zip(input_shape, output_shape):
            if input_dim == 1:
                continue
            assert input_dim == output_dim, "%s, %s"%(input_shape, output_shape)
        yield f"{output_tensor} = {m}.broadcast_to({input_tensor}, shape={output_shape})"
        yield f"assert {output_tensor}.shape == {output_shape}"


class Ones:
    @classmethod
    def paddle(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)

    @classmethod
    def numpy(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)

    @classmethod
    def Call(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        output_tensor_name = ctx.code_gen_spec.output_tensor_name
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"{output_tensor_name} = {m}.ones({output_shape})"



class Zeros:
    @classmethod
    def paddle(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)

    @classmethod
    def numpy(cls, ctx: CodeGenContext):
        yield from cls.Call(ctx)

    @classmethod
    def Call(cls, ctx: CodeGenContext):
        m = ctx.requirement.module_name
        output_tensor_name = ctx.code_gen_spec.output_tensor_name
        output_shape = ctx.code_gen_spec.output_shape
        yield ""
        yield f"{output_tensor_name} = {m}.zeros({output_shape})"