from typing import List
from dataclasses import dataclass, field, fields
from signature_constructor import SignatureConstructor

def InputIdx(idx):
    return field(metadata=dict(input_idx=idx))


def OutputIdx(idx):
    return field(metadata=dict(output_idx=idx))


class SignatureInferer:
    def __init__(self):
        self.current_sources = []

    def Infer(self, signature_constructor: SignatureConstructor):
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        method = getattr(
            BottomUpInferAndConstructSignature, 
            type(dag_gen_instruction).__name__
        )
        return method(self, signature_constructor)


class BottomUpInferAndConstructSignature:
    @classmethod
    def Nope(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        return signature_constructor.Nope()

    @classmethod
    def AddSourceTensor(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        return signature_constructor.AddSourceTensor()

    @classmethod
    def AddSinkTensor(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        signature = signature_constructor.AddSinkTensor()
        input, = _GetInputs(signature)
        signature_inferer.current_sources.append(input)
        return signature

    @classmethod
    def AddUnaryOp(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        output = signature_inferer.current_sources[source_tensor_index]
        signature = signature_constructor.AddUnaryOp(output)
        input, = _GetInputs(signature)
        signature_inferer.current_sources[source_tensor_index] = input
        return signature

    @classmethod
    def AddBinaryOp(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        output = signature_inferer.current_sources[source_tensor_index]
        signature = signature_constructor.AddBinaryOp(output)
        lhs_input, rhs_input = _GetInputs(signature)
        signature_inferer.current_sources[source_tensor_index] = lhs_input
        signature_inferer.current_sources.append(rhs_input)
        return signature

    @classmethod
    def AddBinaryClone(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        lhs_source_tensor_index = dag_gen_instruction.lhs_source_tensor_index
        rhs_source_tensor_index = dag_gen_instruction.rhs_source_tensor_index
        lhs_output = signature_inferer.current_sources[lhs_source_tensor_index]
        rhs_output = signature_inferer.current_sources[rhs_source_tensor_index]
        signature = signature_constructor.AddBinaryClone(lhs_output, rhs_output)
        signature_inferer.current_sources.pop(rhs_source_tensor_index)
        return signature


    @classmethod
    def AddSourceOp(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        output = signature_inferer.current_sources[source_tensor_index]
        signature = signature_constructor.AddSourceOp(output)
        signature_inferer.current_sources.pop(source_tensor_index)
        return signature


class TopDownSignatureInferer:
    def __init__(self):
        self.current_sinks = []

    def Infer(self, signature_constructor: SignatureConstructor):
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        method = getattr(
            TopDownInferAndConstructSignature, 
            type(dag_gen_instruction).__name__
        )
        return method(self, signature_constructor)


class TopDownInferAndConstructSignature:
    @classmethod
    def Nope(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        return signature_constructor.Nope()

    @classmethod
    def AddSourceTensor(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        signature = signature_constructor.AddSourceTensor()
        output, = _GetOutputs(signature)
        signature_inferer.current_sinks.append(output)
        return signature

    @classmethod
    def AddSinkTensor(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        input = signature_inferer.current_sinks[-1]
        signature_inferer.current_sinks.pop(-1)
        return signature_constructor.AddSinkTensor(input)

    @classmethod
    def AddUnaryOp(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        input = signature_inferer.current_sinks[source_tensor_index]
        signature = signature_constructor.AddUnaryOp(input)
        output, = _GetOutputs(signature)
        signature_inferer.current_sinks[source_tensor_index] = output
        return signature

    @classmethod
    def AddBinaryOp(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        lhs_input = signature_inferer.current_sinks[source_tensor_index]
        rhs_input = signature_inferer.current_sinks[-1]
        signature = signature_constructor.AddBinaryOp(lhs_input, rhs_input)
        output, = _GetOutputs(signature)
        signature_inferer.current_sinks[source_tensor_index] = output
        signature_inferer.current_sinks.pop(-1)
        return signature

    @classmethod
    def AddBinaryClone(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        lhs_source_tensor_index = dag_gen_instruction.lhs_source_tensor_index
        rhs_source_tensor_index = dag_gen_instruction.rhs_source_tensor_index
        input = signature_inferer.current_sinks[lhs_source_tensor_index]
        signature = signature_constructor.AddBinaryClone(input)
        lhs_output, rhs_output = _GetOutputs(signature)
        signature_inferer.current_sinks[lhs_source_tensor_index] = lhs_output
        signature_inferer.current_sinks.insert(rhs_source_tensor_index, rhs_output)
        return signature


    @classmethod
    def AddSourceOp(
        cls,
        signature_inferer: SignatureInferer,
        signature_constructor: SignatureConstructor
    ) -> "Signature":
        dag_gen_instruction = signature_constructor.dag_gen_instruction
        source_tensor_index = dag_gen_instruction.source_tensor_index
        signature = signature_constructor.AddSourceOp()
        output, = _GetOutputs(signature)
        signature_inferer.current_sinks.insert(source_tensor_index, output)
        return signature

def _GetInputs(signature):
    signature_fields = fields(signature)
    input_fields = [
        f
        for f in signature_fields
        if ('input_idx' in f.metadata) and (type(f.metadata['input_idx']) is int)
    ]
    input_fields = sorted(input_fields, key=lambda f: f.metadata['input_idx'])
    return [
        getattr(signature, f.name)
        for f in input_fields
    ]


def _GetOutputs(signature):
    signature_fields = fields(signature)
    output_fields = [
        f
        for f in signature_fields
        if ('output_idx' in f.metadata) and (type(f.metadata['output_idx']) is int)
    ]
    output_fields = sorted(output_fields, key=lambda f: f.metadata['output_idx'])
    return [
        getattr(signature, f.name)
        for f in output_fields
    ]