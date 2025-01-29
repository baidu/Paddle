from dataclasses import dataclass, field

class SignatureConstructor:

    def Nope(self):
        return self.Make()

    def AddSourceTensor(self, *args):
        return self.Make(*args)

    def AddSinkTensor(self, *args):
        return self.Make(*args)

    def AddUnaryOp(self, *args):
        return self.Make(*args)

    def AddBinaryOp(self, *args):
        return self.Make(*args)

    def AddBinaryClone(self, *args):
        return self.Make(*args)

    def AddSourceOp(self, *args):
        return self.Make(*args)

    def Make(self, *args):
        raise NotImplementedError("no overrided self.Make() method")

class NaiveSignatureConstructor(SignatureConstructor):
    def __init__(self, dag_gen_instruction, constructor):
        self.dag_gen_instruction=dag_gen_instruction
        self.constructor = constructor

    def Make(self, *args):
        return self.constructor(*args)


@dataclass
class NoneSignature:
    pass


@dataclass
class NoInputNoneSignature(NoneSignature):
    pass

@dataclass
class UnaryInputNoneSignature(NoneSignature):
    input: None = field(
        default_factory=lambda: None,
        metadata=dict(input_idx=0)
    )


@dataclass
class BinaryInputNoneSignature(NoneSignature):
    lhs_input: None = field(
        default_factory=lambda: None,
        metadata=dict(input_idx=0)
    )
    rhs_input: None = field(
        default_factory=lambda: None,
        metadata=dict(input_idx=1)
    )

class NoneSignatureConstructor:

    def __init__(self, dag_gen_instruction: "DAGGenInstruction"):
        self.dag_gen_instruction = dag_gen_instruction

    def Nope(self):
        return NoInputNoneSignature()

    def AddSourceTensor(self):
        return NoInputNoneSignature()

    def AddSinkTensor(self):
        return UnaryInputNoneSignature()

    def AddUnaryOp(self, output):
        return UnaryInputNoneSignature()

    def AddBinaryOp(self, output):
        return BinaryInputNoneSignature()

    def AddBinaryClone(self, lhs_output, rhs_output):
        return UnaryInputNoneSignature()

    def AddSourceOp(self, output):
        return NoInputNoneSignature()
