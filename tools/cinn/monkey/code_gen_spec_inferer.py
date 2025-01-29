from dataclasses import dataclass
import dag_generator as dag_generator
import dims_eq1_signature_inferer as dims_eq1_signature_inferer
import shape_signature_inferer as shape_signature_inferer
import tensor_name_signature_inferer as tensor_name_signature_inferer
from typing import List
from hash_combine import HashCombine

kComponentClasses = (
    dims_eq1_signature_inferer.DimsEq1Signature,
    shape_signature_inferer.ShapeSignature,
    tensor_name_signature_inferer.TensorNameSignature
)

@dataclass
class CodeGenSpec:
    
    @classmethod
    def GetDerivedClassByDAGGenClass(cls, dag_gen_class):
        return kDAGGenClassToDerivedClass[dag_gen_class]


def MergeClass(dag_gen_cls, component_classes):
    class_name = dag_gen_cls.__name__
    base_classes = (CodeGenSpec,)
    base_classes += tuple(
        component_class.GetDerivedClassByDAGGenClass(dag_gen_cls)
        for component_class in component_classes
    )
    return dataclass(type(class_name, base_classes, {}))

Nope = MergeClass(dag_generator.Nope, kComponentClasses)
AddSourceTensor = MergeClass(dag_generator.AddSourceTensor, kComponentClasses)
AddSinkTensor = MergeClass(dag_generator.AddSinkTensor, kComponentClasses)
AddUnaryOp = MergeClass(dag_generator.AddUnaryOp, kComponentClasses)
AddBinaryOp = MergeClass(dag_generator.AddBinaryOp, kComponentClasses)
AddBinaryClone = MergeClass(dag_generator.AddBinaryClone, kComponentClasses)
AddSourceOp = MergeClass(dag_generator.AddSourceOp, kComponentClasses)


kDAGGenClassToDerivedClass = {
    dag_generator.Nope: Nope,
    dag_generator.AddSourceTensor: AddSourceTensor,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

class CodeGenSpecInferer:
    def __init__(self):
        pass
    
    def Infer(
        self,
        dag_gen_instructions: List["DAGGenInstruction"],
        dims_eq1_signatures: List["DimsEq1Signature"],
        shape_signatures: List["ShapeSignature"],
        tensor_name_signatures: List["TensorNameSignature"]
    ) -> List[CodeGenSpec]:
        def CreateCodeGenSpec(dag_gen_instr, *signatures):
            params = dict()
            for signature in signatures:
                params.update(vars(signature))
            dag_gen_cls = type(dag_gen_instr)
            code_gen_spec_class = kDAGGenClassToDerivedClass[dag_gen_cls]
            return code_gen_spec_class(**params)
        return [
            CreateCodeGenSpec(*x)
            for x in zip(
                dag_gen_instructions,
                dims_eq1_signatures,
                shape_signatures,
                tensor_name_signatures)
        ]
