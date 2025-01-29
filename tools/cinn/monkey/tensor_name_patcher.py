from typing import List, Dict
from tensor_name_generator import (
    TensorNameGenRequirement,
    TensorNameGenInstruction,
    TensorNameGenerator
)
from instruction_id import InstructionId
from dag_generator import DAGGenInstruction

class TensorNamePatcher:
    def __init__(self, requirement: TensorNameGenRequirement):
        self.requirement = requirement

    def Patch(
        self,
        dag_gen_instructions: List[DAGGenInstruction],
        instruction_ids: List[InstructionId],
        instruction_id2existed_tensor_name:
            Dict[InstructionId, TensorNameGenInstruction]
    ) -> List[TensorNameGenInstruction]:
        tensor_name_gen = TensorNameGenerator(self.requirement)
        tensor_name_gen_instructions = tensor_name_gen.Generate(dag_gen_instructions)
        def GetTensorName(dag_gen_instr, instruction_id, tensor_name_gen_instr):
            if instruction_id in instruction_id2existed_tensor_name:
                return instruction_id2existed_tensor_name[instruction_id]
            return tensor_name_gen_instr
        return [
            GetTensorName(*triple)
            for triple in zip(
                dag_gen_instructions,
                instruction_ids,
                tensor_name_gen_instructions
            )
        ]