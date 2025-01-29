from typing import List, Dict
from op_name_generator import (
    OpNameGenRequirement,
    OpNameGenInstruction,
    OpNameGenerator
)
from instruction_id import InstructionId
from dag_generator import DAGGenInstruction

class OpNamePatcher:
    def __init__(self, requirement: OpNameGenRequirement):
        self.requirement = requirement

    def Patch(
        self,
        dag_gen_instructions: List[DAGGenInstruction],
        instruction_ids: List[InstructionId],
        instruction_id2existed_op_name: Dict[InstructionId, OpNameGenInstruction]
    ) -> List[OpNameGenInstruction]:
        op_name_gen = OpNameGenerator(self.requirement)
        op_name_gen_instructions = op_name_gen.Generate(dag_gen_instructions)
        def GetOpName(dag_gen_instr, instruction_id, op_name_gen_instr):
            if instruction_id in instruction_id2existed_op_name:
                return instruction_id2existed_op_name[instruction_id]
            return op_name_gen_instr
        return [
            GetOpName(*triple)
            for triple in zip(
                dag_gen_instructions,
                instruction_ids,
                op_name_gen_instructions
            )
        ]