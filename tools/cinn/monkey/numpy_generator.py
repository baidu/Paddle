from dataclasses import dataclass
from unit_test_case_spec import UnitTestCaseSpec
from op_call_code_gen import OpCallCodeGenRequirement
from script import Script
from stmt_generator import StmtGenerator
from defensive_list import DList


@dataclass
class NumpyScript(Script):
    pass

class NumpyGenerator:
    def __init__(self):
        self.requirement = OpCallCodeGenRequirement(
            module_name="numpy"
        )
        self.stmt_generator = StmtGenerator(self.requirement)

    def Generate(
        self,
        instruction_code_gen_spec: DList["Instruction", "CodeGenSepc"]
    ) -> NumpyScript:
        stmts_list = self.stmt_generator.Generate(instruction_code_gen_spec)
        instruction_code_strs = [
            "\n".join(list(stmts))
            for stmts in stmts_list
        ]
        file_content = "\n".join(instruction_code_strs)
        return NumpyScript(
            file_content=file_content
        )