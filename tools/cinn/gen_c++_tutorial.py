# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
This script helps to extract the tutorial content from a C++ source file.
'''

# syntax definition
# The text content locates in the comments with `//!` prefix.
# Some predefined marks:
#  - @h1, @h2, @h3, the nth headline
#  - @IGNORE-NEXT, hide the next line of code
#  - @ROC, the code block inside a C++ multi-line string guard `ROC()ROC`,
#          display as a markdown code block.

from __future__ import annotations

import logging
import sys


class Markdown:
    '''
    A simple markdown generator.
    '''

    def __init__(self):
        self.content: list[str] = []

    def h1(self, title: str):
        self.add_line('# ' + title)

    def h2(self, title: str):
        self.add_line('## ' + title)

    def h3(self, title: str):
        self.add_line('### ' + title)

    def code_block(self, lang: str, block: list[str]):
        # drop the preceding and tailing empty lines to make code block more compact
        pre_valid_offset = 0
        tail_valid_offset = 0
        for x in block:
            if x.strip():
                break
            else:
                pre_valid_offset += 1
        for x in reversed(block):
            if x.strip():
                break
            else:
                tail_valid_offset += 1
        logging.warning(f"block0: {block}")
        block = (
            block[pre_valid_offset:-tail_valid_offset]
            if tail_valid_offset > 0
            else block[pre_valid_offset:]
        )
        logging.warning(f"block1: {block}")
        if not block:
            return

        c = "```" + lang

        # add empty lines to wrap code block
        self.add_line('')
        self.add_line('\n'.join([c, '\n'.join(block), "```"]))
        self.add_line('')

    def add_line(self, content: str):
        self.content.append(content)

    def generate(self):
        return '\n'.join(self.content)


class Mark:
    h1 = "@h1"
    h2 = "@h2"
    h3 = "@h3"
    h4 = "@h4"
    ignore_next = "@IGNORE-NEXT"
    roc = "@ROC"


class ContentGenerator:
    '''
    Interface for some content passed into the parser.
    '''

    def has_next(self) -> bool:
        pass

    def get_line(self) -> str:
        pass


class Parser:
    DOC_COMMENT_PREFIX = "//!"

    def __init__(self):
        self.doc = Markdown()
        self.code_block = []

    def parse(self, content: ContentGenerator):
        while content.has_next():
            line = content.get_line()
            line_striped = line.strip()
            is_doc = False
            if line_striped.startswith(self.DOC_COMMENT_PREFIX):
                is_doc = True
                if self.code_block:
                    self.doc.code_block('c++', self.code_block)
                    self.code_block = []

                line_striped = line_striped[
                    len(self.DOC_COMMENT_PREFIX) :
                ].strip()

                if line_striped.startswith(Mark.h1):
                    self.eat_h1(line_striped)
                elif line_striped.startswith(Mark.h2):
                    self.eat_h2(line_striped)
                elif line_striped.startswith(Mark.h3):
                    self.eat_h3(line_striped)
                elif line_striped.startswith(Mark.h4):
                    self.eat_h4(line_striped)
                elif line_striped.startswith(Mark.ignore_next):
                    self.eat_ignore_next(content)
                elif line_striped.startswith(Mark.roc):
                    self.eat_roc(line_striped, content)
                else:
                    self.doc.add_line(line_striped)

            else:  # normal code
                self.code_block.append(line)

    def eat_h1(self, content: str) -> None:
        self.doc.h1(content[len(Mark.h1) :].strip())

    def eat_h2(self, content: str) -> None:
        self.doc.h2(content[len(Mark.h2) :].strip())

    def eat_h3(self, content: str) -> None:
        self.doc.h3(content[len(Mark.h3) :].strip())

    def eat_ignore_next(self, content: ContentGenerator) -> None:
        content.get_line()

    def eat_roc(self, header: str, content: ContentGenerator) -> None:
        '''
        Get the content from a pair of ROC guards.
        @param header the string contains description of the ROC block.
        @content: the content generator.

        e.g.

        the content:

            //! @ROC[c++]
            auto target_source = R"ROC(
            function fn0 (_A, _B, _tensor)
            {
            }
            ROC);

        The parameter header is `//! @ROC[c++]`.
        '''
        assert "ROC" in header
        lang = header[len("@ROC[") : -1]

        logging.warning("eating ROC")

        assert content.has_next()
        line: str = content.get_line()
        assert "ROC(" in line
        line = content.get_line()
        code_block = []
        while ")ROC" not in line:
            code_block.append(line)
            line: str = content.get_line()

        logging.warning(f"DOC content: {code_block}")

        self.doc.code_block(lang, code_block)

    def generate(self):
        return self.doc.generate()


if __name__ == '__main__':

    class Content(ContentGenerator):
        def __init__(self):
            self.lines = list(sys.stdin)
            self.cur = 0

        def has_next(self):
            return self.cur < len(self.lines)

        def get_line(self):
            assert self.has_next()
            res = self.lines[self.cur]
            self.cur += 1
            return res.rstrip()

    parser = Parser()
    parser.parse(Content())
    sys.stdout.write(parser.generate())
