#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
r"""
usage: coverage_diff.py info_file diff_file > > coverage-diff.info
"""

import sys


def get_diff_file_lines(diff_file):
    r"""
    Args:
        diff_file (str): File to get modified lines.  

    Returns:
        dict: The diff lines of files.
    """
    diff_file_lines = {}

    current_file = None
    current_line = -1

    with open(diff_file) as diff_file:
        for line in diff_file:
            line = line.strip()

            if line.startswith('+++ '):
                current_file = line.lstrip('+++ ')

                diff_file_lines[current_file] = []

                continue

            elif line.startswith('@@ '):
                current_line = line.split()[2]
                current_line = current_line.lstrip('+').split(',')[0]
                current_line = int(current_line)

                continue

            elif line.startswith('-'):
                continue

            elif line.startswith('+'):
                diff_file_lines[current_file].append(current_line)

            current_line += 1

    return diff_file_lines


def get_info_file_lines(info_file, diff_file):
    r"""
    Args:
        info_file (str): File generated by lcov.
        diff_file (str): File to get modified lines.  

    Returns:
        None
    """
    diff_file_lines = get_diff_file_lines(diff_file)

    current_lines = []
    current_lf = 0
    current_lh = 0

    with open(info_file) as info_file:
        for line in info_file:
            line = line.strip()

            if line.startswith('SF:'):
                current_file = line.lstrip('SF:')

                if current_file.startswith('/paddle/'):
                    current_file = current_file[len('/paddle/'):]

                current_lines = diff_file_lines.get(current_file, [])

            elif line.startswith('DA:'):
                da = line.lstrip('DA:').split(',')

                if int(da[0]) in current_lines:
                    current_lf += 1

                    if not line.endswith(',0'):
                        current_lh += 1

                    print(line)

                continue

            elif line.startswith('LF:'):
                print('LF:{}'.format(current_lf))

                continue

            elif line.startswith('LH:'):
                print('LH:{}'.format(current_lh))

                continue

            print(line)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit()

    info_file = sys.argv[1]
    diff_file = sys.argv[2]

    get_info_file_lines(info_file, diff_file)
