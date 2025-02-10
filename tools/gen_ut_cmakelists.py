# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import re
import sys

# port range (21200, 23000) is reserved for dist-ops


# function to process pythonpath env
# append "${PADDLE_BINARY_DIR}/python" to PYTHONPATH
def _process_PYTHONPATH(pythonpath_option):
    pythonpath_option += ":${PADDLE_BINARY_DIR}/python"
    return pythonpath_option


def _process_envs(envs):
    """
    Desc:
        Input a str and output a str with the same function to specify some environment variables.
    Here we can give a specital process for some variable if needed.
    Example 1:
        Input: "http_proxy=;PYTHONPATH=.."
        Output: "http_proxy=;PYTHONPATH=..:${PADDLE_BINARY_DIR}/python"
    Example 2:
        Input: "http_proxy=;https_proxy=123.123.123.123:1230"
        Output: "http_proxy=;https_proxy=123.123.123.123:1230"
    """
    envs = envs.strip()

    envs_parts = envs.split(";")
    processed_envs = []

    for p in envs_parts:
        assert (
            " " not in p
            and re.compile("^[a-zA-Z_][0-9a-zA-Z_]*=").search(p) is not None
        ), f"""The environment option format is wrong. The env variable name can only contains'a-z', 'A-Z', '0-9' and '_',
and the var can not contain space in either env names or values.
However the var's format is '{p}'."""

        # if p starts with "PYTHONPATH=", then process python path
        if re.compile("^PYTHONPATH=").search(p):
            p = _process_PYTHONPATH(p)

        processed_envs.append(p)

    return ";".join(processed_envs)


def _process_conditions(conditions):
    """
    Desc:
        Input condition expression in cmake grammar and return a string wrapped by 'AND ()'.
        If the conditions string is empty, return an empty string.
    Example 1:
        Input: "LINUX"
        Output: "AND (LINUX)"
    Example 2:
        Input: ""
        Output: ""
    """
    if len(conditions.strip()) == 0:
        conditions = []
    else:
        conditions = conditions.strip().split(";")
    return [c.strip() for c in conditions]


def _process_archs(arch):
    """
    desc:
        Input archs options and warp it with 'WITH_', 'OR' and '()' in cmakelist grammar.
        The case is ignored.
        If the input is empty, return "LOCAL_ALL_ARCH".
    Example 1:
        Input: 'gpu'
        Output: '(WITH_GPU)'
    Example 2:
        Input: 'gpu;ROCM'
        Output: '(WITH_GPU OR WITH_ROCM)'
    """
    archs = ""
    arch = arch.upper().strip()
    if len(arch) > 0:
        for a in arch.split(";"):
            if '' == a:
                continue
            assert a in [
                "GPU",
                "ROCM",
                "XPU",
            ], f"""Supported arhc options are "GPU", "ROCM", and "XPU", but the options is {a}"""
            archs += "WITH_" + a.upper() + " OR "
        arch = "(" + archs[:-4] + ")"
    else:
        arch = "LOCAL_ALL_ARCH"
    return arch


def _process_os(os_):
    """
    Desc:
        Input os options and output wrapped options with 'OR' and '()'
        If the input is empty, return "LOCAL_ALL_PLAT"
    Example 1:
        Input: "WIN32"
        Output: "(WIN32)"
    Example 2:
        Input: "WIN32;linux"
        Output: "(WIN32 OR LINUX)"
    """
    os_ = os_.strip()
    if len(os_) > 0:
        os_ = os_.upper()
        for p in os_.split(';'):
            assert p in [
                "WIN32",
                "APPLE",
                "LINUX",
            ], f"""Supported os options are 'WIN32', 'APPLE' and 'LINUX', but the options is {p}"""
        os_ = os_.replace(";", " OR ")
        os_ = "(" + os_ + ")"
    else:
        os_ = "LOCAL_ALL_PLAT"
    return os_


# check whether run_serial is 0, 1 or empty
def _process_run_serial(run_serial):
    rs = run_serial.strip()
    assert rs in [
        "1",
        "0",
        "",
    ], f"""the value of run_serial must be one of 0, 1 or empty. But this value is {rs}"""
    if rs == "":
        return ""
    return rs


def _file_with_extension(prefix, suffixes):
    """
    Desc:
        check whether test file exists.
    """
    for ext in suffixes:
        if os.path.isfile(prefix + ext):
            return True
    return False


def _process_name(name, curdir):
    """
    Desc:
        check whether name is with a legal format and check whether the test file exists.
    """
    name = name.strip()
    assert re.compile("^test_[0-9a-zA-Z_]+").search(name), (
        """If line is not the header of table, the test name must begin with "test_" """
        """and the following substring must include at least one char of "0-9", "a-z", "A-Z" or "_"."""
    )
    filepath_prefix = os.path.join(curdir, name)
    suffix = [".py", ".sh"]
    assert _file_with_extension(
        filepath_prefix, suffix
    ), f""" Please ensure the test file with the prefix '{filepath_prefix}' and one of the suffix {suffix} exists, because you specified a unittest named '{name}'"""

    return name


def _norm_dirs(dirs):
    # reform all dirs' path as normal absolute path
    # abspath() can automatically normalize the path format
    norm_dirs = []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in norm_dirs:
            norm_dirs.append(d)
    return norm_dirs


def _process_run_type(run_type):
    rt = run_type.strip()
    # completely match one of the strings: 'NIGHTLY', 'EXCLUSIVE', 'CINN', 'DIST', 'HYBRID', 'GPUPS', 'INFER', 'EXCLUSIVE:NIGHTLY' and 'DIST:NIGHTLY'
    assert re.compile(
        "^(NIGHTLY|EXCLUSIVE|CINN|DIST|HYBRID|GPUPS|INFER|EXCLUSIVE:NIGHTLY|DIST:NIGHTLY)$"
    ).search(rt), (
        f""" run_type must be one of 'NIGHTLY', 'EXCLUSIVE', 'CINN', 'DIST', 'HYBRID', 'GPUPS', 'INFER', 'EXCLUSIVE:NIGHTLY' and 'DIST:NIGHTLY'"""
        f"""but the run_type is {rt}"""
    )
    return rt


class DistUTPortManager:
    def __init__(self, ignore_dirs=[]):
        self.dist_ut_port = 21200
        self.assigned_ports = {}
        self.last_test_name = ""
        self.last_test_cmake_file = ""
        self.no_cmake_dirs = []
        self.processed_dirs = set()
        self.ignore_dirs = _norm_dirs(ignore_dirs)

    def reset_current_port(self, port=None):
        self.dist_ut_port = 21200 if port is None else port

    def get_current_port(self):
        return self.dist_ut_port

    def gset_port(self, test_name, port):
        '''
        Get and set a port for unit test named test_name. If the test has been already holding a port, return the port it holds.
        Else assign the input port as a new port to the test.
        '''
        if test_name not in self.assigned_ports:
            self.assigned_ports[test_name] = port
        self.dist_ut_port = max(
            self.dist_ut_port, self.assigned_ports[test_name]
        )
        return self.assigned_ports[test_name]

    def process_dist_port_num(self, port_num):
        assert (
            re.compile("^[0-9]+$").search(port_num)
            and int(port_num) > 0
            or port_num.strip() == ""
        ), f"""port_num must be format as a positive integer or empty, but this port_num is '{port_num}'"""
        port_num = port_num.strip()
        if len(port_num) == 0:
            return 0
        port = self.dist_ut_port
        assert port < 23000, "dist port is exhausted"
        self.dist_ut_port += int(port_num)
        return port

    def _init_dist_ut_ports_from_cmakefile(self, cmake_file_name):
        '''
        Desc:
            Find all signed ut ports in cmake_file and update the ASSIGNED_PORTS
            and keep the DIST_UT_PORT max of all assigned ports
        '''
        with open(cmake_file_name) as cmake_file:
            # match lines including 'PADDLE_DIST_UT_PORT=' followed by a number
            port_reg = re.compile("PADDLE_DIST_UT_PORT=[0-9]+")
            lines = cmake_file.readlines()
            for idx, line in enumerate(lines):
                matched = port_reg.search(line)
                if matched is None:
                    continue
                p = matched.span()
                port = int(line[p[0] : p[1]].split("=")[-1])

                # find the test name which the port belongs to
                for k in range(idx, 0, -1):
                    if lines[k].strip() == "START_BASH":
                        break
                name = lines[k - 1].strip()

                # matcg right tets name format, the name must start with 'test_' followed bu at least one char of
                # '0-9'. 'a-z'. 'A-Z' or '_'
                assert re.compile("^test_[0-9a-zA-Z_]+").search(
                    name
                ), f'''we found a test for initial the latest dist_port but the test name '{name}' seems to be wrong
                    at line {k - 1}, in file {cmake_file_name}
                    '''
                self.gset_port(name, port)

                # get the test_name which latest assigned port belongs to
                if self.assigned_ports[name] == self.dist_ut_port:
                    self.last_test_name = name
                    self.last_test_cmake_file = cmake_file_name

    def parse_assigned_dist_ut_ports(self, current_work_dir, depth=0):
        '''
        Desc:
            get all assigned dist ports to keep port of unmodified test fixed.
        '''
        if current_work_dir in self.processed_dirs:
            return

        # if root(depth==0)
        if depth == 0:
            self.processed_dirs.clear()

        self.processed_dirs.add(current_work_dir)
        contents = os.listdir(current_work_dir)
        cmake_file = os.path.join(current_work_dir, "CMakeLists.txt")
        csv = cmake_file.replace("CMakeLists.txt", 'testslist.csv')

        if os.path.isfile(csv) or os.path.isfile(cmake_file):
            if current_work_dir not in self.ignore_dirs:
                if os.path.isfile(cmake_file) and os.path.isfile(csv):
                    self._init_dist_ut_ports_from_cmakefile(cmake_file)
                elif not os.path.isfile(cmake_file):
                    # put the directory which has csv but no cmake into NO_CMAKE_DIR_WARNING
                    self.no_cmake_dirs.append(current_work_dir)

            # recursively process the subdirectories
            for c in contents:
                c_path = os.path.join(current_work_dir, c)
                if os.path.isdir(c_path):
                    self.parse_assigned_dist_ut_ports(c_path, depth + 1)

        if depth == 0:
            # After all directories are scanned and processed
            # 1. Get the num_port of last added test and set DIST_UT_PORT+=num_port
            #    to guarantee the DIST_UT_PORT is not assigned
            # 2. Summary all the directories which include csv but no cmake and show an error
            #    if such a drectory exists

            # step 1
            if (
                len(self.last_test_name) > 0
                and len(self.last_test_cmake_file) > 0
            ):
                with open(
                    self.last_test_cmake_file.replace(
                        "CMakeLists.txt", "testslist.csv"
                    )
                ) as csv_file:
                    found = False
                    for line in csv_file:
                        (
                            name,
                            _,
                            _,
                            _,
                            _,
                            launcher,
                            num_port,
                            _,
                            _,
                            _,
                        ) = line.strip().split(",")
                        if name == self.last_test_name:
                            found = True
                            break
                assert (
                    found
                ), f"no such test named '{self.last_test_name}' in file '{self.last_test_cmake_file}'"
                if launcher[-2:] == ".sh":
                    self.process_dist_port_num(num_port)

            # step 2
            err_msg = """==================[No Old CMakeLists.txt Error]==================================
        Following directories has no CmakeLists.txt files:
    """
            for c in self.no_cmake_dirs:
                err_msg += "   " + c + "\n"
            err_msg += """
        This may cause the dist ports different with the old version.
        If the directories are newly created or there is no CMakeLists.txt before, or ignore this error, you
        must specify the directories using the args option --ignore-cmake-dirs/-i.
        If you want to keep the dist ports of old tests unchanged, please ensure the old
        version CMakeLists.txt file existing before using the gen_ut_cmakelists tool to
        generate new CmakeLists.txt files.
    ====================================================================================
    """
            assert len(self.no_cmake_dirs) == 0, err_msg


class CMakeGenerator:
    def __init__(self, current_dirs, only_check, ignore_dirs):
        self.processed_dirs = set()
        self.port_manager = DistUTPortManager(ignore_dirs)
        self.current_dirs = _norm_dirs(current_dirs)
        self.modified_or_created_files = []
        self._only_check = only_check

    def prepare_dist_ut_port(self):
        for c in self._find_root_dirs():
            self.port_manager.parse_assigned_dist_ut_ports(c, depth=0)

    def parse_csvs(self):
        '''
        parse csv files, return the lists of created or modified files
        '''
        self.modified_or_created_files = []
        for c in self.current_dirs:
            self._gen_cmakelists(c)
        return self.modified_or_created_files

    def _find_root_dirs(self):
        root_dirs = []
        # for each current directory, find its highest ancient directory (at least itself)
        # which includes CMakeLists.txt or testslist.csv.txt in the filesys tree
        for c in self.current_dirs:
            while True:
                ppath = os.path.dirname(c)
                if ppath == c:
                    break
                cmake = os.path.join(ppath, "CMakeLists.txt")
                csv = os.path.join(ppath, "testslist.csv.txt")
                if not (os.path.isfile(cmake) or os.path.isfile(csv)):
                    break
                c = ppath
            if c not in root_dirs:
                root_dirs.append(c)
        return root_dirs

    def _parse_line(self, line, curdir):
        """
        Desc:
            Input a line in csv file and output a string in cmake grammar, adding the specified test and setting its properties.
        Example:
            Input: "test_allreduce,linux,gpu;rocm,120,DIST,test_runner.py,20071,1,PYTHONPATH=..;http_proxy=;https_proxy=,"
            Output:
                "if((WITH_GPU OR WITH_ROCM) AND (LINUX) )
                    py_test_modules(
                    test_allreduce
                    MODULES
                    test_allreduce
                    ENVS
                    "PADDLE_DIST_UT_PORT=20071;PYTHONPATH=..:${PADDLE_BINARY_DIR}/python;http_proxy=;https_proxy=")
                    set_tests_properties(test_allreduce PROPERTIES  TIMEOUT "120" RUN_SERIAL 1)
                endif()"
        """

        (
            name,
            os_,
            archs,
            timeout,
            run_type,
            launcher,
            num_port,
            run_serial,
            envs,
            conditions,
        ) = line.strip().split(",")

        # name == "name" means the line being parsed is the header of the table
        # we should skip this line and return empty here.
        if name == "name":
            return ""
        name = _process_name(name, curdir)

        envs = _process_envs(envs)
        conditions = _process_conditions(conditions)
        archs = _process_archs(archs)
        os_ = _process_os(os_)
        run_serial = _process_run_serial(run_serial)

        cmd = ""

        for c in conditions:
            cmd += f"if ({c})\n"

        time_out_str = (
            f' TIMEOUT "{timeout}"' if len(timeout.strip()) > 0 else ''
        )

        if launcher[-3:] == ".sh":
            run_type = _process_run_type(run_type)
            dist_ut_port = self.port_manager.process_dist_port_num(num_port)
            dist_ut_port = self.port_manager.gset_port(name, dist_ut_port)
            cmd += f'''if({archs} AND {os_})
        bash_test_modules(
        {name}
        START_BASH
        {launcher}
        {time_out_str}
        LABELS
        "RUN_TYPE={run_type}"
        ENVS
        "PADDLE_DIST_UT_PORT={dist_ut_port};{envs}")%s
    endif()
    '''
            run_type_str = ""
        else:
            try:
                run_type = _process_run_type(run_type)
            except Exception as e:
                assert (
                    run_type.strip() == ""
                ), f"{e}\nIf use test_runner.py, the run_type can be ''"
            cmd += f'''if({archs} AND {os_})
        py_test_modules(
        {name}
        MODULES
        {name}
        ENVS
        "{envs}")%s
    endif()
    '''
            run_type_str = (
                "" if len(run_type) == 0 else f' LABELS "RUN_TYPE={run_type}"'
            )
        run_serial_str = (
            f' RUN_SERIAL {run_serial}' if len(run_serial) > 0 else ''
        )
        if (
            len(time_out_str) > 0
            or len(run_serial_str) > 0
            or len(run_type_str) > 0
        ):
            set_properties = f'''
        set_tests_properties({name} PROPERTIES{time_out_str}{run_serial_str}{run_type_str})'''
        else:
            set_properties = ""
        cmd = cmd % set_properties
        for _ in conditions:
            cmd += "endif()\n"
        return cmd

    def _gen_cmakelists(self, current_work_dir, depth=0):
        if depth == 0:
            self.processed_dirs.clear()
        if current_work_dir == "":
            current_work_dir = "."

        contents = os.listdir(current_work_dir)
        contents.sort()
        sub_dirs = []
        for c in contents:
            c_path = os.path.join(current_work_dir, c)
            if c_path in self.processed_dirs:
                return
            if not os.path.isdir(c_path):
                continue
            self.processed_dirs.add(c_path)
            if os.path.isfile(
                os.path.join(current_work_dir, c, "testslist.csv")
            ) or os.path.isfile(
                os.path.join(current_work_dir, c, "CMakeLists.txt")
            ):
                self._gen_cmakelists(
                    os.path.join(current_work_dir, c), depth + 1
                )
                sub_dirs.append(c)

        if not os.path.isfile(os.path.join(current_work_dir, "testslist.csv")):
            return
        cmds = """# This file is generated by ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py.
    # Please don't modify this file manually.
    # If you need to change unittests in this file, please modify testslist.csv in the current directory
    # and then run the command `python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py -f ${CURRENT_DIRECTORY}/testslist.csv`
    set(LOCAL_ALL_ARCH ON)
    set(LOCAL_ALL_PLAT ON)\n"""
        with open(f"{current_work_dir}/testslist.csv") as csv_file:
            for i, line in enumerate(csv_file.readlines()):
                try:
                    cmds += self._parse_line(line, current_work_dir)
                except Exception as e:
                    print("===============PARSE LINE ERRORS OCCUR==========")
                    print(e)
                    print(f"[ERROR FILE]: {current_work_dir}/testslist.csv")
                    print(f"[ERROR LINE {i + 1}]: {line.strip()}")
                    sys.exit(1)

        for sub in sub_dirs:
            cmds += f"add_subdirectory({sub})\n"

        # check whether the generated file are the same with the existing file, ignoring the blank chars
        # if they are same, skip the waiting process
        if os.path.isfile(f"{current_work_dir}/CMakeLists.txt"):
            with open(
                f"{current_work_dir}/CMakeLists.txt", "r"
            ) as old_cmake_file:
                char_seq = old_cmake_file.read().split()
        else:
            char_seq = []
        char_seq = "".join(char_seq)

        if char_seq != "".join(cmds.split()):
            assert (
                f"{current_work_dir}/CMakeLists.txt"
                not in self.modified_or_created_files
            ), f"the file {current_work_dir}/CMakeLists.txt are modified twice, which may cause some error"
            self.modified_or_created_files.append(
                f"{current_work_dir}/CMakeLists.txt"
            )
            if not self._only_check:
                with open(
                    f"{current_work_dir}/CMakeLists.txt", "w"
                ) as cmake_file:
                    print(cmds, end="", file=cmake_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        "-f",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help="Input a list of files named testslist.csv and output files named CmakeLists.txt in the same directories as the csv files respectly",
    )
    parser.add_argument(
        "--dirpaths",
        "-d",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help="Input a list of dir paths including files named testslist.csv and output CmakeLists.txt in these directories respectly",
    )
    parser.add_argument(
        "--ignore-cmake-dirs",
        '-i',
        type=str,
        required=False,
        default=[],
        nargs='*',
        help="To keep dist ports the same with old version cmake, old cmakelists.txt files are needed to parse dist_ports. If a directories are newly created and there is no cmakelists.txt file, the directory path must be specified by this option. The dirs are not recursive.",
    )
    parser.add_argument(
        "--only-check-changed",
        '-o',
        type=lambda x: x.lower() not in ["false", "0", "off"],
        required=False,
        default=False,
        help="Only check whether the CMake files should be rewritten, do not write it even if it should be write",
    )
    args = parser.parse_args()

    assert not (
        len(args.files) == 0 and len(args.dirpaths) == 0
    ), "You must provide at least one file or dirpath"
    current_work_dirs = []
    if len(args.files) >= 1:
        for p in args.files:
            assert (
                os.path.basename(p) == "testslist.csv"
            ), "you must input file named testslist.csv"
        current_work_dirs = current_work_dirs + [
            os.path.dirname(file) for file in args.files
        ]
    if len(args.dirpaths) >= 1:
        current_work_dirs = current_work_dirs + list(args.dirpaths)

    cmake_generator = CMakeGenerator(
        current_work_dirs, args.only_check_changed, args.ignore_cmake_dirs
    )
    cmake_generator.prepare_dist_ut_port()
    created = cmake_generator.parse_csvs()

    # summary the modified files
    for f in created:
        print("modified/new:", f)
