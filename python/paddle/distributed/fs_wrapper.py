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

import paddle.fluid as fluid
import sys
import abc


class FS(object):
    @abc.abstractmethod
    def list_dirs(self, fs_path):
        pass

    @abc.abstractmethod
    def ls(self, fs_path):
        pass

    @abc.abstractmethod
    def stat(self, fs_path):
        pass

    @abc.abstractmethod
    def upload(self, local_path, fs_path):
        pass

    @abc.abstractmethod
    def download(self, fs_path, local_path):
        pass

    @abc.abstractmethod
    def mkdir(self, fs_path):
        pass

    @abc.abstractmethod
    def mv(self, fs_src_path, fs_dst_path):
        pass

    @abc.abstractmethod
    def rmr(self, fs_path):
        pass


class Local(object):
    def list_dirs(self, fs_path):
        return [f for f in os.listdir(fs_path) if os.path.isdir(f)]

    def ls(self, fs_path):
        return [f for f in os.listdir(fs_path)]

    def stat(self, fs_path):
        return os.path.exists(fs_path)

    def upload(self, local_path, fs_path):
        os.symlink(local_path, fs_path)

    def download(self, fs_path, local_path):
        os.symlink(fs_path, local_path)

    def mkdir(self, fs_path):
        if not stat(fs_path):
            os.mkdir(fs_path)

    def mv(self, fs_src_path, fs_dst_path):
        os.rename(fs_src_path, fs_dst_path)

    def rmr(self, fs_path):
        shutil.rmtree(fs_path)


class BDFS(FS):
    def __init__(self,
                 hdfs_name,
                 hdfs_ugi,
                 time_out=20 * 60 * 1000,
                 sleep_inter=1000):
        self._base_cmd = "hadoop fs -Dfs.default.name=\"{}\" -Dhadoop.job.ugi=\"{}\"".format(
            hdfs_name, hdfs_ugi)
        self._time_out = time_out
        self._sleep_inter = sleep_inter

    def _run_cmd(self, cmd):
        ret = fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)
        if len(ret) <= 0:
            return []

        lines = ret.splitlines()
        return lines

    def list_dirs(self, fs_path):
        dirs, _ = self.ls(fs_path)
        return dirs

    def ls(self, fs_path):
        cmd = "{} -ls {}/".format(self._base_cmd, fs_path)
        lines = self._run_cmd(cmd)

        dirs = []
        files = []
        for line in lines:
            #print("line:", line)
            arr = line.split()
            #print(arr)
            if len(arr) != 8:
                continue

            if arr[0][0] == 'd':
                dirs.append(arr[7])
            else:
                files.append(arr[7])

        return dirs, files

    def stat(self, fs_path):
        cmd = "{} -stat {}/".format(self._base_cmd, fs_path)
        lines = self._run_cmd(cmd)
        for line in lines:
            if "No such file or directory" in line:
                return False
        return True

    def upload(self, local_path, fs_path):
        cmd = "{} -put {} {}/".format(self._base_cmd, local_path, fs_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def download(self, fs_path, local_path):
        cmd = "{} -get {} {}/".format(self._base_cmd, fs_path, local_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def mkdir(self, fs_path):
        cmd = "{} -mkdir {}".format(self._base_cmd, fs_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def mv(self, fs_src_path, fs_dst_path):
        cmd = "{} -mv {} {}".format(self._base_cmd, fs_src_path, fs_dst_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def rmr(self, fs_path):
        cmd = "{} -rmr {}".format(self._base_cmd, fs_path)
        return fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)
