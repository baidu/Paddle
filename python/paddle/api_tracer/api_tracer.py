# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import math

import numpy as np
import yaml


class HookAPIMap:
    pass


class ConfigDump:
    def __init__(self):
        pass

    def open_file(self, path):
        self.file = open(path, "a+")

    def dump_config(self, api, input_args, input_kwargs, outputs):
        result = api + "("
        for value in input_args:
            tmp = self.dump_item_str(api, value)
            if tmp == "":
                return
            result = result + tmp + ", "
        for key, value in input_kwargs.items():
            tmp = self.dump_item_str(api, value)
            if tmp == "":
                return
            result = result + key + "=" + tmp + ", "

        result = result + ")"
        # self.file.write(") -> ")
        # if isinstance(outputs, (list, tuple)):
        #     for output in outputs:
        #         self.file.write(self.dump_item_str(api, output) + ", ")
        # else:
        #     self.file.write(self.dump_item_str(api, outputs) + ", ")

        self.file.write(result)
        self.file.write("\n")
        self.file.flush()

    def dump_item_str(self, api, item):
        import paddle

        type_mapping = {
            np.int16: int,
            np.int32: int,
            np.int64: int,
            np.float16: float,
            np.float32: float,
            np.float64: float,
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.complexfloating: complex,
            np.str_: str,
            np.bytes_: bytes,
            # np.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(item, numpy_type):
                item = builtin_type(item)
                break

        if isinstance(item, paddle.Tensor):
            return (
                "Tensor(" + str(item.shape) + ',"' + str(item.dtype)[7:] + '")'
            )
        elif isinstance(item, paddle.base.core.DataType):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.core.VarDesc.VarType):
            return "VarType(" + str(item)[7:] + ")"
        elif isinstance(item, list):
            result = "list["
            for sub_item in item:
                tmp = self.dump_item_str(api, sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + "]"
            return result
        elif isinstance(item, tuple):
            result = "tuple("
            for sub_item in item:
                tmp = self.dump_item_str(api, sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + ")"
            return result
        elif isinstance(item, slice):
            return (
                "slice("
                + str(item.start)
                + ","
                + str(item.stop)
                + ","
                + str(item.step)
                + ")"
            )
        elif isinstance(item, complex):
            return (
                "complex("
                + self.dump_item_str(api, item.real)
                + ","
                + self.dump_item_str(api, item.imag)
                + ")"
            )
        elif item is None:
            return "None"
        elif isinstance(
            item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)
        ):
            return ""
        elif item == math.inf:
            return "math.inf"
        elif item == -math.inf:
            return "-math.inf"
        elif item == math.nan:
            return "math.nan"
        elif item == -math.nan:
            return "-math.nan"
        elif isinstance(item, (bool, int, float)):
            return str(item)
        elif isinstance(item, str):
            return '"' + item + '"'
        elif isinstance(item, type):
            return (
                "type("
                + str(item)[str(item).index("'") + 1 : str(item).rindex("'")]
                + ")"
            )
        else:
            print("[api_tracer error] : dump_item_str ", api, ", item = ", item)
            return ""


config_dump = ConfigDump()


class APITemplate:
    def __init__(self, api_name):
        self.api_name = api_name

    def __call__(self, *args, **kwargs):
        output = getattr(HookAPIMap, self.api_name)(*args, **kwargs)
        try:
            config_dump.dump_config(self.api_name, args, kwargs, output)
        except Exception as err:
            print(
                "[api_tracer error] : config_dump.dump_config ",
                self.api_name,
                str(err),
            )
        return output


def wrapped_api(api_name):
    def api_template(*args, **kwargs):
        return APITemplate(api_name)(*args, **kwargs)

    return api_template


def start_api_tracer(api_path, save_config_path):
    import paddle

    print(paddle.__version__)
    with open(api_path, "r") as f:
        apis = yaml.safe_load(f)
        sample_apis = apis.get("apis")
        f.close()

    for api in sample_apis:
        parent_package, method_name = api.rsplit(".", maxsplit=1)
        try:
            setattr(HookAPIMap, api, getattr(eval(parent_package), method_name))
            setattr(eval(parent_package), method_name, wrapped_api(api))
        except Exception as err:
            print("[api_tracer error] : start_api_tracer ", api, str(err))

    config_dump.open_file(save_config_path)
