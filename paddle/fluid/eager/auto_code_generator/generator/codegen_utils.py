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

import re

import yaml

####################
# Global Variables #
####################
ops_to_fill_zero_for_empty_grads = {
    "split_grad",
    "split_with_num_grad",
    "rnn_grad",
    "matmul_double_grad",
    "matmul_triple_grad",
    "sigmoid_double_grad",
    "sigmoid_triple_grad",
    "add_double_grad",
    "add_triple_grad",
    "multiply_grad",
    "multiply_double_grad",
    "multiply_triple_grad",
    "conv2d_grad_grad",
    "conv2d_transpose_double_grad",
    "batch_norm_double_grad",
    "tanh_grad",
    "tanh_double_grad",
    "tanh_triple_grad",
    "sin_triple_grad",
    "cos_triple_grad",
    "subtract_double_grad",
    "divide_double_grad",
    "log_double_grad",
    "elu_double_grad",
    "leaky_relu_double_grad",
    "sqrt_double_grad",
    "rsqrt_double_grad",
    "square_double_grad",
    "celu_double_grad",
    "pad_double_grad",
    "pad3d_double_grad",
    "squeeze_double_grad",
    "unsqueeze_double_grad",
    "instance_norm_double_grad",
    "conv3d_double_grad",
    "depthwise_conv2d_grad_grad",
    "concat_double_grad",
    "stack_double_grad",
    "expand_grad",
    "argsort_grad",
    "eigh_grad",
    "add_grad",
    "subtract_grad",
    "multiply_grad",
    "divide_grad",
    "matmul_grad",
    "unbind_grad",
}

# For API dispatch used at python-level
# { op_name : [arg_name, ...] }
core_ops_returns_info = {}
core_ops_args_info = {}
core_ops_args_type_info = {}

yaml_types_mapping = {
    'int': 'int',
    'int32_t': 'int32_t',
    'int64_t': 'int64_t',
    'size_t': 'size_t',
    'float': 'float',
    'double': 'double',
    'bool': 'bool',
    'str': 'std::string',
    'str[]': 'std::vector<std::string>',
    'float[]': 'std::vector<float>',
    'bool[]': 'std::vector<bool>',
    'Place': 'paddle::Place',
    'DataLayout': 'phi::DataLayout',
    'DataType': 'phi::DataType',
    'int64_t[]': 'std::vector<int64_t>',
    'int[]': 'std::vector<int>',
    'Tensor': 'Tensor',
    'Tensor[]': 'std::vector<Tensor>',
    'Tensor[Tensor[]]': 'std::vector<std::vector<Tensor>>',
    'Scalar': 'paddle::experimental::Scalar',
    'Scalar(int)': 'paddle::experimental::Scalar',
    'Scalar(int64_t)': 'paddle::experimental::Scalar',
    'Scalar(float)': 'paddle::experimental::Scalar',
    'Scalar(double)': 'paddle::experimental::Scalar',
    'Scalar[]': 'std::vector<phi::Scalar>',
    'IntArray': 'paddle::experimental::IntArray',
}


#########################
#  File Reader Helpers  #
#########################
def AssertMessage(lhs_str, rhs_str):
    return f"lhs: {lhs_str}, rhs: {rhs_str}"


def ReadFwdFile(filepath):
    f = open(filepath, 'r')
    # empty file loaded by yaml is None
    contents = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    # not all fused ops support dygraph
    if (
        filepath.endswith("fused_ops.yaml") is True
        or filepath.endswith("fused_backward.yaml") is True
    ):
        new_apis = [
            api
            for api in contents
            if "support_dygraph_mode" in api
            and api["support_dygraph_mode"] is True
        ]
        contents = new_apis
    return contents if contents is not None else []


def ReadBwdFile(filepath, bw_ops=None):
    f = open(filepath, 'r')
    if bw_ops is None:
        contents = yaml.load(f, Loader=yaml.FullLoader)
        # not all fused ops support dygraph
        if filepath.endswith("fused_backward.yaml") is True:
            new_apis = [
                api
                for api in contents
                if "support_dygraph_mode" in api
                and api["support_dygraph_mode"] is True
            ]
            contents = new_apis
    else:
        contents = bw_ops

    ret = {}
    if contents is not None:
        for content in contents:
            assert 'backward_op' in content.keys(), AssertMessage(
                'backward_op', content.keys()
            )
            if 'backward_op' in content.keys():
                api_name = content['backward_op']

            ret[api_name] = content
    f.close()
    return ret


##############################
#  Generic Helper Functions  #
##############################
def FindGradName(string):
    return string + "_grad"


def FindForwardName(string):
    if not string.endswith("_grad"):
        return None
    return string[:-5]


def IsGradName(string):
    return string.endswith("_grad")


def IsPlainTensorType(string):
    plain_tensor_types = ['Tensor&', 'Tensor', 'const Tensor&', 'const Tensor']
    if string in plain_tensor_types:
        return True
    return False


def IsVectorTensorType(string):
    vector_tensor_types = [
        'std::vector<std::vector<Tensor>>',
        'std::vector<Tensor>',
    ]
    if string in vector_tensor_types:
        return True
    return False


def GetSavedName(string):
    return string + "_"


def GetConstReference(string):
    ret = string
    if not string.startswith("const "):
        ret = "const " + string
    if not string.endswith("&"):
        ret += "&"
    return ret


def RemoveConstAndReference(string):
    ret = string
    if string.startswith("const "):
        ret = ret[6:]
    if string.endswith("&"):
        ret = ret[:-1]

    return ret


def GetGradNodeName(string):
    def str2Hump(text):
        arr = filter(None, text.split('_'))
        res = ''
        for i in arr:
            res = res + i[0].upper() + i[1:]
        return res

    string = str2Hump(string)
    if string.rfind("Grad") == (len(string) - 4):
        string = string[:-4]
    return f"{string}GradNode"


def GetDygraphForwardFunctionName(string):
    return f"{string}_ad_func"


def GetDygraphLogName(string):
    def str2Hump(text):
        arr = filter(None, text.split('_'))
        res = ''
        for i in arr:
            res = res + i.lower()
        return res

    string = str2Hump(string)
    return string


def GetIntermediateAPIFunctionName(string):
    return string + "_intermediate"


def GetAutoGradMetaName(string):
    return f"{string}_autograd_meta"


def GetAutoGradMetaVectorName(string):
    return f"{string}_autograd_meta_vec"


def RemoveSpecialSymbolsInName(string):
    # Remove any name after '@'
    ret = string.split("@")[0]
    return ret


def RecoverBaseNameOfInplaceFunction(function_name):
    return function_name[:-1]


def GetInplacedFunctionName(function_name):
    inplace_func_name = function_name
    if inplace_func_name[-1] != '_':
        inplace_func_name += '_'
    return inplace_func_name


def GetForwardFunctionName(string):
    return f"{string}_ad_func"


def GetIndent(num):
    tab = "  "
    return "".join([tab for i in range(num)])


##################
#  Yaml Parsers  #
##################
def ParseYamlArgs(string):
    # Example: const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y

    # inputs_list = [ [arg_name, arg_type, orig_position], ...]
    inputs_list = []
    # attrs_list = [ [arg_name, arg_type, default_value, orig_position], ...]
    attrs_list = []

    pattern = re.compile(r',(?![^{]*\})')  # support int[] a={1,3}
    args = re.split(pattern, string.strip())
    args = [x.strip() for x in args]
    atype = r'((const )?\S+) '
    aname = r'(.*)'
    pattern = f'{atype}{aname}'
    for i in range(len(args)):
        arg = args[i]
        m = re.search(pattern, arg)
        arg_type = m.group(1).strip()
        arg_name = m.group(3).split("=")[0].strip()
        default_value = (
            m.group(3).split("=")[1].strip()
            if len(m.group(3).split("=")) > 1
            else None
        )

        assert (
            arg_type in yaml_types_mapping.keys()
        ), f"The argument type {arg_type} in yaml config is not supported in yaml_types_mapping."
        if arg_type in ["DataLayout"] and default_value is not None:
            default_value = f"paddle::experimental::{default_value}"
        if arg_type in ["DataType"] and default_value is not None:
            default_value = f"phi::{default_value}"
        arg_type = yaml_types_mapping[arg_type]

        arg_name = RemoveSpecialSymbolsInName(arg_name)
        if "Tensor" in arg_type:
            assert default_value is None
            inputs_list.append([arg_name, arg_type, i])
        else:
            attrs_list.append([arg_name, arg_type, default_value, i])

    return inputs_list, attrs_list


def ParseYamlReturns(string):
    # Example0: Tensor(out), Tensor(out1)
    # Example1: Tensor, Tensor
    # Example2: Tensor[](out), Tensor

    # list = [ [ret_name, ret_type, orig_position], ...]
    returns_list = []

    returns = [x.strip() for x in string.strip().split(",")]

    for i in range(len(returns)):
        ret = returns[i].split("{")[0].strip()

        ret_name = ""
        if "(" in ret and ")" in ret:
            # Remove trailing ')'
            ret = ret[:-1]
            ret_type = ret.split("(")[0].strip()
            ret_name = ret.split("(")[1].strip()
        else:
            ret_type = ret.strip()

        assert (
            ret_type in yaml_types_mapping.keys()
        ), f"The return type {ret_type} in yaml config is not supported in yaml_types_mapping."
        ret_type = yaml_types_mapping[ret_type]

        assert "Tensor" in ret_type, AssertMessage("Tensor", ret_type)
        ret_name = RemoveSpecialSymbolsInName(ret_name)
        returns_list.append([ret_name, ret_type, i])

    return returns_list


def ParseYamlForwardFromBackward(string):
    # Example: matmul (const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y) -> Tensor(out)

    fname = r'(.*?)'
    wspace = r'\s*'
    fargs = r'(.*?)'
    frets = r'(.*)'
    pattern = (
        fr'{fname}{wspace}\({wspace}{fargs}{wspace}\){wspace}->{wspace}{frets}'
    )

    m = re.search(pattern, string)
    function_name = m.group(1)
    function_args = m.group(2)
    function_returns = m.group(3)

    forward_inputs_list, forward_attrs_list = ParseYamlArgs(function_args)
    forward_returns_list = ParseYamlReturns(function_returns)

    return forward_inputs_list, forward_attrs_list, forward_returns_list


def ParseYamlForward(args_str, returns_str):
    # args Example: (const Tensor& x, const Tensor& y, bool transpose_x = false, bool transpose_y = false)
    # returns Example: Tensor, Tensor

    fargs = r'(.*?)'
    wspace = r'\s*'
    args_pattern = fr'^\({fargs}\)$'
    args_str = re.search(args_pattern, args_str.strip()).group(1)

    inputs_list, attrs_list = ParseYamlArgs(args_str)
    returns_list = ParseYamlReturns(returns_str)

    return inputs_list, attrs_list, returns_list


def ParseYamlBackward(args_str, returns_str):
    # args Example: (const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x=false, bool transpose_y=false)
    # returns Example: Tensor(x_grad), Tensor(y_grad)

    fargs = r'(.*?)'
    wspace = r'\s*'
    args_pattern = fr'\({fargs}\)'
    args_str = re.search(args_pattern, args_str).group(1)

    inputs_list, attrs_list = ParseYamlArgs(args_str)
    returns_list = ParseYamlReturns(returns_str)

    return inputs_list, attrs_list, returns_list


def ParseYamlInplaceInfo(string):
    # inplace_map_str: "(x -> out0), (y -> out2)"
    inplace_map = {}
    for pair in string.split(","):
        pair = pair.strip()
        if pair.startswith("("):
            pair = pair[1:]

        if pair.endswith(")"):
            pair = pair[:-1]

        key = pair.split("->")[0].strip()
        val = pair.split("->")[1].strip()
        inplace_map[key] = val
    return inplace_map


def ParseYamlCompositeInfo(string):
    # example:  composite: fun(args1, args2, ...)
    fname = r'(.*?)'
    wspace = r'\s*'
    fargs = r'(.*?)'
    pattern = fr'{fname}{wspace}\({wspace}{fargs}{wspace}\)'

    m = re.search(pattern, string)
    composite_fun_info = {}
    composite_fun_info.update({"name": m.group(1)})
    func_args = m.group(2).split(",")
    for fun_arg in func_args:
        if "args" in composite_fun_info:
            composite_fun_info["args"].append(fun_arg.strip())
        else:
            composite_fun_info.update({"args": [fun_arg.strip()]})

    return composite_fun_info


####################
#  Generator Base  #
####################
class FunctionGeneratorBase:
    def __init__(self, forward_api_contents, namespace):
        self.forward_api_contents = forward_api_contents
        self.namespace = namespace

        self.is_forward_only = (
            False if 'backward' in forward_api_contents.keys() else True
        )

        self.forward_api_name = ""

        self.orig_forward_inputs_list = (
            []
        )  # [ [arg_name, arg_type, orig_position], ...]
        self.orig_forward_attrs_list = (
            []
        )  # [ [attr_name, attr_type, default_value, orig_position], ...]
        self.orig_forward_returns_list = (
            []
        )  # [ [ret_name, ret_type, orig_position], ...]

        # Processed Forward Data
        self.forward_inputs_position_map = (
            {}
        )  # { "name" : [type, fwd_position] }
        self.forward_outputs_position_map = (
            {}
        )  # { "name" : [type, fwd_position] }

        # Special Op Attributes
        self.optional_inputs = []  # [name, ...]
        self.no_need_buffers = []  # [name, ...]
        self.composite_func_info = (
            {}
        )  # {name: func_name, args: [input_name, ...]}
        self.intermediate_outputs = []  # [name, ...]
        self.forward_inplace_map = {}  # {name : name, ...}

    def ParseForwardInplaceInfo(self):
        forward_api_contents = self.forward_api_contents
        if 'inplace' not in forward_api_contents.keys():
            return

        inplace_map_str = forward_api_contents['inplace']
        self.forward_inplace_map = ParseYamlInplaceInfo(inplace_map_str)

    def ParseNoNeedBuffer(self):
        grad_api_contents = self.grad_api_contents

        if 'no_need_buffer' in grad_api_contents.keys():
            no_need_buffer_str = grad_api_contents['no_need_buffer']
            for name in no_need_buffer_str.split(","):
                name = name.strip()
                name = RemoveSpecialSymbolsInName(name)
                self.no_need_buffers.append(name.strip())

    def ParseComposite(self):
        grad_api_contents = self.grad_api_contents

        if 'composite' in grad_api_contents.keys():
            composite_str = grad_api_contents['composite']
            self.composite_func_info = ParseYamlCompositeInfo(composite_str)

    def ParseDispensable(self):
        forward_api_contents = self.forward_api_contents

        if 'optional' in forward_api_contents.keys():
            optional_inputs_str = forward_api_contents['optional']
            for name in optional_inputs_str.split(","):
                name = name.strip()
                name = RemoveSpecialSymbolsInName(name)
                self.optional_inputs.append(name)

    def ParseIntermediate(self):
        forward_api_contents = self.forward_api_contents

        if 'intermediate' in forward_api_contents.keys():
            intermediate_str = forward_api_contents['intermediate']
            for name in intermediate_str.split(","):
                name = name.strip()
                name = RemoveSpecialSymbolsInName(name)
                self.intermediate_outputs.append(name)

    def CollectOriginalForwardInfo(self):
        forward_api_contents = self.forward_api_contents
        assert (
            'op' in forward_api_contents.keys()
            or 'backward_op' in forward_api_contents.keys()
        ), 'Unable to find "op" in forward_api_contents keys'

        if 'op' in forward_api_contents.keys():
            self.forward_api_name = forward_api_contents['op']
        elif 'backward_op' in forward_api_contents.keys():
            self.forward_api_name = forward_api_contents['backward_op']

        assert (
            'args' in forward_api_contents.keys()
        ), 'Unable to find "args" in forward_api_contents keys'

        forward_args_str = forward_api_contents['args']

        assert (
            'output' in forward_api_contents.keys()
        ), 'Unable to find "output" in forward_api_contents keys'

        forward_returns_str = forward_api_contents['output']

        # Collect Original Forward Inputs/Outputs and then perform validation checks
        (
            self.orig_forward_inputs_list,
            self.orig_forward_attrs_list,
            self.orig_forward_returns_list,
        ) = ParseYamlForward(forward_args_str, forward_returns_str)

    def DetermineForwardPositionMap(
        self, forward_inputs_list, forward_returns_list
    ):
        for i in range(len(forward_inputs_list)):
            forward_input = forward_inputs_list[i]
            input_name = forward_input[0]
            input_type = forward_input[1]
            input_pos = forward_input[2]

            self.forward_inputs_position_map[input_name] = [
                input_type,
                input_pos,
            ]

        for i in range(len(forward_returns_list)):
            forward_return = forward_returns_list[i]
            if len(forward_return[0]) == 0:
                if len(forward_returns_list) == 1:
                    return_name = "out"
                else:
                    return_name = f"out_{i + 1}"
            else:
                return_name = forward_return[0]
            return_type = forward_return[1]
            return_pos = forward_return[2]

            self.forward_outputs_position_map[return_name] = [
                return_type,
                return_pos,
            ]


class GeneratorBase:
    def __init__(self, api_yaml_path, fw_ops=None):
        self.namespace = ""
        self.api_yaml_path = api_yaml_path

        self.forward_api_list = fw_ops

    def ParseForwardYamlContents(self):
        api_yaml_path = self.api_yaml_path
        if self.forward_api_list is None:
            self.forward_api_list = ReadFwdFile(api_yaml_path)

    def InferNameSpace(self):
        api_yaml_path = self.api_yaml_path
        if re.search(r"sparse[a-zA-Z0-9_]*\.yaml", api_yaml_path):
            self.namespace = "sparse::"
        elif re.search(r"strings[a-zA-Z0-9_]*\.yaml", api_yaml_path):
            self.namespace = "strings::"
