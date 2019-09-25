#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from .... import core
from ....framework import IrGraph
from ....framework import IrNode

__all__ = ['FakeQAT2MkldnnINT8KernelPass', 'FakeQAT2MkldnnINT8PerfPass']


class FakeQAT2MkldnnINT8KernelPass(object):
    """
    Convert QuantizationFreezePass generated IrGraph to MKL-DNN supported INT8
    IrGraph. Following transformations did in this pass:
        1. Convert int8 range weights with float32 data type, which are generated by
           the QuantizationFreezePass, to float32 range weights with float32 data type
           by using the corresponding scales. This conversion is because MKL-DNN INT8
           conv2d kernel and mul kernel now only support float32 weights input, hence
           weights quantization will happen inside the conv2d and mul INT8 kernel.
        2. Create the new conv2d or mul op with the converted weights and link its output
           to fake_dequantize_abs_max op's output and set conv2d's attribute "force_fp32
           _output" as true
        3. Transform fake_quantize_xx op to quantize op
        4. Remove fake_dequantize_abs_max op
    """

    def __init__(self, _scope=None, _place=None):
        """
        Args:
            scope(fluid.Scope): scope is used to initialize the new parameters.
            place(fluid.CPUPlace): place is used to initialize the new parameters.


        Examples:
        .. code-block:: python
            # The original graph will be rewrite.
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization \
                import FakeQAT2MkldnnINT8KernelPass
            from paddle.fluid.framework import IrGraph
            from paddle.fluid import core

            graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
            place = fluid.CPUPlace()
            mkldnn_pass = FakeQAT2MkldnnINT8KernelPass(fluid.global_scope(),
            place)
            mkldnn_pass.apply(graph)
        """

        self._scope = _scope
        self._place = _place

        self._quantize_type = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_range_abs_max'
        ]
        self._dequantize_type = ['fake_dequantize_max_abs']
        self._quantize_dequantize_type = [
            'fake_quantize_dequantize_moving_average_abs_max'
        ]

        self._quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
        self._pool_ops = ['pool2d']

        self._in_scale = {}
        self._max_range = {}
        self._new_output = {}
        self._s8_max = 127

    def apply(self, graph):
        """
        Quantize the graph for running MKL-DNN INT8 inference. According
        to activation quantization type, the graph will transform fake
        quantize ops to quantize ops and remove the fake dequantize ops.

        Args:
            graph(IrGraph): the applied graph.
        """

        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'
        ops = graph.all_op_nodes()

        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        # Collect the _in_scales and _max_range to calculate the new scales for MKL-DNN
        # INT8 conv2d and mul
        for op_node in ops:
            if op_node.name() in self._dequantize_type:
                input_name = op_node.input("X")[0]
                scale_name = op_node.input("Scale")[0]
                self._in_scale[input_name] = self._load_param(self._scope,
                                                              scale_name)[0]
                self._max_range[input_name] = op_node.op().attr("max_range")
                self._new_output[input_name] = op_node.output("Out")[0]

            if op_node.name() in self._quantize_dequantize_type:
                inputs = op_node.op().input_names()
                attrs = op_node.op().attr_names()
                input_name = op_node.input("X")[0]
                scale_name = op_node.input("InScale")[0]
                self._in_scale[input_name] = self._load_param(self._scope,
                                                              scale_name)[0]
                #  self._max_range[input_name] = op_node.op().attr("max_range")
                self._new_output[input_name] = op_node.output("Out")[0]

        for op_node in ops:
            if op_node.name() in self._quantizable_ops:
                if op_node.name() in self._conv_ops:
                    self._transform_to_conv_mkldnn(graph, op_node)
                elif op_node.name() in self._pool_ops:
                    self._transform_to_pool_mkldnn(graph, op_node)
                else:
                    self._transform_to_mul_mkldnn(graph, op_node)
            elif op_node.name() in self._quantize_type:
                self._transform_to_quantize_mkldnn(graph, op_node)
            elif op_node.name() in self._dequantize_type:
                self._remove_fake_dequantize_op(graph, op_node)
            self._remove_unused_var_nodes(graph)
        return graph

    def _transform_to_pool_mkldnn(self, graph, op):
        output_name = op.output("Out")[0]
        input_name = op.input("X")[0]

    def _transform_to_conv_mkldnn(self, graph, op_node):
        weight_name = op_node.input("Filter")[0]
        output_name = op_node.output("Output")[0]
        # Convert int8 range weights to fp32 range weights
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(
            np.multiply(weight, self._s8_max), self._max_range[output_name])
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("Input")[0])
        weight_var_node = graph._find_node_by_name(op_node.inputs, weight_name)

        # Set fake_dequantize_abs_max's output as new output of conv2d
        output_var_node = graph._find_node_by_name(
            graph.all_var_nodes(), self._new_output[output_name])
        attrs = {
            name: op_node.op().attr(name)
            for name in op_node.op().attr_names()
        }

        conv_op_node = graph.create_op_node(
            op_type='conv2d',
            attrs=attrs,
            inputs={'Input': input_var_node,
                    'Filter': weight_var_node},
            outputs={'Output': output_var_node})

        # Based on the QAT's scales to calculate the scales of MKL-DNN INT8 conv2d
        scale_in = self._s8_max / self._in_scale[output_name]
        scale_w = []
        scale_w = [self._max_range[output_name] / self._s8_max]

        conv_op_node.set_attr("Scale_weights", scale_w)
        conv_op_node.set_attr("Scale_in", scale_in)
        conv_op_node.set_attr("Scale_out", 1.0)
        conv_op_node.set_attr("use_mkldnn", 1)
        conv_op_node.set_attr("force_fp32_output", 1)
        graph.link_to(input_var_node, conv_op_node)
        graph.link_to(weight_var_node, conv_op_node)
        graph.link_to(conv_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _transform_to_mul_mkldnn(self, graph, op_node):
        # For MKL-DNN INT8 mul, input Y should be the weights
        weight_name = op_node.input("Y")[0]
        output_name = op_node.output("Out")[0]
        # Convert int8 range weights to fp32 range weights
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(
            np.multiply(weight, self._s8_max), self._max_range[output_name])
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("X")[0])
        weight_var_node = graph._find_node_by_name(op_node.inputs, weight_name)

        # Set fake_dequantize_abs_max's output as new output of mul
        output_var_node = graph._find_node_by_name(
            graph.all_var_nodes(), self._new_output[output_name])
        attrs = {
            name: op_node.op().attr(name)
            for name in op_node.op().attr_names()
        }

        mul_op_node = graph.create_op_node(
            op_type='mul',
            attrs=attrs,
            inputs={'X': input_var_node,
                    'Y': weight_var_node},
            outputs={'Out': output_var_node})

        # Based on the QAT's scales to calculate MKL-DNN INT8 mul's scales
        scale_in = self._s8_max / self._in_scale[output_name]
        scale_w = []
        scale_w = [self._max_range[output_name] / self._s8_max]

        mul_op_node.set_attr("scale_y", scale_w)
        mul_op_node.set_attr("scale_x", scale_in)
        mul_op_node.set_attr("scale_out", 1.0)
        mul_op_node.set_attr("use_mkldnn", 1)
        mul_op_node.set_attr("force_fp32_output", 1)
        graph.link_to(input_var_node, mul_op_node)
        graph.link_to(weight_var_node, mul_op_node)
        graph.link_to(mul_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _transform_to_quantize_mkldnn(self, graph, op_node):
        """
        Transform fake_quantize_xx op to quantize mkldnn op in the graph.
        """
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("X")[0])
        output_var_node = graph._find_node_by_name(op_node.outputs,
                                                   op_node.output("Out")[0])
        scale_in = self._s8_max / self._load_param(
            self._scope, op_node.input("InScale")[0])[0]
        quant_op_node = graph.create_op_node(
            op_type='quantize',
            attrs={
                'data_format': 'MKLDNNLAYOUT',
                'use_mkldnn': 1,
                'Scale': scale_in,
                'is_negative_input': 1
            },
            inputs={'Input': input_var_node},
            outputs={'Output': output_var_node})
        graph.link_to(input_var_node, quant_op_node)
        graph.link_to(quant_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _remove_fake_dequantize_op(self, graph, op_node):
        input_var_node = graph._find_node_by_name(op_node.inputs,
                                                  op_node.input("X")[0])
        graph.safe_remove_nodes(op_node)

    def _load_param(self, scope, param_name):
        return np.array(scope.find_var(param_name).get_tensor())

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _remove_unused_var_nodes(self, graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(lambda node: node.node not in all_used_vars,
                            graph.all_var_nodes())
        }
        graph.safe_remove_nodes(all_unused_vars)


class FakeQAT2MkldnnINT8PerfPass(object):
    """
    Transform a QAT model IrGraph into MKL-DNN supported INT8 IrGraph.
    The pass consists of the following transformations:
        1. gather scale values from fake quantize/dequantize operators,
        2. extract FP32 inference model graph from the QAT graph, i.e.
            a.  remove fake quantize/dequantize operators,
            b.  dequantize conv2d and mul's weights,
        3. optimize the FP32 graph using standard FP32 optimization fuses
            (e.g. `conv2d`+`bn` -> `conv2d`),
        4. quantize the optimized FP32 graph using standard INT8v2 quantization
            passes (`cpu_quantize_pass`, `cpu_quantize_squash_pass`).
    """

    def __init__(self, _scope=None, _place=None, _core=None, _debug=False):
        self._scope = _scope
        self._place = _place
        self._core = _core
        self._debug = _debug
        self._quantize_types = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_range_abs_max',
            'fake_quantize_dequantize_moving_average_abs_max'
        ]
        self._fake_quantize_types = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_dequantize_moving_average_abs_max'
        ]
        self._fake_dequantize_types = ['fake_dequantize_max_abs']
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
        self._pool_ops = ['pool2d']
        self._mul_ops = ['mul']
        self._fc_ops = ['fc']
        self._weight_scales = {}
        # Collect the Input and Output sclaes from Fake QAT models
        self._var_quant_scales = {}
        self._max_range = {}
        self._s8_max = 127

    def apply(self, graph):
        assert isinstance(graph,
                          IrGraph), 'graph must be the instance of IrGraph.'

        graph = self._gather_scales(graph)
        graph = self._remove_fake_ops(graph)
        graph = self._update_pooling_scales(graph)
        graph = self._dequantize_weights(graph)
        graph = self._optimize_fp32_graph(graph)
        graph = self._compute_weight_scales(graph)
        graph = self._quantize_fp32_graph(graph)
        graph = self._remove_unused_var_nodes(graph)
        return graph

    def _convert_scale2tensor(self, scale):
        tensor = core.LoDTensor()
        tensor.set(scale, core.CPUPlace())
        return tensor

    def _gather_scales(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._quantize_types:
                bit_length = op.op().attr("bit_length")
                assert bit_length == 8, 'Unsupported number quantization bits ({}). Only 8 is supported now.'.format(
                    bit_length)

                input_name = op.input("X")[0]
                scale_name = op.input("InScale")[0]
                # Gather new weights scale after folding batchnorm in convolution
                scale = np.array(1.0 / self._load_param(
                    self._scope, scale_name)[0]).astype(np.float64)
                lod_tensor = self._convert_scale2tensor(scale)
                use_unsigned_int = False
                self._var_quant_scales[input_name] = (use_unsigned_int,
                                                      lod_tensor)

            if op.name() in self._fake_dequantize_types:
                input_name = op.input("X")[0]
                _max_range = op.op().attr("max_range")
                self._weight_scales[input_name] = _max_range
        return graph

    def _update_pooling_scales(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._pool_ops:
                input_name = op.input("X")[0]
                output_name = op.output("Out")[0]
                if input_name in self._var_quant_scales:
                    self._var_quant_scales[
                        output_name] = self._var_quant_scales[input_name]
        return graph

    def _load_param(self, scope, param_name):
        return np.array(scope.find_var(param_name).get_tensor())

    def _remove_fake_ops(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._fake_quantize_types:
                op_out = graph._find_node_by_name(op.outputs,
                                                  op.output("Out")[0])
                self._remove_fake_quantize(graph, op)

        for op in graph.all_op_nodes():
            if op.name() in self._fake_dequantize_types:
                op_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
                self._remove_fake_dequantize(graph, op)
        return graph

    def _remove_fake_quantize(self, graph, op):
        fake_quant_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
        fake_quant_in_scale = graph._find_node_by_name(op.inputs,
                                                       op.input("InScale")[0])
        fake_quant_out = graph._find_node_by_name(op.outputs,
                                                  op.output("Out")[0])
        fake_quant_out_scale = graph._find_node_by_name(
            op.outputs, op.output("OutScale")[0])

        next_ops = fake_quant_out.outputs
        for next_op in next_ops:
            self._swap_inputs(next_op, fake_quant_out, fake_quant_in)
            graph.link_to(fake_quant_in, next_op)
        graph.safe_remove_nodes(
            {op, fake_quant_in_scale, fake_quant_out, fake_quant_out_scale})

        return graph

    def _remove_fake_dequantize(self, graph, op):
        fake_dequant_in = graph._find_node_by_name(op.inputs, op.input("X")[0])
        fake_dequant_out = graph._find_node_by_name(op.outputs,
                                                    op.output("Out")[0])

        next_ops = fake_dequant_out.outputs
        for next_op in next_ops:
            self._swap_inputs(next_op, fake_dequant_out, fake_dequant_in)
            graph.link_to(fake_dequant_in, next_op)
        graph.safe_remove_nodes({op, fake_dequant_out})

        return graph

    def _swap_inputs(self, op, old_input, new_input):
        for input_name in op.op().input_names():
            if old_input.name() in op.input(input_name):
                op.op().set_input(input_name, [
                    new_input.name() if x == old_input.name() else x
                    for x in op.input(input_name)
                ])

    def _dequantize_weights(self, graph):
        for op in graph.all_op_nodes():
            if op.name() in self._conv_ops:
                self._dequantize_conv_weights(graph, op)
            elif op.name() in self._mul_ops:
                self._dequantize_mul_weights(graph, op)
        return graph

    def _dequantize_conv_weights(self, graph, op_node):
        weight_name = op_node.input("Filter")[0]
        output_name = op_node.output("Output")[0]
        # Convert int8 range weights to fp32 range weights
        scales = self._weight_scales[output_name]
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(np.multiply(weight, self._s8_max), scales)
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)

    def _dequantize_mul_weights(self, graph, op_node):
        weight_name = op_node.input("Y")[0]
        output_name = op_node.output("Out")[0]
        scales = self._weight_scales[output_name]
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(np.multiply(weight, self._s8_max), scales)
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _optimize_fp32_graph(self, graph):
        graph = self._apply_pass(graph, 'mkldnn_placement_pass',
                                 ['mkldnn_enabled_op_types'], [set()])
        graph = self._apply_pass(graph, 'depthwise_conv_mkldnn_pass')
        graph = self._apply_pass(graph, 'conv_bn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_eltwiseadd_bn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_bias_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_elementwise_add_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_relu_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'conv_relu6_mkldnn_fuse_pass')
        graph = self._apply_pass(graph, 'fc_fuse_pass')
        return graph

    def _apply_pass(self, graph, pass_name, attrs=None, attr_values=None):
        ir_pass = core.get_pass(pass_name)
        cpp_graph = graph.graph
        if cpp_graph.has('__param_scope__'):
            cpp_graph.erase('__param_scope__')
        cpp_graph.set_not_owned('__param_scope__', self._scope)
        if attrs:
            assert attr_values and len(attrs) == len(
                attr_values
            ), "Different number of pass attributes and their values."
            for attr, value in zip(attrs, attr_values):
                ir_pass.set(attr, value)
        ir_pass.apply(cpp_graph)
        graph = IrGraph(cpp_graph, for_test=True)
        if self._debug:
            graph.draw('.', 'qat_fp32_{}'.format(pass_name),
                       graph.all_op_nodes())
        self._remove_unused_var_nodes(graph)
        return graph

    def _remove_unused_var_nodes(self, graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(lambda node: node.node not in all_used_vars,
                            graph.all_var_nodes())
        }
        graph.safe_remove_nodes(all_unused_vars)
        return graph

    def _compute_weight_scales(self, graph):
        def _compute_var_scales(ops, out_name, w_name, axis):
            for op in graph.all_op_nodes():
                if op.op().type() in ops:
                    weight_var_name = op.input(w_name)[0]
                    weights = np.array(
                        self._load_param(self._scope, weight_var_name))
                    scales = 1.0 / np.amax(
                        np.abs(weights.reshape(weights.shape[0], -1)),
                        axis=axis)

                    lod_tensor = self._convert_scale2tensor(
                        scales.astype(np.float64))
                    use_unsigned_int = False
                    self._var_quant_scales[weight_var_name] = (use_unsigned_int,
                                                               lod_tensor)

        _compute_var_scales(self._conv_ops, "Output", "Filter", axis=1)
        _compute_var_scales(self._fc_ops, "Out", "W", axis=0)
        return graph

    def _find_avg_pooling_ids(self, graph):
        ids = []
        for op in graph.all_op_nodes():
            if op.name() in self._pool_ops:
                if op.op().attr("pooling_type") == "avg":
                    ids.append(op.id())
        return set(ids)

    def _quantize_fp32_graph(self, graph):
        ir_pass = self._core.get_pass('cpu_quantize_placement_pass')
        cpp_graph = graph.graph
        ir_pass.set('quantize_enabled_op_types', {'conv2d', 'pool2d'})
        ir_pass.set('quantize_excluded_op_ids',
                    self._find_avg_pooling_ids(graph))
        ir_pass.apply(cpp_graph)
        graph = IrGraph(cpp_graph, for_test=True)
        if self._debug:
            graph.draw('.', 'qat_int8_{}'.format(ir_pass.type()),
                       graph.all_op_nodes())

        graph = self._apply_pass(graph, 'cpu_quantize_pass',
                                 ['quant_var_scales'],
                                 [self._var_quant_scales])
        graph = self._apply_pass(graph, 'cpu_quantize_squash_pass')
        return graph
