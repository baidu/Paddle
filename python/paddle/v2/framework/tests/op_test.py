import unittest
import numpy as np
import random
import itertools
import paddle.v2.framework.core as core
import collections
from paddle.v2.framework.backward import append_backward_ops
from paddle.v2.framework.op import Operator
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.framework import Program, OpProtoHolder


def create_op(scope, op_type, inputs, outputs, attrs):
    kwargs = dict()

    def __create_var__(name, var_name):
        scope.var(var_name)
        kwargs[name].append(var_name)

    for in_name, in_dup in Operator.get_op_inputs(op_type):
        if in_name in inputs:
            kwargs[in_name] = []
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, _ in sub_in:
                    __create_var__(in_name, sub_in_name)
            else:
                __create_var__(in_name, in_name)

    out_var_names = []

    for out_name, out_dup in Operator.get_op_outputs(op_type):
        if out_name in outputs:
            kwargs[out_name] = []
            if out_dup:
                sub_out = outputs[out_name]
                for sub_out_name, _ in sub_out:
                    __create_var__(out_name, sub_out_name)
                    out_var_names.append(sub_out_name)
            else:
                __create_var__(out_name, out_name)
                out_var_names.append(out_name)

    for attr_name in Operator.get_op_attr_names(op_type):
        if attr_name in attrs:
            kwargs[attr_name] = attrs[attr_name]

    net = core.Net.create()
    net.append_op(Operator(op_type, **kwargs))
    if len(out_var_names) == 1:
        scope.var("__mean_out__")
        net.append_op(Operator('mean', X=out_var_names[0], Out='__mean_out__'))
    else:
        for i in xrange(len(out_var_names)):
            mean_out = "__mean_out__{0}".format(i)
            scope.var(mean_out)
            net.append_op(Operator('mean', X=out_var_names[i], Out=mean_out))
        scope.var("__sum_out__")
        net.append_op(
            Operator(
                'sum',
                X=[
                    "__mean_out__{0}".format(i)
                    for i in xrange(len(out_var_names))
                ],
                Out="__sum_out__"))
        scope.var("__mean_out__")
        net.append_op(Operator('mean', X="__sum_out__", Out="__mean_out__"))
    net.complete_add_op(True)
    return net


def set_input(op_type, scope, op, inputs, place):
    def __set_input__(var_name, var):
        if isinstance(var, tuple) or isinstance(var, np.ndarray):
            tensor = scope.find_var(var_name).get_tensor()
            if isinstance(var, tuple):
                tensor.set_lod(var[1])
                var = var[0]
            tensor.set_dims(var.shape)
            tensor.set(var, place)
        elif isinstance(var, float):
            scope.find_var(var_name).set_float(var)
        elif isinstance(var, int):
            scope.find_var(var_name).set_int(var)

    for in_name, in_dup in Operator.get_op_inputs(op_type):
        if in_name in inputs:
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, sub_in_val in sub_in:
                    __set_input__(sub_in_name, sub_in_val)
            else:
                __set_input__(in_name, inputs[in_name])


def get_numeric_gradient(op_type,
                         scope,
                         op,
                         inputs,
                         input_to_check,
                         delta=0.005,
                         in_place=False):
    set_input(op_type, scope, op, inputs, core.CPUPlace())

    def product(dim):
        return reduce(lambda a, b: a * b, dim, 1)

    ctx = core.DeviceContext.create(core.CPUPlace())

    def get_output():
        op.run(scope, ctx)
        return np.array(scope.find_var('__mean_out__').get_tensor()).mean()

    tensor_to_check = scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.get_dims())
    tensor_to_check_dtype = tensor_to_check.dtype()
    if tensor_to_check_dtype == core.DataType.FP32:
        tensor_to_check_dtype = np.float32
    elif tensor_to_check_dtype == core.DataType.FP64:
        tensor_to_check_dtype = np.float64
    else:
        raise ValueError("Not supported data type " + str(
            tensor_to_check_dtype))

    gradient_flat = np.zeros(shape=(tensor_size, ), dtype=tensor_to_check_dtype)

    def __get_elem__(tensor, i):
        if tensor_to_check_dtype == np.float32:
            return tensor.get_float_element(i)
        else:
            return tensor.get_double_element(i)

    def __set_elem__(tensor, i, e):
        if tensor_to_check_dtype == np.float32:
            tensor.set_float_element(i, e)
        else:
            tensor.set_double_element(i, e)

    # we only compute gradient of one element each time.
    # we use a for loop to compute the gradient of every element.
    for i in xrange(tensor_size):
        if in_place:
            set_input(scope, op, inputs, core.CPUPlace())

        # get one input element throw it's index i.
        origin = __get_elem__(tensor_to_check, i)
        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        __set_elem__(tensor_to_check, i, x_pos)
        y_pos = get_output()

        if in_place:
            set_input(scope, op, inputs, core.CPUPlace())

        x_neg = origin - delta
        __set_elem__(tensor_to_check, i, x_neg)
        y_neg = get_output()

        __set_elem__(tensor_to_check, i, origin)
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    return gradient_flat.reshape(tensor_to_check.get_dims())


def append_input_output(block, op_proto, np_list, is_input):
    '''Insert VarDesc and generate Python variable instance'''
    proto_list = op_proto.inputs if is_input else op_proto.outputs

    def create_var(block, name, np_list, var_proto):
        if name not in np_list:
            assert var_proto.intermediate, "{} not found".format(name)
            shape = None
            lod_level = None
        else:
            np_value = np_list[name]
            if isinstance(np_value, tuple):
                shape = list(np_value[0].shape)
                lod_level = len(np_value[1])
            else:
                shape = list(np_value.shape)
                lod_level = 0
        return block.create_var(
            dtype="float32", shape=shape, lod_level=lod_level, name=name)

    var_dict = {}
    for var_proto in proto_list:
        var_name = str(var_proto.name)
        if is_input:
            if (var_name not in np_list) and var_proto.dispensable:
                continue
            assert (var_name in np_list) or (var_proto.dispensable), \
                "Missing {} as input".format(var_name)
        if var_proto.duplicable:
            assert isinstance(np_list[var_name], list), \
                "Duplicable {} should be set as list".format(var_name)
            var_list = []
            for (name, np_value) in np_list[var_name]:
                var_list.append(
                    create_var(block, name, {name: np_value}, var_proto))
            var_dict[var_name] = var_list
        else:
            var_dict[var_name] = create_var(block, var_name, np_list, var_proto)

    return var_dict


class OpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        np.random.seed(123)
        random.seed(124)

    @classmethod
    def tearDownClass(cls):
        '''Restore random seeds'''
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
                    tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    tensor.set_lod(self.inputs[var_name][1])
                else:
                    tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def check_output_with_place(self, place, atol):
        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

        program = Program()
        block = program.global_block()

        inputs = append_input_output(block, op_proto, self.inputs, True)
        outputs = append_input_output(block, op_proto, self.outputs, False)

        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=self.attrs if hasattr(self, "attrs") else dict())

        fetch_list = []
        for var_name, var in outputs.iteritems():
            if var_name in self.outputs:
                if isinstance(var, list):
                    for v in var:
                        fetch_list.append(v)
                else:
                    fetch_list.append(var)

        feed_map = self.feed_var(inputs, place)

        exe = Executor(place)
        outs = exe.run(program, feed=feed_map, fetch_list=fetch_list)

        for out_name, out_dup in Operator.get_op_outputs(self.op_type):
            if out_name not in self.outputs:
                continue

            def find_actual(target_name, fetch_list):
                found = [
                    i for i, var in enumerate(fetch_list)
                    if var.name == target_name
                ]
                self.assertTrue(
                    len(found) == 1, "Found {} {}".format(
                        len(found), target_name))
                return found[0]

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))
                for sub_out_name, expect in sub_out:
                    idx = find_actual(sub_out_name, fetch_list)
                    actual_t = np.array(outs[idx])
                    expect_t = expect[0] \
                        if isinstance(expect, tuple) else expect
                    self.assertTrue(
                        np.allclose(
                            actual_t, expect_t, atol=atol),
                        "Output (" + sub_out_name + ") has diff at " +
                        str(place))
                    if isinstance(expect, tuple):
                        self.assertListEqual(
                            actual_t.lod(), expect[1], "Output (" + sub_out_name
                            + ") has different lod at " + str(place))
            else:
                idx = find_actual(out_name, fetch_list)
                actual_t = outs[idx]
                expect = self.outputs[out_name]
                expect_t = expect[0] if isinstance(expect, tuple) else expect
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, atol=atol),
                    "Output (" + out_name + ") has diff at " + str(place))
                if isinstance(expect, tuple):
                    self.assertListEqual(actual_t.lod(), expect[1],
                                         "Output (" + out_name +
                                         ") has different lod at " + str(place))

    def check_output(self, atol=1e-5):
        places = [core.CPUPlace()]
        if core.is_compile_gpu() and core.op_support_gpu(self.op_type):
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_output_with_place(place, atol)

    def __assert_is_close(self, numeric_grads, analytic_grads, names,
                          max_relative_error, msg_prefix):

        for a, b, name in itertools.izip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return ("%s Variable %s max gradient diff %f over limit %f, "
                        "the first error element is %d") % (
                            msg_prefix, name, max_diff, max_relative_error,
                            offset)

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   in_place=False,
                   max_relative_error=0.005):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()
        self.op = create_op(self.scope, self.op_type, op_inputs, op_outputs,
                            op_attrs)

        if no_grad_set is None:
            no_grad_set = set()

        if not type(output_names) is list:
            output_names = [output_names]

        numeric_grads = [
            get_numeric_gradient(
                self.op_type,
                self.scope,
                self.op,
                self.inputs,
                input_to_check,
                in_place=in_place) for input_to_check in inputs_to_check
        ]
        cpu_place = core.CPUPlace()
        cpu_analytic_grads = self._get_gradient(inputs_to_check, cpu_place,
                                                output_names, no_grad_set)

        self.__assert_is_close(numeric_grads, cpu_analytic_grads,
                               input_to_check, max_relative_error,
                               "Gradient Check On %s" % str(cpu_place))

        if core.is_compile_gpu() and self.op.support_gpu():
            gpu_place = core.GPUPlace(0)
            gpu_analytic_grads = self._get_gradient(inputs_to_check, gpu_place,
                                                    output_names, no_grad_set)
            self.__assert_is_close(numeric_grads, gpu_analytic_grads,
                                   input_to_check, max_relative_error,
                                   "Gradient Check On %s" % str(gpu_place))

    @staticmethod
    def _create_var_descs_(block, var_dict):
        for param_name in var_dict:
            var = var_dict[param_name]
            if not isinstance(var, list) and not isinstance(var, tuple):
                var = [(param_name, var, None)]
            if not isinstance(var[0], list) and not isinstance(var[0], tuple):
                var = [(param_name, var[0], var[1])]

            for i, item in enumerate(var):
                if not isinstance(item[0], basestring):
                    item = [[param_name] + list(item)]
                if len(item) == 2:
                    # only set var name and value, set lod to None
                    var[i] = list(item) + [None]

            var_descs = [(block.create_var(
                name=name, shape=each.shape, dtype=each.dtype), each, lod)
                         for name, each, lod in var]

            yield param_name, var_descs

    @staticmethod
    def _merge_list(iterable):
        return reduce(lambda a, b: list(a) + list(b), iterable, [])

    @staticmethod
    def _numpy_to_lod_tensor(np_value, lod, place):
        tensor = core.LoDTensor()
        tensor.set(np_value, place)
        if lod is not None:
            tensor.set_lod(lod)
        return tensor

    def _get_gradient(self, input_to_check, place, output_names, no_grad_set):
        prog = Program()
        block = prog.global_block()
        inputs_with_np = {
            key: value
            for (key, value) in OpTest._create_var_descs_(
                block, getattr(self, 'inputs', {}))
        }
        outputs_with_np = {
            key: val
            for (key, val) in OpTest._create_var_descs_(
                block, getattr(self, 'outputs', {}))
        }
        inputs = {
            k: [item[0] for item in inputs_with_np[k]]
            for k in inputs_with_np
        }
        outputs = {
            k: [item[0] for item in outputs_with_np[k]]
            for k in outputs_with_np
        }

        block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=getattr(self, 'attrs', {}))

        mean_inputs = OpTest._merge_list((outputs[oname]
                                          for oname in output_names))

        loss_list = []
        for loss in mean_inputs:
            avg_loss = block.create_var(dtype=loss.data_type, shape=[1])

            block.append_op(
                inputs={"X": loss}, outputs={"Out": avg_loss}, type='mean')
            loss_list.append(avg_loss)

        if len(loss_list) == 1:
            loss = loss_list[0]
        else:
            sum_loss = block.create_var(dtype=loss_list[0].data_type, shape=[1])
            block.append_op(
                inputs={"X": loss_list}, outputs={"Out": sum_loss}, type="sum")
            loss = block.create_var(dtype=sum_loss.data_type, shape=[1])
            block.append_op(
                inputs={"X": sum_loss},
                outputs={"Out": loss},
                type="scale",
                attrs={"scale": 1 / float(len(loss_list))})

        param_grad_list = append_backward_ops(
            loss=loss, parameter_list=input_to_check, no_grad_set=no_grad_set)

        feed_dict = {
            item[0].name: OpTest._numpy_to_lod_tensor(item[1], item[2], place)
            for p_name in inputs_with_np for item in inputs_with_np[p_name]
        }

        fetch_list = [g for p, g in param_grad_list]
        executor = Executor(place)
        result = executor.run(prog, feed_dict, fetch_list)
        return map(np.array, result)
