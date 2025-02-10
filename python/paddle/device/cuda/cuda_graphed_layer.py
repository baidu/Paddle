# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
from collections import deque
from enum import Enum

import paddle
from paddle.base import log_helper

from .graphs import CUDAGraph

# CUDAGraphedLayer Debug tools
enable_debug_print = bool(
    int(os.getenv('PADDLE_DEBUG_ENABLE_CUDAGRAPH_LAYER_LOGGING', '0'))
)
debug_cudagraphedlayer_fallback_to_default = bool(
    int(os.getenv('PADDLE_DEBUG_CUDAGRAPHEDLAYER_FALLBACK_TO_DEFAULT', '0'))
)

logger = log_helper.get_logger(
    __name__, logging.INFO, fmt='[%(levelname)s] %(message)s'
)


def debug_print(x):
    if not enable_debug_print:
        return
    logger.info(x)


def print_tensor(
    t,
    name="Unnamed",
    print_meta=True,
    print_ptr=False,
    print_hash=True,
    hash=None,
):
    output = []
    if name:
        output.append(name)
    if hash is None:
        hash = lambda t: float((t.astype('float32') * 1000).sum())

    if t is None:
        debug_print(f"{name} is None")
    elif isinstance(t, paddle.Tensor):
        if print_meta:
            output.append(f"shape = {t.shape}")
            output.append(f"place = {t.place}")
        if print_ptr:
            output.append(f"ptr = {hex(t.data_ptr())}")
        if print_hash:
            output.append(f"hash = {hash(t)}")
        debug_print(" | ".join(output))


def printer(x, banner="printer"):
    if not enable_debug_print:
        return
    debug_print(banner.center(100, "-"))
    recursive_apply(print_tensor, x)


# We need this function, for any kind of inputs with iterables
# we recursively apply the function to the leave nodes
def recursive_apply(function, input_var):
    if isinstance(input_var, list):
        return [recursive_apply(function, item) for item in input_var]
    elif isinstance(input_var, tuple):
        return tuple(recursive_apply(function, item) for item in input_var)
    elif isinstance(input_var, dict):
        return {
            key: recursive_apply(function, value)
            for key, value in input_var.items()
        }
    else:
        return function(input_var)


def detach_tensor(tensor):
    # Detach an individual tensor and preserve its 'stop_gradient' property
    if isinstance(tensor, paddle.Tensor):
        detached_tensor = tensor.detach()
        detached_tensor.stop_gradient = tensor.stop_gradient
        return detached_tensor
    return tensor


# We try our best to flatten the input to list of tensors
# example: args = ((t1,t2),(t3,(t4,t5))) -> [t1, t2, t3, t4, t5]
def recursive_flatten(target):
    ret = []

    def append(arg):
        if isinstance(arg, paddle.Tensor):
            # [NOTE] sometimes unnecessary tensors, such as the constant `mask` tensor in the PP layer, is passed into subsequent layers.
            # When a tensor is marked with `stop_gradient=True`, it indicates that it does not contribute to gradient calculations,
            # suggesting it's unrelated to the main computational process.
            # Therefore, I try to eliminate the copying of such tensors in the to optimize performance.
            # if not arg.stop_gradient:
            # [NOTE] However, `stop_gradient=True` propagation rules within the framework appear to be flawed, so directly eliminate stop_gradient may cause bug
            ret.append(arg)

    recursive_apply(append, target)
    return ret


# input any kind of args / kwargs structure, output list of tensor
def recursive_flatten_args_kwargs(args, kwargs):
    return [
        *recursive_flatten(args),
        *recursive_flatten(tuple(kwargs.values())),
    ]


detach = lambda x: recursive_apply(detach_tensor, x)


def get_grad_tensor(x):
    """Returns the gradient of a Paddle Tensor if it's a tensor; otherwise, returns the input."""
    if isinstance(x, paddle.Tensor):
        if x.stop_gradient:
            return None
        else:
            return x.grad
    return None


# CUDA Graph with Static Input and Output
class CUDAGraphWithStaticInputOutput:
    def __init__(self, num_warmup_steps):
        self.num_warmup_steps = num_warmup_steps
        self.graph = CUDAGraph()

        self.has_recorded = False

        self.has_preserved_inputs = False
        self.args_static = None
        self.kwargs_static = None

        # inputs is the recursively flattened args and kwargs
        self.inputs_static = None
        self.outputs_static = None

    def preserve_or_copy(self, args, kwargs):
        """
        For the CUDA Graph, it is crucial that the buffer remains address-stable,
        meaning that the buffer addresses for any inputs to the CUDA Graph should not change.
        One solution to achieve this is to preserve all input tensors.

        This function attempts to recursively flatten the input arguments and keyword arguments
        to identify all tensors passed to the layer (though it may still miss some due to other implicit
        ways inputs can be passed to a layer). It then preserves references to these input tensors
        as `self.inputs_static` so that the buffer pointers can be reused later.

        When this method is called subsequently, it copies the values back to the preserved input tensors
        to ensure the buffers are reused.
        """
        if not self.has_preserved_inputs:
            self.has_preserved_inputs = True
            self.args_static = args
            self.kwargs_static = kwargs
            self.inputs_static = recursive_flatten_args_kwargs(
                self.args_static, self.kwargs_static
            )
        else:
            inputs = recursive_flatten_args_kwargs(args, kwargs)
            for x_static, x in zip(self.inputs_static, inputs):
                x_static.copy_(x, True)

    def record(self, f, *args, **kwargs):
        self.preserve_or_copy(args, kwargs)

        self.graph.capture_begin()
        self.outputs_static = f(*self.args_static, **self.kwargs_static)
        self.graph.capture_end()
        debug_print(
            "[CUDAGraph] Record-Replay Start (Graph is replayed for the first time)"
        )
        self.graph.replay()

        self.has_recorded = True

        return self.outputs_static

    def set_output_static(self, outputs_static):
        self.outputs_static = outputs_static

    def replay(self, *args, **kwargs):
        if not self.has_recorded:
            raise RuntimeError("Graph should be recorded first")

        self.preserve_or_copy(args, kwargs)

        debug_print("[CUDAGraph] Replay Start")
        self.graph.replay()
        return self.outputs_static

    def save(self, name):
        logging.info(f"save graph to {name}")
        self.graph.print_to_dot_files(name)


# CUDA Graph Layer Status Enumeration
class CUDAGraphLayerStatus(Enum):
    """Enum to represent the status of a CUDA Graph Layer."""

    WARMUP = 1
    RECORD = 2
    CUDAGRAPH = 3


class CUDAGraphForwardBackward:
    def __init__(self, num_warmup_steps):
        self.forward_graph = CUDAGraphWithStaticInputOutput(num_warmup_steps)
        self.backward_graph = CUDAGraphWithStaticInputOutput(num_warmup_steps)
        self.status = CUDAGraphLayerStatus.RECORD

    def capture_end(self):
        self.status = CUDAGraphLayerStatus.CUDAGRAPH

    def is_record_step(self):
        return self.status == CUDAGraphLayerStatus.RECORD

    def is_cuda_graph_step(self):
        return self.status == CUDAGraphLayerStatus.CUDAGRAPH


class CUDAGraphContext:
    """
    Manages the context for CUDA graph execution in layers. This includes handling
    the state of CUDA graph layers, managing forward and backward graphs, and
    tracking the execution steps.
    """

    def __init__(self, layer, num_warmup_steps):
        """
        Initializes the CUDA graph context.
        :param layer: The layer to be used in the CUDA graph.
        :param num_warmup_steps: Number of warmup steps before recording starts.
        """
        self.layer = layer
        self.num_warmup_steps = num_warmup_steps

        # The state of context is in either WARMUP or CUDAGRAPH
        self._step = 0
        self.status = CUDAGraphLayerStatus.WARMUP

        # Queue to support 1f1b/interleaved scheduler, assuming FIFO order
        # data queue
        self.data_queue = deque()

        # graph queue
        self.graph_queue = deque()

    # Graph Operations
    def get_graph(self):
        if len(self.graph_queue) == 0:
            return CUDAGraphForwardBackward(self.num_warmup_steps)
        else:
            return self.graph_queue.popleft()

    def reuse_graph(self, g):
        self.graph_queue.append(g)

    # Tensor Queue Operations
    def push_data(self, args):
        self.data_queue.append(args)

    def pop_data(self):
        return self.data_queue.popleft()

    # Finite State Machine of Layer State
    def warmup_step(self):
        self._step += 1
        if self._step == self.num_warmup_steps:
            self.status = CUDAGraphLayerStatus.CUDAGRAPH

    def is_warmup_step(self):
        return self.status == CUDAGraphLayerStatus.WARMUP

    def is_cuda_graph_step(self):
        return self.status == CUDAGraphLayerStatus.CUDAGRAPH


def select_y_with_grad(ys, dys):
    # [TODO] when there is multiple output tensor, we support only one y that allows backward
    y, dy = None, None
    if isinstance(ys, paddle.Tensor):
        y, dy = ys, dys[0]
    elif isinstance(ys, (list, tuple)):
        for v, dv in zip(ys, dys):
            if isinstance(v, paddle.Tensor) and (not v.stop_gradient):
                y, dy = v, dv
                break
    assert isinstance(y, paddle.Tensor) and isinstance(dy, paddle.Tensor)
    return y, dy


# we get the output of the backward from the detached inputs after the backward is calculated
# we save it to the graph itself
def get_args_grad(inputs):
    grad_inputs, detached_grad_inputs = inputs
    args_grad = []
    for x, detached_x in zip(grad_inputs, detached_grad_inputs):
        # if required grad
        if not x.stop_gradient:
            if detached_x.grad is None:
                # if input requires grad but we don't have grad, we just allocate some zeros
                # x.stop_gradient = True
                args_grad.append(paddle.zeros(detached_x.shape))
                # args_grad.append(None)
            else:
                args_grad.append(detached_x.grad)
        else:
            args_grad.append(None)
    return tuple(args_grad)


class _CUDAGraphedLayer(paddle.autograd.PyLayer):
    """
    A custom layer that integrates CUDA Graph recording and execution into PaddlePaddle's autograd system.
    It handles forward and backward operations differently based on the CUDA graph layer status.
    """

    @staticmethod
    def forward(ctx, context, arg_tuple, *grad_inputs):
        """
        Handles the forward pass of the layer. It operates differently based on the
        context's status: warmup, recording, or CUDA graph step.
        """
        args, kwargs = arg_tuple
        # Detach all inputs from the computational graph
        args = detach(args)
        kwargs = detach(kwargs)

        detached_grad_inputs = recursive_flatten_args_kwargs(args, kwargs)
        inputs = (grad_inputs, detached_grad_inputs)

        printer(detached_grad_inputs, "Forward input")
        if (
            context.is_warmup_step()
            or debug_cudagraphedlayer_fallback_to_default
        ):
            debug_print("[CUDAGraph] Forward Step (Default)")

            with paddle.enable_grad():
                y = context.layer(*args, **kwargs)

            context.push_data((CUDAGraphLayerStatus.WARMUP, None, inputs, y))
        else:
            graph = context.get_graph()
            if graph.is_record_step():
                # In record step, record the forward pass in CUDA graph
                debug_print(f"[CUDAGraph] Forward Step (Record) id {id(graph)}")

                def forward(*args, **kwargs):
                    with paddle.enable_grad():
                        return context.layer(*args, **kwargs)

                y = graph.forward_graph.record(forward, *args, **kwargs)

                context.push_data(
                    (CUDAGraphLayerStatus.RECORD, graph, inputs, y)
                )
            else:
                debug_print(f"[CUDAGraph] Forward Step (Graph) id {id(graph)}")
                y = graph.forward_graph.replay(*args, **kwargs)

                context.push_data(
                    (CUDAGraphLayerStatus.CUDAGRAPH, graph, None, y)
                )

        debug_print("[CUDAGraph] Forward Step End")

        ctx.save_for_backward(context)
        printer(y, "Forward output")
        return detach(y)

    @staticmethod
    def backward(ctx, *dys):
        """
        Handles the backward pass of the layer. Similar to forward, it handles
        backward based on the context's status: warmup, record, or CUDAGraph.
        """
        (context,) = ctx.saved_tensor()

        (status, graph, inputs, ys) = context.pop_data()
        y, dy = select_y_with_grad(ys, dys)

        printer((y, dy), "Backward input")

        if status == CUDAGraphLayerStatus.WARMUP:
            debug_print("[CUDAGraph] Backward Step (Default)")

            # In warmup step, perform standard backward operation
            y.backward(dy)
            args_grad = get_args_grad(inputs)

            context.warmup_step()
        elif status == CUDAGraphLayerStatus.RECORD:
            debug_print(f"[CUDAGraph] Backward Step (Record) id {id(graph)}")

            # In record step, record the backward pass in CUDA graph
            def backward(y, dy):
                y.backward(dy)

            graph.backward_graph.record(backward, y, dy)

            # [NOTE] the get_args_grad should not put inside backward
            # the args_grad should be calculated after graph is replayed
            args_grad = get_args_grad(inputs)
            graph.backward_graph.set_output_static(args_grad)
            graph.capture_end()

            context.reuse_graph(graph)
        elif status == CUDAGraphLayerStatus.CUDAGRAPH:
            debug_print(f"[CUDAGraph] Backward Step (Graph) id {id(graph)}")

            # In CUDA graph step, replay the recorded graph for backward pass
            args_grad = graph.backward_graph.replay(y, dy)
            context.reuse_graph(graph)
        else:
            raise RuntimeError("Unknown cuda graph status")

        debug_print("[CUDAGraph] Backward Step End")

        printer(args_grad, "Backward output")
        return args_grad


class CUDAGraphedLayer(paddle.nn.Layer):
    """
    CUDAGraphedLayer: A PaddlePaddle Layer to convert an eager mode model to utilize CUDA Graphs.

    CUDA Graphs provide a way to capture kernel-level operations of a model and play
    them back efficiently, allowing for potential speedups in repetitive computations,
    such as those during training iterations. This layer is a wrapper that enables
    the usage of CUDA Graphs with PaddlePaddle models.

    Overview:
    - The layer encapsulates another layer (the model to be converted).
    - During the first few (num_warmup_steps) iterations, the layer operates in
      eager mode without any CUDA Graphs.
    - After the warmup steps, the layer captures the forward and backward computations
      and replays them using CUDA Graphs in subsequent iterations.

    Usage:
        model = Model()
        graphed_model = CUDAGraphedLayer(model)

    Parameters:
    - layer (paddle.nn.Layer): The PaddlePaddle model/layer to be converted.
    - num_warmup_steps (int): The number of iterations before the CUDA Graph
      capture begins. Default is 3.

    Notes:
    - Restrictions:
        * CPU-GPU Synchronization: Operations that synchronize the CPU with the GPU, like device to host transfers, are not allowed.
        * CPU Work: Any operations on the CPU within the captured graph are not recorded.
        * Memory Address (Pointer) Consistency: Replays consistently read from and write to identical virtual memory addresses.
        * Dynamic Operations:
            - Control Flow: Dynamic control flows, especially those based on CPU data like if/else statements, are prohibited.
            - Tensor Shapes: Dynamic tensor shapes are not supported.

    - Allowed Operations:
        * CUDA RNG Operations: CUDA-based Random Number Generation operations are allowed.
    """

    def __init__(self, layer: paddle.nn.Layer, num_warmup_steps=3):
        super().__init__()
        self.context = CUDAGraphContext(layer, num_warmup_steps)
        self.add_sublayer(f"Graphed {type(layer).__name__}", layer)

    def forward(self, *args, **kwargs):
        # We collect them into a list of tensor that required grad
        grad_inputs = recursive_flatten_args_kwargs(args, kwargs)
        return _CUDAGraphedLayer.apply(
            self.context, (args, kwargs), *grad_inputs
        )

    def is_warmup_step(self):
        return self.context.is_warmup_step()

    def is_cuda_graph_step(self):
        return self.context.is_cuda_graph_step()
