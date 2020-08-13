#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.tensor import Variable, fill_constant

from ...fluid.layers import core
# TODO: define the common functions to build a neural network
from ...fluid.layers import dropout  # DEFINE_ALIAS
from ...fluid.layers import label_smooth  # DEFINE_ALIAS
from ...fluid import one_hot  # DEFINE_ALIAS
from ...fluid.layers import pad  # DEFINE_ALIAS
from ...fluid.layers import pad2d  # DEFINE_ALIAS
from ...fluid.layers import unfold  # DEFINE_ALIAS
from ...fluid.layers import assign  # DEFINE_ALIAS
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
from ...fluid.framework import Variable, in_dygraph_mode

# from ...fluid.layers import fc  #DEFINE_ALIAS
from ...fluid.layers import pad_constant_like  # DEFINE_ALIAS

__all__ = [
    'dropout',
    'embedding',
    #       'fc',
    'label_smooth',
    'one_hot',
    'pad',
    'pad_constant_like',
    'pad2d',
    'unfold',
    #       'bilinear_tensor_product',
    'assign',
    'interpolate'
]


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=False,
                align_mode=1,
                data_format='NCHW',
                name=None):
    """
	:alias_main: paddle.nn.functional.interpolate
	:alias: paddle.nn.functional.interpolate,paddle.nn.functional.common.interpolate

    This op resizes a batch of images.
    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    and the resizing only applies on the three dimensions(depth, height and width).
    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the
    future and only use :attr:`out_shape` instead.
    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation

    Linear interpolation is the method of using a line connecting two known quantities
    to determine the value of an unknown quantity between the two known quantities.

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    Align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Example:

    .. code-block:: text

        For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)

        Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}

        Nearest neighbor interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})
          else:
              align_corners = True
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

        Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    For details of linear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Linear_interpolation.

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.

    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    Parameters:
        input (Variable): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Variable|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w)
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor.
             Default: None. If a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale_factor (float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale_factor` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale_factor`.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'nearest', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`,  `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).
    Raises:
        TypeError: size should be a list or tuple or Variable.
        ValueError: The 'mode' of image_resize can only be 'linear', 'bilinear',
                    'trilinear', 'bicubic', or 'nearest' currently.
        ValueError: 'linear' only support 3-D tensor.
        ValueError: 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        ValueError: 'trilinear' only support 5-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 1 for input 3-D tensor.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: size length should be 3 for input 5-D tensor.
        ValueError: scale_factor should be greater than zero.
        TypeError: align_corners should be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python

	    #declarative mode
	    import paddle
	    import numpy as np
	    input = fluid.data(name="input", shape=[None,3,6,10])
	    #1
	    output = paddle.nn.functional.interpolate(input=input, size=[12,12])
	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = paddle.nn.functional.interpolate(input=input, size=[12,dim1])
	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = paddle.nn.functional.interpolate(input=input, size=shape_tensor)
	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = paddle.nn.functional.interpolate(input=input, scale_factor=scale_tensor)
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    input_data = np.random.rand(2,3,6,10).astype("float32")
	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)

	    print(output_data[0].shape)
	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)
	    #imperative mode
	    import paddle.fluid.dygraph as dg
	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = paddle.nn.functional.interpolate(input=input, size=[12,12])
    		print(output.shape)
		# [2L, 3L, 12L, 12L]
    """
    data_format = data_format.upper()
    resample = mode.upper()
    resample_type = mode.lower()

    resample_methods = [
        'LINEAR',
        'BILINEAR',
        'TRILINEAR',
        'NEAREST',
        'BICUBIC',
    ]
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'linaer', 'bilinear', 'trilinear', "
            " 'bicubic' or 'nearest' currently.")

    if resample in ['LINEAR'] and len(input.shape) != 3:
        raise ValueError("'linear' only support 3-D tensor.")

    if resample in ['BILINEAR', 'NEAREST', 'BICUBIC'] and len(input.shape) != 4:
        raise ValueError(
            "'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.")
    if resample == 'TRILINEAR' and len(input.shape) != 5:
        raise ValueError("'trilinear'only support 5-D tensor.")

    if size is None and scale_factor is None:
        raise ValueError("One of size and scale_factor must not be None.")

    if not isinstance(align_corners, bool):
        raise TypeError("Attr align_corners should be a bool value")

    if align_mode != 0 and align_mode != 1:
        raise ValueError("align_mode can only be 0 or 1")

    helper = LayerHelper('{}_interp'.format(resample_type), **locals())
    dtype = helper.input_dtype()

    if len(input.shape) == 3 and data_format not in ['NCW', 'NWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCW` or `NWC` supported for 3-D input.")
    elif len(input.shape) == 4 and data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCHW` or `NHWC` supported for 4-D input.")
    elif len(input.shape) == 5 and data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCDHW` or `NDHWC` supported for 5-D input.")

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if data_format == 'NCHW' or data_format == 'NCDHW' or data_format == 'NCW':
        data_layout = 'NCHW'
    if data_format == 'NHWC' or data_format == 'NDHWC' or data_format == 'NWC':
        data_layout = 'NHWC'

    inputs = {"X": input}
    attrs = {
        "out_d": -1,
        "out_h": -1,
        "out_w": -1,
        "interp_method": resample_type,
        "align_corners": align_corners,
        "align_mode": align_mode,
        "data_layout": data_layout
    }

    out_shape = size
    scale = scale_factor
    if out_shape is not None:
        if isinstance(out_shape, Variable):
            out_shape.stop_gradient = True
            inputs['OutSize'] = out_shape
        else:
            if not (_is_list_or_turple_(out_shape)):
                raise TypeError(
                    "out_shape should be a list or tuple or Variable.")
            # Validate the shape
            contain_var = False
            for dim_idx, dim_size in enumerate(out_shape):
                if isinstance(dim_size, Variable):
                    contain_var = True
                    continue
                assert dim_size > 0, (
                    "Each dimension size given in out_shape must be greater than 0."
                )

            if contain_var:
                new_size_tensor = []
                size_list = []
                for dim in out_shape:
                    if isinstance(dim, Variable):
                        dim.stop_gradient = True
                        new_size_tensor.append(dim)
                        size_list.append(-1)
                    else:
                        assert (isinstance(dim, int))
                        temp_out = helper.create_variable_for_type_inference(
                            'int32')
                        fill_constant(
                            [1], 'int32', dim, force_cpu=True, out=temp_out)
                        new_size_tensor.append(temp_out)
                        size_list.append(dim)
                inputs['SizeTensor'] = new_size_tensor

            if len(input.shape) == 3:
                if len(out_shape) != 1:
                    raise ValueError(
                        "out_shape length should be 2 for input 3-D tensor")
                if contain_var:
                    attrs['out_w'] = size_list[0]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_w'] = out_shape[0]
            if len(input.shape) == 4:
                if len(out_shape) != 2:
                    raise ValueError("out_shape length should be 2 for "
                                     "input 4-D tensor.")
                if contain_var:
                    attrs['out_h'] = size_list[0]
                    attrs['out_w'] = size_list[1]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_h'] = out_shape[0]
                    attrs['out_w'] = out_shape[1]
            if len(input.shape) == 5:
                if len(out_shape) != 3:
                    raise ValueError("out_shape length should be 3 for "
                                     "input 5-D tensor.")
                if contain_var:
                    attrs['out_d'] = size_list[0]
                    attrs['out_h'] = size_list[1]
                    attrs['out_w'] = size_list[2]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_d'] = out_shape[0]
                    attrs['out_h'] = out_shape[1]
                    attrs['out_w'] = out_shape[2]

    else:
        if isinstance(scale, Variable):
            scale.stop_gradient = True
            inputs["Scale"] = scale
        elif isinstance(scale, float) or isinstance(scale, int):
            if scale <= 0:
                raise ValueError("Attr(scale) should be greater than zero.")
            attrs['scale'] = float(scale)
        else:
            raise TypeError(
                "Attr(scale)'s type should be float, int or Variable.")

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='{}_interp'.format(resample_type),
        inputs=inputs,
        outputs={"Out": out},
        attrs=attrs)
    return out


def embedding(input, weight, padding_idx=None, is_sparse=True, name=None):
    """
        The operator is used to lookup embeddings vector of ids provided by :attr:`input` .
        It automatically constructs a 2D embedding matrix based on the
        input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

        This OP requires the last dimension of Tensor shape must be equal to 1. The shape
        of output Tensor is generated by replacing the last dimension of the input Tensor shape
        with emb_size.

        **Note:** The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` ,
        otherwise the program will throw an exception and exit.

        .. code-block:: text

            Case 1:

            input is a Tensor. padding_idx = -1
                input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
                input.shape = [3, 2, 1]
            Given size = [128, 16]
            output is a Tensor:
                out.shape = [3, 2, 16]
                out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                            [0.345421456, 0.524563927, ..., 0.144534654]],

                            [[0.345249859, 0.124939536, ..., 0.194353745],
                            [0.945345345, 0.435394634, ..., 0.435345365]],

                            [[0.945345345, 0.435394634, ..., 0.435345365],
                            [0.0,         0.0,         ..., 0.0        ]]]  # padding data
            The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
            It will pad all-zero data when ids is 127.

            Case 2:

            input is a LoDTensor with 1-level LoD. padding_idx = 0
                input.lod = [[2, 3]]
                input.data = [[1], [3], [2], [4], [0]]
                input.shape = [5, 1]
            Given size = [128, 16]
            output is a LoDTensor:
                out.lod = [[2, 3]]
                out.shape = [5, 16]
                out.data = [[0.129435295, 0.244512452, ..., 0.436322452],
                            [0.345421456, 0.524563927, ..., 0.144534654],
                            [0.345249859, 0.124939536, ..., 0.194353745],
                            [0.945345345, 0.435394634, ..., 0.435345365],
                            [0.0,         0.0,         ..., 0.0        ]]  # padding data
            It will pad all-zero data when ids is 0.

        Args:
            input(Variable): A Tensor or LoDTensor with type int64, which contains the id information.
                The last dimension of Tensor shape must be equal to 1. The value of the input id should
                satisfy :math:`0<= id < size[0]` .
            weight (Variable): The weight. A Tensor with shape of lookup table parameter. It should have two elements which
                indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
            is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
                affects the performance of the backwards gradient update. It is recommended to set
                True because sparse update is faster. But some optimizer does not support sparse update,
                such as :ref:`api_fluid_optimizer_AdadeltaOptimizer` , :ref:`api_fluid_optimizer_AdamaxOptimizer` ,
                :ref:`api_fluid_optimizer_DecayedAdagradOptimizer` , :ref:`api_fluid_optimizer_FtrlOptimizer` ,
                :ref:`api_fluid_optimizer_LambOptimizer` and :ref:`api_fluid_optimizer_LarsMomentumOptimizer` .
                In these case, is_sparse must be False. Default: False.
            padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size).
                If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
                to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
                encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
                If set None, it makes no effect to output. Default: None.
            name(str|None): For detailed information, please refer
               to :ref:`api_guide_Name`. Usually name is no need to set and
               None by default.
        Returns:
            Variable: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              import numpy as np
              data = fluid.data(name='x', shape=[None, 1], dtype='int64')

              # example: load custom or pre-trained word vectors
              # word vectors with numpy format
              weight_data = np.random.random(size=(128, 100))
              weight = fluid.ParamAttr(
                  name="emb_weight",
                  learning_rate=0.5,
                  initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
                  trainable=True)
              emb = paddle.nn.functional.embedding(input=data, weight=weight, is_sparse=True, name="sparse_embedding")
    """
    if in_dygraph_mode():
        return core.ops.lookup_table_v2(
            weight, input, 'is_sparse', is_sparse, 'is_distributed', False,
            'remote_prefetch', False, 'padding_idx', padding_idx)
    else:
        helper = LayerHelper('embedding', **locals())
        dtype = helper.input_dtype()

        check_variable_and_dtype(input, 'input', ['int64'], 'embedding')
        check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'embedding')

        is_distributed = False
        remote_prefetch = is_sparse and (not is_distributed)

        tmp = helper.create_variable_for_type_inference(dtype)
        padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
            weight.shape[0] + padding_idx)

        helper.append_op(
            type='lookup_table_v2',
            inputs={'Ids': input,
                    'W': weight},
            outputs={'Out': tmp},
            attrs={
                'is_sparse': is_sparse,
                'is_distributed': is_distributed,
                'remote_prefetch': remote_prefetch,
                'padding_idx': padding_idx
            })
        return tmp
