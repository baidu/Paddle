#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..helper import is_complex, is_real, complex_variable_exists
from ...fluid.framework import ComplexVariable
from ...fluid import layers
from ...tensor import math

__all__ = [
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'trace',
    'sum',
]


def elementwise_add(x, y, axis=-1, name=None):
    """
    The element-wise addition layer for complex number inputs. At least one of 
    inputs :attr:`x` and :attr:`y` must be a ComplexVariable. See the detailed 
    description for the function and other arguments 
    in :ref:`api_fluid_layers_elementwise_add` . 

    Args:
        x (Variable|ComplexVariable): The first input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.
        y (Variable|ComplexVariable): The second input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.

    Examples:
        .. code-block:: python
    
            import numpy as np
            import paddle
            import paddle.fluid.dygraph as dg

            a = np.array([[1.0+1.0j, 2.0+1.0j], [3.0+1.0j, 4.0+1.0j]])
            b = np.array([[5.0+2.0j, 6.0+2.0j], [7.0+2.0j, 8.0+2.0j]])
            with dg.guard():
                x = dg.to_variable(a)
                y = dg.to_variable(b)
                out = paddle.complex.elementwise_add(x, y)
                print(out.numpy())
                # [[ 6.+3.j  8.+3.j]
                #  [10.+3.j 12.+3.j]]
    """
    complex_variable_exists([x, y], "elementwise_add")
    (x_real, x_imag) = (x.real, x.imag) if is_complex(x) else (x, None)
    (y_real, y_imag) = (y.real, y.imag) if is_complex(y) else (y, None)
    real = layers.elementwise_add(x_real, y_real, axis=axis, name=name)
    if is_real(x_imag) and is_real(y_imag):
        imag = layers.elementwise_add(x_imag, y_imag, axis=axis, name=name)
    elif is_real(x_imag):
        imag = layers.assign(x_imag)
    else:
        imag = layers.elementwise_add(
            layers.zeros_like(x_real), y_imag, axis=axis, name=name)
    return ComplexVariable(real, imag)


def elementwise_sub(x, y, axis=-1, name=None):
    """
    The element-wise subtraction layer for complex number inputs. At least one of 
    inputs :attr:`x` and :attr:`y` must be a ComplexVariable. See the detailed 
    description for the function and other arguments 
    in :ref:`api_fluid_layers_elementwise_sub` . 

    Args:
        x (Variable|ComplexVariable): The first input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.
        y (Variable|ComplexVariable): The second input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.

    Examples:
        .. code-block:: python
    
            import numpy as np
            import paddle
            import paddle.fluid.dygraph as dg

            a = np.array([[1.0+1.0j, 2.0+1.0j], [3.0+1.0j, 4.0+1.0j]])
            b = np.array([[5.0+2.0j, 6.0+2.0j], [7.0+2.0j, 8.0+2.0j]])
            with dg.guard():
                x = dg.to_variable(a)
                y = dg.to_variable(b)
                out = paddle.complex.elementwise_sub(x, y)
                print(out.numpy())
                # [[-4.-1.j -4.-1.j]
                #  [-4.-1.j -4.-1.j]]
    """
    complex_variable_exists([x, y], "elementwise_sub")
    (x_real, x_imag) = (x.real, x.imag) if is_complex(x) else (x, None)
    (y_real, y_imag) = (y.real, y.imag) if is_complex(y) else (y, None)
    real = layers.elementwise_sub(x_real, y_real, axis=axis, name=name)
    if is_real(x_imag) and is_real(y_imag):
        imag = layers.elementwise_sub(x_imag, y_imag, axis=axis, name=name)
    elif is_real(x_imag):
        imag = layers.assign(x_imag)
    else:
        imag = layers.elementwise_sub(
            layers.zeros_like(x_real), y_imag, axis=axis, name=name)
    return ComplexVariable(real, imag)


def elementwise_mul(x, y, axis=-1, name=None):
    """
    The element-wise multiplication layer for complex number inputs. At least 
    one of inputs :attr:`x` and :attr:`y` must be a ComplexVariable. See the 
    detailed description for the function and other arguments 
    in :ref:`api_fluid_layers_elementwise_mul` . 

    Args:
        x (Variable|ComplexVariable): The first input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.
        y (Variable|ComplexVariable): The second input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.

    Examples:
        .. code-block:: python
    
            import numpy as np
            import paddle
            import paddle.fluid.dygraph as dg

            a = np.array([[1.0+1.0j, 2.0+1.0j], [3.0+1.0j, 4.0+1.0j]])
            b = np.array([[5.0+2.0j, 6.0+2.0j], [7.0+2.0j, 8.0+2.0j]])
            with dg.guard():
                x = dg.to_variable(a)
                y = dg.to_variable(b)
                out = paddle.complex.elementwise_mul(x, y)
                print(out.numpy())
                # [[ 3. +7.j 10.+10.j]
                #  [19.+13.j 30.+16.j]]
    """
    complex_variable_exists([x, y], "elementwise_mul")
    # (a + bi)(c + di) = (ac - bd) + (bc + ad)i
    (a, b) = (x.real, x.imag) if is_complex(x) else (x, None)
    (c, d) = (y.real, y.imag) if is_complex(y) else (y, None)

    ac = layers.elementwise_mul(a, c, axis=axis, name=name)
    bd = layers.elementwise_mul(
        b, d, axis=axis, name=name) if is_real(b) and is_real(d) else None
    bc = layers.elementwise_mul(
        b, c, axis=axis, name=name) if is_real(b) else None
    ad = layers.elementwise_mul(
        a, d, axis=axis, name=name) if is_real(d) else None
    real = ac - bd if is_real(bd) else ac
    imag = bc + ad if is_real(bc) and is_real(ad) else bc if is_real(bc) else ad
    return ComplexVariable(real, imag)


def elementwise_div(x, y, axis=-1, name=None):
    """
    The element-wise division layer for complex number inputs. At least one of 
    inputs :attr:`x` and :attr:`y` must be a ComplexVariable. See the detailed 
    description for the function and other arguments 
    in :ref:`api_fluid_layers_elementwise_div` . 

    Args:
        x (Variable|ComplexVariable): The first input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.
        y (Variable|ComplexVariable): The second input Variable or ComplexVariable 
            with any number of dimensions. The supported data types include float32 
            and float64 when it is a Variable. Otherwise the supported data types 
            are complex64 or complex128.

    Examples:
        .. code-block:: python
    
            import numpy as np
            import paddle
            import paddle.fluid.dygraph as dg

            a = np.array([[1.0+1.0j, 2.0+1.0j], [3.0+1.0j, 4.0+1.0j]])
            b = np.array([[5.0+2.0j, 6.0+2.0j], [7.0+2.0j, 8.0+2.0j]])
            with dg.guard():
                x = dg.to_variable(a)
                y = dg.to_variable(b)
                out = paddle.complex.elementwise_div(x, y)
                print(out.numpy())
                # [[0.24137931+0.10344828j 0.35      +0.05j      ]
                #  [0.43396226+0.01886792j 0.5       +0.j        ]]
    """
    complex_variable_exists([x, y], "elementwise_div")
    # (a + bi)/(c + di) = (a + bi)(c - di)/(c^2 + d^2)
    (c, d) = (y.real, y.imag) if is_complex(y) else (y, None)
    y_conj = ComplexVariable(c, -d) if is_real(d) else c
    e = 1 / (layers.pow(c, 2.0) + layers.pow(d, 2.0)
             ) if is_real(d) else 1 / layers.pow(c, 2.0)
    return elementwise_mul(
        elementwise_mul(
            x, y_conj, axis=axis, name=name),
        e,
        axis=axis,
        name=name)


def trace(input, offset=0, dim1=0, dim2=1):
    """
    The trace layer for complex number inputs. inputs :attr:`input` must be a ComplexVariable. 
    See the detailed description for the function and other arguments 
    in :ref:`api_tensor_trace` . 
    
    Args:
        input(ComplexVariable): The input ComplexVariable. Must be at least 2-dimensional. 
            The supported data types include complex64 and complex128.
    
    Returns:
        ComplexVariable, the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle.complex.tensor as tensor
            import paddle.fluid.dygraph as dg
            import numpy as np
            
            case1 = np.random.randn(3, 10, 10).astype('float64') + 1j * np.random.randn(3, 10, 10).astype('float64')
            
            with dg.guard():
                case1 = dg.to_variable(case1)
                data1 = tensor.trace(case1, offset=1, dim1=1, dim2=2) # data2.shape = [3]
    """
    complex_variable_exists([input], "trace")
    (input_real, input_imag) = (input.real, input.imag)
    real = math.trace(input_real, offset, dim1, dim2)
    imag = math.trace(input_imag, offset, dim1, dim2)

    return ComplexVariable(real, imag)


def sum(input, dim=None, dtype=None, keep_dim=False, name=None):
    """
    The sum layer for complex number inputs. inputs :attr:`input` must be a ComplexVariable. 
    See the detailed description for the function and other arguments 
    in :ref:`api_tensor_sum` . 

    Args:
        input(ComplexVariable): The input ComplexVariable with any number of dimensions. 
            The supported data types include complex64 and complex128.

    Returns:
        ComplexVariable, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input.

    Raises:
        ValueError, the :attr:`dtype` must be float64 or int64.
    
    Examples:
        .. code-block:: python

            import paddle.complex as cpx
            import paddle.fluid.dygraph as dg
            import numpy as np

            with dg.guard():
                # x is a Tensor variable with following elements:
                #    [[0.2, 0.3, 0.5, 0.9], 
                #     [0.1, 0.2, 0.6, 0.7]]
                # Each example is followed by the corresponding output tensor.
                x = np.array([[0.2, 0.3, 0.5, 0.9],[0.1, 0.2, 0.6, 0.7]]) + 1j * np.array([[0.3, 0.4, 0.5, 0.2],[0.3, 0.6, 0.8, 0.3]])
                x = dg.to_variable(x)
                out1 = cpx.sum(x)  # [3.5+3.4j]
                out2 = cpx.sum(x, dim=0)  # [0.3+0.6j, 0.5+1.j, 1.1+1.3j, 1.6+0.5j]
                out3 = cpx.sum(x, dim=-1)  # [1.9+1.4j, 1.6+2.j]
                out4 = cpx.sum(x, dim=1, keep_dim=True)  # [[1.9+1.4j], [1.6+2.j]]

                # y is a Tensor variable with shape [2, 2, 2] and elements as below:
                #      [[[1, 2], [3, 4]],
                #      [[5, 6], [7, 8]]]
                # Each example is followed by the corresponding output tensor.
                y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) + 1j * np.array([[[4, 3], [2, 1]], [[8, 7], [6, 5]]])
                y = dg.to_variable(y)
                out5 = cpx.sum(y, dim=[1, 2]) # [10.+10.j, 26.+26.j]
                out6 = cpx.sum(y, dim=[0, 1]) # [16.+20.j, 20.+16.j]

    """
    complex_variable_exists([input], "sum")
    (input_real, input_imag) = (input.real, input.imag)
    real = math.sum(input_real, dim, dtype, keep_dim, name)
    imag = math.sum(input_imag, dim, dtype, keep_dim, name)

    return ComplexVariable(real, imag)
