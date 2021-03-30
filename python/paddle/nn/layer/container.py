# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from ...fluid.dygraph.layers import Layer

__all__ = ['LayerDict', ]


class LayerDict(Layer):
    """
    LayerDict holds sublayers in the ordered dictionary, and sublayers it contains are properly registered.
    Holded sublayers can be accessed like a regular ordered python dictionary. 

    Parameters:
        sublayers (iterable of Layer, optional): sublayers to hold

    Examplex:
        .. code-block:: python

            import paddle
            import numpy as np
            from collections import OrderedDict

            sublayers = OrderedDict([
                ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
            ])

            layers_dict = paddle.nn.LayerDict(sublayers=sublayers)

            l = layers_dict['conv1d']

            for k in layers_dict:
                l = layers_dict[k]

            len(layers_dict)
            #3

            del layers_dict['conv2d']
            len(layers_dict)
            #2

            conv1d = layers_dict.pop('conv1d')
            len(layers_dict)
            #1

            layers_dict.clear()
            len(layers_dict)
            #0

    """

    def __init__(self, sublayers=None):
        super(LayerDict, self).__init__()
        if sublayers is not None:
            self._sub_layers.update(sublayers)

    def __getitem__(self, key: str):
        return self._sub_layers[key]

    def __setitem__(self, key: str, sublayer):
        return self.add_sublayer(key, sublayer)

    def __delitem__(self, key: str):
        del self._sub_layers[key]

    def __len__(self):
        return len(self._sub_layers)

    def __iter__(self):
        return iter(self._sub_layers)

    def __contains__(self, key: str):
        return key in self._sub_layers

    def clear(self):
        """
        Clear all the sublayers in the LayerDict.

        Parameters:
            None.

        Examplex:
            .. code-block:: python

                import paddle
                from collections import OrderedDict

                sublayers = OrderedDict([
                    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                ])

                layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                len(layer_dict)
                #3

                layer_dict.clear()
                len(layer_dict)
                #0

        """
        self._sub_layers.clear()

    def pop(self, key: str):
        """
        Remove the key from the LayerDict and return the layer of the key.

        Parameters:
            key (str): the key to be removed.

        Examples:
            .. code-block:: python

                import paddle
                from collections import OrderedDict

                sublayers = OrderedDict([
                    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                ])

                layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                len(layer_dict)
                #3

                layer_dict.pop('conv2d')
                len(layer_dict)
                #2

        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        """
        Return the iterable of the keys in LayerDict.

        Parameters:
            None.
        
        Examples:
            .. code-block:: python

                import paddle
                from collections import OrderedDict

                sublayers = OrderedDict([
                    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                ])

                layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                for k in layer_dict.keys():
                    print(k)
                
                #conv1d
                #conv2d
                #conv3d

        """
        return self._sub_layers.keys()

    def items(self):
        """
        Return the iterable of the key/value pairs in LayerDict.

        Parameters:
            None.
        
        Examples:
            .. code-block:: python

                import paddle
                from collections import OrderedDict

                sublayers = OrderedDict([
                    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                ])

                layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                for k, v in layer_dict.items():
                    print(k, ":", v)

                #conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
                #conv2d : Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
                #conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)

        """
        return self._sub_layers.items()

    def values(self):
        """
        Return the iterable of the values in LayerDict.

        Parameters:
            None.
        
        Examples:
            .. code-block:: python

                import paddle
                from collections import OrderedDict

                sublayers = OrderedDict([
                    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                ])

                layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
                for v in layer_dict.values():
                    print(v)

                #Conv1D(3, 2, kernel_size=[3], data_format=NCL)
                #Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
                #Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)

        """
        return self._sub_layers.values()

    def update(self, sublayers):
        """
        Update the key/values pairs in sublayers to the LayerDict, overwriting the existing keys.

        Parameters:
            sublayers (iterable of Layer): iterable of sublayers to update.
        
        Examples:
            .. code-block:: python

                import paddle
                from collections import OrderedDict

                sublayers = OrderedDict([
                    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
                    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
                ])

                new_sublayers = OrderedDict([
                    ('relu', paddle.nn.ReLU()),
                    ('conv2d', paddle.nn.Conv2D(4, 2, 4)),
                ])
                layer_dict = paddle.nn.LayerDict(sublayers=sublayers)

                layer_dict.update(new_sublayers)
                
                for k, v in layer_dict.items():
                    print(k, ":", v)
                #conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
                #conv2d : Conv2D(4, 2, kernel_size=[4, 4], data_format=NCHW)
                #conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)
                #relu : ReLU()

        """
        for key, layer in sublayers.items():
            self.add_sublayer(key, layer)
        return self
