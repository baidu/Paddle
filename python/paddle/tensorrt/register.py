# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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


class ConverterOpRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, op_name, trt_version=None):
        def decorator(func):
            if op_name not in self._registry:
                self._registry[op_name] = []
            self._registry[op_name].append((trt_version, func))
            return func

        return decorator

    def get(self, op_name, trt_version=None):
        if op_name not in self._registry:
            return None
        for version_range, func in self._registry[op_name]:
            if self._version_match(trt_version, version_range):
                return func
        return self._registry.get(op_name)

    def _version_match(self, trt_version, version_range):
        def _normalize_version(version):
            """Normalize version string to a tuple of three parts."""
            parts = version.split('.')
            while len(parts) < 3:
                parts.append('0')
            return tuple(map(int, parts))

        def _compare_versions(trt_version_tuple, ref_version_tuple, comparator):
            """Compare TRT version tuple with reference version tuple."""
            if comparator == 'ge':
                return trt_version_tuple >= ref_version_tuple
            elif comparator == 'le':
                return trt_version_tuple <= ref_version_tuple

        trt_version_tuple = _normalize_version(trt_version)

        if version_range.startswith('trt_version_ge='):
            min_version = version_range.split('=')[1]
            min_version_tuple = _normalize_version(min_version)
            return _compare_versions(trt_version_tuple, min_version_tuple, 'ge')

        elif version_range.startswith('trt_version_le='):
            max_version = version_range.split('=')[1]
            max_version_tuple = _normalize_version(max_version)
            return _compare_versions(trt_version_tuple, max_version_tuple, 'le')

        elif 'x' in version_range:
            major_version = int(version_range.split('.')[0])
            return trt_version_tuple[0] == major_version

        return False  # If version_range doesn't match the expected formats


converter_registry = ConverterOpRegistry()
