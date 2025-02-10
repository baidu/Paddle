# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
Print all signature of a python module in alphabet order.

Usage:
    python tools/print_signature.py "paddle" > API.spec
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import logging
import pkgutil
import re
import sys
from typing import Literal

import paddle

SpecFields = Literal[
    "args",
    "varargs",
    "varkw",
    "defaults",
    "kwonlyargs",
    "kwonlydefaults",
    "annotations",
    "document",
]

member_dict = collections.OrderedDict()

visited_modules = set()

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler(sys.stderr)
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
    )
)


def md5(doc):
    try:
        hashinst = hashlib.md5()
        hashinst.update(str(doc).encode('utf-8'))
        md5sum = hashinst.hexdigest()
    except UnicodeDecodeError as e:
        md5sum = None
        print(
            f"Error({e}) occurred when `md5({doc})`, discard it.",
            file=sys.stderr,
        )

    return md5sum


ErrorSet = set()
IdSet = set()
skiplist = []


def visit_all_module(mod):
    mod_name = mod.__name__
    if mod_name != 'paddle' and not mod_name.startswith('paddle.'):
        return

    if mod_name.startswith('paddle.base.core'):
        return

    if mod in visited_modules:
        return
    visited_modules.add(mod)

    member_names = dir(mod)
    if hasattr(mod, "__all__"):
        member_names += mod.__all__
    for member_name in member_names:
        if member_name.startswith('_'):
            continue
        cur_name = mod_name + '.' + member_name
        if cur_name in skiplist:
            continue
        try:
            instance = getattr(mod, member_name)
            if inspect.ismodule(instance):
                visit_all_module(instance)
            else:
                instance_id = id(instance)
                if instance_id in IdSet:
                    continue
                IdSet.add(instance_id)
                if (
                    hasattr(instance, '__name__')
                    and member_name != instance.__name__
                ):
                    print(
                        f"Found alias API, alias name is: {member_name}, original name is: {instance.__name__}",
                        file=sys.stderr,
                    )
        except:
            if cur_name not in ErrorSet and cur_name not in skiplist:
                ErrorSet.add(cur_name)


# all from gen_doc.py
api_info_dict = {}  # used by get_all_api


# step 1: walkthrough the paddle package to collect all the apis in api_set
def get_all_api(root_path='paddle', attr="__all__"):
    """
    walk through the paddle package to collect all the apis.
    """
    global api_info_dict
    api_counter = 0
    for filefinder, name, ispkg in pkgutil.walk_packages(
        path=paddle.__path__, prefix=paddle.__name__ + '.'
    ):
        try:
            if name in sys.modules:
                m = sys.modules[name]
            else:
                # importlib.import_module(name)
                m = eval(name)
                continue
        except AttributeError:
            logger.warning("AttributeError occurred when `eval(%s)`", name)
        else:
            api_counter += process_module(m, attr)

    api_counter += process_module(paddle, attr)

    logger.info(
        '%s: collected %d apis, %d distinct apis.',
        attr,
        api_counter,
        len(api_info_dict),
    )

    return [
        (sorted(api_info['all_names'])[0], md5(api_info['docstring']))
        for api_info in api_info_dict.values()
    ]


def insert_api_into_dict(full_name, gen_doc_anno=None):
    """
    insert add api into the api_info_dict
    Return:
        api_info object or None
    """
    try:
        obj = eval(full_name)
        fc_id = id(obj)
    except AttributeError:
        logger.warning("AttributeError occurred when `id(eval(%s))`", full_name)
        return None
    except Exception as e:
        logger.warning(
            "Exception(%s) occurred when `id(eval(%s))`", str(e), full_name
        )
        return None
    else:
        logger.debug("adding %s to api_info_dict.", full_name)
        if fc_id in api_info_dict:
            api_info_dict[fc_id]["all_names"].add(full_name)
        else:
            api_info_dict[fc_id] = {
                "all_names": {full_name},
                "id": fc_id,
                "object": obj,
                "type": type(obj).__name__,
                "docstring": '',
            }
            docstr = inspect.getdoc(obj)
            if docstr:
                api_info_dict[fc_id]["docstring"] = inspect.cleandoc(docstr)
            if gen_doc_anno:
                api_info_dict[fc_id]["gen_doc_anno"] = gen_doc_anno
            if inspect.isfunction(obj):
                api_info_dict[fc_id]["signature"] = inspect.getfullargspec(obj)
        return api_info_dict[fc_id]


# step 1 fill field : `id` & `all_names`, type, docstring
def process_module(m, attr="__all__"):
    api_counter = 0
    if hasattr(m, attr):
        # may have duplication of api
        for api in set(getattr(m, attr)):
            if api[0] == '_':
                continue
            # Exception occurred when `id(eval(paddle.dataset.conll05.test, get_dict))`
            if ',' in api:
                continue

            # api's fullname
            full_name = m.__name__ + "." + api
            api_info = insert_api_into_dict(full_name)
            if api_info is not None:
                api_counter += 1
                if inspect.isclass(api_info['object']):
                    for name, value in inspect.getmembers(api_info['object']):
                        if (not name.startswith("_")) and hasattr(
                            value, '__name__'
                        ):
                            method_full_name = (
                                full_name + '.' + name
                            )  # value.__name__
                            method_api_info = insert_api_into_dict(
                                method_full_name, 'class_method'
                            )
                            if method_api_info is not None:
                                api_counter += 1
    return api_counter


def check_allmodule_callable():
    modulelist = [paddle]
    for m in modulelist:
        visit_all_module(m)

    return member_dict


class ApiSpecFormatter:
    def __init__(self, show_fields: SpecFields):
        self.show_fields = show_fields

    def format_spec(self, spec: inspect.FullArgSpec | None) -> str:
        if spec is None:
            return "ArgSpec()"
        inner_str = ", ".join(
            f"{field}={getattr(spec, field)!r}"
            for field in spec._fields
            if field in self.show_fields
        )
        return f"ArgSpec({inner_str})"

    def format_doc(self, doc: str) -> str:
        if "document" not in self.show_fields:
            return "('document', '**********')"
        return f"('document', '{md5(doc)}')"

    def format(self, api_name: str, spec: inspect.FullArgSpec, doc: str) -> str:
        return f"{api_name} ({self.format_spec(spec)}, {self.format_doc(doc)})"


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Print Apis Signatures')
    parser.add_argument('module', type=str, help='module', default='paddle')
    parser.add_argument(
        '--skipped',
        dest='skipped',
        type=str,
        help='Skip Checking submodules, support regex',
        default=r'paddle\.base\.libpaddle\.(eager|pir)\.ops',
    )
    parser.add_argument(
        '--show-fields',
        type=str,
        default="args,varargs,varkw,defaults,kwonlyargs,kwonlydefaults,annotations,document",
        help="show fields in arg spec, separated by comma, e.g. 'args,varargs'",
    )
    args = parser.parse_args()
    return args


def create_api_filter(skipped_regex: str):
    if not skipped_regex:
        return lambda api_name: True
    skipped_pattern = re.compile(skipped_regex)

    def api_filter(api_name: str) -> bool:
        return not skipped_pattern.match(api_name)

    return api_filter


if __name__ == '__main__':
    args = parse_args()
    check_allmodule_callable()
    get_all_api(args.module)
    api_filter = create_api_filter(args.skipped)
    spec_formatter = ApiSpecFormatter(args.show_fields.split(','))

    all_api_names_to_k = {}
    for k, api_info in api_info_dict.items():
        # 1. the shortest suggested_name may be renamed;
        # 2. some api's fullname is not accessible, the module name of it is overridden by the function with the same name;
        api_name = sorted(api_info['all_names'])[0]
        all_api_names_to_k[api_name] = k
    all_api_names_sorted = sorted(all_api_names_to_k.keys())
    for api_name in all_api_names_sorted:
        if not api_filter(api_name):
            continue
        api_info = api_info_dict[all_api_names_to_k[api_name]]

        print(
            spec_formatter.format(
                api_name,
                api_info.get('signature'),
                api_info['docstring'],
            )
        )

    if len(ErrorSet) == 0:
        sys.exit(0)
    else:
        for erroritem in ErrorSet:
            print(
                f"Error, new function {erroritem} is unreachable",
                file=sys.stderr,
            )
        sys.exit(1)
