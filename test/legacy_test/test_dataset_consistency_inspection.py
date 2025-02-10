#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
TestCases for Dataset consistency inspection of use_var_list and data_generator.
"""

import math
import os
import tempfile
import unittest

import paddle
from paddle import base
from paddle.distributed import fleet

# paddle.enable_static()
# base.disable_dygraph()
base.disable_dygraph()
url_schema_len = 5
query_schema = [
    'Q_query_basic',
    'Q_query_phrase',
    'Q_quq',
    'Q_timelevel',
    'Q_context_title_basic1',
    'Q_context_title_basic2',
    'Q_context_title_basic3',
    'Q_context_title_basic4',
    'Q_context_title_basic5',
    'Q_context_title_phrase1',
    'Q_context_title_phrase2',
    'Q_context_title_phrase3',
    'Q_context_title_phrase4',
    'Q_context_title_phrase5',
    'Q_context_site1',
    'Q_context_site2',
    'Q_context_site3',
    'Q_context_site4',
    'Q_context_site5',
]


class CTRDataset(fleet.MultiSlotDataGenerator):
    def __init__(self, mode):
        self.test = mode

    def generate_sample(self, line):
        def reader():
            ins = line.strip().split(';')
            label_pos_num = int(ins[1].split(' ')[0])
            label_neg_num = int(ins[1].split(' ')[1])

            # query fea parse
            bias = 2
            query_len = 0
            sparse_query_feature = []
            for index in range(len(query_schema)):
                pos = index + bias
                sparse_query_feature.append(
                    [int(x) for x in ins[pos].split(' ')]
                )
                if index == 0:
                    query_len = len(ins[pos].split(' '))
                    query_len = 1.0 / (1 + pow(2.7182818, 3 - 1.0 * query_len))

            # positive url fea parse
            bias = 2 + len(query_schema)
            pos_url_feas = []
            pos_click_feas = []
            pos_context_feas = []
            for k in range(label_pos_num):
                pos_url_fea = []
                pos = 0
                for index in range(url_schema_len - 1):
                    pos = bias + k * (url_schema_len) + index
                    pos_url_fea.append([int(x) for x in ins[pos].split(' ')])
                # click info
                if ins[pos + 1] == '':
                    continue
                item = ins[pos + 1].split(' ')
                if len(item) != 17:
                    continue
                stat_fea = [
                    [max(float(item[i]), 0.0)]
                    for i in range(len(item))
                    if not (
                        i == 5
                        or i == 9
                        or i == 13
                        or i == 14
                        or i == 15
                        or i == 16
                    )
                ]
                pos_url_feas.append(pos_url_fea)
                pos_click_feas.append(stat_fea)

                query_search = float(item[5])
                if query_search > 0.0:
                    query_search = min(math.log(query_search), 10.0) / 10.0
                pos_context_fea = [[query_search], [query_len]]
                pos_context_feas.append(pos_context_fea)

            # negative url fea parse
            bias = 2 + len(query_schema) + label_pos_num * (url_schema_len)
            neg_url_feas = []
            neg_click_feas = []
            neg_context_feas = []
            for k in range(label_neg_num):
                neg_url_fea = []
                pos = 0
                for index in range(url_schema_len - 1):
                    pos = bias + k * (url_schema_len) + index
                    neg_url_fea.append([int(x) for x in ins[pos].split(' ')])
                if ins[pos + 1] == '':
                    continue
                item = ins[pos + 1].split(' ')
                # zdf_tmp
                if len(item) != 17:
                    continue
                    # print ins[pos + 1]
                stat_fea = [
                    [max(float(item[i]), 0.0)]
                    for i in range(len(item))
                    if not (
                        i == 5
                        or i == 9
                        or i == 13
                        or i == 14
                        or i == 15
                        or i == 16
                    )
                ]
                neg_click_feas.append(stat_fea)
                neg_url_feas.append(neg_url_fea)

                query_search = float(item[5])
                if query_search > 0.0:
                    query_search = min(math.log(query_search), 10.0) / 10.0
                neg_context_fea = [[query_search], [query_len]]
                neg_context_feas.append(neg_context_fea)

            # make train data
            if self.test == 1:
                for p in range(len(pos_url_feas)):
                    # feature_name = ["click"] + query_schema + url_schema[:4] + click_info_schema[:11] + context_schema[:2]
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    pos_url_fea = pos_url_feas[p]
                    pos_click_fea = pos_click_feas[p]
                    pos_context_fea = pos_context_feas[p]
                    yield zip(
                        feature_name,
                        [
                            [1],
                            *sparse_query_feature,
                            *pos_url_fea,
                            *pos_click_fea,
                            *pos_context_fea,
                            *pos_url_fea,
                            *pos_click_fea,
                            *pos_context_fea,
                        ],
                    )
                for n in range(len(neg_url_feas)):
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    neg_url_fea = neg_url_feas[n]
                    neg_click_fea = neg_click_feas[n]
                    neg_context_fea = neg_context_feas[n]
                    yield zip(
                        feature_name,
                        [
                            [0],
                            *sparse_query_feature,
                            *neg_url_fea,
                            *neg_click_fea,
                            *neg_context_fea,
                            *neg_url_fea,
                            *neg_click_fea,
                            *neg_context_fea,
                        ],
                    )
            elif self.test == 0:
                for p in range(len(pos_url_feas)):
                    # feature_name = ["click"] + query_schema + url_schema[:4] + click_info_schema[:11] + context_schema[:2] + url_schema[4:] + click_info_schema[11:] + context_schema[2:]
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    # print("#######")
                    # print(feature_name)
                    # print("#######")
                    pos_url_fea = pos_url_feas[p]
                    pos_click_fea = pos_click_feas[p]
                    pos_context_fea = pos_context_feas[p]
                    for n in range(len(neg_url_feas)):
                        # prob = get_rand()
                        # if prob < sample_rate:
                        neg_url_fea = neg_url_feas[n]
                        neg_click_fea = neg_click_feas[n]
                        neg_context_fea = neg_context_feas[n]
                        # print("q:", query_feas)
                        # print("pos:", pos_url_fea)
                        # print("neg:", neg_url_fea)
                        # yield zip(feature_name[:3], sparse_query_feature[:3])
                        yield list(
                            zip(
                                feature_name,
                                [
                                    [1],
                                    *sparse_query_feature,
                                    *pos_url_fea,
                                    *pos_click_fea,
                                    *pos_context_fea,
                                    *neg_url_fea,
                                    *neg_click_fea,
                                    *neg_context_fea,
                                ],
                            )
                        )
            elif self.test == 2:
                for p in range(len(pos_url_feas)):
                    # feature_name = ["click"] + query_schema + url_schema[:4] + click_info_schema[:11] + context_schema[:2] + url_schema[4:] + click_info_schema[11:] + context_schema[2:]
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    # print("#######")
                    # print(feature_name)
                    # print("#######")
                    pos_url_fea = pos_url_feas[p]
                    pos_click_fea = pos_click_feas[p]
                    pos_context_fea = pos_context_feas[p]
                    for n in range(len(neg_url_feas)):
                        # prob = get_rand()
                        # if prob < sample_rate:
                        neg_url_fea = neg_url_feas[n]
                        neg_click_fea = neg_click_feas[n]
                        neg_context_fea = neg_context_feas[n]
                        # print("q:", query_feas)
                        # print("pos:", pos_url_fea)
                        # print("neg:", neg_url_fea)
                        # yield zip(feature_name[:3], sparse_query_feature[:3])
                        yield list(
                            zip(
                                feature_name,
                                [
                                    [1],
                                    [2],
                                    *sparse_query_feature,
                                    *pos_url_fea,
                                    *pos_click_fea,
                                    *pos_context_fea,
                                    *neg_url_fea,
                                    *neg_click_fea,
                                    *neg_context_fea,
                                ],
                            )
                        )
            elif self.test == 3:
                for p in range(len(pos_url_feas)):
                    # feature_name = ["click"] + query_schema + url_schema[:4] + click_info_schema[:11] + context_schema[:2] + url_schema[4:] + click_info_schema[11:] + context_schema[2:]
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    # print("#######")
                    # print(feature_name)
                    # print("#######")
                    pos_url_fea = pos_url_feas[p]
                    pos_click_fea = pos_click_feas[p]
                    pos_context_fea = pos_context_feas[p]
                    for n in range(len(neg_url_feas)):
                        # prob = get_rand()
                        # if prob < sample_rate:
                        neg_url_fea = neg_url_feas[n]
                        neg_click_fea = neg_click_feas[n]
                        neg_context_fea = neg_context_feas[n]
                        # print("q:", query_feas)
                        # print("pos:", pos_url_fea)
                        # print("neg:", neg_url_fea)
                        # yield zip(feature_name[:3], sparse_query_feature[:3])
                        yield list(
                            zip(
                                feature_name,
                                [
                                    [1],
                                    [2.0],
                                    *sparse_query_feature,
                                    *pos_url_fea,
                                    *pos_click_fea,
                                    *pos_context_fea,
                                    *neg_url_fea,
                                    *neg_click_fea,
                                    *neg_context_fea,
                                ],
                            )
                        )
            elif self.test == 4:
                for p in range(len(pos_url_feas)):
                    # feature_name = ["click"] + query_schema + url_schema[:4] + click_info_schema[:11] + context_schema[:2] + url_schema[4:] + click_info_schema[11:] + context_schema[2:]
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    # print("#######")
                    # print(feature_name)
                    # print("#######")
                    pos_url_fea = pos_url_feas[p]
                    pos_click_fea = pos_click_feas[p]
                    pos_context_fea = pos_context_feas[p]
                    for n in range(len(neg_url_feas)):
                        # prob = get_rand()
                        # if prob < sample_rate:
                        neg_url_fea = neg_url_feas[n]
                        neg_click_fea = neg_click_feas[n]
                        neg_context_fea = neg_context_feas[n]
                        # print("q:", query_feas)
                        # print("pos:", pos_url_fea)
                        # print("neg:", neg_url_fea)
                        # yield zip(feature_name[:3], sparse_query_feature[:3])
                        yield list(
                            zip(
                                feature_name,
                                [
                                    [],
                                    [2.0],
                                    *sparse_query_feature,
                                    *pos_url_fea,
                                    *pos_click_fea,
                                    *pos_context_fea,
                                    *neg_url_fea,
                                    *neg_click_fea,
                                    *neg_context_fea,
                                ],
                            )
                        )
            elif self.test == 5:
                for p in range(len(pos_url_feas)):
                    # feature_name = ["click"] + query_schema + url_schema[:4] + click_info_schema[:11] + context_schema[:2] + url_schema[4:] + click_info_schema[11:] + context_schema[2:]
                    feature_name = ["click"]
                    for i in range(1, 54):
                        feature_name.append(str(i))
                    # print("#######")
                    # print(feature_name)
                    # print("#######")
                    pos_url_fea = pos_url_feas[p]
                    pos_click_fea = pos_click_feas[p]
                    pos_context_fea = pos_context_feas[p]
                    for n in range(len(neg_url_feas)):
                        # prob = get_rand()
                        # if prob < sample_rate:
                        neg_url_fea = neg_url_feas[n]
                        neg_click_fea = neg_click_feas[n]
                        neg_context_fea = neg_context_feas[n]
                        # print("q:", query_feas)
                        # print("pos:", pos_url_fea)
                        # print("neg:", neg_url_fea)
                        # yield zip(feature_name[:3], sparse_query_feature[:3])
                        yield list(
                            zip(
                                feature_name,
                                sparse_query_feature
                                + pos_url_fea
                                + pos_click_fea
                                + pos_context_fea
                                + neg_url_fea
                                + neg_click_fea
                                + neg_context_fea,
                            )
                        )

        return reader


class TestDataset(unittest.TestCase):
    """TestCases for Dataset."""

    def setUp(self):
        pass
        # use_data_loader = False
        # epoch_num = 10
        # drop_last = False

    def test_var_consistency_inspection(self):
        """
        Testcase for InMemoryDataset of consistency inspection of use_var_list and data_generator.
        """

        temp_dir = tempfile.TemporaryDirectory()
        dump_a_path = os.path.join(temp_dir.name, 'test_run_with_dump_a.txt')

        with open(dump_a_path, "w") as f:
            # data = "\n"
            # data += "\n"
            data = "2 1;1 9;20002001 20001240 20001860 20003611 20000723;20002001 20001240 20001860 20003611 20000723;0;40000001;20002001 20001240 20001860 20003611 20000157 20000723 20000070 20002616 20000157 20000005;20002001 20001240 20001860 20003611 20000157 20001776 20000070 20002616 20000157 20000005;20002001 20001240 20001860 20003611 20000723 20000070 20002001 20001240 20001860 20003611 20012788 20000157;20002001 20001240 20001860 20003611 20000623 20000251 20000157 20000723 20000070 20000001 20000057;20002640 20004695 20000157 20000723 20000070 20002001 20001240 20001860 20003611;20002001 20001240 20001860 20003611 20000157 20000723 20000070 20003519 20000005;20002001 20001240 20001860 20003611 20000157 20001776 20000070 20003519 20000005;20002001 20001240 20001860 20003611 20000723 20000070 20002001 20001240 20001860 20003611 20131464;20002001 20001240 20001860 20003611 20018820 20000157 20000723 20000070 20000001 20000057;20002640 20034154 20000723 20000070 20002001 20001240 20001860 20003611;10000200;10000200;10063938;10000008;10000177;20002001 20001240 20001860 20003611 20010833 20000210 20000500 20000401 20000251 20012198 20001023 20000157;20002001 20001240 20001860 20003611 20012396 20000500 20002513 20012198 20001023 20000157;10000123;30000004;0.623 0.233 0.290 0.208 0.354 49.000 0.000 0.000 0.000 -1.000 0.569 0.679 0.733 53 17 2 0;20002001 20001240 20001860 20003611 20000723;20002001 20001240 20001860 20003611 20000723;10000047;30000004;0.067 0.000 0.161 0.005 0.000 49.000 0.000 0.000 0.000 -1.000 0.000 0.378 0.043 0 6 0 0;20002001 20001240 20001860 20003611 20000157 20000723 20000070 20002616 20000157 20000005;20002001 20001240 20001860 20003611 20000157 20000723 20000070 20003519 20000005;10000200;30000001;0.407 0.111 0.196 0.095 0.181 49.000 0.000 0.000 0.000 -1.000 0.306 0.538 0.355 48 8 0 0;20002001 20001240 20001860 20003611 20000157 20001776 20000070 20002616 20000157 20000005;20002001 20001240 20001860 20003611 20000157 20001776 20000070 20003519 20000005;10000200;30000001;0.226 0.029 0.149 0.031 0.074 49.000 0.000 0.000 0.000 -1.000 0.220 0.531 0.286 26 6 0 0;20002001 20001240 20001860 20003611 20000723 20000070 20002001 20001240 20001860 20003611 20012788 20000157;20002001 20001240 20001860 20003611 20000723 20000070 20002001 20001240 20001860 20003611 20131464;10063938;30000001;0.250 0.019 0.138 0.012 0.027 49.000 0.000 0.000 0.000 -1.000 0.370 0.449 0.327 7 2 0 0;20002001 20001240 20001860 20003611 20000723;20002001 20001240 20001860 20003611 20000723;10000003;30000002;0.056 0.000 0.139 0.003 0.000 49.000 0.000 0.000 0.000 -1.000 0.000 0.346 0.059 15 3 0 0;20002001 20001240 20001860 20003611 20000623 20000251 20000157 20000723 20000070 20000001 20000057;20002001 20001240 20001860 20003611 20018820 20000157 20000723 20000070 20000001 20000057;10000008;30000001;0.166 0.004 0.127 0.001 0.004 49.000 0.000 0.000 0.000 -1.000 0.103 0.417 0.394 10 3 0 0;20002640 20004695 20000157 20000723 20000070 20002001 20001240 20001860 20003611;20002640 20034154 20000723 20000070 20002001 20001240 20001860 20003611;10000177;30000001;0.094 0.008 0.157 0.012 0.059 49.000 0.000 0.000 0.000 -1.000 0.051 0.382 0.142 21 0 0 0;20002001 20001240 20001860 20003611 20000157 20001776 20000070 20000157;20002001 20001240 20001860 20003611 20000157 20001776 20000070 20000157;10000134;30000001;0.220 0.016 0.181 0.037 0.098 49.000 0.000 0.000 0.000 -1.000 0.192 0.453 0.199 17 1 0 0;20002001 20001240 20001860 20003611 20002640 20004695 20000157 20000723 20000070 20002001 20001240 20001860 20003611;20002001 20001240 20001860 20003611 20002640 20034154 20000723 20000070 20002001 20001240 20001860 20003611;10000638;30000001;0.000 0.000 0.000 0.000 0.000 49.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0 0 0 0;\n"
            data += "2 1;1 11;20000025 20000404;20001923;20000002 20000157 20000028 20004205 20000500 20028809 20000571 20000007 20027523 20004940 20000651 20000043 20000051 20000520 20015398 20000066 20004720 20000070 20001648;40000001;20000025 20000404 20000571 20004940 20000001 20000017;20000025 20000404 20000029 20000500 20001408 20000404 20000001 20000017;0;0;0;20001923 20011130 20000027;20001923 20000029 20000500 20001408 20000404 20000027;0;0;0;10000005;10000005;0;0;0;20003316 20000392 20001979 20000474 20000025 20000194 20000025 20000404 20000019 20000109;20016528 20024913 20004748 20001923 20000019 20000109;10000015;30000002;0.572 0.043 0.401 0.352 0.562 32859.000 0.005 0.060 0.362 -1.000 0.448 0.673 0.222 16316 991 89 0;20000025 20000404 20000571 20004940 20000001 20000017;20001923 20011130 20000027;10000005;30000001;0.495 0.024 0.344 0.285 0.379 32859.000 0.002 0.050 0.362 -1.000 0.423 0.764 0.254 19929 896 72 0;20000202 20000026 20001314 20004289 20000025 20000404 20000451 20000089 20000007;20000202 20000026 20014094 20001314 20004289 20001923 20000451 20000089 20000007;10000035;30000003;0.133 0.006 0.162 0.042 0.174 32859.000 0.003 0.037 0.362 -1.000 0.363 0.542 0.122 14763 664 53 0;20000202 20000026 20001314 20004289 20000025 20000404;20000202 20000026 20014094 20001314 20004289 20001923;10000021;30000001;0.058 0.004 0.133 0.017 0.120 32859.000 0.000 0.006 0.362 -1.000 0.168 0.437 0.041 -1 -1 -1 -1;20000025 20000404 20000018 20012461 20001699 20000446 20000174 20000062 20000133 20003172 20000240 20007877 20067375 20000111 20000164 20001410 20000204 20016958;20001923 20000018 20012461 20001699 20007717 20000062 20000133 20003172 20000240 20007877 20067375 20000111 20000164 20001410 20000204 20016958;10000002;30000001;0.017 0.000 0.099 0.004 0.072 32859.000 0.000 0.009 0.362 -1.000 0.058 0.393 0.025 -1 -1 -1 -1;20000025 20000404;20001923;10000133;30000005;0.004 0.000 0.122 0.000 0.000 32859.000 0.000 0.000 0.362 -1.000 0.000 0.413 0.020 0 444 35 0;20000025 20000404;20001923;10005297;30000004;0.028 0.000 0.138 0.002 0.000 32859.000 0.000 0.000 0.362 -1.000 0.000 0.343 0.024 0 600 48 0;20000025 20000404;20001923;10000060;30000005;0.107 0.000 0.110 0.027 0.077 32859.000 0.000 0.005 0.362 -1.000 0.095 0.398 0.062 1338 491 39 0;20002960 20005534 20000043 20000025 20000404 20000025 20000007;20002960 20005534 20000043 20001923 20000025 20000007;10000020;30000003;0.041 0.000 0.122 0.012 0.101 32859.000 0.001 0.025 0.362 -1.000 0.302 0.541 0.065 9896 402 35 0;20000025 20000404 20000259 20000228 20000235 20000142;20001923 20000259 20000264 20000142;10000024;30000003;0.072 0.002 0.156 0.026 0.141 32859.000 0.002 0.032 0.362 -1.000 0.386 0.569 0.103 9896 364 35 0;20000025 20000404 20000029 20000500 20001408 20000404 20000001 20000017;20001923 20000029 20000500 20001408 20000404 20000027;10000005;30000001;0.328 0.006 0.179 0.125 0.181 32859.000 0.003 0.058 0.362 -1.000 0.300 0.445 0.141 9896 402 32 0;20000025 20000404;20001923;10012839;30000002;0.012 0.000 0.108 0.002 0.048 32859.000 0.000 0.000 0.362 -1.000 0.021 0.225 0.016 2207 120 12 0;\n"
            # data += ""
            f.write(data)

        slot_data = []
        label = paddle.static.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
        )
        slot_data.append(label)

        # sprase_query_feat_names
        len_sparse_query = 19
        for feat_name in range(1, len_sparse_query + 1):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name),
                    shape=[-1, 1],
                    dtype='int64',
                )
            )

        # sparse_url_feat_names
        for feat_name in range(len_sparse_query + 1, len_sparse_query + 5):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name),
                    shape=[-1, 1],
                    dtype='int64',
                )
            )

        # dense_feat_names
        for feat_name in range(len_sparse_query + 5, len_sparse_query + 16):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name), shape=[-1, 1], dtype='float32'
                )
            )

        # context_feat_names
        for feat_name in range(len_sparse_query + 16, len_sparse_query + 18):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name), shape=[-1, 1], dtype='float32'
                )
            )

        # neg sparse_url_feat_names
        for feat_name in range(len_sparse_query + 18, len_sparse_query + 22):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name),
                    shape=[-1, 1],
                    dtype='int64',
                )
            )

        # neg dense_feat_names
        for feat_name in range(len_sparse_query + 22, len_sparse_query + 33):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name), shape=[-1, 1], dtype='float32'
                )
            )

        # neg context_feat_names
        for feat_name in range(len_sparse_query + 33, len_sparse_query + 35):
            slot_data.append(
                paddle.static.data(
                    name=str(feat_name), shape=[-1, 1], dtype='float32'
                )
            )

        dataset = paddle.distributed.InMemoryDataset()

        print("========================================")
        generator_class = CTRDataset(mode=0)
        try:
            dataset._check_use_var_with_data_generator(
                slot_data, generator_class, dump_a_path
            )
            print("case 1: check passed!")
        except Exception as e:
            print("warning: catch expected error")
            print(e)
        print("========================================")
        print("\n")

        print("========================================")
        generator_class = CTRDataset(mode=2)
        try:
            dataset._check_use_var_with_data_generator(
                slot_data, generator_class, dump_a_path
            )
        except Exception as e:
            print("warning: case 2 catch expected error")
            print(e)
        print("========================================")
        print("\n")

        print("========================================")
        generator_class = CTRDataset(mode=3)
        try:
            dataset._check_use_var_with_data_generator(
                slot_data, generator_class, dump_a_path
            )
        except Exception as e:
            print("warning: case 3 catch expected error")
            print(e)
        print("========================================")
        print("\n")

        print("========================================")
        generator_class = CTRDataset(mode=4)
        try:
            dataset._check_use_var_with_data_generator(
                slot_data, generator_class, dump_a_path
            )
        except Exception as e:
            print("warning: case 4 catch expected error")
            print(e)
        print("========================================")
        print("\n")

        print("========================================")
        generator_class = CTRDataset(mode=5)
        try:
            dataset._check_use_var_with_data_generator(
                slot_data, generator_class, dump_a_path
            )
        except Exception as e:
            print("warning: case 5 catch expected error")
            print(e)
        print("========================================")

        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
