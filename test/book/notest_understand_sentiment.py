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

import contextlib
import math
import os
import sys
import unittest

import numpy as np

# TODO: remove sys.path.append
sys.path.append("../legacy_test")
import nets

import paddle
from paddle import base


def convolution_net(
    data, label, input_dim, class_dim=2, emb_dim=32, hid_dim=32
):
    emb = paddle.nn.Embedding(
        num_embeddings=input_dim, embedding_dim=emb_dim, sparse=True
    )(data)
    conv_3 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="sqrt",
    )
    conv_4 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="tanh",
        pool_type="sqrt",
    )
    prediction = paddle.static.nn.fc(
        x=[conv_3, conv_4], size=class_dim, activation="softmax"
    )
    cost = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(cost)
    accuracy = paddle.static.accuracy(input=prediction, label=label)
    return avg_cost, accuracy, prediction


def train(
    word_dict,
    net_method,
    use_cuda,
    parallel=False,
    save_dirname=None,
    is_local=True,
):
    BATCH_SIZE = 128
    PASS_NUM = 5
    dict_dim = len(word_dict)
    class_dim = 2

    data = paddle.static.data(name="words", shape=[-1, 1], dtype="int64")
    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    if not parallel:
        cost, acc_out, prediction = net_method(
            data, label, input_dim=dict_dim, class_dim=class_dim
        )
    else:
        raise NotImplementedError

    adagrad = paddle.optimizer.Adagrad(learning_rate=0.002)
    adagrad.minimize(cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000
        ),
        batch_size=BATCH_SIZE,
    )
    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)
    feeder = base.DataFeeder(feed_list=[data, label], place=place)

    def train_loop(main_program):
        exe.run(base.default_startup_program())

        for pass_id in range(PASS_NUM):
            for data in train_data():
                cost_val, acc_val = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[cost, acc_out],
                )
                print("cost=" + str(cost_val) + " acc=" + str(acc_val))
                if cost_val < 0.4 and acc_val > 0.8:
                    if save_dirname is not None:
                        paddle.static.io.save_inference_model(
                            save_dirname, data, prediction, exe
                        )
                    return
                if math.isnan(float(cost_val)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError(f"Cost is too large for {net_method.__name__}")

    if is_local:
        train_loop(base.default_main_program())
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        t = paddle.distributed.transpiler.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(
                current_endpoint, pserver_prog
            )
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(word_dict, use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)

    inference_scope = base.core.Scope()
    with base.scope_guard(inference_scope):
        # Use paddle.static.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(save_dirname, exe)

        word_dict_len = len(word_dict)

        # Setup input by creating DenseTensor to represent sequence of words.
        # Here each word is the basic element of the DenseTensor and the shape of
        # each word (base_shape) should be [1] since it is simply an index to
        # look up for the corresponding word vector.
        # Suppose the recursive_sequence_lengths info is set to [[3, 4, 2]],
        # which has only one level of detail. Then the created DenseTensor will have only
        # one higher level structure (sequence of words, or sentence) than the basic
        # element (word). Hence the DenseTensor will hold data for three sentences of
        # length 3, 4 and 2, respectively.
        # Note that recursive_sequence_lengths should be a list of lists.
        # The range of random integers is [low, high]
        tensor_words = paddle.randint(
            low=0, high=word_dict_len - 1, shape=[9, 1], dtype=paddle.int64
        )

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        assert feed_target_names[0] == "words"
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_words},
            fetch_list=fetch_targets,
            return_numpy=False,
        )
        np_data = np.array(results[0])
        print("Inference Shape: ", np_data.shape)
        print("Inference results: ", np_data)


def main(word_dict, net_method, use_cuda, parallel=False, save_dirname=None):
    if use_cuda and not base.core.is_compiled_with_cuda():
        return

    train(
        word_dict,
        net_method,
        use_cuda,
        parallel=parallel,
        save_dirname=save_dirname,
    )
    infer(word_dict, use_cuda, save_dirname)


class TestUnderstandSentiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word_dict = paddle.dataset.imdb.word_dict()

    @contextlib.contextmanager
    def new_program_scope(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                yield

    def test_conv_cpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=False,
                save_dirname="understand_sentiment_conv.inference.model",
            )

    def test_conv_cpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=False,
                parallel=True,
            )

    def test_conv_gpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=True,
                save_dirname="understand_sentiment_conv.inference.model",
            )

    def test_conv_gpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=True,
                parallel=True,
            )


if __name__ == '__main__':
    unittest.main()
