#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import math
import os
import sys
import tempfile
import unittest

import numpy as np

import paddle
from paddle import base

paddle.enable_static()


def get_place(target):
    if target == "cuda":
        return base.CUDAPlace(0)
    elif target == "xpu":
        return base.XPUPlace(0)
    elif target == "cpu":
        return base.CPUPlace()
    else:
        raise ValueError(
            f"Target `{target}` is not on the support list: `cuda`, `xpu` and `cpu`."
        )


def train(
    target,
    is_sparse,
    is_parallel,
    save_dirname,
    is_local=True,
    use_bf16=False,
    pure_bf16=False,
):
    PASS_NUM = 100
    EMBED_SIZE = 32
    HIDDEN_SIZE = 256
    N = 5
    BATCH_SIZE = 32
    IS_SPARSE = is_sparse

    def __network__(words):
        embed_first = paddle.static.nn.embedding(
            input=words[0],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
        )
        embed_second = paddle.static.nn.embedding(
            input=words[1],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
        )
        embed_third = paddle.static.nn.embedding(
            input=words[2],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
        )
        embed_forth = paddle.static.nn.embedding(
            input=words[3],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
        )

        concat_embed = paddle.concat(
            [embed_first, embed_second, embed_third, embed_forth], axis=1
        )
        hidden1 = paddle.static.nn.fc(
            x=concat_embed, size=HIDDEN_SIZE, activation='sigmoid'
        )
        predict_word = paddle.static.nn.fc(
            x=hidden1, size=dict_size, activation='softmax'
        )
        cost = paddle.nn.functional.cross_entropy(
            input=predict_word,
            label=words[4],
            reduction='none',
            use_softmax=False,
        )
        avg_cost = paddle.mean(cost)
        return avg_cost, predict_word

    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    first_word = paddle.static.data(name='firstw', shape=[-1, 1], dtype='int64')
    second_word = paddle.static.data(
        name='secondw', shape=[-1, 1], dtype='int64'
    )
    third_word = paddle.static.data(name='thirdw', shape=[-1, 1], dtype='int64')
    forth_word = paddle.static.data(name='forthw', shape=[-1, 1], dtype='int64')
    next_word = paddle.static.data(name='nextw', shape=[-1, 1], dtype='int64')

    if not is_parallel:
        avg_cost, predict_word = __network__(
            [first_word, second_word, third_word, forth_word, next_word]
        )
    else:
        raise NotImplementedError

    sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
    if use_bf16:
        sgd_optimizer = paddle.static.amp.bf16.decorate_bf16(
            sgd_optimizer,
            amp_lists=paddle.static.amp.bf16.AutoMixedPrecisionListsBF16(
                custom_fp32_list={'softmax', 'concat'},
            ),
            use_bf16_guard=False,
            use_pure_bf16=pure_bf16,
        )

    sgd_optimizer.minimize(avg_cost, base.default_startup_program())

    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE
    )

    place = get_place(target)
    exe = base.Executor(place)
    feeder = base.DataFeeder(
        feed_list=[first_word, second_word, third_word, forth_word, next_word],
        place=place,
    )

    def train_loop(main_program):
        exe.run(base.default_startup_program())
        if pure_bf16:
            sgd_optimizer.amp_init(exe.place)

        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_cost_np = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[avg_cost]
                )
                if avg_cost_np[0] < 5.0:
                    if save_dirname is not None and not pure_bf16:
                        paddle.static.io.save_inference_model(
                            save_dirname,
                            [first_word, second_word, third_word, forth_word],
                            [predict_word],
                            exe,
                        )
                    return
                if math.isnan(float(avg_cost_np[0])):
                    sys.exit("got NaN loss, training failed.")

        raise AssertionError(f"Cost is too large {avg_cost_np[0]:2.2}")

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


def infer(target, save_dirname=None):
    if save_dirname is None:
        return

    place = get_place(target)
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

        word_dict = paddle.dataset.imikolov.build_dict()
        dict_size = len(word_dict)

        # Setup inputs by creating 4 DenseTensors representing 4 words. Here each word
        # is simply an index to look up for the corresponding word vector and hence
        # the shape of word (base_shape) should be [1]. The recursive_sequence_lengths,
        # which is length-based level of detail (lod) of each DenseTensor, should be [[1]]
        # meaning there is only one level of detail and there is only one sequence of
        # one word on this level.
        # Note that recursive_sequence_lengths should be a list of lists.
        shape = [1, 1]
        low = 0
        high = dict_size - 1
        # The range of random integers is [low, high]
        first_word = paddle.randint(
            low=low, high=high, shape=shape, dtype=paddle.int64
        )
        second_word = paddle.randint(
            low=low, high=high, shape=shape, dtype=paddle.int64
        )
        third_word = paddle.randint(
            low=low, high=high, shape=shape, dtype=paddle.int64
        )
        fourth_word = paddle.randint(
            low=low, high=high, shape=shape, dtype=paddle.int64
        )

        assert feed_target_names[0] == 'firstw'
        assert feed_target_names[1] == 'secondw'
        assert feed_target_names[2] == 'thirdw'
        assert feed_target_names[3] == 'forthw'

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={
                feed_target_names[0]: first_word,
                feed_target_names[1]: second_word,
                feed_target_names[2]: third_word,
                feed_target_names[3]: fourth_word,
            },
            fetch_list=fetch_targets,
            return_numpy=False,
        )

        infer_inputs = [first_word, second_word, third_word, fourth_word]
        infer_config = base.core.NativeConfig()
        infer_config.prog_file = save_dirname + ".pdmodel"
        infer_config.param_file = save_dirname + ".pdiparams"
        if target == "cuda":
            infer_config.use_gpu = True
            infer_config.device = 0
            infer_config.fraction_of_gpu_memory = 0.15
        elif target == "xpu":
            infer_config.use_xpu = True
        compiled_program = base.compiler.CompiledProgram(inference_program)
        compiled_program._with_inference_optimize(infer_config)
        assert compiled_program._is_inference is True
        infer_outputs = exe.run(compiled_program, feed=infer_inputs)
        np_data = np.array(results[0])
        infer_out = infer_outputs[0].data.float_data()
        for a, b in zip(np_data[0], infer_out):
            assert np.isclose(a, b, rtol=5e-5), f"a: {a}, b: {b}"


def main(target, is_sparse, is_parallel, use_bf16, pure_bf16):
    if target == "cuda" and not base.core.is_compiled_with_cuda():
        return
    if target == "xpu" and not base.core.is_compiled_with_xpu():
        return

    if use_bf16 and not base.core.is_compiled_with_mkldnn():
        return

    temp_dir = tempfile.TemporaryDirectory()
    if not is_parallel:
        save_dirname = os.path.join(temp_dir.name, "word2vec_inference_model")
    else:
        save_dirname = None

    if target == "xpu":
        # This model cannot be trained with xpu temporarily,
        # so only inference is turned on.
        train("cpu", is_sparse, is_parallel, save_dirname)
    else:
        train(
            target,
            is_sparse,
            is_parallel,
            save_dirname,
            use_bf16=use_bf16,
            pure_bf16=pure_bf16,
        )
    infer(target, save_dirname)
    temp_dir.cleanup()


FULL_TEST = os.getenv('FULL_TEST', '0').lower() in [
    'true',
    '1',
    't',
    'y',
    'yes',
    'on',
]
SKIP_REASON = "Only run minimum number of tests in CI server, to make CI faster"


class W2VTest(unittest.TestCase):
    pass


def inject_test_method(
    target, is_sparse, is_parallel, use_bf16=False, pure_bf16=False
):
    fn_name = "test_{}_{}_{}{}".format(
        target,
        "sparse" if is_sparse else "dense",
        "parallel" if is_parallel else "normal",
        "_purebf16" if pure_bf16 else "_bf16" if use_bf16 else "",
    )

    def __impl__(*args, **kwargs):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                main(target, is_sparse, is_parallel, use_bf16, pure_bf16)

    if (
        not base.core.is_compiled_with_cuda() or target == "cuda"
    ) and is_sparse:
        fn = __impl__
    else:
        # skip the other test when on CI server
        fn = unittest.skipUnless(condition=FULL_TEST, reason=SKIP_REASON)(
            __impl__
        )

    setattr(W2VTest, fn_name, fn)


for target in ("cuda", "cpu", "xpu"):
    for is_sparse in (False, True):
        for is_parallel in (False,):
            inject_test_method(target, is_sparse, is_parallel)
inject_test_method("cpu", False, False, True)
inject_test_method("cpu", False, False, True, True)

if __name__ == '__main__':
    unittest.main()
