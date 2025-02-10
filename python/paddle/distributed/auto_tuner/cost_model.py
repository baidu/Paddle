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


def all_params(mp, pp, sharding, h, l, V):
    # TODO: TBD - add some fixed structure models.
    return 1


def full_recompute_acts(mp, pp, s, b, h, l):
    # TODO: TBD - add some fixed structure models.
    return 1


def all_acts(mp, pp, s, b, h, l, a):
    # TODO: TBD - add some fixed structure models.
    return 1


def to_gb(p):
    return p / (2**30)


def get_mem(total_cards, parallel_cfg, l, h, a, V, s, gbs):
    """Estimate the memory of model unset parallel strategy."""
    sharding = parallel_cfg["sharding_degree"]
    mp = parallel_cfg["mp_degree"]
    b = parallel_cfg["micro_batch_size"]
    pp = parallel_cfg["pp_degree"]
    vpp = parallel_cfg["vpp_degree"]
    use_recompute = parallel_cfg["use_recompute"]

    sep = 1

    lbs = int(gbs / sharding / s)
    lbs = int(lbs / pp) * pp
    assert s % sep == 0
    s_sep = s // sep
    assert a % (sep * mp) == 0, f'{a} vs {sep * mp}'

    vpp_ratio = 1
    if vpp > 1:
        assert l % (pp * vpp) == 0
        vpp_ratio = 1 + (pp - 1) / (pp * vpp)

    params = to_gb(all_params(mp, pp, sharding, h, l, V))

    acts = 0
    assert l % pp == 0

    if use_recompute:
        acts = to_gb(full_recompute_acts(mp, pp, s_sep, b, h, l)) * vpp_ratio
    else:
        acts = to_gb(all_acts(mp, pp, s, b, h, l, a)) * vpp_ratio
    assert acts > 0

    peak_mem = params + acts
    return peak_mem


def divisor(num, reverse=False):
    """Get the divisor of a given number."""
    results = set()
    i = 1
    mid = num // 2 + 1
    while i < mid:
        if num % i == 0:
            results.add(i)
            results.add(num // i)
        i += 1
    results = list(results)
    return sorted(results, reverse=reverse)


def get_not_oom_cfgs(cfgs, tuner_cfg):
    """Get not OOM parallel strategies."""
    total_cards, l, h, a, V, s, gbs, per_card_memory = (
        tuner_cfg["search_algo"]["estimated_num_gpus"],
        tuner_cfg["model_cfg"]["num_layers"],
        tuner_cfg["model_cfg"]["hidden_size"],
        tuner_cfg["model_cfg"]["num_attention_heads"],
        tuner_cfg["model_cfg"]["vocab_size"],
        tuner_cfg["model_cfg"]["seq_length"],
        tuner_cfg["model_cfg"]["global_batch_size"],
        tuner_cfg.get("per_card_memory", 80),
    )
    pruned_cfgs = []
    for cfg in cfgs:
        mp = cfg["mp_degree"]
        sharding = cfg["sharding_degree"]
        mbs = cfg["micro_batch_size"]
        pp = cfg["pp_degree"]
        vpp = cfg["vpp_degree"]
        dp = cfg["dp_degree"]
        use_recompute = cfg["use_recompute"]

        if mp * sharding * pp * dp != total_cards:
            continue
        if gbs % sharding != 0:
            continue
        if gbs // sharding % dp != 0:
            continue
        if gbs // sharding // dp % mbs != 0:
            continue
        if l % pp != 0:
            continue
        if l // pp % vpp != 0:
            continue
        if vpp != 1 and pp <= 2:
            continue
        if a % mp != 0 or V % mp != 0 or h % mp != 0:
            continue

        pruned_cfgs.append(cfg)
    valid_cfgs = []
    for cfg in pruned_cfgs:
        mem = get_mem(total_cards, cfg, l, h, a, V, s, gbs)
        # TODO: Uncomment when it is actually implemented.
        # if (
        #     mem < per_card_memory
        #     and mem
        #     > tuner_cfg.get(
        #         "search_algo", {"name": "dp_estimation", "threshold": 0.7}
        #     ).get("threshold", 0.7)
        #     * per_card_memory
        # ):
        # cfg["memory_cost"] = mem
        # valid_cfgs.append(cfg)
        cfg["memory_cost"] = mem
        valid_cfgs.append(cfg)
    assert valid_cfgs
    return valid_cfgs
