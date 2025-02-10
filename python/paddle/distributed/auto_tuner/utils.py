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
from __future__ import annotations

import copy
import csv
import itertools
import logging
import os
import re

import paddle

from .prune import _PRUNE_FUNC

__SUPPORTED_RECOMPUTE_GRANULARITY__ = ["full", "full_attn", "core_attn"]

logger = logging.getLogger('auto_tuner')


def divisor(num, reverse=False):
    """Return the divisor of the given number."""
    if num == 1:
        return [num]
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


def dist_degree_with_customized_range(
    mode, num_gpus, num_nodes, customized_range, tuner_cfg=None
):
    """Return the degree of different parallel modes by gpus and nodes num with customized range."""
    dist_degree_all = dist_degree(mode, num_gpus, num_nodes, tuner_cfg)
    return [degree for degree in dist_degree_all if degree in customized_range]


def dist_degree(mode, num_gpus, num_nodes, tuner_cfg=None):
    """Return the degree of different parallel modes by gpus and nodes num."""
    assert mode in [
        "dp_degree",
        "mp_degree",
        "pp_degree",
        "sharding_degree",
        "micro_batch_size",
        "vpp_degree",
    ]
    results = []
    prune_results = []
    if mode == "dp_degree":
        if tuner_cfg.get("schedule_mode", "memory") != "performance":
            results = divisor(num_gpus, reverse=False)
        else:
            results = divisor(num_gpus, reverse=True)

    elif mode == "pp_degree":
        if num_nodes > 1 and tuner_cfg.get("enable_pp_prune", True):
            results = list(range(num_nodes + 1, 0, -1))
        else:
            results = divisor(num_gpus, reverse=True)
        for pp_degree in results:
            prune_flag = False
            num_layers = tuner_cfg["model_cfg"].get("num_layers", None)

            if num_layers:
                if num_layers % pp_degree != 0:
                    prune_flag = True

            if not prune_flag:
                prune_results.append(pp_degree)
        results = prune_results

    elif mode == "mp_degree":
        if tuner_cfg.get("enable_mp_prune", True):
            gpus_per_node = num_gpus // num_nodes
            if tuner_cfg.get("schedule_mode", "memory") != "performance":
                results = divisor(gpus_per_node, reverse=True)
            else:
                results = divisor(gpus_per_node, reverse=False)
        else:
            if tuner_cfg.get("schedule_mode", "memory") != "performance":
                results = divisor(num_gpus, reverse=True)
            else:
                results = divisor(num_gpus, reverse=False)

        for mp_degree in results:
            prune_flag = False
            hidden_size = tuner_cfg["model_cfg"].get("hidden_size", None)
            vocab_size = tuner_cfg["model_cfg"].get("vocab_size", None)
            num_attention_heads = tuner_cfg["model_cfg"].get(
                "num_attention_heads", None
            )
            seq_length = tuner_cfg["model_cfg"].get("seq_length", None)
            use_sequence_parallel = tuner_cfg.get(
                "use_sequence_parallel", False
            )

            if hidden_size and hidden_size % mp_degree != 0:
                prune_flag = True

            if vocab_size and vocab_size % mp_degree != 0:
                prune_flag = True

            if num_attention_heads and num_attention_heads % mp_degree != 0:
                prune_flag = True

            if (
                seq_length
                and seq_length % mp_degree != 0
                and use_sequence_parallel
            ):
                prune_flag = True

            if not prune_flag:
                prune_results.append(mp_degree)
        results = prune_results

    elif mode == "sharding_degree":
        results = divisor(num_gpus, reverse=True)

    elif mode == "micro_batch_size":
        if tuner_cfg.get("schedule_mode", "memory") != "performance":
            results = divisor(
                tuner_cfg["model_cfg"]["global_batch_size"], reverse=False
            )
        else:
            results = divisor(
                tuner_cfg["model_cfg"]["global_batch_size"], reverse=True
            )

    elif mode == "vpp_degree":
        if tuner_cfg.get("schedule_mode", "memory") != "performance":
            results = divisor(
                tuner_cfg["model_cfg"]["num_layers"], reverse=False
            )
        else:
            results = divisor(
                tuner_cfg["model_cfg"]["num_layers"], reverse=True
            )

    return results


def default_candidates(tuner_cfg):
    """Return the default candidates of every hyper param which user defined auto"""
    candidates = {}
    estimated_num_gpus = None
    if (
        "search_algo" in tuner_cfg
        and "estimated_num_gpus" in tuner_cfg["search_algo"]
    ):
        estimated_num_gpus = tuner_cfg["search_algo"]["estimated_num_gpus"]
    num_gpus = (
        tuner_cfg["num_gpus"]
        if estimated_num_gpus is None
        else estimated_num_gpus
    )
    num_nodes = (
        tuner_cfg["nodes"]
        if estimated_num_gpus is None
        else estimated_num_gpus // tuner_cfg["gpus_per_node"]
    )
    assert num_gpus > 0

    for strategy in ["dp_degree", "mp_degree", "pp_degree", "sharding_degree"]:
        strategy_customized_range = _param2range(
            tuner_cfg.get(strategy, None), num_gpus, strategy
        )
        candidates[strategy] = dist_degree_with_customized_range(
            strategy, num_gpus, num_nodes, strategy_customized_range, tuner_cfg
        )

    vpp_degree_customized_range = _param2range(
        tuner_cfg.get("vpp_degree", None),
        tuner_cfg["model_cfg"]["num_layers"],
        "vpp_degree",
    )
    candidates["vpp_degree"] = dist_degree_with_customized_range(
        "vpp_degree",
        num_gpus,
        num_nodes,
        vpp_degree_customized_range,
        tuner_cfg,
    )

    mbs_customized_range = _param2range(
        tuner_cfg.get("micro_batch_size", None),
        tuner_cfg["model_cfg"]["global_batch_size"],
        "micro_batch_size",
    )
    candidates["micro_batch_size"] = dist_degree_with_customized_range(
        "micro_batch_size", num_gpus, num_nodes, mbs_customized_range, tuner_cfg
    )

    schedule_mode = tuner_cfg.get("schedule_mode", "memory")
    sharding_stage_customized_range = _param2range(
        tuner_cfg.get("sharding_stage", None), 3, "sharding_stage"
    )
    candidates["sharding_stage"] = [
        stage for stage in [3, 2, 1] if stage in sharding_stage_customized_range
    ]
    if schedule_mode != "performance":
        candidates["sharding_stage"] = sorted(
            candidates["sharding_stage"], reverse=True
        )
    else:
        candidates["sharding_stage"] = sorted(
            candidates["sharding_stage"], reverse=False
        )

    use_recompute = tuner_cfg.get("use_recompute", None)
    if isinstance(use_recompute, str) and use_recompute.lower() == "auto":
        candidates["use_recompute"] = (
            [True, False] if schedule_mode != "performance" else [False, True]
        )
    elif isinstance(use_recompute, bool):
        candidates["use_recompute"] = [use_recompute]
    elif isinstance(use_recompute, list):
        if len(use_recompute) == 0:
            candidates["use_recompute"] = [None]
        else:
            candidates["use_recompute"] = []
            for recompute_setting in use_recompute:
                if recompute_setting not in [True, False]:
                    raise ValueError(
                        f"use_recompute only supports auto/True/False, but got {recompute_setting}"
                    )
                else:
                    candidates["use_recompute"].append(recompute_setting)
            if len(candidates["use_recompute"]) == 0:
                candidates["use_recompute"] = [None]
    # TODO: should remove this case in the future
    elif use_recompute is None:
        candidates["use_recompute"] = [None]
    else:
        raise ValueError("use_recompute supports auto/True/False")

    recompute_granularity = tuner_cfg.get("recompute_granularity", None)
    if isinstance(recompute_granularity, str):
        if recompute_granularity.lower() == "auto":
            candidates["recompute_granularity"] = (
                __SUPPORTED_RECOMPUTE_GRANULARITY__
                if schedule_mode != "performance"
                else list(reversed(__SUPPORTED_RECOMPUTE_GRANULARITY__))
            )
        elif (
            recompute_granularity.lower() in __SUPPORTED_RECOMPUTE_GRANULARITY__
        ):
            candidates["recompute_granularity"] = [
                recompute_granularity.lower()
            ]
        else:
            raise ValueError(
                f"recompute_granularity only supports auto/{'/'.join(__SUPPORTED_RECOMPUTE_GRANULARITY__)}, but got {recompute_granularity}"
            )
    elif isinstance(recompute_granularity, list):
        if len(recompute_granularity) == 0:
            candidates["recompute_granularity"] = [None]
        else:
            candidates["recompute_granularity"] = []
            for granularity in recompute_granularity:
                if (
                    granularity.lower()
                    not in __SUPPORTED_RECOMPUTE_GRANULARITY__
                ):
                    raise ValueError(
                        f"recompute_granularity only supports auto/{'/'.join(__SUPPORTED_RECOMPUTE_GRANULARITY__)}, but got {granularity}"
                    )
                else:
                    candidates["recompute_granularity"].append(
                        granularity.lower()
                    )
            if len(candidates["recompute_granularity"]) == 0:
                candidates["recompute_granularity"] = [None]
    # TODO: should remove this case in the future
    elif recompute_granularity is None:
        candidates["recompute_granularity"] = [None]
    else:
        raise ValueError(
            f"recompute_granularity only supports auto/{'/'.join(__SUPPORTED_RECOMPUTE_GRANULARITY__)}, but got {recompute_granularity}"
        )
    custom_search_dim = tuner_cfg.get("custom_search_dim", None)
    if custom_search_dim is not None:
        candidates["custom_search_dim"] = []
        for key, value in custom_search_dim.items():
            candidates["custom_search_dim"].append(value["value"])
    return candidates


def search_all(tuner_cfg):
    """Permutate the candidates of all hyper params."""
    candidates = tuner_cfg["candidates"]
    # Order: dp -> sharding -> mbs -> pp -> mp  -> recompute
    dp_degree_candidates = candidates["dp_degree"]
    mp_degree_candidates = candidates["mp_degree"]
    pp_degree_candidates = candidates["pp_degree"]
    vpp_degree_candidates = candidates["vpp_degree"]
    mbs_candidates = candidates["micro_batch_size"]
    sharding_stage_candidates = candidates["sharding_stage"]
    sharding_degree_candidates = candidates["sharding_degree"]
    use_recompute_candidates = candidates["use_recompute"]
    recompute_granularity_candidates = candidates["recompute_granularity"]

    num_gpus = (
        tuner_cfg["num_gpus"]
        if "search_algo" not in tuner_cfg
        or "estimated_num_gpus" not in tuner_cfg["search_algo"]
        else tuner_cfg["search_algo"]["estimated_num_gpus"]
    )
    valid_degrees = []

    for mp_degree in mp_degree_candidates:
        degrees = []
        if num_gpus % mp_degree != 0:
            continue
        degrees.append(mp_degree)
        sharding_res = num_gpus // mp_degree

        for sharding_degree in sharding_degree_candidates:
            if sharding_res % sharding_degree != 0:
                continue
            degrees.append(sharding_degree)
            pp_res = sharding_res // sharding_degree

            for pp_degree in pp_degree_candidates:
                if pp_res % pp_degree != 0:
                    continue
                degrees.append(pp_degree)
                dp_res = pp_res // pp_degree

                for dp_degree in dp_degree_candidates:
                    if dp_res != dp_degree:
                        continue
                    degrees.append(dp_degree)
                    assert len(degrees) == 4
                    valid_degrees.append(copy.deepcopy(degrees))
                    degrees.pop()
                degrees.pop()
            degrees.pop()

    other_dim_cfgs = list(
        itertools.product(
            sharding_stage_candidates,
            mbs_candidates,
            vpp_degree_candidates,
            use_recompute_candidates,
            recompute_granularity_candidates,
        )
    )

    custom_search_dim = tuner_cfg.get("custom_search_dim", None)
    if custom_search_dim is not None:
        custom_search_dim_candidates = candidates["custom_search_dim"]
        custom_dim_cfgs = list(itertools.product(*custom_search_dim_candidates))
        other_cfgs_without_cumtom = other_dim_cfgs
        other_dim_cfgs = []
        for cfg_without_cumtom in other_cfgs_without_cumtom:
            for custom_cfg in custom_dim_cfgs:
                cfg = list(cfg_without_cumtom) + list(custom_cfg)
                other_dim_cfgs.append(cfg)

    all_cfgs = []
    refined_recompute = tuner_cfg.get("refined_recompute", None)
    for valid_degree in valid_degrees:
        for other_dim_cfg in other_dim_cfgs:
            mp_degree, sharding_degree, pp_degree, dp_degree = valid_degree
            (
                sharding_stage,
                mbs,
                vpp,
                use_recompute,
                recompute_granularity,
            ) = list(other_dim_cfg[:5])
            if (
                tuner_cfg["model_cfg"]["global_batch_size"]
                % (mbs * sharding_degree * dp_degree)
                != 0
            ):
                continue
            if tuner_cfg["model_cfg"]["num_layers"] % (pp_degree * vpp) != 0:
                continue

            if refined_recompute is not None:
                # if refine recompute is not valid, set 0 for all rr op.
                if (
                    (pp_degree == 1)
                    or (not use_recompute)
                    or (use_recompute and recompute_granularity != "full")
                ):
                    cfg = (
                        list(valid_degree)
                        + list(other_dim_cfg)
                        + [0 for i in range(len(refined_recompute))]
                    )
                    if cfg not in all_cfgs:
                        all_cfgs.append(cfg)
                else:
                    max_value = (
                        tuner_cfg["model_cfg"]["num_layers"] // pp_degree
                    )
                    rr_valid_values = list(range(0, max_value + 1))
                    # The previous operator has reached its maximum value, and the current operator can only be turned on
                    op_count = len(refined_recompute)

                    # first op values
                    rr_dim_cfgs = []
                    for value in rr_valid_values:
                        cfg = [value]
                        cfg.extend([0 for _ in range(op_count - 1)])
                        if cfg not in rr_dim_cfgs:
                            rr_dim_cfgs.append(cfg)
                    # other ops values
                    i = 1
                    while i < op_count:
                        for value in rr_valid_values:
                            cfg = [max_value for _ in range(i)]
                            cfg.extend([value])
                            cfg.extend([0 for _ in range(op_count - i - 1)])
                            if cfg not in rr_dim_cfgs:
                                rr_dim_cfgs.append(cfg)
                        i += 1

                    if tuner_cfg.get("schedule_mode") != "performance":
                        # memory sort
                        for rr_dim_cfg in rr_dim_cfgs:
                            cfg = (
                                list(valid_degree)
                                + list(other_dim_cfg)
                                + list(rr_dim_cfg)
                            )
                            if cfg not in all_cfgs:
                                all_cfgs.append(cfg)
                    else:
                        rr_dim_cfgs.sort(reverse=True)
                        for rr_dim_cfg in rr_dim_cfgs:
                            cfg = (
                                list(valid_degree)
                                + list(other_dim_cfg)
                                + list(rr_dim_cfg)
                            )
                            if cfg not in all_cfgs:
                                all_cfgs.append(cfg)
            else:
                cfg = list(valid_degree) + list(other_dim_cfg)
                all_cfgs.append(cfg)

    mapping = {
        0: "mp_degree",
        1: "sharding_degree",
        2: "pp_degree",
        3: "dp_degree",
        4: "sharding_stage",
        5: "micro_batch_size",
        6: "vpp_degree",
        7: "use_recompute",
        8: "recompute_granularity",
    }

    if custom_search_dim is not None:
        for key, _ in custom_search_dim.items():
            mapping[len(mapping)] = key

    if refined_recompute is not None:
        for dim in refined_recompute:
            mapping[len(mapping)] = dim
    new_all_cfgs = []
    for cfg in all_cfgs:
        new_cfg = {}
        for idx, val in enumerate(cfg):
            new_cfg[mapping[idx]] = val
        new_all_cfgs.append(new_cfg)
    search_space_size_before_prune = len(new_all_cfgs)
    pruned_all_cfgs = []
    tuner_cfg["num_gpus"] = num_gpus
    for cur_cfg in new_all_cfgs:
        pruned = False
        for func in _PRUNE_FUNC:
            result = func(tuner_cfg, cur_cfg, pruned_all_cfgs)
            if result:
                pruned = True
                break
        if not pruned:
            pruned_all_cfgs.append(cur_cfg)
    search_space_size_after_prune = len(pruned_all_cfgs)
    logger.info(
        f"{search_space_size_before_prune - search_space_size_after_prune} tasks are pruned before launching."
    )
    if tuner_cfg.get("schedule_prior", False):
        pruned_all_cfgs = sort_by_special(pruned_all_cfgs, tuner_cfg)
    return pruned_all_cfgs


def sort_by_special(cfgs, tuner_cfg):
    assert tuner_cfg.get("schedule_prior", False)
    prior_strategy = tuner_cfg["schedule_prior"]
    prior_strategy.sort(reverse=True)
    for strategy in prior_strategy:
        idx = 0
        matched_count = 0
        while idx < len(cfgs):
            cfg = cfgs[idx]
            if _matched(cfg, strategy):
                cfgs.pop(idx)
                cfgs.insert(0, cfg)
                matched_count += 1
            idx += 1
        tmp = cfgs[:matched_count]
        tmp.reverse()
        cfgs[:matched_count] = tmp
    return cfgs


def memory_sort(cfg):
    # ascending order in default
    return (
        -cfg['mp_degree'],
        -cfg['pp_degree'],
        -cfg['vpp_degree'],
        -cfg["sharding_degree"],
        -cfg["sharding_stage"],
        cfg["micro_batch_size"],
        -cfg["use_recompute"],
    )


def performance_sort(cfg):
    return -cfg["micro_batch_size"]


def _matched(cur_cfg, strategy):
    mapping = {
        "dp_degree": "dp",
        "mp_degree": "mp",
        "pp_degree": "pp",
        "vpp_degree": "vpp",
        "micro_batch_size": "mbs",
        "sharding_degree": "sharding",
        "sharding_stage": "stage",
        "use_recompute": "recompute",
        "recompute_granularity": "granularity",
    }
    granularity_mapping = {0: "full", 1: "full_attn", 2: "core_attn"}
    reversed_mapping = {}
    for key in mapping:
        reversed_mapping[mapping[key]] = key

    assert isinstance(strategy, str)
    dims = strategy.split("_")
    has_matched = 0
    for dim in dims:
        matched = None
        for key in reversed_mapping:
            if dim.startswith(key):
                matched = key
                break
        if matched:
            value = dim[len(matched)]
            # * means this strategy turned on
            if matched in ["dp", "mp", "pp", "vpp", "sharding"]:
                if value == "*":
                    if cur_cfg[reversed_mapping[matched]] > 1:
                        has_matched += 1
                        continue
                else:
                    value = int(value)
                    if cur_cfg[reversed_mapping[matched]] == value:
                        has_matched += 1
                        continue
            elif matched == "recompute":
                if value == "*":
                    if cur_cfg[reversed_mapping[matched]]:
                        has_matched += 1
                        continue
                else:
                    value = bool(int(value))
                    if cur_cfg[reversed_mapping[matched]] == value:
                        has_matched += 1
                        continue
            elif matched == "stage":
                if value == "*":
                    if cur_cfg[reversed_mapping["sharding"]] > 1:
                        has_matched += 1
                        continue
                else:
                    value = int(value)
                    if cur_cfg[reversed_mapping[matched]] == value:
                        has_matched += 1
                        continue
            elif matched == "mbs":
                if value == "*":
                    has_matched += 1
                    continue
                else:
                    value = int(value)
                    if cur_cfg[reversed_mapping[matched]] == value:
                        has_matched += 1
                        continue
            elif matched == "granularity":
                if value == "*":
                    if cur_cfg[reversed_mapping["use_recompute"]]:
                        has_matched += 1
                        continue
                else:
                    value = int(value)
                    granularity = granularity_mapping[value]
                    if cur_cfg[reversed_mapping[matched]] == granularity:
                        has_matched += 1
                        continue
    if has_matched == len(dims):
        return True
    return False


def _param2range(param_from_json_file, max_value, param_key):
    """Convert a param from json file to candidates range."""
    selected_range = None
    if isinstance(param_from_json_file, str):
        if "auto" in param_from_json_file.lower():
            selected_range = list(range(1, max_value + 1))
        else:
            raise ValueError(
                f"Illegal param found: {param_key}, only support auto in str type."
            )
    elif isinstance(param_from_json_file, dict):
        customized_min_value = param_from_json_file.get("min", None)
        customized_max_value = param_from_json_file.get("max", None)
        if not (customized_min_value and customized_max_value):
            raise ValueError(
                f"Illegal param found: {param_key}, min and max should be specified in dict type."
            )
        selected_range = list(
            range(customized_min_value, customized_max_value + 1)
        )
    elif isinstance(param_from_json_file, list):
        selected_range = param_from_json_file
    elif isinstance(param_from_json_file, int):
        selected_range = [param_from_json_file]
    elif param_from_json_file is None:
        selected_range = [1]
    else:
        raise ValueError(
            f"Illegal param found: {param_key}, only support str, dict, list and int type."
        )
    return selected_range


def search_by_dp_estimation(tuner_cfg):
    all_cfgs = search_all(tuner_cfg)
    estimated_num_gpus = tuner_cfg["search_algo"].get(
        "estimated_num_gpus", None
    )
    assert estimated_num_gpus is not None
    # change global_batch_size, dp_degree, sharding_degree
    new_all_cfgs = []
    for task in all_cfgs:
        task["estimated_dp_degree"] = int(
            task["dp_degree"] * task["sharding_degree"]
        )
        task["dp_degree"] = 1
        task["sharding_degree"] = 1
        task["sharding_stage"] = 1
        task["num_gpus"] = task["mp_degree"] * task["pp_degree"]
        actual_cards = task["num_gpus"]
        if actual_cards <= tuner_cfg["gpus_per_node"]:
            nnodes = 1
        elif actual_cards % tuner_cfg["gpus_per_node"] == 0:
            nnodes = actual_cards // tuner_cfg["gpus_per_node"]
        else:
            for i in range(2, tuner_cfg["nodes"] + 1):
                if (
                    actual_cards % i == 0
                    and actual_cards // i <= tuner_cfg["gpus_per_node"]
                ):
                    nnodes = i
                    break
        assert actual_cards % nnodes == 0
        task["nodes"] = nnodes
        task["global_batch_size"] = (
            tuner_cfg["model_cfg"]["global_batch_size"]
            // task["estimated_dp_degree"]
        )
        if task not in new_all_cfgs and task["nodes"] <= tuner_cfg["nodes"]:
            new_all_cfgs.append(task)

    # expanding sharding degree to run overlap and non-overlap to calculate overlap benefits
    sharding_all_cfgs = []
    if tuner_cfg["search_algo"].get("sharding_overlap", None):
        for task in new_all_cfgs:
            new_task = copy.deepcopy(task)
            given_num_gpus = tuner_cfg["nodes"] * tuner_cfg["gpus_per_node"]
            sharding_degree = given_num_gpus // task["num_gpus"]
            if sharding_degree > 1:
                new_task["sharding_degree"] = sharding_degree
                new_task["sharding_stage"] = 1
                new_task["estimated_dp_degree"] = None
                new_task["num_gpus"] = (
                    new_task["mp_degree"]
                    * new_task["pp_degree"]
                    * new_task["sharding_degree"]
                )
                actual_cards = new_task["num_gpus"]
                if actual_cards <= tuner_cfg["gpus_per_node"]:
                    nnodes = 1
                elif actual_cards % tuner_cfg["gpus_per_node"] == 0:
                    nnodes = actual_cards // tuner_cfg["gpus_per_node"]
                else:
                    for i in range(2, tuner_cfg["nodes"] + 1):
                        if (
                            actual_cards % i == 0
                            and actual_cards // i <= tuner_cfg["gpus_per_node"]
                        ):
                            nnodes = i
                            break
                assert actual_cards % nnodes == 0
                new_task["nodes"] = nnodes
                new_task["global_batch_size"] = (
                    task["global_batch_size"] * sharding_degree
                )
                new_task["sharding_overlap"] = False
                sharding_all_cfgs.append(new_task)

                overlap_new_task = copy.deepcopy(new_task)
                overlap_new_task["sharding_overlap"] = True
                sharding_all_cfgs.append(overlap_new_task)

    new_all_cfgs.extend(sharding_all_cfgs)
    return new_all_cfgs


def add_overlap_performance(cur_cfg, tuner_cfg, history_cfgs):
    """
    In single dp search scenario,
    the overlay acceleration ratio is obtained by automatically running overlap and non overlap tasks,
    and the estimated performance of the multi dp after overlap is obtained.
    """
    if cur_cfg[tuner_cfg['metric_cfg']['name']]:
        non_overlap_cfg = None
        raw_cfg = None
        for cfg in history_cfgs:
            keys = [
                "dp_degree",
                "mp_degree",
                "pp_degree",
                "vpp_degree",
                "micro_batch_size",
                "use_recompute",
                "recompute_granularity",
                "sharding_stage",
            ]
            same = True
            for key in keys:
                if cfg[key] != cur_cfg[key]:
                    same = False
                    break
            if same:
                if "sharding_overlap" not in cfg:
                    raw_cfg = cfg
                elif not cfg["sharding_overlap"]:
                    if cfg["sharding_degree"] == cur_cfg["sharding_degree"]:
                        non_overlap_cfg = cfg
        assert non_overlap_cfg is not None
        assert raw_cfg is not None

        before_overlap_performance = non_overlap_cfg[
            tuner_cfg['metric_cfg']['name']
        ]
        overlap_performance = cur_cfg[tuner_cfg['metric_cfg']['name']]
        raw_performance = raw_cfg[tuner_cfg['metric_cfg']['name']]
        if (
            raw_performance
            and overlap_performance
            and before_overlap_performance
        ):
            ratio = (
                overlap_performance - before_overlap_performance
            ) / before_overlap_performance
            keys = copy.deepcopy(list(raw_cfg.keys()))
            for key in keys:
                if key.startswith("bw_") and raw_cfg[key]:
                    mew_key = "overlap_" + key
                    raw_cfg[mew_key] = round(raw_cfg[key] * (1 + ratio), 5)


def gen_sharding_overlap_args_of_grid_search(res_args, cfg, tuner_cfg):
    """Generate args of sharding overlap."""
    if "sharding_overlap" not in tuner_cfg["search_algo"]:
        return
    cmd = copy.deepcopy(tuner_cfg["search_algo"]["sharding_overlap"])
    valid_hybrid_strategy = [
        "sharding_mp",
        "sharding_pp",
        "sharding_mp_pp",
        "no_overlap",
    ]
    for key in cmd:
        if key not in valid_hybrid_strategy:
            raise ValueError(
                f"Only support {valid_hybrid_strategy}, but got {key}."
            )
    sharding_degree = cfg["sharding_degree"]
    mp_degree = cfg["mp_degree"]
    pp_degree = cfg["pp_degree"]
    arg = None
    if mp_degree > 1 and pp_degree == 1 and sharding_degree > 1:
        arg = "sharding_mp"
    elif mp_degree == 1 and pp_degree > 1 and sharding_degree > 1:
        arg = "sharding_pp"
    elif mp_degree > 1 and pp_degree > 1 and sharding_degree > 1:
        arg = "sharding_mp_pp"
    else:
        arg = "no_overlap"
    assert arg is not None
    if arg in cmd:
        if "--" in cmd[arg][0]:
            arg_map_len = len(cmd[arg])
            assert arg_map_len % 2 == 0
            i = 0
            while i < arg_map_len:
                new_arg = [cmd[arg][i], str(cmd[arg][i + 1])]
                res_args.extend(new_arg)
                i += 2
        elif "-o" in cmd[arg][0]:
            res_args.extend(cmd[arg])
        elif ".json" in cmd[arg][0]:
            import json

            file_path = cmd[arg][0]
            try:
                with open(file_path, "r") as f:
                    cmd_cfg = json.load(f)
            except:
                raise ValueError(
                    "Please check your auto tuner json whether valid."
                )
            keys = cmd[arg][1].split(".")
            value = None
            for key in keys[: len(keys) - 1]:
                if value:
                    value = value[key]
                else:
                    value = cmd_cfg[key]
            if value:
                value[keys[-1]] = cmd[arg][2]
            else:
                cmd_cfg[keys[-1]] = cmd[arg][2]
            json.dump(cmd_cfg, open(cmd[arg][0], "w"))

        elif ".yaml" in cmd[arg][0]:
            import yaml

            file_path = cmd[arg][0]
            try:
                with open(file_path, "r") as f:
                    cmd_cfg = yaml.safe_load(f)
            except:
                raise ValueError(
                    "Please check your auto tuner json whether valid."
                )
            arg_map_len = len(cmd[arg]) - 1
            assert arg_map_len % 2 == 0

            i = 1
            while i < arg_map_len:
                keys = cmd[arg][i].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    i += 1
                    value[keys[-1]] = cmd[arg][i]
                else:
                    i += 1
                    cmd_cfg[keys[-1]] = cmd[arg][i]
                i += 1
            yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))


def gen_sharding_overlap_args(res_args, cfg, tuner_cfg):
    """Generate args of sharding overlap."""
    if "sharding_overlap" not in tuner_cfg["search_algo"]:
        return
    cmd = copy.deepcopy(tuner_cfg["search_algo"]["sharding_overlap"])
    if "sharding_overlap" in cfg:
        valid_hybrid_strategy = ["sharding_mp", "sharding_pp", "sharding_mp_pp"]
        for key in cmd:
            if key not in valid_hybrid_strategy:
                raise ValueError(
                    f"Only support {valid_hybrid_strategy}, but got {key}."
                )
        sharding_degree = cfg["sharding_degree"]
        assert sharding_degree > 1
        mp_degree = cfg["mp_degree"]
        pp_degree = cfg["pp_degree"]
        arg = None
        if mp_degree > 1 and pp_degree == 1:
            arg = "sharding_mp"
        elif mp_degree == 1 and pp_degree > 1:
            arg = "sharding_pp"
        elif mp_degree > 1 and pp_degree > 1:
            arg = "sharding_mp_pp"
        else:
            return
        assert arg is not None
        if arg in cmd:
            if "--" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = (
                        cmd[arg][2] if cfg["sharding_overlap"] else cmd[arg][3]
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        cmd[arg][2] if cfg["sharding_overlap"] else cmd[arg][3]
                    )
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))


def gen_new_args(raw_args, cfg, tuner_cfg, run_best=False):
    """Generate new script args."""
    cfg = copy.deepcopy(cfg)

    def _get_new_cfg(arg, cmg, cfg, tuner_cfg):
        if arg == "local_batch_size" and arg in cmd:
            global_batch_size = (
                cfg["global_batch_size"]
                if "global_batch_size" in cfg
                else tuner_cfg["model_cfg"]["global_batch_size"]
            )
            local_batch_size = (
                global_batch_size // cfg["sharding_degree"] // cfg["dp_degree"]
            )
            cfg["local_batch_size"] = local_batch_size

        if arg == "gradient_accumulation_steps" and arg in cmd:
            try:
                global_batch_size = (
                    cfg["global_batch_size"]
                    if "global_batch_size" in cfg
                    else tuner_cfg["model_cfg"]["global_batch_size"]
                )
                gradient_accumulation_steps = (
                    global_batch_size
                    // cfg["sharding_degree"]
                    // cfg["dp_degree"]
                    // cfg["micro_batch_size"]
                )
                cfg["gradient_accumulation_steps"] = gradient_accumulation_steps
            except:
                return

        if arg == "sequence_parallel" and arg in cmd:
            try:
                sequence_parallel = 1 if cfg["mp_degree"] > 1 else 0
                cfg["sequence_parallel"] = sequence_parallel
            except:
                return

        if arg == "global_batch_size" and arg in cmd:
            try:
                global_batch_size = (
                    cfg["global_batch_size"]
                    if "global_batch_size" in cfg
                    else tuner_cfg["model_cfg"]["global_batch_size"]
                )
                cfg["global_batch_size"] = global_batch_size
            except:
                return

    def _gen_new_arg(arg, cmd, cfg, res_args, tuner_cfg):
        if arg in cmd and arg in cfg:
            if "--" in cmd[arg][0]:
                cmd[arg][1] = cmd[arg][1] + str(cfg[arg])
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                cmd[arg][1] = cmd[arg][1] + "=" + str(cfg[arg])
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
                if (
                    tuner_cfg["run_cmd"].get("generate_launch_cfg", True)
                    and not run_best
                ):
                    new_cmd_apth = (
                        os.path.splitext(cmd[arg][0])[0]
                        + "_"
                        + cfg["log_dir_name"]
                        + ".json"
                    )
                    json.dump(cmd_cfg, open(new_cmd_apth, "w"))

            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))
                if (
                    tuner_cfg["run_cmd"].get("generate_launch_cfg", True)
                    and not run_best
                ):
                    new_cmd_apth = (
                        os.path.splitext(cmd[arg][0])[0]
                        + cfg["log_dir_name"]
                        + ".yaml"
                    )
                    yaml.dump(cmd_cfg, open(new_cmd_apth, "w"))

        elif arg == "refined_recompute" and arg in cmd:
            if "--" in cmd["refined_recompute"][0]:
                raise NotImplementedError(
                    "refined recompute is not supported by command in autotuner."
                )
            elif "-o" in cmd["refined_recompute"][0]:
                raise NotImplementedError(
                    "refined recompute is not supported by '-o' in autotuner."
                )
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                if len(cmd[arg]) >= 3:
                    raise ValueError(
                        "The 3rd arg is not supported in refined_recompute"
                    )
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                rr_values = {}
                rr = tuner_cfg.get("refined_recompute", None)
                if not rr:
                    return
                for key in rr:
                    rr_values[key] = cfg[key]
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = rr_values
                else:
                    cmd_cfg[keys[-1]] = rr_values
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
                if (
                    tuner_cfg["run_cmd"].get("generate_launch_cfg", True)
                    and not run_best
                ):
                    new_cmd_apth = (
                        os.path.splitext(cmd[arg][0])[0]
                        + cfg["log_dir_name"]
                        + ".json"
                    )
                    json.dump(cmd_cfg, open(new_cmd_apth, "w"))

            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                if len(cmd[arg]) >= 3:
                    raise ValueError(
                        "The 3rd arg is not supported in refined_recompute"
                    )
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                rr_values = {}
                rr = tuner_cfg.get("refined_recompute", None)
                if not rr:
                    return
                for key in rr:
                    rr_values[key] = cfg[key]
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = rr_values
                else:
                    cmd_cfg[keys[-1]] = rr_values
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))
                if (
                    tuner_cfg["run_cmd"].get("generate_launch_cfg", True)
                    and not run_best
                ):
                    new_cmd_apth = (
                        os.path.splitext(cmd[arg][0])[0]
                        + cfg["log_dir_name"]
                        + ".yaml"
                    )
                    yaml.dump(cmd_cfg, open(new_cmd_apth, "w"))

    assert "run_cmd" in tuner_cfg
    cmd = copy.deepcopy(tuner_cfg["run_cmd"])
    res_args = copy.deepcopy(raw_args)

    new_args = [
        "dp_degree",
        "mp_degree",
        "pp_degree",
        "vpp_degree",
        "micro_batch_size",
        "sharding_degree",
        "sharding_stage",
        "use_recompute",
        "recompute_granularity",
        "local_batch_size",
        "gradient_accumulation_steps",
        "global_batch_size",
        "sequence_parallel",
        "refined_recompute",
    ]

    if "custom_search_dim" in tuner_cfg:
        for key in tuner_cfg["custom_search_dim"]:
            new_args.append(key)

    for arg in new_args:
        _get_new_cfg(arg, cmd, cfg, tuner_cfg)
        _gen_new_arg(arg, cmd, cfg, res_args, tuner_cfg)

    if tuner_cfg["run_cmd"].get("search_stage", None) and not run_best:
        cmd = copy.deepcopy(tuner_cfg["run_cmd"]["search_stage"])
        for arg in cmd:
            if "--" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

    if tuner_cfg["run_cmd"].get("run_best_stage", None) and run_best:
        cmd = copy.deepcopy(tuner_cfg["run_cmd"]["run_best_stage"])
        for arg in cmd:
            if "--" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

    # sharding overlap args
    if tuner_cfg["search_algo"]["name"] == "grid":
        gen_sharding_overlap_args_of_grid_search(res_args, cfg, tuner_cfg)
    else:
        gen_sharding_overlap_args(res_args, cfg, tuner_cfg)

    return res_args


def gen_new_ctx(ctx, cur_cfg, tuner_cfg):
    """Generate new running context."""
    new_ctx = copy.deepcopy(ctx)
    if (
        "search_algo" in tuner_cfg
        and "estimated_num_gpus" in tuner_cfg["search_algo"]
    ):
        assert cur_cfg["dp_degree"] == 1
        assert cur_cfg["sharding_stage"] == 1
        actual_cards = (
            cur_cfg["mp_degree"]
            * cur_cfg["pp_degree"]
            * cur_cfg["sharding_degree"]
        )
        if actual_cards <= tuner_cfg["gpus_per_node"]:
            new_ctx.args.devices = ",".join(
                [str(i) for i in range(actual_cards)]
            )
            if new_ctx.args.master:
                new_ctx.args.nnodes = "1:1"
        else:
            if actual_cards % tuner_cfg["gpus_per_node"] == 0:
                nnodes = actual_cards // tuner_cfg["gpus_per_node"]
            else:
                for i in range(2, tuner_cfg["nodes"] + 1):
                    if (
                        actual_cards % i == 0
                        and actual_cards // i <= tuner_cfg["gpus_per_node"]
                    ):
                        nnodes = i
                        break
            assert actual_cards % nnodes == 0
            new_ctx.args.devices = ",".join(
                [str(i) for i in range(actual_cards // nnodes)]
            )
            new_ctx.args.nnodes = f"{nnodes}:{nnodes}"
    return new_ctx


def read_metric_log(
    path, file="workerlog.0", target_metric='step/s'
) -> tuple[float, int]:
    """For extracting metric from log file."""
    """
    return:
        metric: average metric of last 10 steps
        err_code:
            00: no error
            01: no metric
            10: out of memory
    """
    err_code = 0
    target_file = path + "/" + file
    if not os.path.exists(target_file):
        return (0.0, 1)
    with open(target_file, "r") as f:
        # read file
        re_metric_pattern = (
            target_metric + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + target_metric
        )
        re_out_of_memory_pattern = (
            r"out of memory"
            if paddle.device.is_compiled_with_custom_device('npu')
            else r"Out of memory error on"
        )
        out_of_memory_flag = 0
        metric_list = []
        lines = f.readlines()
        for line in lines:
            metric = re.findall(re_metric_pattern, line)
            out_of_memory = re.findall(re_out_of_memory_pattern, line)
            if metric:
                value = None
                for item in metric[0]:
                    try:
                        value = float(item)
                        metric_list.append(value)
                        break
                    except:
                        continue
                assert value is not None

            if out_of_memory:
                out_of_memory_flag = 1

        if out_of_memory_flag:
            metric_ave = 0.0
            err_code = err_code | (out_of_memory_flag << 1)
        if not metric_list:
            metric_ave = 0.0
            err_code = err_code | 1
        elif len(metric_list) < 10:
            metric_ave = metric_list[-1]
        elif len(metric_list) < 20:
            metric_ave = sum(metric_list[9:]) / (len(metric_list[9:]))
        else:
            metric_ave = sum(metric_list[-10:]) / 10
        # round to 5 decimal places
        metric_ave = round(metric_ave, 5)
    res = metric_ave, err_code
    return res


def read_step_time_log(
    path, file="workerlog.0", target_metric='interval_runtime'
) -> tuple[float, int]:
    target_file = path + "/" + file
    if not os.path.exists(target_file):
        return None
    with open(target_file, "r") as f:
        # read file
        re_metric_pattern = (
            target_metric + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + target_metric
        )
        metric_list = []
        lines = f.readlines()
        for line in lines:
            metric = re.findall(re_metric_pattern, line)
            if metric:
                value = None
                for item in metric[0]:
                    try:
                        value = float(item)
                        metric_list.append(value)
                        break
                    except:
                        continue
                assert value is not None
        if not metric_list:
            metric_ave = None
            return None
        elif len(metric_list) < 10:
            metric_ave = metric_list[-1]
        elif len(metric_list) < 20:
            metric_ave = sum(metric_list[9:]) / (len(metric_list[9:]))
        else:
            metric_ave = sum(metric_list[-10:]) / 10
        # round to 5 decimal places
        metric_ave = round(metric_ave, 5)
    res = metric_ave
    return res


def read_allocated_memory_log(
    path, file="workerlog.0", target_metric='max_memory_allocated'
):
    target_file = path + "/" + file
    if not os.path.exists(target_file):
        return None
    with open(target_file, "r") as f:
        # read file
        re_metric_pattern = (
            target_metric + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + target_metric
        )
        metric_list = []
        lines = f.readlines()
        for line in lines:
            metric = re.findall(re_metric_pattern, line)
            if metric:
                value = None
                for item in metric[0]:
                    try:
                        value = int(float(item))
                        metric_list.append(value)
                        break
                    except:
                        continue
                assert value is not None
        if not metric_list:
            return None
        else:
            metric_list.sort()
            return metric_list[-1]


def read_memory_log(path, file) -> tuple[float, bool]:
    log_path = os.path.join(path, file)
    if not os.path.exists(log_path):
        return (0.0, True)
    memory_used = []
    utilization_gpu = []
    indices = []

    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        flag = False
        # skip headers
        while not flag:
            # show the first line of reader
            row = next(reader)
            if len(row) == 6 and 'memory_used' in row:
                flag = True
        for row in reader:
            # If row length is 6 then it's a utilization data row
            # skip header
            if len(row) == 6:
                index, util_gpu, _, mem_used, _, _ = row
                indices.append(int(index))
                memory_used.append(int(mem_used))
                utilization_gpu.append(int(util_gpu))
    return max(memory_used), False


def read_completed(path):
    """
    check if training is completed
    return:
        True: completed
        False: not completed
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("workerlog"):
                continue
            target_file = path + "/" + file
            if not os.path.exists(target_file):
                return False
            with open(target_file, "r") as f:
                # read file
                re_completed_pattern = r"Training completed."
                lines = f.readlines()
                for line in lines:
                    completed = re.findall(
                        re_completed_pattern, line, re.IGNORECASE
                    )
                    if completed:
                        return True
    return False


def read_log(
    path,
    metric_file="workerlog.0",
    target_metric='step/s',
    memory_file="0.gpu.log",
) -> tuple[float, float, int]:
    """
    extract metric and max memory usage from log file
    return:
        metric: average metric of last 10 steps
        memory: max memory used
        err_code: 00: no error, 01: no metric, 10: out of memory, 100: no memory log
    """
    err_code = 0
    # check out of memory
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("workerlog"):
                continue
            metric, metric_flag = read_metric_log(path, file, target_metric)
            if metric_flag:
                err_code = (metric_flag & 2) | err_code

    # read metric
    res_metric, metric_flag = read_metric_log(path, metric_file, target_metric)
    err_code = metric_flag | err_code
    # check max memory usage
    try:
        res_memory, memory_flag = read_memory_log(path, memory_file)
        err_code = (memory_flag << 2) | err_code
    except:
        res_memory = 0.0
        err_code = (1 << 2) | err_code
    return res_metric, res_memory, err_code


def get_error_info(filename):
    """
    get error info from log file
    return:
        error_info: Specific error message
    """
    error_infos = []
    error_pattern = r"Error"
    with open(filename, 'r') as file:
        lines = file.readlines()
        last_lines = lines[-100:]
        for line in last_lines:
            error_info = re.findall(error_pattern, line, re.IGNORECASE)
            if error_info:
                if "Out of memory" in line:
                    error_infos.append("Out of memory")
                else:
                    error_infos.append(line)
    return list(set(error_infos))


def find_error_from_log(path):
    """
    find error infos from log directory
    return:
        error_info: all error message on log directory
    """
    unique_error_info = ""
    all_error_infos = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("workerlog"):
                continue
            error_infos = get_error_info(path + "/" + file)
            all_error_infos += error_infos
    all_error_infos = list(set(all_error_infos))
    for info in all_error_infos:
        unique_error_info = unique_error_info + info + ","
    unique_error_info = unique_error_info[:-1]
    return unique_error_info


def three_mul_combinations(target):
    """Return the combinations of three numbers which product is target."""
    results = []
    for i in range(1, target // 3 + 1):
        if target % i == 0:
            for j in range(i, target // 2 + 1):
                if (target // i) % j == 0:
                    results.append((i, j, target // i // j))
    return results


def gbs_dp_mp_pp_candidates(tuner_cfg, num_gpus, num_nodes):
    """Return middle candidates of dp, mp, pp"""

    start = round(num_gpus ** (1 / 3))

    # find factors that can be evenly distributed
    for i in range(start, 0, -1):
        if num_gpus % i == 0:
            remaining = num_gpus // i
            # find the square root as a factor for the remaining part
            j = round(remaining**0.5)
            while remaining % j != 0:
                j -= 1
            return i, j, remaining // j

    raise ValueError("Cannot distribute GPUs equally")


def gbs_default_candidates(tuner_cfg):
    """Return the default candidates of every hyper param which user defined auto"""
    candidates = {}
    num_gpus = tuner_cfg["num_gpus"]
    num_nodes = tuner_cfg["nodes"]
    assert num_gpus > 0
    global_batch_size = tuner_cfg.get("model_cfg", {}).get(
        "global_batch_size", "auto"
    )
    if global_batch_size == "auto":
        dp_candidate, mp_candidate, pp_candidate = gbs_dp_mp_pp_candidates(
            tuner_cfg, num_gpus, num_nodes
        )
        sharding_degree_candidate = dp_candidate
        candidates["dp_degree"] = [1]
        candidates["mp_degree"] = [mp_candidate]
        candidates["pp_degree"] = [pp_candidate]
        candidates["sharding_degree"] = [sharding_degree_candidate]
        candidates["sharding_stage"] = [1]
        candidates["use_recompute"] = [False]
        candidates["recompute_granularity"] = [None]
        candidates["micro_batch_size"] = [2**i for i in range(0, 10)]
        candidates["global_batch_size"] = [
            pp_candidate * dp_candidate * e
            for e in candidates["micro_batch_size"]
        ]
    return candidates


def gbs_search_all(tuner_cfg):
    """Permutate the candidates of all hyper params."""
    candidates = tuner_cfg["candidates"]
    # Order: dp -> mp -> pp -> mbs -> sharding-> recompute
    dp_degree_candidates = candidates["dp_degree"]
    mp_degree_candidates = candidates["mp_degree"]
    pp_degree_candidates = candidates["pp_degree"]
    mbs_candidates = candidates["micro_batch_size"]
    sharding_stage_candidates = candidates["sharding_stage"]
    sharding_degree_candidates = candidates["sharding_degree"]
    use_recompute_candidates = candidates["use_recompute"]
    recompute_granularity_candidates = candidates["recompute_granularity"]
    # gbs_candidates = candidates["global_batch_size"]
    all_cfgs = list(
        itertools.product(
            dp_degree_candidates,
            mp_degree_candidates,
            pp_degree_candidates,
            mbs_candidates,
            sharding_degree_candidates,
            sharding_stage_candidates,
            use_recompute_candidates,
            recompute_granularity_candidates,
            # gbs_candidates,
        )
    )
    mapping = {
        0: "dp_degree",
        1: "mp_degree",
        2: "pp_degree",
        3: "micro_batch_size",
        5: "sharding_stage",
        4: "sharding_degree",
        6: "use_recompute",
        7: "recompute_granularity",
        # 8: "global_batch_size",
    }
    new_all_cfgs = []
    for cfg in all_cfgs:
        new_cfg = {}
        for idx, val in enumerate(cfg):
            new_cfg[mapping[idx]] = val
        new_cfg["global_batch_size"] = (
            new_cfg["pp_degree"]
            * new_cfg["dp_degree"]
            * new_cfg["micro_batch_size"]
        )
        new_all_cfgs.append(new_cfg)
    return new_all_cfgs


def load_configs_from_csv(configs_csv):
    """Load the configs from csv file."""
    all_configs = []
    extract_keys_integer = [
        "dp_degree",
        "mp_degree",
        "pp_degree",
        "vpp_degree",
        "micro_batch_size",
        "sharding_degree",
        "sharding_stage",
    ]
    extract_keys_string = ["use_recompute", "recompute_granularity"]
    with open(configs_csv, "r") as f:
        reader = csv.DictReader(f)
        raw_configs = list(reader)
    for raw_config in raw_configs:
        config = {}
        for extract_key in extract_keys_integer:
            val = raw_config.get(extract_key, "")
            try:
                config[extract_key] = int(val)
            except ValueError:
                raise ValueError(
                    f"{extract_key} must be integer, but got {val}"
                )

        use_recompute = raw_config.get("use_recompute", "")
        assert use_recompute.lower() in [
            "true",
            "false",
        ], f"{use_recompute} must be true or false, but got {use_recompute}"
        config["use_recompute"] = use_recompute.lower() == "true"

        recompute_granularity = raw_config.get("recompute_granularity", "")
        assert (
            recompute_granularity == ""
            or recompute_granularity.lower()
            in __SUPPORTED_RECOMPUTE_GRANULARITY__
        ), f"{recompute_granularity} must be one of {__SUPPORTED_RECOMPUTE_GRANULARITY__}, but got {recompute_granularity}."
        config["recompute_granularity"] = (
            recompute_granularity if recompute_granularity != "" else None
        )

        all_configs.append(config)

    return all_configs
