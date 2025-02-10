# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import logging
import os
import re
import time
from enum import IntEnum, unique

import paddle
from paddle.distributed.launch.context.node import Node
from paddle.distributed.launch.utils.kv_client import KVClient
from paddle.distributed.launch.utils.kv_server import KVServer
from paddle.distributed.launch.utils.topology import SingleNodeTopology

from ...utils.log_utils import get_logger


@unique
class DeviceType(IntEnum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2
    XPU = 3
    DCU = 5
    NIC = 6


@unique
class LinkType(IntEnum):
    UNKNOWN = 0
    LOC = 1
    SYS = 2
    PHB = 3
    PIX = 4
    PIB = 5
    NVL = 6
    NVB = 7
    NET = 8


class Mesh:
    def __init__(self, id, name):
        self._id = id
        self._name = name
        self._type = None  # GPU/XPU
        self._full_type = None  # GPU-V100-XSB-40G/GPU-A100-XSB-80G
        self._machines = {}
        self._links = {}

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def full_type(self):
        return self._full_type

    @full_type.setter
    def full_type(self, value):
        self._full_type = value

    @property
    def machines(self):
        return self._machines

    @property
    def links(self):
        return self._links

    @machines.setter
    def machines(self, value):
        self._machines = value

    def add_machine(self, machine):
        self._machines[machine.id] = machine

    def get_machine(self, id):
        return self._machines.get(id, None)

    def get_num_machines(self):
        return len(self._machines)

    def add_link(self, link):
        self._links[(link.source, link.target)] = link

    def get_link(self, source, target):
        return self._links.get((source, target), None)

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "machines": [x.to_json() for x in self.machines.values()],
            "links": [x.to_json() for x in self.links.values()],
        }


class MeshGroup:
    def __init__(self):
        self._meshes = {}
        self._links = {}
        self._global_device_num = 0

    @property
    def meshes(self):
        return self._meshes

    @property
    def links(self):
        return self._links

    def add_mesh(self, mesh):
        self._meshes[mesh.id] = mesh

    def get_mesh(self, id):
        return self._meshes.get(id, None)

    def add_link(self, link):
        self._links[(link.source, link.target)] = link

    def get_link(self, source, target):
        return self._links.get((source, target), None)

    def generate_global_device_id(self):
        curr_device_id = self._global_device_num
        self._global_device_num += 1
        return curr_device_id

    def to_json(self):
        return {
            "meshes": [x.to_json() for x in self.meshes.values()],
            "links": [x.to_json() for x in self.links.values()],
        }


class Device:
    NON_ACCELERATOR_TYPE = [DeviceType.CPU, DeviceType.NIC, DeviceType.UNKNOWN]

    def __init__(self, global_id, local_id, machine, mesh=None):
        self._global_id = global_id
        self._local_id = local_id
        self._machine = machine
        self._mesh = mesh
        self._type = None
        # Different device have different models, such as
        # "Tesla V100-SXM2-32GB" and "A100-SXM4-40GB" etc.
        self._model = None
        # Double precision GFLOPS
        self._dp_gflops = None
        # Single precision GFLOPS
        self._sp_gflops = None
        # Half precision GFLOPS
        self._hp_gflops = None
        # Memory is stored by GB
        self._memory = None
        self._links = {}

    @property
    def global_id(self):
        return self._global_id

    @global_id.setter
    def global_id(self, value):
        self._global_id = value

    @property
    def local_id(self):
        return self._local_id

    @local_id.setter
    def local_id(self, value):
        self._local_id = value

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, value):
        self._machine = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def dp_gflops(self):
        return self._dp_gflops

    @dp_gflops.setter
    def dp_gflops(self, value):
        self._dp_gflops = value

    @property
    def sp_gflops(self):
        return self._sp_gflops

    @sp_gflops.setter
    def sp_gflops(self, value):
        self._sp_gflops = value

    @property
    def hp_gflops(self):
        return self._hp_gflops

    @hp_gflops.setter
    def hp_gflops(self, value):
        self._hp_gflops = value

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    def add_link(self, link):
        self._links[(link.source, link.target)] = link

    def to_json(self):
        return {
            "global_id": self.global_id,
            "local_id": self.local_id,
            "type": self.type,
            "model": self.model,
            "sp_gflops": self.sp_gflops,
            "dp_gflops": self.dp_gflops,
            "memory": self.memory,
        }

    def __str__(self):
        str = ""
        str += f"global_id: {self.global_id}, local_id: {self.local_id}, machine_id: {self.machine.id}, type: {self.type.name}, model: {self.model}, dp_flops: {self.dp_gflops}, sp_flops: {self.sp_gflops}, hp_flops: {self.hp_gflops}, memory: {self.memory}"
        return str

    def __repr__(self):
        return self.__str__()


class Link:
    default_hop = 1
    default_nic_bandwidth = 24

    def __init__(self, source, target, topo=False):
        self._src = source
        self._tgt = target
        self._type = None
        # bandwidth is stored by GB/s
        self._bandwidth = None
        # latency is stored by millisecond
        self._latency = None
        # linked between mesh, machine, device
        self._link_level = None
        self._hop = None
        self._topo = topo

    @property
    def source(self):
        return self._src

    @source.setter
    def source(self, value):
        self._source = value

    @property
    def target(self):
        return self._tgt

    @target.setter
    def target(self, value):
        self._target = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value

    @property
    def hop(self):
        return self._hop

    @hop.setter
    def hop(self, value):
        self._hop = value

    @property
    def link_level(self):
        return self._link_level

    @link_level.setter
    def link_level(self, value):
        self._link_level = value

    def to_json(self):
        return {
            "source_id": self.source,
            "target_id": self.target,
            "type": self.type,
            "bandwidth": self.bandwidth,
            "latency": self.latency,
        }

    def __str__(self):
        str = ""
        source_id = self.source if self._topo else self.source.global_id
        target_id = self.target if self._topo else self.target.global_id
        str += f"source_global_id: {source_id}, target_global_id: {target_id}, type: {self.type}, bandwidth: {self.bandwidth}, latency: {self.latency}"
        return str

    def __repr__(self):
        return self.__str__()


class Machine:
    def __init__(self, id, mesh=None, topo=False):
        self._id = id
        self._hostname = None
        self._addr = None
        # Double precision GFLOPS
        self._dp_gflops = None
        # Single precision GFLOPS
        self._sp_gflops = None
        self._memory = None
        self._bandwidth = None
        self._latency = None
        self._port = None
        self._devices = {}
        self._links = {}
        self._accelerators = {}
        self._non_accelerator_cumulative_count = 0

        self._topo_links = {}
        self._mesh = mesh
        self._topo = topo

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def hostname(self):
        return self._hostname

    @hostname.setter
    def hostname(self, value):
        self._hostname = value

    @property
    def addr(self):
        return self._addr

    @addr.setter
    def addr(self, value):
        self._addr = value

    @property
    def sp_gflops(self):
        return self._sp_gflops

    @sp_gflops.setter
    def sp_gflops(self, value):
        self._sp_gflops = value

    @property
    def dp_gflops(self):
        return self._dp_gflops

    @dp_gflops.setter
    def dp_gflops(self, value):
        self._dp_gflops = value

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def devices(self):
        return self._devices

    @property
    def links(self):
        if self._topo:
            return self._topo_links
        return self._links

    @property
    def accelerators(self):
        return self._accelerators

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def add_device(self, device):
        # Use the device global_id as the key
        self._devices[device.global_id] = device
        if device.type not in Device.NON_ACCELERATOR_TYPE:
            self._accelerators[device.global_id] = device

    def get_device(self, id):
        return self._devices.get(id, None)

    def add_link(self, link):
        # Use the source device global_id and target device global_id as the key
        if self._topo:
            self._topo_links[(link.source, link.target)] = link
        else:
            self._links[(link.source.global_id, link.target.global_id)] = link

    def get_link(self, source_global_id, target_global_id):
        if self._topo:
            return self._topo_links.get(
                (source_global_id, target_global_id), None
            )
        return self._links.get((source_global_id, target_global_id), None)

    def to_json(self):
        return {
            "id": self.id,
            "hostname": self.hostname,
            "addr": self.addr,
            "dp_gflops": self.dp_gflops,
            "sp_gflops": self.sp_gflops,
            "memory": self.memory,
            "bandwidth": self.bandwidth,
            "latency": self.latency,
            "devices": [x.to_json() for x in self.devices.values()],
            "links": [x.to_json() for x in self.links.values()],
        }

    def __str__(self):
        str = ""
        for device in self.devices.values():
            str += f", device: {device}"
        for link in self.links.values():
            str += f", link: {link}"
        return str

    def __repr__(self):
        return self.__str__()


class AlphaLatency:
    def __init__(self, alpha_latency):
        assert isinstance(alpha_latency, dict)
        self._base = alpha_latency.get("base", None)
        self._inter = alpha_latency.get("inter", None)
        self._intra = alpha_latency.get("intra", None)
        self._switch = alpha_latency.get("switch", None)
        if self._switch is not None:
            try:
                self._switch = float(self._switch)
            except:
                raise TypeError("The switch latency must be float")
        self._base_ring = (
            self._base.get("ring", None) if self._base is not None else None
        )
        self._base_tree = (
            self._base.get("tree", None) if self._base is not None else None
        )
        self._base_inter = (
            self._base.get("inter", None) if self._base is not None else None
        )
        if self._base_ring is not None:
            try:
                self._base_ring = float(self._base_ring)
            except:
                raise TypeError("The base ring latency must be float.")
        if self._base_tree is not None:
            try:
                self._base_tree = float(self._base_tree)
            except:
                raise TypeError("The base ring latency must be float.")

        self._inter_ring = self._inter.get("ring", None)
        self._inter_tree = self._inter.get("tree", None)
        self._intra_ring = self._intra.get("ring", None)
        self._intra_tree = self._intra.get("tree", None)

        if self._inter_ring is not None:
            if isinstance(self._inter_ring, str):
                assert self._inter_ring in ["NET"]
                self._inter_ring = LinkType[self._inter_ring]
            else:
                try:
                    self._inter_ring = float(self._inter_ring)
                except:
                    raise TypeError("The inter ring latency must be float.")

        if self._inter_tree is not None:
            if isinstance(self._inter_tree, str):
                assert self._inter_tree in ["NET"]
                self._inter_tree = LinkType[self._inter_tree]
            else:
                try:
                    self._inter_tree = float(self._inter_tree)
                except:
                    raise TypeError("The inter tree latency must be float.")

        if self._intra_ring is not None:
            if isinstance(self._intra_ring, str):
                assert self._intra_ring in ["NVL", "PHB"]
                self._intra_ring = LinkType[self._intra_ring]
            else:
                try:
                    self._intra_ring = float(self._intra_ring)
                except:
                    raise TypeError("The intra ring latency must be float.")

        if self._intra_tree is not None:
            if isinstance(self._intra_tree, str):
                assert self._intra_tree in ["NVL", "PHB"]
                self._intra_tree = LinkType[self._intra_tree]
            else:
                try:
                    self._intra_tree = float(self._intra_tree)
                except:
                    raise TypeError("The intra tree latency must be float.")

    @property
    def base_ring(self):
        return self._base_ring

    @property
    def base_tree(self):
        return self._base_tree

    @property
    def switch(self):
        return self._switch

    @property
    def inter_ring(self):
        return self._inter_ring

    @property
    def inter_tree(self):
        return self._inter_tree

    @property
    def intra_ring(self):
        return self._intra_ring

    @property
    def intra_tree(self):
        return self._intra_tree


class Cluster:
    """
    The cluster is an abstract of the hardware resource for training, which contains the cluster topology and
    related hardware information. It will serve the task mapping, cost model and auto searching.
    """

    def __init__(self):
        self._num_meshes = 0
        # Used to compute machine id
        self._num_machines = 0
        # Store all machines within the cluster
        self._machines = {}
        # Cluster graph topology
        self._topology = None
        # Latency for communication cost model
        self._alpha_latency = None
        self._rank_to_device_id = {}
        self._device_id_to_rank = {}
        # This property only be valid when the cluster consists of machines,
        # which have the same number accelerators.
        self._num_devices_per_machine = None
        self._gpu_model = None
        self._initialized = False
        self._mesh_group = None
        self._topo = False
        self._hetero = False

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, value):
        self._initialized = value

    def gen_default_config_cluster(
        self,
        gpu_model="V100",
        cpu_model="6271C",
        node_count=1,
        device_count=1,
        gpu_memory=32,
        cpu_memory=503,
        inter_bandwidth=24,
        intra_bandwidth=235,
        gpu_dp_gflops=7800,
        gpu_sp_gflops=15700,
        gpu_hp_gflops=31400,
        cpu_dp_gflops=75,
        cpu_sp_gflops=150,
    ):
        """Generate cluster by default config."""
        gpu_models = ["V100", "A100", "H100", "A2", "A10", "A16", "A30", "A40"]
        xpu_models = ["XPU"]
        dcu_models = ["DCU"]
        all_gpu_models = gpu_models + xpu_models + dcu_models
        self._num_devices_per_machine = device_count
        self._gpu_model = gpu_model

        def _convert_to_type(gpu_model):
            type = None
            if gpu_model in gpu_models:
                type = "GPU"
            elif gpu_model in xpu_models:
                type = "XPU"
            elif gpu_model in dcu_models:
                type = "DCU"
            else:
                type = "GPU"
            assert type is not None

            return type

        def _convert_to_model(gpu_model, gpu_memory):
            model = None
            if gpu_model == "V100":
                model = "Tesla V100-SXM2-" + str(gpu_memory) + "GB"
            elif gpu_model == "A100":
                model = "Tesla A100-SXM-" + str(gpu_memory) + "GB"
            elif gpu_model == "A30":
                model = "Tesla A30-SXM-" + str(gpu_memory) + "GB"
            else:
                model = gpu_model + str(gpu_memory) + "GB"
            assert model is not None

            return model

        def _convert_to_cpu_info(cpu_model):
            arch, vendor, model = None, None, None
            if cpu_model == "6271C":
                arch = "x86_64"
                vendor = "GenuineIntel"
                model = "Intel(R) Xeon(R) Gold 6271C CPU @ 2.60G"
            elif cpu_model == "6148":
                arch = "x86_64"
                vendor = "GenuineIntel"
                model = "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40G"
            assert arch is not None
            assert vendor is not None
            assert model is not None

            return arch, vendor, model

        cluster_info = {}
        cluster_info["machines"] = []
        global_id = 0
        global_id_to_device_type = {}
        global_id_to_node = {}
        # NOTE: It will support NPU, XPU, DCU models in the future, it is just a fake value now
        for i in range(node_count):
            machine = {}
            # NOTE: The hostname is host_0, host_1, ...
            machine["hostname"] = "host_" + str(i)
            # NOTE: The addr is localhost, if need actual addr, it should be reset manually
            machine["addr"] = "127.0.0.1"
            # NOTE: The port is a default value
            machine["port"] = 60009
            machine["links"] = []

            devices = []
            local_id = 0

            for j in range(device_count):
                device = {}
                global_id = global_id if i == 0 and j == 0 else global_id + 1

                local_id += 1
                type = _convert_to_type(gpu_model)
                model = _convert_to_model(gpu_model, gpu_memory)
                memory = gpu_memory

                device["global_id"] = global_id
                device["local_id"] = local_id
                device["type"] = type
                device["model"] = model
                device["memory"] = memory
                device["sp_gflops"] = gpu_sp_gflops
                device["dp_gflops"] = gpu_dp_gflops
                device["hp_gflops"] = gpu_hp_gflops
                # hard code
                device["type"] = "GPU"
                global_id_to_device_type[global_id] = type
                global_id_to_node[global_id] = i
                devices.append(device)

            # add cpu device and nic device, just one cpu
            cpu_device = {}
            arch, vendor, model = _convert_to_cpu_info(cpu_model)
            sp_gflops = cpu_sp_gflops
            dp_gflops = cpu_dp_gflops
            global_id += 1
            local_id = 0
            memory = cpu_memory
            type = "CPU"
            cpu_device["arch"] = arch
            cpu_device["vendor"] = vendor
            cpu_device["model"] = model
            cpu_device["sp_gflops"] = sp_gflops
            cpu_device["dp_gflops"] = dp_gflops
            cpu_device["global_id"] = global_id
            cpu_device["local_id"] = local_id
            cpu_device["memory"] = memory
            cpu_device["type"] = type
            global_id_to_node[global_id] = i
            global_id_to_device_type[global_id] = type
            devices.append(cpu_device)

            nic_device = {}
            global_id += 1

            # add NIC
            type = "NIC"
            width = 12.5
            ip = "127.0.0.1"
            local_id = 0
            nic_device["type"] = type
            nic_device["local_id"] = type
            nic_device["global_id"] = global_id
            global_id_to_device_type[global_id] = type
            global_id_to_node[global_id] = i
            devices.append(nic_device)
            machine["devices"] = devices
            cluster_info["machines"].append(machine)

        # build link
        for i in range(0, global_id + 1):
            for j in range(0, global_id + 1):
                if i == j:
                    continue
                node_id_i = global_id_to_node[i]
                node_id_j = global_id_to_node[j]
                device_type_i = global_id_to_device_type[i]
                device_type_j = global_id_to_device_type[j]
                link = {}
                source_global_id = i
                target_global_id = j
                link["source_global_id"] = source_global_id
                link["target_global_id"] = target_global_id
                # the same node and device_type, set intra_bandwidth, NVL
                if node_id_i == node_id_j and device_type_i == device_type_j:
                    link["type"] = "NVL"
                    link["bandwidth"] = intra_bandwidth
                else:
                    link["type"] = "PHB"
                    link["bandwidth"] = inter_bandwidth
                cluster_info["machines"][node_id_i]["links"].append(link)

        self._build_from_dict(cluster_info)

    @property
    def rank_to_device_id(self):
        return self._rank_to_device_id

    @property
    def device_id_to_rank(self):
        return self._device_id_to_rank

    @property
    def mesh_group(self):
        return self._mesh_group

    @mesh_group.setter
    def mesh_group(self, value):
        self._mesh_group = value

    @property
    def machines(self):
        return self._machines

    def add_machine(self, machine):
        assert isinstance(machine, Machine)
        self._machines[machine.id] = machine

        # map rank to device id and map device id to rank
        if machine.id != 0:
            prev_machine = self._machines[machine.id - 1]
            offset = prev_machine._non_accelerator_cumulative_count
            for global_id in machine.devices:
                if (
                    machine.devices[global_id].type
                    not in Device.NON_ACCELERATOR_TYPE
                ):
                    rank_id = global_id - offset
                    self._rank_to_device_id[rank_id] = global_id
                    self._device_id_to_rank[global_id] = rank_id
            machine._non_accelerator_cumulative_count = (
                len(machine.devices)
                - len(machine.accelerators)
                + prev_machine._non_accelerator_cumulative_count
            )
        else:
            for global_id in machine.devices:
                if (
                    machine.devices[global_id].type
                    not in Device.NON_ACCELERATOR_TYPE
                ):
                    rank_id = global_id
                    self._rank_to_device_id[rank_id] = global_id
                    self._device_id_to_rank[global_id] = rank_id
                    machine.accelerators[global_id] = machine.devices[global_id]
            machine._non_accelerator_cumulative_count = len(
                machine.devices
            ) - len(machine.accelerators)

    @property
    def alpha_latency(self):
        return self._alpha_latency

    def add_device(self, device):
        assert isinstance(device, Device)
        device.machine.add_device(device)

    def add_link(self, link):
        assert isinstance(link, Link)
        # Only add the link to the source machine
        link.source.machine.add_link(link)

    def get_device(self, device_global_id):
        device = None
        if self._topo:
            target_machines = []
            for mesh in self.mesh_group.meshes.values():
                target_machines.extend(mesh.machines.values())
        else:
            target_machines = self.machines.values()

        for machine in target_machines:
            if device_global_id in machine.devices.keys():
                device = machine.devices[device_global_id]
        return device

    def _build_from_dict(self, cluster_info):
        machines_info = cluster_info["machines"]
        for machine_info in machines_info:
            machine_id = self._generate_machine_id()
            machine = Machine(machine_id)
            machine.hostname = machine_info.get("hostname")
            machine.addr = machine_info.get("addr")
            machine.port = machine_info.get("port")
            devices_info = machine_info.get("devices", [])
            for device_info in devices_info:
                device_global_id = device_info.get("global_id")
                device_local_id = device_info.get("local_id")
                device = Device(device_global_id, device_local_id, machine)
                device_type = device_info.get("type", None)
                if device_type is not None:
                    device_type = DeviceType[device_type]
                else:
                    device_type = DeviceType.UNKNOWN
                device.type = device_type
                device.model = device_info.get("model", None)
                device.dp_gflops = float(device_info.get("dp_gflops", 0))
                device.sp_gflops = float(device_info.get("sp_gflops", 0))
                device.hp_gflops = float(device_info.get("hp_gflops", 0))
                device.memory = float(device_info.get("memory", 0))
                self.add_device(device)
            self.add_machine(machine)
        for machine_info in machines_info:
            links_info = machine_info.get("links", [])
            for link_info in links_info:
                source_global_id = link_info.get("source_global_id")
                target_global_id = link_info.get("target_global_id")
                source = self.get_device(source_global_id)
                target = self.get_device(target_global_id)
                link = Link(source, target)
                link_type = link_info.get("type", None)
                if link_type is not None:
                    link_type = LinkType[link_type]
                else:
                    link_type = LinkType.UNKNOWN
                link.type = link_type
                link.bandwidth = float(link_info.get("bandwidth", 0))
                link.latency = float(link_info.get("latency", 0))
                link.hop = link_info.get("hop", None)
                if link.hop is None:
                    # Set the default of hop: If in the same machine, hop is 0. And if in the different machine, hop is 1.
                    source_machine = source.machine
                    target_machine = target.machine
                    if source_machine.id == target_machine.id:
                        link.hop = 0
                    else:
                        link.hop = Link.default_hop
                self.add_link(link)

        if "alpha_latency" in cluster_info:
            self._alpha_latency = AlphaLatency(
                cluster_info.get("alpha_latency")
            )
        else:
            self._alpha_latency = None

    def _build_from_topo(self, topo_info, local_size):
        self.mesh_group = MeshGroup()
        for mesh_key, mesh_val in topo_info.items():
            # parse mesh
            mesh_id = self._generate_mesh_id()
            mesh = Mesh(mesh_id, mesh_key)
            mesh_fields = mesh_key.split("-")
            mesh.type = mesh_fields[0]
            mesh.full_type = "-".join(mesh_fields[1:])

            # parse machine
            machine_ids = list(range(len(mesh_val)))

            for machine_id in range(len(mesh_val)):
                machine_val = mesh_val[machine_id]
                machine = Machine(id=machine_id, mesh=mesh, topo=True)
                machine.hostname = machine_val.get("hostname")
                machine.addr = machine_val.get("addr")
                machine.sp_gflops = int(machine_val.get("sp_gflops"))
                machine.dp_gflops = int(machine_val.get("dp_gflops"))
                machine.memory = int(machine_val.get("memory"))
                machine.bandwidth = int(machine_val.get("bandwidth"))
                machine.latency = int(machine_val.get("latency"))

                # parse device
                self._num_devices_per_machine = len(machine_val.get("devices"))
                for device_val in machine_val.get("devices"):
                    device = Device(
                        device_val.get("global_id"),
                        device_val.get("local_id"),
                        machine,
                        mesh,
                    )
                    device.type = device_val.get("type")
                    device.model = device_val.get("model")
                    device.sp_gflops = int(device_val.get("sp_gflops"))
                    device.dp_gflops = int(device_val.get("dp_gflops"))
                    device.memory = int(device_val.get("memory"))
                    machine.add_device(device)

                for link_val in machine_val.get("links"):
                    source_device_id = link_val.get("source_global_id")
                    target_device_id = link_val.get("target_global_id")
                    device_link = Link(
                        source=source_device_id,
                        target=target_device_id,
                        topo=True,
                    )
                    device_link.type = link_val.get("type")
                    device_link.bandwidth = int(link_val.get("bandwidth"))
                    device_link.latency = int(link_val.get("latency"))
                    device_link.link_level = "device"
                    device_link.hop = link_val.get("hop", None)
                    if device_link.hop is None:
                        # Set the default of hop: If in the same machine, hop is 0. And if in the different machine, hop is 1.
                        if source_device_id == target_device_id:
                            device_link.hop = 0
                        else:
                            device_link.hop = Link.default_hop
                        machine.add_link(device_link)
                mesh.add_machine(machine)
            for i in mesh.machines:
                for j in mesh.machines:
                    if i == j:
                        continue
                    machine_link = Link(i, j, topo=True)
                    machine_link.type = "NET"
                    machine_link.bandwidth = 12
                    machine_link.latency = 0.5
                    machine_link.link_level = "machine"
                    mesh.add_link(machine_link)
            self.mesh_group.add_mesh(mesh)
        for i in self.mesh_group.meshes:
            for j in self.mesh_group.meshes:
                if i == j:
                    continue
                mesh_link = Link(i, j, topo=True)
                mesh_link.type = "NET"
                mesh_link.bandwidth = 12
                mesh_link.latency = 0.5
                mesh_link.link_level = "mesh"
                self.mesh_group.add_link(mesh_link)
        self._topo = True

    def build_from_file(self, json_file_path):
        with open(json_file_path) as json_file:
            cluster_info = json.load(json_file)
        self._build_from_dict(cluster_info)

    def _generate_mesh_id(self):
        cur_mesh_id = self._num_meshes
        self._num_meshes += 1
        return cur_mesh_id

    def _generate_machine_id(self):
        cur_machine_id = self._num_machines
        self._num_machines += 1
        return cur_machine_id

    def get_all_devices(self, device_type):
        devices = []
        if self._topo:
            target_machines = []
            for mesh in self.mesh_group.meshes():
                target_machines.extend(mesh.machines.values())
        else:
            target_machines = self.machines.values()

        for machine in target_machines:
            for device in machine.devices.values():
                if device.type == DeviceType[device_type]:
                    devices.append(device)
        return devices

    def get_beta_topo(self, source_device_id, target_device_id):
        beta = None
        convert_base = 1000
        src_device = self.get_device(source_device_id)
        tgt_device = self.get_device(target_device_id)

        src_machine = src_device.machine
        tgt_machine = tgt_device.machine

        src_mesh = src_machine.mesh
        tgt_mesh = tgt_machine.mesh

        if src_mesh.id != tgt_mesh.id:
            link = self.mesh_group.get_link(src_mesh.id, tgt_mesh.id)
        elif src_machine.id != tgt_machine.id:
            mesh = self.mesh_group.get_mesh(src_mesh.id)
            link = mesh.get_link(src_machine.id, tgt_machine.id)
        else:
            mesh = self.mesh_group.get_mesh(src_mesh.id)
            machine = mesh.get_machine(src_machine.id)
            link = machine.get_link(source_device_id, target_device_id)
        return link

    def get_beta(self, source_device_id, target_device_id):
        if self._topo:
            link = self.get_beta_topo(source_device_id, target_device_id)
        else:
            device = self.get_device(source_device_id)
            machine = device.machine
            link = machine.get_link(source_device_id, target_device_id)
        # beta means the time transferring a byte, us/B
        beta = None
        convert_base = 1000
        bandwidth = None
        # None means the source and target are not connected directly, set NIC in default
        if link is None:
            bandwidth = Link.default_nic_bandwidth
        else:
            bandwidth = link.bandwidth

        if bandwidth == 0.0:
            beta = 0
        else:
            beta = 1 / (bandwidth * (convert_base**3 / 10**6))

        return beta

    def get_hop(self, source_device_id, target_device_id):
        beta = None
        hop = None
        device = self.get_device(source_device_id)
        machine = device.machine
        link = machine.get_link(source_device_id, target_device_id)
        if link is not None:
            hop = link.hop
        else:
            hop = Link.default_hop
        return hop

    def cross_machine(self, device_ids):
        machine_ids = set()
        mesh_ids = set()

        for device_id in device_ids:
            device = self.get_device(device_id)
            machine_id = device.machine.id
            machine_ids.add(machine_id)
            if self._topo:
                mesh_id = device.machine.mesh.id

                mesh_ids.add(mesh_id)
        if self._topo:
            if len(mesh_ids) == 1 and len(machine_ids) == 1:
                return False
            return True
        elif len(machine_ids) == 1:
            return False
        else:
            return True

    def convert_rank_to_device_id(self, group_ranks):
        # group_ranks is global id of the rank in paddle
        # task will use all of machine in this cluster with accelerators in default
        if self._topo:
            return group_ranks

        device_ids = []
        for rank in group_ranks:
            device_ids.append(self.rank_to_device_id[rank])
        return device_ids

    def get_involved_machine_count(self, device_ids):
        machine_ids = set()
        for device_id in device_ids:
            device = self.get_device(device_id)
            machine_id = device.machine.id
            machine_ids.add(machine_id)
        count = len(machine_ids)
        assert count > 0
        return count

    def get_num_machines(self):
        if self._topo:
            n = 0
            for mesh in self.mesh_group.meshes.values():
                n += mesh.get_num_machines()
            return n
        else:
            return len(self._machines)

    def get_num_devices_per_machine(self):
        # Only return the number of accelerators of each machine.
        # All machines must has the same number of devices and same type of devices.
        assert self._num_devices_per_machine
        return self._num_devices_per_machine

    def __str__(self):
        str = ""
        for machine in self.machines.values():
            str += f"machine: {machine}\n"
        return str

    def __repr__(self):
        return self.__str__()


logger = get_logger(logging.INFO)


def get_default_cluster(json_config=None, auto_config=None):
    def is_by_json_config(json_config):
        if not json_config:
            return False
        if "cluster" not in json_config:
            return False
        else:
            if "path" not in json_config["cluster"]:
                if "num_nodes" not in json_config["cluster"]:
                    return False
                if "num_gpus" not in json_config["cluster"]:
                    return False
                if "gpu_model" not in json_config["cluster"]:
                    return False
                if "gpu_memory" not in json_config["cluster"]:
                    return False
                return True
            else:
                return True

    cluster = Cluster()
    if json_config and is_by_json_config(json_config):
        # Get GPU info by json config
        if "path" in json_config["cluster"]:
            cluster.build_from_file(json_config["cluster"]["path"])
            return cluster
        else:
            node_count = json_config["cluster"]["num_nodes"]
            local_device_count = json_config["cluster"]["num_gpus"]
            gpu_model = json_config["cluster"]["gpu_model"]
            memory = json_config["cluster"]["gpu_memory"]
    elif auto_config:
        master_endpoint = os.getenv("PADDLE_MASTER")
        local_topo = SingleNodeTopology()
        local_topo.detect()

        nnodes = int(os.getenv("PADDLE_NNODES"))
        curr_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")

        global_rank = int(os.getenv("PADDLE_GLOBAL_RANK"))
        local_rank = int(os.getenv("PADDLE_LOCAL_RANK"))
        local_size = int(os.getenv("PADDLE_LOCAL_SIZE"))
        node_id = int((global_rank - local_rank) / local_size)

        if nnodes > 0 and master_endpoint is not None:
            node = Node()
            master_ip, _ = master_endpoint.split(":")
            # TODO how to generate the same free port in all process
            free_port = 12346
            server_endpoint = f"{master_ip}:{free_port}"

            if local_rank == 0 and master_ip in curr_endpoint:
                server = KVServer(free_port)
                server.start()
                logger.info(f"server start at: {server_endpoint}")

            client = KVClient(server_endpoint)
            device_type = local_topo.machine["device_type_full"]
            # only local rank 0 need put topo data
            if local_rank == 0:
                resp = False
                while not resp:
                    resp = client.put(
                        key=f"/topo/data/{device_type}/{node_id}",
                        value=local_topo.json_object,
                    )
            # all rank need get topo data
            retry = True
            while retry:
                global_topo = client.get_prefix(key="/topo/data")
                if global_topo and len(global_topo) == nnodes:
                    topo_dict = {}
                    for key, value in global_topo.items():
                        _, _, _, mesh_type, idx = key.split("/")
                        if mesh_type not in topo_dict:
                            topo_dict[mesh_type] = []
                        mesh_idx = len(topo_dict[mesh_type])
                        global_topo_value = json.loads(value)
                        topo_dict[mesh_type].append(global_topo_value)
                    cluster._build_from_topo(topo_dict, local_size)
                    retry = False
                else:
                    global_size = len(global_topo) if global_topo else 0
                    logger.info(
                        f"get global_topo failed, actual size: {global_size}, expected size: {nnodes}, retry later!"
                    )
                    time.sleep(1)

            resp = False
            while not resp:
                resp = client.put(key=f"/topo/status/{global_rank}", value="ok")
                if not resp:
                    logger.info(
                        f"put ok status for rank {global_rank} failed, retry later!"
                    )
            if global_rank == 0:
                retry = True
                global_size = int(os.getenv("PADDLE_GLOBAL_SIZE"))
                while retry:
                    resp = client.get_prefix(key="/topo/status")
                    if resp and len(resp) == global_size:
                        server.stop()
                        retry = False
                        logger.info("server stopped success")
                    else:
                        logger.info("server stopped failed! retry later")
                        time.sleep(1)
            logger.info(
                f'cluster_topo_info: {json.dumps(cluster.mesh_group.to_json(), indent=3)}'
            )
            name = None
            for mesh in cluster.mesh_group.meshes.values():
                if name is None:
                    name = mesh.name
                else:
                    if name != mesh.name:
                        cluster._hetero = True
            return cluster
        else:
            # when single machine, use topo directory
            topo_dict = {
                local_topo.machine["device_type_full"]: {
                    0: local_topo.machine,
                }
            }
            cluster._build_from_topo(topo_dict, local_size)
            cluster._hetero = False
            logger.info(
                f'cluster_topo_info: {json.dumps(cluster.mesh_group.to_json(), indent=3)}'
            )
            return cluster
    else:
        # Get GPU info by get_device_properties
        local_device_count = os.getenv("PADDLE_LOCAL_SIZE")
        if local_device_count is None:
            local_device_count = 1
        else:
            local_device_count = int(local_device_count)

        global_device_count = os.getenv("PADDLE_GLOBAL_SIZE")
        if global_device_count is None:
            node_count = 1
        else:
            global_device_count = int(global_device_count)
            assert global_device_count % local_device_count == 0
            node_count = int(global_device_count) // local_device_count

        if os.getenv("PADDLE_DISTRI_BACKEND", None) == "xccl":
            gpu_name = os.getenv("PADDLE_XCCL_BACKEND", None)
            gpu_model = gpu_name
            memory = int(
                paddle.base.core.libpaddle._get_device_total_memory(gpu_name)
            ) // (1000**3)
        else:
            gpu_info = paddle.device.cuda.get_device_properties()
            assert gpu_info, "Auto parallel just runs on gpu now."

            gpu_name = gpu_info.name
            try:
                re_result = re.split(r'[ , -]', gpu_name)
                gpu_model = re_result[1]
                memory = int(re_result[-1][:-2])
            except:
                memory = int(gpu_info.total_memory) // (1000**3)
                gpu_model = gpu_name

    logger.info(
        "Node Count: {}, Local Device Size: {}, GPU Model: {}, GPU Memory: {}GB, World size: {}, EndPoint: {}.".format(
            node_count,
            local_device_count,
            gpu_model,
            memory,
            paddle.distributed.get_world_size(),
            os.getenv("PADDLE_CURRENT_ENDPOINT", None),
        )
    )

    gflops_info = {
        "V100": {"dp": 7800, "sp": 15700, "hp": 125000},
        "A100": {"dp": 9700, "sp": 19500, "hp": 624000},
    }
    default_gflops = (
        gflops_info["A100"] if gpu_model == "A100" else gflops_info["V100"]
    )

    cluster.gen_default_config_cluster(
        node_count=node_count,
        device_count=local_device_count,
        gpu_model=gpu_model,
        gpu_memory=memory,
        gpu_dp_gflops=default_gflops["dp"],
        gpu_sp_gflops=default_gflops["sp"],
        gpu_hp_gflops=default_gflops["hp"],
    )
    return cluster
