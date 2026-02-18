"""通信协议成本模型（对齐 DS_TPU 口径）."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numba


# ============== Numba JIT 加速的通信协议计算 ==============


@numba.jit(nopython=True, cache=True)
def _hierarchical_group_size_numba(tp: int) -> int:
    """计算层次化分组大小"""
    if tp == 16:
        return 2
    return 4


@numba.jit(nopython=True, cache=True)
def _allreduce_numba(
    tp: int,
    comm_bytes: float,
    intra_bw: float,
    inter_bw: float,
    bw_urate: float,
    start_lat: float,
    sync_lat: float,
    switch_latency: float,
    cable_latency: float,
) -> Tuple[float, float]:
    """AllReduce 通信延迟和通信量计算（Numba JIT）"""
    if tp in (8, 16, 32):
        group_size = _hierarchical_group_size_numba(tp)
        num_groups = tp // group_size

        comm_1 = 2.0 * (group_size - 1) / group_size * comm_bytes
        comm_2 = 2.0 * (num_groups - 1) / num_groups * comm_bytes
        comm_3 = float(comm_bytes)

        lat_1 = (comm_1 / intra_bw / bw_urate) * 1e6 + (group_size - 1) * (
            start_lat + sync_lat
        )
        lat_2 = (comm_2 / inter_bw / bw_urate) * 1e6 + (num_groups - 1) * (
            start_lat + sync_lat + switch_latency + cable_latency
        )
        lat_3 = (comm_3 / intra_bw / bw_urate) * 1e6 + (group_size - 1) * (
            start_lat + sync_lat
        )

        latency_us = max(lat_1, max(lat_2, lat_3))
        comm_size = comm_1 * num_groups + comm_2 + comm_3 * num_groups
        return latency_us, comm_size

    comm_size = 2.0 * (tp - 1) / max(1, tp) * comm_bytes
    latency_us = (comm_size / intra_bw / bw_urate) * 1e6 + (tp - 1) * (
        start_lat + sync_lat
    )
    return latency_us, comm_size


@numba.jit(nopython=True, cache=True)
def _allgather_numba(
    tp: int,
    comm_bytes: float,
    intra_bw: float,
    inter_bw: float,
    bw_urate: float,
    start_lat: float,
    sync_lat: float,
    switch_latency: float,
    cable_latency: float,
) -> Tuple[float, float]:
    """AllGather 通信延迟和通信量计算（Numba JIT）"""
    if tp in (8, 16, 32):
        group_size = _hierarchical_group_size_numba(tp)
        num_groups = tp // group_size

        comm_1 = float((group_size - 1) * comm_bytes)
        comm_2 = float((num_groups - 1) * comm_bytes)
        comm_3 = 0.0

        lat_1 = (comm_1 / intra_bw / bw_urate) * 1e6 + (group_size - 1) * (
            start_lat + sync_lat
        )
        lat_2 = (comm_2 / inter_bw / bw_urate) * 1e6 + (num_groups - 1) * (
            start_lat + sync_lat + switch_latency + cable_latency
        )

        latency_us = max(lat_1, max(lat_2, comm_3))
        comm_size = comm_1 * num_groups + comm_2 + comm_3 * num_groups
        return latency_us, comm_size

    comm_size = float((tp - 1) * comm_bytes)
    latency_us = (comm_size / intra_bw / bw_urate) * 1e6 + (tp - 1) * (
        start_lat + sync_lat
    )
    return latency_us, comm_size


@numba.jit(nopython=True, cache=True)
def _reducescatter_numba(
    tp: int,
    comm_bytes: float,
    intra_bw: float,
    inter_bw: float,
    bw_urate: float,
    start_lat: float,
    sync_lat: float,
    switch_latency: float,
    cable_latency: float,
) -> Tuple[float, float]:
    """ReduceScatter 通信延迟和通信量计算（Numba JIT）"""
    if tp in (8, 16, 32):
        group_size = _hierarchical_group_size_numba(tp)
        num_groups = tp // group_size

        comm_1 = (group_size - 1) / group_size * comm_bytes
        comm_2 = (num_groups - 1) / num_groups * comm_bytes
        comm_3 = 0.0

        lat_1 = (comm_1 / intra_bw / bw_urate) * 1e6 + (group_size - 1) * (
            start_lat + sync_lat
        )
        lat_2 = (comm_2 / inter_bw / bw_urate) * 1e6 + (num_groups - 1) * (
            start_lat + sync_lat + switch_latency + cable_latency
        )

        latency_us = max(lat_1, max(lat_2, comm_3))
        comm_size = comm_1 * num_groups + comm_2 + comm_3 * num_groups
        return latency_us, comm_size

    comm_size = (tp - 1) / max(1, tp) * comm_bytes
    latency_us = (comm_size / intra_bw / bw_urate) * 1e6 + (tp - 1) * (
        start_lat + sync_lat
    )
    return latency_us, comm_size


# ============== 数据类 ==============


@dataclass
class CommArchSpec:
    """通信硬件口径

    Attributes:
        intra_bw: 片内/板内带宽（bytes/s）
        inter_bw: 板间/节点间带宽（bytes/s）
    """

    intra_bw: float
    inter_bw: float


@dataclass
class CommProtocolParams:
    """通信协议参数（可由外部配置注入）"""

    c2c_lat: float = 0.15
    ddr_r_lat: float = 0.15
    ddr_w_lat: float = 0.01
    noc_lat: float = 0.05
    d2d_lat: float = 0.04
    sync_lat: float = 0.0
    bw_urate: float = 0.95
    switch_latency: float = 0.25
    cable_latency: float = 0.025
    cpu_fetch_delay: float = 0.0
    topk: int = 8
    prefill_factor: float = 8 / 128

    @classmethod
    def from_mapping(cls, config: Mapping[str, float | int]) -> CommProtocolParams:
        """从外部字典读取通信协议参数.

        输入:
            - config: 由 merge_specs() 输出的标准 key name 字典。
        输出:
            - CommProtocolParams
        关键步骤:
            - merge_specs() 保证标准 key name 存在，直接读取。
            - cpu_fetch_delay / topk / prefill_factor 为评估协议可选参数，保留默认值。
        """
        defaults = cls()

        return cls(
            c2c_lat=float(config["c2c_latency_us"]),
            ddr_r_lat=float(config["memory_read_latency_us"]),
            ddr_w_lat=float(config["memory_write_latency_us"]),
            noc_lat=float(config["noc_latency_us"]),
            d2d_lat=float(config["die_to_die_latency_us"]),
            sync_lat=float(config["sync_lat_us"]),
            bw_urate=float(config["bw_utilization"]),
            switch_latency=float(config["switch_latency_us"]),
            cable_latency=float(config["cable_latency_us"]),
            # 以下为评估协议可选参数，保留默认值
            cpu_fetch_delay=float(config.get("cpu_fetch_delay_us", defaults.cpu_fetch_delay)),
            topk=int(config.get("moe_topk", defaults.topk)),
            prefill_factor=float(config.get("prefill_topk_factor", defaults.prefill_factor)),
        )


class CommProtocolCostModel:
    """通信协议成本模型集合（近似 DS_TPU 实现）"""

    def __init__(
        self,
        arch: CommArchSpec,
        params: CommProtocolParams | None = None,
    ) -> None:
        self.arch = arch
        self.params = params or CommProtocolParams()

    @property
    def start_lat(self) -> float:
        return (
            2 * self.params.c2c_lat
            + self.params.ddr_r_lat
            + self.params.ddr_w_lat
            + self.params.noc_lat
            + 2 * self.params.d2d_lat
        )

    @property
    def dispatch_start_lat(self) -> float:
        """dispatch/combine 启动延迟（含交换机/线缆延迟，对齐 CHIPMathica）"""
        return (
            self.start_lat
            + 2 * self.params.switch_latency
            + 2 * self.params.cable_latency
        )

    def allreduce(self, tp: int, comm_bytes: int, comm_protocol: int) -> tuple[float, float]:
        return _allreduce_numba(
            tp=tp, comm_bytes=float(comm_bytes),
            intra_bw=self.arch.intra_bw, inter_bw=self.arch.inter_bw,
            bw_urate=self.params.bw_urate, start_lat=self.start_lat,
            sync_lat=self.params.sync_lat,
            switch_latency=self.params.switch_latency,
            cable_latency=self.params.cable_latency,
        )

    def allgather(self, tp: int, comm_bytes: int, comm_protocol: int) -> tuple[float, float]:
        return _allgather_numba(
            tp=tp, comm_bytes=float(comm_bytes),
            intra_bw=self.arch.intra_bw, inter_bw=self.arch.inter_bw,
            bw_urate=self.params.bw_urate, start_lat=self.start_lat,
            sync_lat=self.params.sync_lat,
            switch_latency=self.params.switch_latency,
            cable_latency=self.params.cable_latency,
        )

    def reducescatter(
        self, tp: int, comm_bytes: int, comm_protocol: int
    ) -> tuple[float, float]:
        return _reducescatter_numba(
            tp=tp, comm_bytes=float(comm_bytes),
            intra_bw=self.arch.intra_bw, inter_bw=self.arch.inter_bw,
            bw_urate=self.params.bw_urate, start_lat=self.start_lat,
            sync_lat=self.params.sync_lat,
            switch_latency=self.params.switch_latency,
            cable_latency=self.params.cable_latency,
        )

    def dispatch(
        self,
        moe_tp: int,
        ep: int,
        comm_bytes: int,
        bs: int,
        comm_protocol: int,
        is_prefill: bool,
    ) -> tuple[float, float]:
        t_us = (
            (comm_bytes / self.arch.inter_bw / self.params.bw_urate) * 1e6
            + self.dispatch_start_lat
            + self.params.cpu_fetch_delay
        )

        ag_lat, ag_comm = self.allgather(moe_tp, comm_bytes, comm_protocol)
        t_us += ag_lat
        return t_us, comm_bytes + ag_comm

    def combine(
        self,
        moe_tp: int,
        ep: int,
        comm_bytes: int,
        bs: int,
        comm_protocol: int,
        is_prefill: bool,
    ) -> tuple[float, float]:
        t_us = (
            (comm_bytes / self.arch.inter_bw / self.params.bw_urate) * 1e6
            + self.dispatch_start_lat
            + self.params.cpu_fetch_delay
        )

        ag_lat, ag_comm = self.allgather(moe_tp, comm_bytes, comm_protocol)
        t_us += ag_lat
        return t_us, comm_bytes + ag_comm
