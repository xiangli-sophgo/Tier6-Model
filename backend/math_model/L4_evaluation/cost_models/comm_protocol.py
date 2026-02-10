"""通信协议成本模型（对齐 DS_TPU 口径）."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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
    link_delay: float = 0.0
    switch_delay: float = 0.25
    cable_delay: float = 0.025
    cpu_fetch_delay: float = 0.0
    topk: int = 8
    prefill_factor: float = 8 / 128

    @classmethod
    def from_mapping(cls, config: Mapping[str, float | int]) -> CommProtocolParams:
        """从外部字典读取通信协议参数.

        输入:
            - config: 参数字典，支持 `comm_` 前缀或无前缀键。
        输出:
            - CommProtocolParams
        关键步骤:
            - 优先读取 `comm_xxx`，缺失时回退读取 `xxx`，都缺失则使用默认值。

        注意: 此方法的默认值用于向后兼容和可选配置，通信协议参数通常为可选。
        """
        defaults = cls()

        def _read(default: float | int, *keys: str) -> float | int:
            for key in keys:
                if key in config:
                    return config[key]
            return default

        return cls(
            c2c_lat=float(_read(defaults.c2c_lat, "c2c_lat_us", "comm_c2c_lat", "c2c_lat")),
            ddr_r_lat=float(
                _read(defaults.ddr_r_lat, "ddr_r_lat_us", "comm_ddr_r_lat", "ddr_r_lat")
            ),
            ddr_w_lat=float(
                _read(defaults.ddr_w_lat, "ddr_w_lat_us", "comm_ddr_w_lat", "ddr_w_lat")
            ),
            noc_lat=float(_read(defaults.noc_lat, "noc_lat_us", "comm_noc_lat", "noc_lat")),
            d2d_lat=float(_read(defaults.d2d_lat, "d2d_lat_us", "comm_d2d_lat", "d2d_lat")),
            sync_lat=float(_read(defaults.sync_lat, "sync_lat_us", "comm_sync_lat", "sync_lat")),
            bw_urate=float(
                _read(defaults.bw_urate, "bw_utilization", "comm_bw_urate", "bw_urate")
            ),
            link_delay=float(
                _read(defaults.link_delay, "link_delay_us", "comm_link_delay", "link_delay")
            ),
            switch_delay=float(
                _read(
                    defaults.switch_delay,
                    "switch_delay_us",
                    "comm_switch_delay",
                    "switch_delay",
                )
            ),
            cable_delay=float(
                _read(defaults.cable_delay, "cable_delay_us", "comm_cable_delay", "cable_delay")
            ),
            cpu_fetch_delay=float(
                _read(
                    defaults.cpu_fetch_delay,
                    "cpu_fetch_delay_us",
                    "comm_cpu_fetch_delay",
                    "cpu_fetch_delay",
                )
            ),
            topk=int(_read(defaults.topk, "moe_topk", "comm_topk", "topk")),
            prefill_factor=float(
                _read(
                    defaults.prefill_factor,
                    "prefill_topk_factor",
                    "comm_prefill_factor",
                    "prefill_factor",
                )
            ),
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

    def _hierarchical_group_size(self, tp: int) -> int:
        if tp == 16:
            return 2
        return 4

    def allreduce(self, tp: int, comm_bytes: int, comm_protocol: int) -> tuple[float, float]:
        if tp in {8, 16, 32}:
            group_size = self._hierarchical_group_size(tp)
            num_groups = tp // group_size
            comm_1 = 2 * (group_size - 1) / group_size * comm_bytes
            comm_2 = 2 * (num_groups - 1) / num_groups * comm_bytes
            comm_3 = comm_bytes
            lat_1 = (comm_1 / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
                group_size - 1
            ) * (self.start_lat + self.params.sync_lat)
            lat_2 = (comm_2 / self.arch.inter_bw / self.params.bw_urate) * 1e6 + (
                num_groups - 1
            ) * (self.start_lat + self.params.sync_lat + self.params.link_delay)
            lat_3 = (comm_3 / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
                group_size - 1
            ) * (self.start_lat + self.params.sync_lat)
            latency_us = max(lat_1, lat_2, lat_3)
            comm_size = comm_1 * num_groups + comm_2 + comm_3 * num_groups
            return latency_us, comm_size

        comm_size = 2 * (tp - 1) / max(1, tp) * comm_bytes
        latency_us = (comm_size / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
            tp - 1
        ) * (self.start_lat + self.params.sync_lat)
        return latency_us, comm_size

    def allgather(self, tp: int, comm_bytes: int, comm_protocol: int) -> tuple[float, float]:
        if tp in {8, 16, 32}:
            group_size = self._hierarchical_group_size(tp)
            num_groups = tp // group_size
            comm_1 = (group_size - 1) * comm_bytes
            comm_2 = (num_groups - 1) * comm_bytes
            comm_3 = 0.0
            lat_1 = (comm_1 / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
                group_size - 1
            ) * (self.start_lat + self.params.sync_lat)
            lat_2 = (comm_2 / self.arch.inter_bw / self.params.bw_urate) * 1e6 + (
                num_groups - 1
            ) * (self.start_lat + self.params.sync_lat + self.params.link_delay)
            latency_us = max(lat_1, lat_2, comm_3)
            comm_size = comm_1 * num_groups + comm_2 + comm_3 * num_groups
            return latency_us, comm_size

        comm_size = (tp - 1) * comm_bytes
        latency_us = (comm_size / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
            tp - 1
        ) * (self.start_lat + self.params.sync_lat)
        return latency_us, comm_size

    def reducescatter(
        self, tp: int, comm_bytes: int, comm_protocol: int
    ) -> tuple[float, float]:
        if tp in {8, 16, 32}:
            group_size = self._hierarchical_group_size(tp)
            num_groups = tp // group_size
            comm_1 = (group_size - 1) / group_size * comm_bytes
            comm_2 = (num_groups - 1) / num_groups * comm_bytes
            comm_3 = 0.0
            lat_1 = (comm_1 / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
                group_size - 1
            ) * (self.start_lat + self.params.sync_lat)
            lat_2 = (comm_2 / self.arch.inter_bw / self.params.bw_urate) * 1e6 + (
                num_groups - 1
            ) * (self.start_lat + self.params.sync_lat + self.params.link_delay)
            latency_us = max(lat_1, lat_2, comm_3)
            comm_size = comm_1 * num_groups + comm_2 + comm_3 * num_groups
            return latency_us, comm_size

        comm_size = (tp - 1) / max(1, tp) * comm_bytes
        latency_us = (comm_size / self.arch.intra_bw / self.params.bw_urate) * 1e6 + (
            tp - 1
        ) * (self.start_lat + self.params.sync_lat)
        return latency_us, comm_size

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
            + self.start_lat
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
            + self.start_lat
            + self.params.cpu_fetch_delay
        )

        ag_lat, ag_comm = self.allgather(moe_tp, comm_bytes, comm_protocol)
        t_us += ag_lat
        return t_us, comm_bytes + ag_comm
