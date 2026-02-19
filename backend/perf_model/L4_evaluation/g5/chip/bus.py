"""Bus NxM 总线互联模型

对标 TPUPerf simple_bus:
- 2D mesh 拓扑
- Manhattan 距离延迟: delay = distance * base_latency_cycles / frequency
- SG2262: 8x8 mesh, 45 cycles/hop, 1.0 GHz

参考: docs/plans/2026-02-19-g5-full-architecture-design.md Section 6.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perf_model.L4_evaluation.g5.kernel.stats import StatGroup


class BusModel:
    """NxM 总线互联 (2D mesh)"""

    def __init__(
        self,
        core_count: int,
        mesh_dims: tuple[int, int],
        base_latency_cycles: int,
        frequency_ghz: float,
        parent_stats: StatGroup | None = None,
    ) -> None:
        self._core_count = core_count
        self._cols, self._rows = mesh_dims
        self._base_latency_cycles = base_latency_cycles
        self._frequency_ghz = frequency_ghz

        if self._cols * self._rows < core_count:
            raise ValueError(
                f"Mesh {mesh_dims} ({self._cols * self._rows} slots) "
                f"cannot hold {core_count} cores"
            )

        # 统计注册
        self._stat_transfers = None
        self._stat_bytes = None
        self._stat_hop_total = None
        if parent_stats is not None:
            from perf_model.L4_evaluation.g5.kernel.stats import StatGroup as SG
            self._bus_stats = SG("bus", parent=parent_stats)
            self._stat_transfers = self._bus_stats.scalar(
                "total_transfers", "总传输次数"
            )
            self._stat_bytes = self._bus_stats.scalar(
                "total_bytes", "总传输数据量"
            )
            self._stat_hop_total = self._bus_stats.scalar(
                "hop_total", "总跳数"
            )

    def _core_to_xy(self, core_id: int) -> tuple[int, int]:
        """core_id -> (x, y) 坐标"""
        x = core_id % self._cols
        y = core_id // self._cols
        return (x, y)

    def manhattan_distance(self, src_core: int, dst_core: int) -> int:
        """两核间 Manhattan 距离"""
        sx, sy = self._core_to_xy(src_core)
        dx, dy = self._core_to_xy(dst_core)
        return abs(sx - dx) + abs(sy - dy)

    def get_delay_ns(self, src_core: int, dst_core: int, data_bytes: int) -> float:
        """计算 Bus 延迟 (ns)

        delay = manhattan_distance * base_latency_cycles / frequency_ghz
        同核通信返回 0.
        """
        if src_core == dst_core:
            return 0.0
        dist = self.manhattan_distance(src_core, dst_core)
        # 累加统计
        if self._stat_transfers is not None:
            self._stat_transfers.inc()
            self._stat_bytes.inc(data_bytes)
            self._stat_hop_total.inc(dist)
        return dist * self._base_latency_cycles / self._frequency_ghz
