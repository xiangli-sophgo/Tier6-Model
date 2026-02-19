"""仿真事件 -> EngineResult 适配器 (G5 指令级仿真模式)

按 source_op_id 分组 SimRecord:
    - t_compute = sum(TIU + HAU durations) (ns -> ms)
    - t_comm = sum(SDMA durations) (ns -> ms)
    - t_wait = span - t_compute - t_comm
    - flops = sum(TIU FLOPs)
    - bytes_read = sum(DDR_TO_LMEM data_bytes)
    - bytes_write = sum(LMEM_TO_DDR data_bytes)

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import DMADirection
from perf_model.L4_evaluation.common.metrics import (
    Aggregates,
    BottleneckTag,
    EngineResult,
    Granularity,
    StepMetrics,
)
from perf_model.L4_evaluation.g5.sim_engine import SimRecord


class G5ResultAdapter:
    """将 SimRecord 列表转换为 EngineResult"""

    def __init__(self, chip: ChipSpecImpl) -> None:
        self._chip = chip

    def convert(
        self,
        records: list[SimRecord],
        stats: dict[str, Any] | None = None,
    ) -> EngineResult:
        """转换 SimRecord -> EngineResult

        Args:
            records: 仿真记录列表
            stats: 统计框架 dump 数据 (可选)

        Returns:
            EngineResult
        """
        if not records:
            return EngineResult(granularity=Granularity.LANE)

        # 按 source_op_id 分组
        groups: dict[str, list[SimRecord]] = defaultdict(list)
        for rec in records:
            groups[rec.source_op_id].append(rec)

        step_metrics: list[StepMetrics] = []
        total_flops = 0
        total_bytes = 0
        total_compute_ns = 0.0
        total_comm_ns = 0.0

        # 全局时间跨度
        global_start = min(r.start_ns for r in records)
        global_end = max(r.end_ns for r in records)

        for op_id, recs in groups.items():
            tiu_duration_ns = 0.0
            hau_duration_ns = 0.0
            sdma_duration_ns = 0.0
            flops = 0
            bytes_read = 0
            bytes_write = 0

            op_start = min(r.start_ns for r in recs)
            op_end = max(r.end_ns for r in recs)

            for rec in recs:
                duration = rec.end_ns - rec.start_ns
                if rec.engine == "TIU":
                    tiu_duration_ns += duration
                    flops += rec.flops
                elif rec.engine == "DMA":
                    if rec.direction == DMADirection.DDR_TO_LMEM:
                        bytes_read += rec.data_bytes
                    elif rec.direction == DMADirection.LMEM_TO_DDR:
                        bytes_write += rec.data_bytes
                elif rec.engine == "SDMA":
                    sdma_duration_ns += duration
                    bytes_read += rec.data_bytes
                elif rec.engine == "HAU":
                    hau_duration_ns += duration

            span_ns = op_end - op_start
            compute_ns = tiu_duration_ns + hau_duration_ns
            comm_ns = sdma_duration_ns
            t_compute_ms = compute_ns / 1e6
            t_comm_ms = comm_ns / 1e6
            t_wait_ms = max(0.0, (span_ns - compute_ns - comm_ns) / 1e6)

            # 瓶颈判断
            dma_total_ns = span_ns - compute_ns - comm_ns
            if comm_ns > 0 and comm_ns >= compute_ns and comm_ns >= dma_total_ns:
                tag = BottleneckTag.BW_BOUND
            elif compute_ns > 0 and (compute_ns >= dma_total_ns or dma_total_ns <= 0):
                tag = BottleneckTag.COMPUTE_BOUND
            else:
                tag = BottleneckTag.BW_BOUND

            step_metrics.append(StepMetrics(
                op_id=op_id,
                t_compute=t_compute_ms,
                t_comm=t_comm_ms,
                t_wait=t_wait_ms,
                bottleneck_tag=tag,
                flops=flops,
                bytes_read=bytes_read,
                bytes_write=bytes_write,
            ))

            total_flops += flops
            total_bytes += bytes_read + bytes_write
            total_compute_ns += compute_ns
            total_comm_ns += comm_ns

        # 聚合指标
        total_time_ms = (global_end - global_start) / 1e6
        total_compute_ms = total_compute_ns / 1e6
        total_comm_ms = total_comm_ns / 1e6
        total_wait_ms = max(0.0, total_time_ms - total_compute_ms - total_comm_ms)

        # MFU 计算: achieved FLOPS / peak FLOPS
        peak_flops = self._chip.get_peak_flops("BF16")
        mfu = 0.0
        if peak_flops > 0 and total_time_ms > 0:
            achieved_flops_per_sec = total_flops / (total_time_ms / 1e3)
            mfu = achieved_flops_per_sec / peak_flops

        # MBU 计算: required bandwidth / peak bandwidth
        gmem_bw_gbps = self._chip.get_gmem_bandwidth()
        mbu = 0.0
        if gmem_bw_gbps > 0 and total_time_ms > 0:
            required_bw_gbps = total_bytes / (total_time_ms * 1e6)
            mbu = required_bw_gbps / gmem_bw_gbps

        # 瓶颈统计
        bottleneck_counts: dict[str, int] = defaultdict(int)
        for sm in step_metrics:
            bottleneck_counts[sm.bottleneck_tag.name] += 1

        aggregates = Aggregates(
            total_time=total_time_ms,
            total_compute_time=total_compute_ms,
            total_comm_time=total_comm_ms,
            total_wait_time=total_wait_ms,
            total_flops=total_flops,
            total_bytes=total_bytes,
            num_steps=len(step_metrics),
            mfu=mfu,
            mbu=mbu,
            bottleneck_summary=dict(bottleneck_counts),
        )

        trace_meta: dict[str, Any] = {
            "chip": self._chip.name,
            "total_records": len(records),
            "total_time_ns": global_end - global_start,
        }
        if stats:
            trace_meta["stats"] = stats

        return EngineResult(
            step_metrics=step_metrics,
            aggregates=aggregates,
            granularity=Granularity.LANE,
            trace_meta=trace_meta,
        )
