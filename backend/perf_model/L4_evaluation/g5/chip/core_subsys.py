"""CoreSubsys: 单核子系统

封装 TIU + GDMA + SDMA + HAU 四引擎, 管理同步信号和指令调度。
对标 TPUPerf TpuSubsys。

同步模型 (与现有 sim_engine.py 完全一致):
  TIU:  cmd.cmd_id_dep <= tdma_sync_id (固定依赖 DMA)
  DMA:  cmd.cmd_id_dep <= tiu_sync_id  (固定依赖 TIU)
  SDMA: cmd.dep_engine 指定依赖引擎
  HAU:  cmd.dep_engine 指定依赖引擎

参考: docs/plans/2026-02-19-g5-full-architecture-design.md Section 6.1
"""

from __future__ import annotations

from typing import Callable

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import CoreInstructions, DMADirection
from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel
from perf_model.L4_evaluation.g5.kernel.sim_object import SimObject
from perf_model.L4_evaluation.g5.kernel.sim_record import SimRecord
from perf_model.L4_evaluation.g5.kernel.stats import StatGroup
from perf_model.L4_evaluation.g5.tiu import calc_tiu_latency
from perf_model.L4_evaluation.g5.dma import calc_dma_latency
from perf_model.L4_evaluation.g5.sdma import calc_sdma_latency
from perf_model.L4_evaluation.g5.hau import calc_hau_latency


class CoreSubsys(SimObject):
    """单核子系统: 4 引擎并行 + 同步信号"""

    def __init__(
        self,
        kernel: SimKernel,
        chip: ChipSpecImpl,
        core_id: int,
        clock_name: str,
        bus_delay_fn: Callable[[int, int, int], float] | None = None,
    ) -> None:
        super().__init__(kernel=kernel, name=f"core_{core_id}", clock_name=clock_name)
        self._chip = chip
        self.core_id = core_id
        self._bus_delay_fn = bus_delay_fn  # (src_core, dst_core, bytes) -> delay_ns

        # 同步信号
        self._tiu_sync_id: int = 0
        self._tdma_sync_id: int = 0
        self._sdma_sync_id: int = 0
        self._hau_sync_id: int = 0

        # 引擎忙状态
        self._tiu_busy: bool = False
        self._dma_busy: bool = False
        self._sdma_busy: bool = False
        self._hau_busy: bool = False

        # 指令队列索引
        self._tiu_idx: int = 0
        self._dma_idx: int = 0
        self._sdma_idx: int = 0
        self._hau_idx: int = 0

        # 指令队列
        self._tiu_cmds: list = []
        self._dma_cmds: list = []
        self._sdma_cmds: list = []
        self._hau_cmds: list = []

        # 仿真记录
        self._records: list[SimRecord] = []

        # ---- 统计注册 ----
        self.stats = StatGroup(f"core{core_id}", parent=kernel.stats)

        # 核级统计
        self._stat_total_instr = self.stats.scalar(
            "total_instructions", "指令总数"
        )
        self._stat_cmd_by_engine = self.stats.vector(
            "cmd_count_by_engine", "按引擎分的指令数"
        )

        # TIU 引擎统计
        self._tiu_stats = StatGroup("tiu", parent=self.stats)
        self._stat_tiu_cmd = self._tiu_stats.scalar("cmd_count", "TIU 指令数")
        self._stat_tiu_compute_cycles = self._tiu_stats.scalar(
            "compute_cycles", "纯计算周期"
        )
        self._stat_tiu_init_cycles = self._tiu_stats.scalar(
            "init_cycles", "流水线填充周期"
        )
        self._stat_tiu_flops = self._tiu_stats.scalar("total_flops", "总 FLOPs")
        self._stat_tiu_by_prec = self._tiu_stats.vector(
            "cycles_by_prec", "按精度分的周期"
        )
        self._stat_tiu_by_op = self._tiu_stats.vector(
            "cmd_by_op", "按操作类型分的指令数"
        )

        # DMA 引擎统计
        self._dma_stats = StatGroup("dma", parent=self.stats)
        self._stat_dma_cmd = self._dma_stats.scalar("cmd_count", "DMA 指令数")
        self._stat_dma_bytes_read = self._dma_stats.scalar(
            "bytes_read", "DDR->LMEM 读取字节数"
        )
        self._stat_dma_bytes_write = self._dma_stats.scalar(
            "bytes_write", "LMEM->DDR 写入字节数"
        )
        self._stat_dma_startup_ns = self._dma_stats.scalar(
            "startup_ns", "启动延迟总和"
        )
        self._stat_dma_transfer_ns = self._dma_stats.scalar(
            "transfer_ns", "数据传输时间总和"
        )
        self._stat_dma_by_dir = self._dma_stats.vector(
            "bytes_by_dir", "按方向分的字节数"
        )

        # SDMA 引擎统计
        self._sdma_stats = StatGroup("sdma", parent=self.stats)
        self._stat_sdma_cmd = self._sdma_stats.scalar("cmd_count", "SDMA 指令数")
        self._stat_sdma_bytes = self._sdma_stats.scalar(
            "total_bytes", "总传输字节数"
        )
        self._stat_sdma_bus_ns = self._sdma_stats.scalar(
            "bus_latency_ns", "Bus 路由延迟总和"
        )
        self._stat_sdma_transfer_ns = self._sdma_stats.scalar(
            "transfer_ns", "数据传输时间总和"
        )
        self._stat_sdma_hops = self._sdma_stats.scalar("hop_total", "总跳数")
        self._stat_sdma_by_type = self._sdma_stats.vector(
            "cmd_by_type", "按操作类型分的指令数"
        )

        # HAU 引擎统计
        self._hau_stats = StatGroup("hau", parent=self.stats)
        self._stat_hau_cmd = self._hau_stats.scalar("cmd_count", "HAU 指令数")
        self._stat_hau_elements = self._hau_stats.scalar(
            "total_elements", "处理的总元素数"
        )
        self._stat_hau_cycles = self._hau_stats.scalar(
            "total_cycles", "总排序周期"
        )
        self._stat_hau_by_op = self._hau_stats.vector(
            "cmd_by_op", "按操作类型分的指令数"
        )

    def load_instructions(self, instr: CoreInstructions) -> None:
        """加载指令并立即尝试发射"""
        self._tiu_cmds = list(instr.tiu_cmds)
        self._dma_cmds = list(instr.dma_cmds)
        self._sdma_cmds = list(instr.sdma_cmds)
        self._hau_cmds = list(instr.hau_cmds)
        self._tiu_idx = 0
        self._dma_idx = 0
        self._sdma_idx = 0
        self._hau_idx = 0
        # 在当前时间尝试发射
        self.schedule(0.0, self._try_issue_all)

    def get_records(self) -> list[SimRecord]:
        """获取仿真记录"""
        return list(self._records)

    def get_sync_id(self, engine: str) -> int:
        """获取指定引擎的 sync_id"""
        if engine == "tdma":
            return self._tdma_sync_id
        if engine == "tiu":
            return self._tiu_sync_id
        if engine == "sdma":
            return self._sdma_sync_id
        if engine == "hau":
            return self._hau_sync_id
        return 0

    # ---- 内部: 发射逻辑 ----

    def _try_issue_all(self) -> None:
        self._try_issue_dma()
        self._try_issue_tiu()
        self._try_issue_sdma()
        self._try_issue_hau()

    def _try_issue_tiu(self) -> None:
        if self._tiu_busy or self._tiu_idx >= len(self._tiu_cmds):
            return
        cmd = self._tiu_cmds[self._tiu_idx]
        if cmd.cmd_id_dep <= self._tdma_sync_id:
            self._tiu_idx += 1
            self._tiu_busy = True
            result = calc_tiu_latency(cmd, self._chip)
            start = self.now()
            end = start + result.latency_ns
            self._records.append(SimRecord(
                engine="TIU", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                flops=result.flops,
                source_op_id=cmd.source_op_id,
            ))
            # 累加统计
            self._stat_tiu_cmd.inc()
            self._stat_tiu_compute_cycles.inc(result.compute_cycles)
            self._stat_tiu_init_cycles.inc(result.init_cycles)
            self._stat_tiu_flops.inc(result.flops)
            self._stat_tiu_by_prec.inc(cmd.precision, result.cycles)
            self._stat_tiu_by_op.inc(cmd.op_type.name, 1)
            self._stat_total_instr.inc()
            self._stat_cmd_by_engine.inc("TIU")
            # 捕获 cmd_id 避免闭包陷阱
            cid = cmd.cmd_id
            self.schedule_at(end, lambda: self._on_tiu_finish(cid))

    def _on_tiu_finish(self, cmd_id: int) -> None:
        self._tiu_sync_id = cmd_id
        self._tiu_busy = False
        self._try_issue_all()

    def _try_issue_dma(self) -> None:
        if self._dma_busy or self._dma_idx >= len(self._dma_cmds):
            return
        cmd = self._dma_cmds[self._dma_idx]
        if cmd.cmd_id_dep <= self._tiu_sync_id:
            self._dma_idx += 1
            self._dma_busy = True
            result = calc_dma_latency(cmd, self._chip)
            start = self.now()
            end = start + result.latency_ns
            self._records.append(SimRecord(
                engine="DMA", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                data_bytes=result.data_bytes,
                direction=cmd.direction,
                source_op_id=cmd.source_op_id,
            ))
            # 累加统计
            self._stat_dma_cmd.inc()
            self._stat_dma_startup_ns.inc(result.startup_ns)
            self._stat_dma_transfer_ns.inc(result.transfer_ns)
            dir_name = cmd.direction.name if cmd.direction else "UNKNOWN"
            self._stat_dma_by_dir.inc(dir_name, result.data_bytes)
            if cmd.direction == DMADirection.DDR_TO_LMEM:
                self._stat_dma_bytes_read.inc(result.data_bytes)
            elif cmd.direction == DMADirection.LMEM_TO_DDR:
                self._stat_dma_bytes_write.inc(result.data_bytes)
            self._stat_total_instr.inc()
            self._stat_cmd_by_engine.inc("DMA")
            cid = cmd.cmd_id
            self.schedule_at(end, lambda: self._on_dma_finish(cid))

    def _on_dma_finish(self, cmd_id: int) -> None:
        self._tdma_sync_id = cmd_id
        self._dma_busy = False
        self._try_issue_all()

    def _try_issue_sdma(self) -> None:
        if self._sdma_busy or self._sdma_idx >= len(self._sdma_cmds):
            return
        cmd = self._sdma_cmds[self._sdma_idx]
        dep_val = self.get_sync_id(cmd.dep_engine)
        if cmd.cmd_id_dep <= dep_val:
            self._sdma_idx += 1
            self._sdma_busy = True
            result = calc_sdma_latency(cmd, self._chip)
            # Bus 距离延迟
            bus_delay = 0.0
            if self._bus_delay_fn is not None:
                bus_delay = self._bus_delay_fn(
                    cmd.src_core_id, cmd.dst_core_id, cmd.data_bytes
                )
            start = self.now()
            end = start + result.latency_ns + bus_delay
            self._records.append(SimRecord(
                engine="SDMA", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                data_bytes=result.data_bytes,
                source_op_id=cmd.source_op_id,
            ))
            # 累加统计
            self._stat_sdma_cmd.inc()
            self._stat_sdma_bytes.inc(result.data_bytes)
            self._stat_sdma_bus_ns.inc(bus_delay)
            self._stat_sdma_transfer_ns.inc(result.latency_ns)
            cmd_type_name = cmd.cmd_type.name if hasattr(cmd.cmd_type, "name") else str(cmd.cmd_type)
            self._stat_sdma_by_type.inc(cmd_type_name)
            self._stat_total_instr.inc()
            self._stat_cmd_by_engine.inc("SDMA")
            cid = cmd.cmd_id
            self.schedule_at(end, lambda: self._on_sdma_finish(cid))

    def _on_sdma_finish(self, cmd_id: int) -> None:
        self._sdma_sync_id = cmd_id
        self._sdma_busy = False
        self._try_issue_all()

    def _try_issue_hau(self) -> None:
        if self._hau_busy or self._hau_idx >= len(self._hau_cmds):
            return
        cmd = self._hau_cmds[self._hau_idx]
        dep_val = self.get_sync_id(cmd.dep_engine)
        if cmd.cmd_id_dep <= dep_val:
            self._hau_idx += 1
            self._hau_busy = True
            result = calc_hau_latency(cmd, self._chip)
            start = self.now()
            end = start + result.latency_ns
            self._records.append(SimRecord(
                engine="HAU", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                source_op_id=cmd.source_op_id,
            ))
            # 累加统计
            self._stat_hau_cmd.inc()
            self._stat_hau_elements.inc(cmd.num_elements)
            self._stat_hau_cycles.inc(result.cycles)
            self._stat_hau_by_op.inc(cmd.op_type.name, 1)
            self._stat_total_instr.inc()
            self._stat_cmd_by_engine.inc("HAU")
            cid = cmd.cmd_id
            self.schedule_at(end, lambda: self._on_hau_finish(cid))

    def _on_hau_finish(self, cmd_id: int) -> None:
        self._hau_sync_id = cmd_id
        self._hau_busy = False
        self._try_issue_all()
