"""SingleChip: 单芯片多核组装

组装流程:
  1. 创建 SimKernel
  2. 创建 BusModel (2D mesh, Manhattan 距离延迟)
  3. 创建 N 个 CoreSubsys (共享 kernel + bus)
  4. 加载各核指令
  5. kernel.run()
  6. 收集所有 SimRecord + 统计 dump

对标: TPUPerf tpuManyCore.cc
参考: docs/plans/2026-02-19-g5-full-architecture-design.md Section 8.1
"""

from __future__ import annotations

from typing import Any

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import CoreProgram
from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel
from perf_model.L4_evaluation.g5.chip.core_subsys import CoreSubsys
from perf_model.L4_evaluation.g5.chip.bus import BusModel
from perf_model.L4_evaluation.g5.kernel.sim_record import SimRecord

# SG2262 NoC 默认参数
_DEFAULT_MESH_COLS = 8
_DEFAULT_MESH_ROWS = 8
_DEFAULT_BUS_LATENCY_CYCLES = 45


class SingleChipSim:
    """单芯片多核仿真器"""

    def __init__(self, chip: ChipSpecImpl) -> None:
        self._chip = chip
        self._last_stats: dict[str, Any] = {}

    def simulate(self, program: CoreProgram) -> list[SimRecord]:
        """仿真 CoreProgram (多核)

        Args:
            program: 多核指令程序

        Returns:
            所有核的 SimRecord 列表 (按时间排序)
        """
        self._last_stats = {}

        if not program.cores:
            return []

        core_count = len(program.cores)
        tiu_freq = self._chip.get_tiu_frequency()

        # 1. 创建 SimKernel + 注册时钟
        kernel = SimKernel()
        kernel.add_clock("tpu", frequency_ghz=tiu_freq)

        # 2. 创建 BusModel (从 ChipSpec NoC 配置读取参数)
        noc = self._chip.noc_config
        mesh_cols = noc.get("mesh_cols", _DEFAULT_MESH_COLS)
        mesh_rows = noc.get("mesh_rows", _DEFAULT_MESH_ROWS)
        bus_latency_cycles = noc.get("base_latency_cycles", _DEFAULT_BUS_LATENCY_CYCLES)

        # 单核时 mesh 大小不影响
        if core_count == 1:
            mesh_cols = max(mesh_cols, 1)
            mesh_rows = max(mesh_rows, 1)

        bus = BusModel(
            core_count=core_count,
            mesh_dims=(mesh_cols, mesh_rows),
            base_latency_cycles=bus_latency_cycles,
            frequency_ghz=tiu_freq,
            parent_stats=kernel.stats,
        )

        # 3. 创建各核 CoreSubsys
        cores: list[CoreSubsys] = []
        for core_instr in program.cores:
            core = CoreSubsys(
                kernel=kernel,
                chip=self._chip,
                core_id=core_instr.core_id,
                clock_name="tpu",
                bus_delay_fn=bus.get_delay_ns,
            )
            core.load_instructions(core_instr)
            cores.append(core)

        # 4. 运行仿真
        kernel.run()

        # 5. 收集统计
        self._last_stats = kernel.stats.dump()

        # 6. 收集所有记录, 按开始时间排序
        all_records: list[SimRecord] = []
        for core in cores:
            all_records.extend(core.get_records())
        all_records.sort(key=lambda r: (r.start_ns, r.cmd_id))
        return all_records

    def get_stats(self) -> dict[str, Any]:
        """获取最近一次仿真的统计数据"""
        return dict(self._last_stats)
