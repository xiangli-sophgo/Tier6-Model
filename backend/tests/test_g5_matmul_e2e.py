"""G5 指令级仿真器端到端测试

测试用例: 单核 MatMul 512x2048x4096, BF16, SG2262
验证: DistributedOp -> CoreProgram -> SimRecord -> EngineResult
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# 添加 backend 到 sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.instruction_emitter import G5InstructionEmitter
from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine
from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter


@dataclass
class MockDistributedOp:
    """模拟 DistributedOp"""
    op_id: str
    op_type: str
    local_shape: dict[str, int]
    attrs: dict[str, Any] = field(default_factory=dict)
    deps: list[str] = field(default_factory=list)
    chip_ids: list[int] = field(default_factory=lambda: [0])


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    """加载芯片配置"""
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def test_g5_matmul_single_core():
    """端到端测试: 512x2048x4096 BF16 MatMul on SG2262"""
    print("=" * 60)
    print("G5 MatMul E2E Test: 512x2048x4096, BF16, SG2262")
    print("=" * 60)

    # 1. 加载芯片
    chip = load_chip("SG2262")
    print(f"\n[1] Chip loaded: {chip.name}")
    print(f"    cores={chip.core_count}, lanes={chip.lane_per_core}")
    print(f"    cube_m={chip.cube_m}, cube_k={chip.cube_k}, cube_n={chip.cube_n}")
    print(f"    freq={chip.frequency_ghz} GHz")
    print(f"    LMEM total={chip.get_total_sram()} bytes")
    print(f"    SRAM utilization={chip.sram_utilization}")

    # 2. 创建 op
    M, N, K = 512, 2048, 4096
    op = MockDistributedOp(
        op_id="mm0",
        op_type="matmul",
        local_shape={"M": M, "N": N, "K": K},
        attrs={"input_dtype_bytes": "2", "weight_dtype_bytes": "2", "output_dtype_bytes": "2"},
    )

    # 3. 指令生成 (Step 2)
    emitter = G5InstructionEmitter(chip)
    program = emitter.emit([op])

    n_tiu = program.total_tiu_cmds()
    n_dma = program.total_dma_cmds()
    print(f"\n[2] Instruction Generation:")
    print(f"    cores={len(program.cores)}")
    print(f"    TIU commands={n_tiu}")
    print(f"    DMA commands={n_dma}")
    print(f"    metadata={program.metadata}")

    assert len(program.cores) == 1, f"Expected 1 core, got {len(program.cores)}"
    assert n_tiu > 0, "Expected TIU commands > 0"
    assert n_dma > 0, "Expected DMA commands > 0"
    print("    [PASS] Instruction generation OK")

    # 4. 仿真 (Step 3)
    engine = G5SimEngine(chip)
    records = engine.simulate(program)

    print(f"\n[3] Simulation:")
    print(f"    records={len(records)}")

    tiu_records = [r for r in records if r.engine == "TIU"]
    dma_records = [r for r in records if r.engine == "DMA"]
    print(f"    TIU records={len(tiu_records)}")
    print(f"    DMA records={len(dma_records)}")

    if tiu_records:
        first_tiu = tiu_records[0]
        print(f"    First TIU: start={first_tiu.start_ns:.1f}ns, end={first_tiu.end_ns:.1f}ns, "
              f"duration={first_tiu.end_ns - first_tiu.start_ns:.1f}ns, flops={first_tiu.flops}")
    if dma_records:
        first_dma = dma_records[0]
        print(f"    First DMA: start={first_dma.start_ns:.1f}ns, end={first_dma.end_ns:.1f}ns, "
              f"duration={first_dma.end_ns - first_dma.start_ns:.1f}ns, bytes={first_dma.data_bytes}")

    assert len(records) > 0, "Expected records > 0"
    print("    [PASS] Simulation OK")

    # 5. 适配输出
    adapter = G5ResultAdapter(chip)
    result = adapter.convert(records)

    print(f"\n[4] Result Adaptation:")
    print(f"    step_metrics count={len(result.step_metrics)}")

    expected_flops = 2 * M * N * K
    actual_flops = result.step_metrics[0].flops
    print(f"    expected_flops={expected_flops}")
    print(f"    actual_flops={actual_flops}")

    agg = result.aggregates
    print(f"\n[5] Aggregates:")
    print(f"    total_time={agg.total_time:.4f} ms")
    print(f"    total_compute_time={agg.total_compute_time:.4f} ms")
    print(f"    total_wait_time={agg.total_wait_time:.4f} ms")
    print(f"    total_flops={agg.total_flops}")
    print(f"    total_bytes={agg.total_bytes}")
    print(f"    MFU={agg.mfu:.4%}")
    print(f"    MBU={agg.mbu:.4%}")
    print(f"    bottleneck={agg.bottleneck_summary}")

    assert actual_flops == expected_flops, (
        f"FLOPs mismatch: expected {expected_flops}, got {actual_flops}"
    )
    assert agg.total_time > 0, "total_time should be > 0"
    assert agg.mfu > 0, "MFU should be > 0"
    print("\n    [PASS] Result adaptation OK")

    print(f"\n{'=' * 60}")
    print(f"[PASS] All checks passed!")
    print(f"{'=' * 60}")

    return result


def test_import_only():
    """仅验证 import"""
    from perf_model.L3_mapping.g5.program import CoreProgram, TIUCommand, DMACommand
    from perf_model.L3_mapping.g5.instruction_tiler import tile_matmul, TilingResult
    from perf_model.L3_mapping.g5.instruction_emitter import G5InstructionEmitter
    from perf_model.L4_evaluation.g5.memory import lmem_budget_per_core
    from perf_model.L4_evaluation.g5.tiu import calc_tiu_latency
    from perf_model.L4_evaluation.g5.dma import calc_dma_latency
    from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine
    from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter
    print("[PASS] All imports successful")


if __name__ == "__main__":
    test_import_only()
    test_g5_matmul_single_core()
