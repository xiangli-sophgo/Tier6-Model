"""G5 指令级仿真器 MoE 流程端到端测试

测试用例:
    1. HAU Top-K 独立测试
    2. SDMA 传输独立测试
    3. MoE 完整流程: Gating(TIU) -> Top-K(HAU) -> Dispatch(SDMA) -> Expert(TIU) -> Combine(SDMA)
    4. MatMul 回归测试 (确认 Step 2-3 未被破坏)
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
from perf_model.L3_mapping.g5.program import (
    HAUCommand,
    HAUMsgAction,
    HAUOpType,
    SDMACommand,
    SDMACommandType,
)
from perf_model.L3_mapping.g5.instruction_emitter import G5InstructionEmitter
from perf_model.L4_evaluation.g5.hau import calc_hau_latency
from perf_model.L4_evaluation.g5.sdma import calc_sdma_latency
from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine
from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter


@dataclass
class MockDistributedOp:
    """模拟 DistributedOp"""
    op_id: str
    op_type: str
    local_shape: dict[str, int] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    deps: list[str] = field(default_factory=list)
    chip_ids: list[int] = field(default_factory=lambda: [0])
    comm_bytes: int = 0


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    """加载芯片配置"""
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


# ========== Test 1: HAU Top-K ==========

def test_hau_topk():
    """HAU Top-K 256 experts, K=8"""
    print("=" * 60)
    print("Test 1: HAU Top-K (256 experts, K=8)")
    print("=" * 60)

    chip = load_chip("SG2262")
    cmd = HAUCommand(
        cmd_id=1,
        cmd_id_dep=0,
        dep_engine="tiu",
        op_type=HAUOpType.TOP_K,
        src_addr=0,
        dst_addr=1024,
        num_elements=256,
        top_k=8,
        data_format="BF16",
        msg_action=HAUMsgAction.SEND,
        msg_id=1,
        source_op_id="topk0",
    )
    result = calc_hau_latency(cmd, chip)
    assert result.latency_ns > 0, "latency should be > 0"
    assert result.cycles > 0, "cycles should be > 0"

    # ceil(256/16) * ceil(log2(8)) * 1 + 20 = 16 * 3 + 20 = 68 cycles
    expected_cycles = 68
    assert result.cycles == expected_cycles, (
        f"Expected {expected_cycles} cycles, got {result.cycles}"
    )
    # HAU 使用 tiu_frequency_ghz (1.0 GHz for SG2262)
    tiu_freq = chip.get_tiu_frequency()
    expected_latency = expected_cycles / tiu_freq
    assert abs(result.latency_ns - expected_latency) < 0.01, (
        f"Expected {expected_latency:.2f} ns, got {result.latency_ns:.2f} ns"
    )

    print(f"  HAU Top-K latency: {result.latency_ns:.1f} ns ({result.cycles} cycles)")
    print("  [PASS] HAU Top-K test passed")
    return result


# ========== Test 2: SDMA Transfer ==========

def test_sdma_transfer():
    """SDMA 1MB transfer"""
    print("\n" + "=" * 60)
    print("Test 2: SDMA 1MB Transfer")
    print("=" * 60)

    chip = load_chip("SG2262")
    cmd = SDMACommand(
        cmd_id=1,
        cmd_id_dep=0,
        dep_engine="tiu",
        cmd_type=SDMACommandType.TENSOR,
        src_addr=0,
        dst_addr=0,
        data_bytes=1024 * 1024,
        elem_bytes=2,
        src_core_id=0,
        dst_core_id=0,
        source_op_id="sdma0",
    )
    result = calc_sdma_latency(cmd, chip)
    assert result.latency_ns > 0, "latency should be > 0"
    assert result.data_bytes == 1024 * 1024, "data_bytes mismatch"

    # 120 + 1048576 / (64 * 0.85) = 120 + 19277.6 = 19397.6 ns
    expected = 120.0 + 1048576.0 / (64.0 * 0.85)
    assert abs(result.latency_ns - expected) < 1.0, (
        f"Expected ~{expected:.1f} ns, got {result.latency_ns:.1f} ns"
    )

    print(f"  SDMA 1MB latency: {result.latency_ns:.1f} ns")
    print("  [PASS] SDMA transfer test passed")
    return result


# ========== Test 3: MoE E2E Flow ==========

def test_g5_moe_flow():
    """MoE: Gating -> Top-K -> Dispatch -> Expert -> Combine"""
    print("\n" + "=" * 60)
    print("Test 3: MoE E2E Flow")
    print("=" * 60)

    chip = load_chip("SG2262")

    # 使用 M=32 (batch tokens) 确保 >= cube_m=16
    batch_tokens = 32
    hidden = 4096
    num_experts = 256
    expert_ffn_dim = 11008

    ops = [
        # 1. Gating MatMul: [32, 4096] x [4096, 256] -> [32, 256]
        MockDistributedOp(
            op_id="gating",
            op_type="matmul",
            local_shape={"M": batch_tokens, "N": num_experts, "K": hidden},
            attrs={
                "input_dtype_bytes": "2",
                "moe_gating": "true",
                "num_experts": str(num_experts),
                "top_k": "8",
            },
        ),
        # 2. Dispatch (SDMA SCATTER)
        MockDistributedOp(
            op_id="dispatch",
            op_type="dispatch",
            comm_bytes=batch_tokens * hidden * 2,
        ),
        # 3. Expert FFN: [32, 4096] x [4096, 11008] -> [32, 11008]
        MockDistributedOp(
            op_id="expert_gate",
            op_type="matmul",
            local_shape={"M": batch_tokens, "N": expert_ffn_dim, "K": hidden},
            attrs={"input_dtype_bytes": "2"},
        ),
        # 4. Combine (SDMA GATHER)
        MockDistributedOp(
            op_id="combine",
            op_type="combine",
            comm_bytes=batch_tokens * hidden * 2,
        ),
    ]

    # Instruction generation
    emitter = G5InstructionEmitter(chip)
    program = emitter.emit(ops)

    core = program.cores[0]
    n_tiu = len(core.tiu_cmds)
    n_dma = len(core.dma_cmds)
    n_hau = len(core.hau_cmds)
    n_sdma = len(core.sdma_cmds)

    print(f"  Instruction counts: TIU={n_tiu}, DMA={n_dma}, HAU={n_hau}, SDMA={n_sdma}")

    assert n_tiu > 0, "Should have TIU commands (gating + expert)"
    assert n_dma > 0, "Should have DMA commands (data load/store)"
    assert n_hau > 0, "Should have HAU commands (Top-K)"
    assert n_sdma > 0, "Should have SDMA commands (dispatch + combine)"
    print("  [PASS] Instruction generation OK")

    # Simulation
    engine = G5SimEngine(chip)
    records = engine.simulate(program)

    tiu_recs = [r for r in records if r.engine == "TIU"]
    dma_recs = [r for r in records if r.engine == "DMA"]
    sdma_recs = [r for r in records if r.engine == "SDMA"]
    hau_recs = [r for r in records if r.engine == "HAU"]

    print(f"  SimRecord counts: TIU={len(tiu_recs)}, DMA={len(dma_recs)}, "
          f"SDMA={len(sdma_recs)}, HAU={len(hau_recs)}")

    assert len(records) > 0, "Expected records > 0"
    assert len(tiu_recs) > 0, "Expected TIU records"
    assert len(hau_recs) > 0, "Expected HAU records"
    assert len(sdma_recs) > 0, "Expected SDMA records"
    print("  [PASS] Simulation OK")

    # Adaptation
    adapter = G5ResultAdapter(chip)
    result = adapter.convert(records)

    agg = result.aggregates
    print(f"\n  Aggregates:")
    print(f"    total_time={agg.total_time:.4f} ms")
    print(f"    compute={agg.total_compute_time:.4f} ms (TIU + HAU)")
    print(f"    comm={agg.total_comm_time:.4f} ms (SDMA)")
    print(f"    wait={agg.total_wait_time:.4f} ms")
    print(f"    MFU={agg.mfu:.4%}")

    assert agg.total_time > 0, "total_time should be > 0"
    assert agg.total_compute_time > 0, "compute_time should be > 0 (TIU + HAU)"
    assert agg.total_comm_time > 0, "comm_time should be > 0 (SDMA)"
    print("\n  [PASS] MoE E2E flow test passed")

    return result


# ========== Test 4: MatMul Regression ==========

def test_matmul_regression():
    """MatMul 512x2048x4096 回归测试"""
    print("\n" + "=" * 60)
    print("Test 4: MatMul Regression (512x2048x4096)")
    print("=" * 60)

    chip = load_chip("SG2262")

    M, N, K = 512, 2048, 4096
    op = MockDistributedOp(
        op_id="mm0",
        op_type="matmul",
        local_shape={"M": M, "N": N, "K": K},
        attrs={"input_dtype_bytes": "2"},
    )

    emitter = G5InstructionEmitter(chip)
    program = emitter.emit([op])

    n_tiu = program.total_tiu_cmds()
    n_dma = program.total_dma_cmds()
    n_sdma = program.total_sdma_cmds()
    n_hau = program.total_hau_cmds()
    print(f"  TIU={n_tiu}, DMA={n_dma}, SDMA={n_sdma}, HAU={n_hau}")

    assert n_tiu > 0, "Expected TIU commands > 0"
    assert n_dma > 0, "Expected DMA commands > 0"
    assert n_sdma == 0, "Expected no SDMA commands for pure MatMul"
    assert n_hau == 0, "Expected no HAU commands for pure MatMul"

    engine = G5SimEngine(chip)
    records = engine.simulate(program)
    assert len(records) > 0, "Expected records > 0"

    adapter = G5ResultAdapter(chip)
    result = adapter.convert(records)

    expected_flops = 2 * M * N * K
    actual_flops = result.step_metrics[0].flops
    assert actual_flops == expected_flops, (
        f"FLOPs mismatch: expected {expected_flops}, got {actual_flops}"
    )

    agg = result.aggregates
    assert agg.total_time > 0, "total_time should be > 0"
    assert agg.mfu > 0, "MFU should be > 0"
    assert agg.total_comm_time == 0.0, "comm_time should be 0 for pure MatMul"

    print(f"  total_time={agg.total_time:.4f} ms, MFU={agg.mfu:.4%}")
    print("  [PASS] MatMul regression test passed")

    return result


# ========== Test 5: Import Verification ==========

def test_import_all():
    """验证所有新模块可以正确 import"""
    print("\n" + "=" * 60)
    print("Test 5: Import Verification")
    print("=" * 60)

    from perf_model.L3_mapping.g5.program import (
        CoreProgram, TIUCommand, DMACommand, SDMACommand, HAUCommand,
        HAUOpType, SDMACommandType, HAUMsgAction,
    )
    from perf_model.L3_mapping.g5.instruction_emitter import G5InstructionEmitter
    from perf_model.L4_evaluation.g5.hau import calc_hau_latency, HAUResult
    from perf_model.L4_evaluation.g5.sdma import calc_sdma_latency, SDMAResult
    from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine, SimRecord
    from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter

    print("  [PASS] All imports successful")


if __name__ == "__main__":
    test_import_all()
    test_hau_topk()
    test_sdma_transfer()
    test_g5_moe_flow()
    test_matmul_regression()
    print("\n" + "=" * 60)
    print("[PASS] ALL TESTS PASSED!")
    print("=" * 60)
