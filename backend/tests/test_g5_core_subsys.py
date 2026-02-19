"""CoreSubsys + BusModel 单元测试"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml
from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreInstructions, TIUCommand, TIUOpType,
    DMACommand, DMADirection,
)
from perf_model.L4_evaluation.g5.kernel import SimKernel
from perf_model.L4_evaluation.g5.chip.core_subsys import CoreSubsys
from perf_model.L4_evaluation.g5.chip.bus import BusModel
from perf_model.L4_evaluation.g5.sim_engine import SimRecord


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


# ==================== BusModel 测试 ====================


def test_manhattan_distance():
    """Manhattan 距离计算"""
    bus = BusModel(core_count=64, mesh_dims=(8, 8),
                   base_latency_cycles=45, frequency_ghz=1.0)
    assert bus.manhattan_distance(0, 0) == 0
    d = bus.manhattan_distance(0, 1)
    assert d > 0
    print(f"[PASS] manhattan_distance: d(0,1)={d}")


def test_same_core_zero_delay():
    """同核通信零延迟"""
    bus = BusModel(core_count=64, mesh_dims=(8, 8),
                   base_latency_cycles=45, frequency_ghz=1.0)
    delay = bus.get_delay_ns(src_core=0, dst_core=0, data_bytes=1024)
    assert delay == 0.0
    print("[PASS] same core -> 0 delay")


def test_cross_core_delay():
    """跨核通信有延迟"""
    bus = BusModel(core_count=64, mesh_dims=(8, 8),
                   base_latency_cycles=45, frequency_ghz=1.0)
    delay_diag = bus.get_delay_ns(src_core=0, dst_core=63, data_bytes=1024)
    delay_adj = bus.get_delay_ns(src_core=0, dst_core=1, data_bytes=1024)
    assert delay_diag > 0.0
    assert delay_diag > delay_adj
    print(f"[PASS] cross-core: adj={delay_adj:.1f}ns, diag={delay_diag:.1f}ns")


def test_small_mesh():
    """2x2 小网格验证"""
    bus = BusModel(core_count=4, mesh_dims=(2, 2),
                   base_latency_cycles=10, frequency_ghz=1.0)
    assert bus.manhattan_distance(0, 1) == 1
    assert bus.manhattan_distance(0, 3) == 2
    assert bus.get_delay_ns(0, 1, 0) == 10.0
    print("[PASS] 2x2 mesh OK")


# ==================== CoreSubsys 测试 ====================


def test_core_subsys_single_tiu():
    """单条 TIU 指令仿真"""
    chip = load_chip()
    kernel = SimKernel()
    kernel.add_clock("tpu", frequency_ghz=chip.get_tiu_frequency())

    core = CoreSubsys(kernel=kernel, chip=chip, core_id=0, clock_name="tpu")
    instr = CoreInstructions(
        core_id=0,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0, op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16", source_op_id="test_op",
            ),
        ],
        dma_cmds=[], sdma_cmds=[], hau_cmds=[],
    )
    core.load_instructions(instr)
    kernel.run()

    records = core.get_records()
    assert len(records) == 1
    r = records[0]
    assert r.engine == "TIU"
    assert r.cmd_id == 1
    assert r.start_ns == 0.0
    assert r.end_ns > 0.0
    assert r.flops == 2 * 16 * 8 * 32
    print(f"[PASS] single TIU: {r.end_ns:.1f}ns, {r.flops} flops")


def test_core_subsys_tiu_dma_overlap():
    """TIU 和 DMA 并行执行 (无依赖)"""
    chip = load_chip()
    kernel = SimKernel()
    kernel.add_clock("tpu", frequency_ghz=chip.get_tiu_frequency())

    core = CoreSubsys(kernel=kernel, chip=chip, core_id=0, clock_name="tpu")
    instr = CoreInstructions(
        core_id=0,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0, op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16", source_op_id="op0",
            ),
        ],
        dma_cmds=[
            DMACommand(
                cmd_id=1, cmd_id_dep=0,
                direction=DMADirection.DDR_TO_LMEM,
                src_addr=0, dst_addr=0,
                data_bytes=4096, elem_bytes=2, source_op_id="op0",
            ),
        ],
        sdma_cmds=[], hau_cmds=[],
    )
    core.load_instructions(instr)
    kernel.run()

    records = core.get_records()
    assert len(records) == 2
    tiu_r = [r for r in records if r.engine == "TIU"][0]
    dma_r = [r for r in records if r.engine == "DMA"][0]
    assert tiu_r.start_ns == 0.0
    assert dma_r.start_ns == 0.0
    print(f"[PASS] overlap: TIU=[0, {tiu_r.end_ns:.1f}], DMA=[0, {dma_r.end_ns:.1f}]")


def test_core_subsys_dependency_chain():
    """TIU -> DMA 依赖链"""
    chip = load_chip()
    kernel = SimKernel()
    kernel.add_clock("tpu", frequency_ghz=chip.get_tiu_frequency())

    core = CoreSubsys(kernel=kernel, chip=chip, core_id=0, clock_name="tpu")
    instr = CoreInstructions(
        core_id=0,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0, op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16", source_op_id="op0",
            ),
        ],
        dma_cmds=[
            DMACommand(
                cmd_id=1, cmd_id_dep=1,
                direction=DMADirection.LMEM_TO_DDR,
                src_addr=0, dst_addr=0,
                data_bytes=4096, elem_bytes=2, source_op_id="op0",
            ),
        ],
        sdma_cmds=[], hau_cmds=[],
    )
    core.load_instructions(instr)
    kernel.run()

    records = core.get_records()
    tiu_r = [r for r in records if r.engine == "TIU"][0]
    dma_r = [r for r in records if r.engine == "DMA"][0]
    assert dma_r.start_ns >= tiu_r.end_ns, (
        f"DMA start {dma_r.start_ns} should be >= TIU end {tiu_r.end_ns}"
    )
    print(f"[PASS] dep chain: TIU end={tiu_r.end_ns:.1f}, DMA start={dma_r.start_ns:.1f}")


if __name__ == "__main__":
    test_manhattan_distance()
    test_same_core_zero_delay()
    test_cross_core_delay()
    test_small_mesh()
    test_core_subsys_single_tiu()
    test_core_subsys_tiu_dma_overlap()
    test_core_subsys_dependency_chain()
    print("[PASS] All CoreSubsys + BusModel tests passed")
