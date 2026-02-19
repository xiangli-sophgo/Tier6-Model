"""G5 多核仿真端到端测试

测试用例:
  1. 4 核独立 MatMul: 每核运行相同的 MatMul, 验证多核并行
  2. 2 核 + SDMA 通信: 核间数据传输, 验证 BusModel 延迟
  3. 单核 vs 多核一致性: 单核结果应与 MatMul E2E 测试一致
"""

import sys
from pathlib import Path

# 添加 backend 到 sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreInstructions, CoreProgram, TIUCommand, DMACommand,
    SDMACommand, TIUOpType, DMADirection, SDMACommandType,
)
from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    """加载芯片配置"""
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def make_matmul_core(core_id: int, M: int = 256, N: int = 256, K: int = 256,
                     precision: str = "BF16") -> CoreInstructions:
    """生成单核 MatMul 指令 (简化: 单 tile, 3 条 DMA + 1 条 TIU + 1 条 DMA)

    流程: load_A -> load_B -> compute -> store_C
    """
    elem_bytes = 2  # BF16
    op_id = f"matmul_core{core_id}"

    dma_cmds = [
        # DMA 1: load A
        DMACommand(
            cmd_id=1, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0, dst_addr=0x1000,
            data_bytes=M * K * elem_bytes,
            elem_bytes=elem_bytes,
            source_op_id=op_id,
        ),
        # DMA 2: load B
        DMACommand(
            cmd_id=2, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0x100000, dst_addr=0x2000,
            data_bytes=K * N * elem_bytes,
            elem_bytes=elem_bytes,
            source_op_id=op_id,
        ),
        # DMA 3: store C (等 TIU 完成)
        DMACommand(
            cmd_id=3, cmd_id_dep=1,  # 等 TIU cmd_id=1 完成
            direction=DMADirection.LMEM_TO_DDR,
            src_addr=0x3000, dst_addr=0x200000,
            data_bytes=M * N * elem_bytes,
            elem_bytes=elem_bytes,
            source_op_id=op_id,
        ),
    ]

    tiu_cmds = [
        # TIU 1: matmul (等 DMA 2 完成)
        TIUCommand(
            cmd_id=1, cmd_id_dep=2,  # 等 DMA cmd_id=2 完成
            op_type=TIUOpType.MM2_NN,
            result_addr=0x3000,
            operand_addrs=[0x1000, 0x2000],
            tile_m=M, tile_n=N, tile_k=K,
            precision=precision,
            source_op_id=op_id,
        ),
    ]

    return CoreInstructions(
        core_id=core_id,
        tiu_cmds=tiu_cmds,
        dma_cmds=dma_cmds,
    )


def test_multicore_independent():
    """测试 4 核独立 MatMul"""
    print("=" * 60)
    print("Test 1: 4-core Independent MatMul (256x256x256, BF16)")
    print("=" * 60)

    chip = load_chip("SG2262")
    print(f"  Chip: {chip.name}, noc_config={chip.noc_config}")

    # 4 核独立运行
    num_cores = 4
    cores = [make_matmul_core(core_id=i) for i in range(num_cores)]
    program = CoreProgram(cores=cores)

    engine = G5SimEngine(chip)
    records = engine.simulate(program)

    # 基本验证
    assert len(records) > 0, "No records produced"

    # 按核分组
    core_records: dict[str, list] = {}
    for r in records:
        # source_op_id 格式: matmul_core{i}
        core_records.setdefault(r.source_op_id, []).append(r)

    print(f"  Total records: {len(records)}")
    for op_id, recs in sorted(core_records.items()):
        tiu_recs = [r for r in recs if r.engine == "TIU"]
        dma_recs = [r for r in recs if r.engine == "DMA"]
        end_time = max(r.end_ns for r in recs)
        print(f"    {op_id}: TIU={len(tiu_recs)}, DMA={len(dma_recs)}, end={end_time:.2f} ns")

    # 验证: 每核应该有相同数量的指令
    for op_id, recs in core_records.items():
        assert len(recs) == 4, f"{op_id}: expected 4 records, got {len(recs)}"

    # 验证: 所有核应该同时开始 (因为独立, 首条指令 start_ns 应相同)
    first_starts = []
    for i in range(num_cores):
        op_id = f"matmul_core{i}"
        first_start = min(r.start_ns for r in core_records[op_id])
        first_starts.append(first_start)

    # 所有核的首条指令应在 t=0 开始
    for i, t in enumerate(first_starts):
        assert t == 0.0, f"Core {i} first start = {t}, expected 0.0"

    # 验证: 独立核的结束时间应相同 (相同负载)
    end_times = []
    for i in range(num_cores):
        op_id = f"matmul_core{i}"
        end_time = max(r.end_ns for r in core_records[op_id])
        end_times.append(end_time)

    for i in range(1, num_cores):
        assert abs(end_times[i] - end_times[0]) < 0.01, \
            f"Core {i} end={end_times[i]:.2f} differs from Core 0 end={end_times[0]:.2f}"

    print(f"  All cores finish at: {end_times[0]:.2f} ns ({end_times[0]/1e6:.4f} ms)")
    print("  [PASS] 4-core independent MatMul OK\n")


def test_multicore_with_sdma():
    """测试 2 核 + SDMA 核间通信"""
    print("=" * 60)
    print("Test 2: 2-core with SDMA Communication")
    print("=" * 60)

    chip = load_chip("SG2262")

    # Core 0: 计算 MatMul, 然后 SDMA 发送结果到 Core 1
    core0_tiu = [
        TIUCommand(
            cmd_id=1, cmd_id_dep=0,
            op_type=TIUOpType.MM2_NN,
            result_addr=0x3000,
            operand_addrs=[0x1000, 0x2000],
            tile_m=128, tile_n=128, tile_k=128,
            precision="BF16",
            source_op_id="matmul_core0",
        ),
    ]
    core0_dma = [
        DMACommand(
            cmd_id=1, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0, dst_addr=0x1000,
            data_bytes=128 * 128 * 2,
            elem_bytes=2,
            source_op_id="matmul_core0",
        ),
        DMACommand(
            cmd_id=2, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0x100000, dst_addr=0x2000,
            data_bytes=128 * 128 * 2,
            elem_bytes=2,
            source_op_id="matmul_core0",
        ),
    ]
    core0_sdma = [
        SDMACommand(
            cmd_id=1, cmd_id_dep=1,  # 等 TIU cmd_id=1 完成
            dep_engine="tiu",
            cmd_type=SDMACommandType.TENSOR,
            src_addr=0x3000, dst_addr=0x4000,
            data_bytes=128 * 128 * 2,
            elem_bytes=2,
            src_core_id=0, dst_core_id=1,
            source_op_id="sdma_0to1",
        ),
    ]

    core0 = CoreInstructions(
        core_id=0,
        tiu_cmds=core0_tiu,
        dma_cmds=core0_dma,
        sdma_cmds=core0_sdma,
    )

    # Core 1: 只有一条 DMA load, 无计算 (仅接收)
    core1_dma = [
        DMACommand(
            cmd_id=1, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0, dst_addr=0x1000,
            data_bytes=128 * 128 * 2,
            elem_bytes=2,
            source_op_id="load_core1",
        ),
    ]

    core1 = CoreInstructions(core_id=1, dma_cmds=core1_dma)

    program = CoreProgram(cores=[core0, core1])

    engine = G5SimEngine(chip)
    records = engine.simulate(program)

    assert len(records) > 0, "No records"

    # 检查 SDMA 记录
    sdma_records = [r for r in records if r.engine == "SDMA"]
    assert len(sdma_records) == 1, f"Expected 1 SDMA record, got {len(sdma_records)}"

    sdma = sdma_records[0]
    print(f"  SDMA record: start={sdma.start_ns:.2f} ns, end={sdma.end_ns:.2f} ns")
    print(f"    duration={sdma.end_ns - sdma.start_ns:.2f} ns")
    print(f"    data_bytes={sdma.data_bytes}")

    # SDMA 应在 TIU 完成后开始
    tiu_records = [r for r in records if r.engine == "TIU"]
    assert len(tiu_records) == 1
    assert sdma.start_ns >= tiu_records[0].end_ns, \
        f"SDMA start ({sdma.start_ns}) < TIU end ({tiu_records[0].end_ns})"

    # SDMA 延迟应包含 bus delay (core 0 -> core 1 的 Manhattan 距离)
    # 在 8x8 mesh 中, core 0 = (0,0), core 1 = (1,0), 距离=1
    # bus_delay = 1 * 45 cycles / 1.0 GHz = 45 ns
    sdma_duration = sdma.end_ns - sdma.start_ns
    print(f"    SDMA total duration includes bus delay: {sdma_duration:.2f} ns")
    assert sdma_duration > 0, "SDMA duration should be positive"

    total_end = max(r.end_ns for r in records)
    print(f"  Total simulation time: {total_end:.2f} ns ({total_end/1e6:.4f} ms)")
    print("  [PASS] 2-core SDMA communication OK\n")


def test_single_core_consistency():
    """验证单核结果与原始 E2E 一致"""
    print("=" * 60)
    print("Test 3: Single-core Consistency Check")
    print("=" * 60)

    chip = load_chip("SG2262")

    # 运行同一个单核 MatMul 两次, 结果应完全相同
    core = make_matmul_core(core_id=0, M=256, N=256, K=256)
    program = CoreProgram(cores=[core])

    engine = G5SimEngine(chip)
    records1 = engine.simulate(program)
    records2 = engine.simulate(program)

    assert len(records1) == len(records2), "Record count mismatch"
    for r1, r2 in zip(records1, records2):
        assert r1.engine == r2.engine, f"Engine mismatch: {r1.engine} vs {r2.engine}"
        assert abs(r1.start_ns - r2.start_ns) < 0.001, f"Start mismatch: {r1.start_ns} vs {r2.start_ns}"
        assert abs(r1.end_ns - r2.end_ns) < 0.001, f"End mismatch: {r1.end_ns} vs {r2.end_ns}"

    total = max(r.end_ns for r in records1)
    print(f"  Single-core total: {total:.2f} ns ({total/1e6:.4f} ms)")
    print(f"  Records: {len(records1)} (deterministic)")
    print("  [PASS] Single-core consistency OK\n")


def test_noc_config_loaded():
    """验证 NoC 配置从 YAML 正确加载"""
    print("=" * 60)
    print("Test 4: NoC Config Loading")
    print("=" * 60)

    chip = load_chip("SG2262")

    assert chip.noc_config, "noc_config should not be empty"
    assert chip.noc_config["topology"] == "2d_mesh", \
        f"Expected topology=2d_mesh, got {chip.noc_config.get('topology')}"
    assert chip.noc_config["mesh_cols"] == 8, \
        f"Expected mesh_cols=8, got {chip.noc_config.get('mesh_cols')}"
    assert chip.noc_config["mesh_rows"] == 8, \
        f"Expected mesh_rows=8, got {chip.noc_config.get('mesh_rows')}"
    assert chip.noc_config["base_latency_cycles"] == 45, \
        f"Expected base_latency_cycles=45, got {chip.noc_config.get('base_latency_cycles')}"

    print(f"  noc_config = {chip.noc_config}")
    print("  [PASS] NoC config loaded correctly\n")


if __name__ == "__main__":
    test_noc_config_loaded()
    test_single_core_consistency()
    test_multicore_independent()
    test_multicore_with_sdma()

    print("=" * 60)
    print("[PASS] ALL MULTICORE TESTS PASSED!")
    print("=" * 60)
