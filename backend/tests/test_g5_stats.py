"""G5 统计框架测试

测试用例:
  1. StatGroup 基础功能 (scalar/vector/dump/reset)
  2. 层次化嵌套 dump
  3. 单核仿真统计验证 (MatMul 256x256x256)
  4. 多核仿真统计验证 (4 核独立 MatMul)
  5. SDMA 通信统计验证 (2 核 + SDMA)
"""

import sys
from pathlib import Path

# 添加 backend 到 sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml

from perf_model.L4_evaluation.g5.kernel.stats import ScalarStat, VectorStat, StatGroup
from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreInstructions, CoreProgram, TIUCommand, DMACommand,
    SDMACommand, TIUOpType, DMADirection, SDMACommandType,
)
from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def test_statgroup_basic():
    """测试 StatGroup 基础功能"""
    print("=" * 60)
    print("Test 1: StatGroup Basic (scalar/vector/dump/reset)")
    print("=" * 60)

    root = StatGroup("root")
    s1 = root.scalar("counter", "A counter")
    v1 = root.vector("dist", "A distribution")

    # Scalar 累加
    s1.inc(10)
    s1.inc(5)
    assert s1.value == 15.0, f"Expected 15.0, got {s1.value}"

    # Scalar set_max
    s2 = root.scalar("peak", "Peak value")
    s2.set_max(5)
    s2.set_max(3)
    s2.set_max(8)
    assert s2.value == 8.0, f"Expected 8.0, got {s2.value}"

    # Vector 累加
    v1.inc("A", 10)
    v1.inc("B", 20)
    v1.inc("A", 5)
    assert v1.bins == {"A": 15.0, "B": 20.0}, f"Got {v1.bins}"

    # dump
    result = root.dump()
    assert result["root.counter"] == 15.0
    assert result["root.peak"] == 8.0
    assert result["root.dist"] == {"A": 15.0, "B": 20.0}
    print(f"  dump result: {result}")

    # reset
    root.reset()
    assert s1.value == 0.0
    assert s2.value == 0.0
    assert v1.bins == {}
    print("  reset: all stats cleared")

    print("  [PASS] StatGroup basic OK\n")


def test_statgroup_hierarchy():
    """测试层次化嵌套"""
    print("=" * 60)
    print("Test 2: StatGroup Hierarchy")
    print("=" * 60)

    root = StatGroup("kernel")
    core0 = StatGroup("core0", parent=root)
    tiu = StatGroup("tiu", parent=core0)

    root_s = root.scalar("total_events", "events")
    core_s = core0.scalar("total_instr", "instructions")
    tiu_s = tiu.scalar("cmd_count", "tiu commands")
    tiu_v = tiu.vector("by_prec", "by precision")

    root_s.inc(100)
    core_s.inc(10)
    tiu_s.inc(5)
    tiu_v.inc("BF16", 3)
    tiu_v.inc("INT8", 2)

    result = root.dump()
    print(f"  dump keys: {sorted(result.keys())}")

    assert result["kernel.total_events"] == 100.0
    assert result["kernel.core0.total_instr"] == 10.0
    assert result["kernel.core0.tiu.cmd_count"] == 5.0
    assert result["kernel.core0.tiu.by_prec"] == {"BF16": 3.0, "INT8": 2.0}

    # 验证 key 格式
    for key in result:
        assert key.startswith("kernel."), f"Key '{key}' should start with 'kernel.'"

    print(f"  result: {result}")
    print("  [PASS] StatGroup hierarchy OK\n")


def test_single_core_stats():
    """测试单核 MatMul 统计"""
    print("=" * 60)
    print("Test 3: Single-core MatMul Stats (256x256x256, BF16)")
    print("=" * 60)

    chip = load_chip("SG2262")

    tiu_cmds = [
        TIUCommand(
            cmd_id=1, cmd_id_dep=2,
            op_type=TIUOpType.MM2_NN,
            result_addr=0x3000,
            operand_addrs=[0x1000, 0x2000],
            tile_m=256, tile_n=256, tile_k=256,
            precision="BF16",
            source_op_id="matmul",
        ),
    ]
    dma_cmds = [
        DMACommand(
            cmd_id=1, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0, dst_addr=0x1000,
            data_bytes=256 * 256 * 2,
            elem_bytes=2, source_op_id="matmul",
        ),
        DMACommand(
            cmd_id=2, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0x100000, dst_addr=0x2000,
            data_bytes=256 * 256 * 2,
            elem_bytes=2, source_op_id="matmul",
        ),
        DMACommand(
            cmd_id=3, cmd_id_dep=1,
            direction=DMADirection.LMEM_TO_DDR,
            src_addr=0x3000, dst_addr=0x200000,
            data_bytes=256 * 256 * 2,
            elem_bytes=2, source_op_id="matmul",
        ),
    ]

    core = CoreInstructions(core_id=0, tiu_cmds=tiu_cmds, dma_cmds=dma_cmds)
    program = CoreProgram(cores=[core])

    engine = G5SimEngine(chip)
    records = engine.simulate(program)
    stats = engine.get_stats()

    assert len(stats) > 0, "Stats should not be empty"

    # 验证 kernel 级统计
    assert stats["kernel.total_events"] > 0
    assert stats["kernel.total_sim_time_ns"] > 0
    print(f"  kernel.total_events = {stats['kernel.total_events']}")
    print(f"  kernel.total_sim_time_ns = {stats['kernel.total_sim_time_ns']:.2f}")

    # 验证 core0 级统计
    assert stats["kernel.core0.total_instructions"] == 4  # 1 TIU + 3 DMA
    assert stats["kernel.core0.cmd_count_by_engine"] == {"TIU": 1.0, "DMA": 3.0}
    print(f"  core0.total_instructions = {stats['kernel.core0.total_instructions']}")
    print(f"  core0.cmd_count_by_engine = {stats['kernel.core0.cmd_count_by_engine']}")

    # 验证 TIU 统计
    assert stats["kernel.core0.tiu.cmd_count"] == 1
    assert stats["kernel.core0.tiu.compute_cycles"] > 0
    assert stats["kernel.core0.tiu.init_cycles"] == 44  # TIU_INIT_CYCLES
    assert stats["kernel.core0.tiu.total_flops"] == 2 * 256 * 256 * 256
    assert "BF16" in stats["kernel.core0.tiu.cycles_by_prec"]
    assert "MM2_NN" in stats["kernel.core0.tiu.cmd_by_op"]
    print(f"  tiu.compute_cycles = {stats['kernel.core0.tiu.compute_cycles']}")
    print(f"  tiu.init_cycles = {stats['kernel.core0.tiu.init_cycles']}")
    print(f"  tiu.total_flops = {stats['kernel.core0.tiu.total_flops']}")
    print(f"  tiu.cycles_by_prec = {stats['kernel.core0.tiu.cycles_by_prec']}")

    # 验证 DMA 统计
    assert stats["kernel.core0.dma.cmd_count"] == 3
    assert stats["kernel.core0.dma.bytes_read"] == 2 * 256 * 256 * 2  # 2 loads
    assert stats["kernel.core0.dma.bytes_write"] == 256 * 256 * 2    # 1 store
    assert stats["kernel.core0.dma.startup_ns"] > 0
    assert stats["kernel.core0.dma.transfer_ns"] > 0
    assert "DDR_TO_LMEM" in stats["kernel.core0.dma.bytes_by_dir"]
    assert "LMEM_TO_DDR" in stats["kernel.core0.dma.bytes_by_dir"]
    print(f"  dma.cmd_count = {stats['kernel.core0.dma.cmd_count']}")
    print(f"  dma.bytes_read = {stats['kernel.core0.dma.bytes_read']}")
    print(f"  dma.bytes_write = {stats['kernel.core0.dma.bytes_write']}")
    print(f"  dma.bytes_by_dir = {stats['kernel.core0.dma.bytes_by_dir']}")

    print("  [PASS] Single-core stats OK\n")


def test_multicore_stats():
    """测试多核统计"""
    print("=" * 60)
    print("Test 4: 4-core Independent MatMul Stats")
    print("=" * 60)

    chip = load_chip("SG2262")

    def make_core(core_id):
        return CoreInstructions(
            core_id=core_id,
            tiu_cmds=[TIUCommand(
                cmd_id=1, cmd_id_dep=1,
                op_type=TIUOpType.MM2_NN,
                result_addr=0x3000,
                operand_addrs=[0x1000, 0x2000],
                tile_m=128, tile_n=128, tile_k=128,
                precision="BF16",
                source_op_id=f"mm_core{core_id}",
            )],
            dma_cmds=[DMACommand(
                cmd_id=1, cmd_id_dep=0,
                direction=DMADirection.DDR_TO_LMEM,
                src_addr=0, dst_addr=0x1000,
                data_bytes=128 * 128 * 2,
                elem_bytes=2,
                source_op_id=f"mm_core{core_id}",
            )],
        )

    cores = [make_core(i) for i in range(4)]
    program = CoreProgram(cores=cores)

    engine = G5SimEngine(chip)
    engine.simulate(program)
    stats = engine.get_stats()

    # 验证 4 个核都有统计
    for i in range(4):
        key = f"kernel.core{i}.total_instructions"
        assert key in stats, f"Missing {key}"
        assert stats[key] == 2  # 1 TIU + 1 DMA
        tiu_key = f"kernel.core{i}.tiu.cmd_count"
        assert stats[tiu_key] == 1
        print(f"  core{i}: total_instr={stats[key]}, tiu_cmd={stats[tiu_key]}")

    # 验证各核 TIU flops 相同
    flops_0 = stats["kernel.core0.tiu.total_flops"]
    for i in range(1, 4):
        flops_i = stats[f"kernel.core{i}.tiu.total_flops"]
        assert flops_i == flops_0, f"Core {i} flops mismatch"

    print(f"  All cores TIU flops = {flops_0}")
    print("  [PASS] 4-core stats OK\n")


def test_sdma_bus_stats():
    """测试 SDMA + Bus 统计"""
    print("=" * 60)
    print("Test 5: SDMA + Bus Stats (core 0 -> core 1)")
    print("=" * 60)

    chip = load_chip("SG2262")

    core0 = CoreInstructions(
        core_id=0,
        tiu_cmds=[TIUCommand(
            cmd_id=1, cmd_id_dep=0,
            op_type=TIUOpType.MM2_NN,
            result_addr=0x3000,
            operand_addrs=[0x1000, 0x2000],
            tile_m=64, tile_n=64, tile_k=64,
            precision="BF16",
            source_op_id="compute_core0",
        )],
        sdma_cmds=[SDMACommand(
            cmd_id=1, cmd_id_dep=1,
            dep_engine="tiu",
            cmd_type=SDMACommandType.TENSOR,
            src_addr=0x3000, dst_addr=0x4000,
            data_bytes=64 * 64 * 2,
            elem_bytes=2,
            src_core_id=0, dst_core_id=1,
            source_op_id="sdma_0to1",
        )],
    )
    core1 = CoreInstructions(core_id=1)
    program = CoreProgram(cores=[core0, core1])

    engine = G5SimEngine(chip)
    engine.simulate(program)
    stats = engine.get_stats()

    # 验证 SDMA 统计
    assert stats["kernel.core0.sdma.cmd_count"] == 1
    assert stats["kernel.core0.sdma.total_bytes"] == 64 * 64 * 2
    assert stats["kernel.core0.sdma.bus_latency_ns"] > 0  # core 0->1 有 bus delay
    assert stats["kernel.core0.sdma.transfer_ns"] > 0
    assert stats["kernel.core0.sdma.cmd_by_type"] == {"TENSOR": 1.0}
    print(f"  sdma.cmd_count = {stats['kernel.core0.sdma.cmd_count']}")
    print(f"  sdma.total_bytes = {stats['kernel.core0.sdma.total_bytes']}")
    print(f"  sdma.bus_latency_ns = {stats['kernel.core0.sdma.bus_latency_ns']:.2f}")
    print(f"  sdma.transfer_ns = {stats['kernel.core0.sdma.transfer_ns']:.2f}")

    # 验证 Bus 统计
    assert stats["kernel.bus.total_transfers"] == 1
    assert stats["kernel.bus.total_bytes"] == 64 * 64 * 2
    assert stats["kernel.bus.hop_total"] >= 1  # Manhattan distance >= 1
    print(f"  bus.total_transfers = {stats['kernel.bus.total_transfers']}")
    print(f"  bus.total_bytes = {stats['kernel.bus.total_bytes']}")
    print(f"  bus.hop_total = {stats['kernel.bus.hop_total']}")

    print("  [PASS] SDMA + Bus stats OK\n")


def test_adapter_passes_stats():
    """测试 Adapter 透传统计到 EngineResult"""
    print("=" * 60)
    print("Test 6: Adapter Stats Passthrough")
    print("=" * 60)

    from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter

    chip = load_chip("SG2262")

    core = CoreInstructions(
        core_id=0,
        tiu_cmds=[TIUCommand(
            cmd_id=1, cmd_id_dep=0,
            op_type=TIUOpType.MM2_NN,
            result_addr=0x3000,
            operand_addrs=[0x1000, 0x2000],
            tile_m=64, tile_n=64, tile_k=64,
            precision="BF16",
            source_op_id="test_op",
        )],
    )
    program = CoreProgram(cores=[core])

    engine = G5SimEngine(chip)
    records = engine.simulate(program)
    stats = engine.get_stats()

    adapter = G5ResultAdapter(chip)
    result = adapter.convert(records, stats=stats)

    # 验证 stats 在 trace_meta 中
    assert "stats" in result.trace_meta, "trace_meta should contain 'stats'"
    result_stats = result.trace_meta["stats"]
    assert "kernel.total_sim_time_ns" in result_stats
    assert "kernel.core0.tiu.cmd_count" in result_stats
    print(f"  trace_meta.stats keys count: {len(result_stats)}")
    print(f"  kernel.total_sim_time_ns = {result_stats['kernel.total_sim_time_ns']:.2f}")
    print(f"  core0.tiu.cmd_count = {result_stats['kernel.core0.tiu.cmd_count']}")

    print("  [PASS] Adapter stats passthrough OK\n")


if __name__ == "__main__":
    test_statgroup_basic()
    test_statgroup_hierarchy()
    test_single_core_stats()
    test_multicore_stats()
    test_sdma_bus_stats()
    test_adapter_passes_stats()

    print("=" * 60)
    print("[PASS] ALL STATS TESTS PASSED!")
    print("=" * 60)
