"""SingleChip 多核组装测试"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml
from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreProgram, CoreInstructions,
    TIUCommand, TIUOpType, DMACommand, DMADirection,
)
from perf_model.L4_evaluation.g5.top.single_chip import SingleChipSim


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def test_single_core_via_single_chip():
    """单核 via SingleChip"""
    chip = load_chip()

    program = CoreProgram(
        cores=[
            CoreInstructions(
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
            ),
        ],
        comm_schedule=[], metadata={},
    )

    sim = SingleChipSim(chip)
    records = sim.simulate(program)

    tiu_records = [r for r in records if r.engine == "TIU"]
    dma_records = [r for r in records if r.engine == "DMA"]
    assert len(tiu_records) == 1
    assert len(dma_records) == 1
    print(f"[PASS] single core: {len(records)} records")


def test_multi_core_independent():
    """2 核独立执行: 并行"""
    chip = load_chip()

    program = CoreProgram(
        cores=[
            CoreInstructions(
                core_id=i,
                tiu_cmds=[
                    TIUCommand(
                        cmd_id=1, cmd_id_dep=0, op_type=TIUOpType.MM2_NN,
                        result_addr=0, operand_addrs=[0, 0],
                        tile_m=16, tile_n=8, tile_k=32,
                        precision="BF16", source_op_id=f"core{i}_op",
                    ),
                ],
                dma_cmds=[], sdma_cmds=[], hau_cmds=[],
            )
            for i in range(2)
        ],
        comm_schedule=[], metadata={},
    )

    sim = SingleChipSim(chip)
    records = sim.simulate(program)

    assert len(records) == 2
    # 两核都从 t=0 开始
    assert records[0].start_ns == 0.0
    assert records[1].start_ns == 0.0
    # 相同计算, 相同结束时间
    assert records[0].end_ns == records[1].end_ns
    print(f"[PASS] 2-core independent: {records[0].end_ns:.1f}ns each")


def test_empty_program():
    """空程序返回空列表"""
    chip = load_chip()
    sim = SingleChipSim(chip)
    records = sim.simulate(CoreProgram(cores=[], comm_schedule=[], metadata={}))
    assert records == []
    print("[PASS] empty program")


if __name__ == "__main__":
    test_single_core_via_single_chip()
    test_multi_core_independent()
    test_empty_program()
    print("[PASS] All SingleChip tests passed")
