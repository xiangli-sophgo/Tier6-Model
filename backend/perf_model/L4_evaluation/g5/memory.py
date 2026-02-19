"""LMEM + DDR 最小化存储模型 (G5 指令级仿真模式)

提供:
    - lmem_budget_per_core: 单核 LMEM 可用预算计算
    - validate_lmem_usage: 校验 LMEM 使用量不超预算

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

from perf_model.L2_arch.chip import ChipSpecImpl


def lmem_budget_per_core(chip: ChipSpecImpl) -> int:
    """计算单核 LMEM 可用预算 (bytes)

    公式: per_core = lmem_total / core_count * utilization

    Args:
        chip: 芯片规格

    Returns:
        单核可用 LMEM 字节数
    """
    lmem_total = chip.get_total_sram()
    if lmem_total <= 0:
        raise ValueError(f"Chip '{chip.name}' has no LMEM (lmem total={lmem_total})")
    if chip.core_count <= 0:
        raise ValueError(f"Chip '{chip.name}' has invalid core_count={chip.core_count}")

    per_core = lmem_total / chip.core_count
    budget = int(per_core * chip.sram_utilization)

    if budget <= 0:
        raise ValueError(
            f"LMEM budget is 0 for chip '{chip.name}': "
            f"lmem_total={lmem_total}, core_count={chip.core_count}, "
            f"sram_utilization={chip.sram_utilization}"
        )
    return budget


def validate_lmem_usage(chip: ChipSpecImpl, usage_bytes: int) -> None:
    """校验 LMEM 使用量不超预算

    Args:
        chip: 芯片规格
        usage_bytes: 实际使用字节数

    Raises:
        ValueError: 如果 usage 超出预算
    """
    budget = lmem_budget_per_core(chip)
    if usage_bytes > budget:
        raise ValueError(
            f"LMEM usage ({usage_bytes} bytes) exceeds budget ({budget} bytes) "
            f"for chip '{chip.name}' "
            f"(lmem_total={chip.get_total_sram()}, "
            f"core_count={chip.core_count}, "
            f"utilization={chip.sram_utilization})"
        )
