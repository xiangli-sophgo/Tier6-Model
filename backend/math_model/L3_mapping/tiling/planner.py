"""TilingPlanner - 片内映射实现."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from math_model.L2_arch.chip import ChipSpecImpl
from math_model.L3_mapping.plan.distributed_model import CommType, DistributedModel, DistributedOp, NodeRole
from math_model.L3_mapping.tiling.evaluators import (
    ElementwiseTilingEvaluator,
    FA2TilingEvaluator,
    L4TileEvaluator,
    MatmulTilingEvaluator,
    TilingEvaluatorRegistry,
)
from math_model.L1_workload.specs import TileConfig


@dataclass
class TilePlan:
    """片内映射结果

    Attributes:
        tile_configs: op_id -> TileConfig
        kernel_configs: op_id -> kernel 配置占位
        intra_chip_comms: 片内通信需求
    """

    tile_configs: dict[str, TileConfig] = field(default_factory=dict)
    kernel_configs: dict[str, dict[str, str]] = field(default_factory=dict)
    intra_chip_comms: list[DistributedOp] = field(default_factory=list)


class TilingPlanner:
    """片内映射规划器（按设计文档的轻量剪枝实现）"""

    def __init__(
        self,
        chip: ChipSpecImpl,
        l4_evaluator: L4TileEvaluator
        | Callable[[DistributedOp, TileConfig, ChipSpecImpl], dict[str, int]]
        | None = None,
    ) -> None:
        self.chip = chip
        self.registry = TilingEvaluatorRegistry(
            [
                MatmulTilingEvaluator(chip),
                FA2TilingEvaluator(chip),
                ElementwiseTilingEvaluator(chip),
            ]
        )
        self.l4_evaluator = l4_evaluator
        self._tile_cache: dict[tuple, tuple[TileConfig, dict[str, int]]] = {}

    def plan(self, dist_model: DistributedModel) -> TilePlan:
        """生成片内映射结果"""
        plan = TilePlan()
        lmem_budget = self._get_lmem_budget()

        for op in dist_model.get_compute_ops():
            if not op.local_shape:
                continue

            cache_key = self._build_cache_key(op)
            cached = self._tile_cache.get(cache_key)
            if cached is None:
                evaluator = self.registry.select(op)
                if evaluator is not None:
                    result = evaluator.select_tile(op, lmem_budget)
                else:
                    result = None
                if result is None:
                    tile_config, tile_meta = self._fallback_tile(op)
                else:
                    tile_config, tile_meta = result
                self._tile_cache[cache_key] = (tile_config, tile_meta)
            else:
                tile_config, tile_meta = cached

            if self.l4_evaluator is not None:
                if hasattr(self.l4_evaluator, "evaluate_tile"):
                    refined = self.l4_evaluator.evaluate_tile(op, tile_config, self.chip)
                else:
                    refined = self.l4_evaluator(op, tile_config, self.chip)
                if refined:
                    tile_meta = {**tile_meta, **refined}
            plan.tile_configs[op.op_id] = tile_config
            plan.kernel_configs[op.op_id] = {
                "kernel": "default",
                "traffic": str(tile_meta.get("traffic", 0)),
                "lmem_bytes": str(tile_meta.get("lmem_bytes", 0)),
            }

            if self._needs_intra_reduce(op.local_shape, tile_config):
                plan.intra_chip_comms.append(self._build_intra_reduce(op, tile_config))

        return plan

    def _get_lmem_budget(self) -> int:
        """获取片内 LMEM 预算（按 0.45 * LMEM）"""
        try:
            lmem = self.chip.memory_hierarchy.get_level("lmem")
            per_core = lmem.capacity_bytes / self.chip.core_count if self.chip.core_count else 0
            return int(per_core * 0.45)
        except KeyError:
            total = self.chip.get_total_sram()
            per_core = total / self.chip.core_count if self.chip.core_count else total
            return int(per_core * 0.45) if per_core else 0

    def _fallback_tile(self, op: DistributedOp) -> tuple[TileConfig, dict[str, int]]:
        cube_m = getattr(self.chip, "cube_m", 0) or 16
        cube_n = getattr(self.chip, "cube_n", 0) or 8
        cube_k = getattr(self.chip, "cube_k", 0) or 32
        tile_config = TileConfig(tile_m=cube_m, tile_n=cube_n, tile_k=cube_k)
        return tile_config, {"traffic": 0, "lmem_bytes": 0}

    def _build_cache_key(self, op: DistributedOp) -> tuple:
        shape_items = tuple(sorted(op.local_shape.items()))
        dtype_key = (
            op.attrs.get("input_dtype_bytes"),
            op.attrs.get("weight_dtype_bytes"),
            op.attrs.get("output_dtype_bytes"),
            op.attrs.get("accum_dtype_bytes"),
        )
        chip_key = (
            self.chip.name,
            self.chip.core_count,
            self.chip.lane_per_core,
        )
        return (op.op_type, shape_items, dtype_key, chip_key)

    def _needs_intra_reduce(self, shape: dict[str, int], tile: TileConfig) -> bool:
        k = int(shape.get("K", 0))
        return k > 0 and tile.tile_k is not None and tile.tile_k < k

    def _build_intra_reduce(self, op: DistributedOp, tile: TileConfig) -> DistributedOp:
        m = op.local_shape.get("M", 0)
        n = op.local_shape.get("N", 0)
        dtype_bytes = self._resolve_comm_dtype_bytes(op)
        comm_bytes = int(m * n * dtype_bytes)
        return DistributedOp(
            op_id=f"{op.op_id}_tiling_reduce",
            op_type="intra_reduce",
            role=NodeRole.COMM,
            comm_type=CommType.P2P,
            comm_bytes=comm_bytes,
            scope="intra_chip",
            cause="tiling_reduce",
            topology_path_key="intra_noc",
            participants=[self.chip.chip_id],
            algo_hint="intra_noc",
            stage_id=op.stage_id,
            chip_ids=[self.chip.chip_id],
            deps=[op.op_id],
            trigger_edge_id=op.op_id,
            reason="tiling_reduce_placeholder",
        )

    def _resolve_comm_dtype_bytes(self, op: DistributedOp) -> int:
        for key in ("output_dtype_bytes", "accum_dtype_bytes", "input_dtype_bytes"):
            value = op.attrs.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 2
