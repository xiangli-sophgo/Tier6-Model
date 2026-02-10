"""TilingPlanner - 片内映射实现."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

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
from math_model.L4_evaluation.evaluators.tile_cache import TilePersistentCache

logger = logging.getLogger(__name__)


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
        enable_persistent_cache: bool = True,
        memory_bandwidth_gbps: float = 0.0,
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

        # 持久化缓存
        self._persistent_cache: Optional[TilePersistentCache] = None
        if enable_persistent_cache:
            try:
                self._persistent_cache = TilePersistentCache.from_chip(
                    chip, memory_bandwidth_gbps=memory_bandwidth_gbps,
                )
            except Exception as e:
                logger.warning("Failed to initialize persistent tile cache: %s", e)

    def plan(self, dist_model: DistributedModel) -> TilePlan:
        """生成片内映射结果

        当 l4_evaluator 存在时，在搜索阶段即用 L4 精评估打分选择最优 tile，
        tile_meta 中直接包含精评估结果（t_compute_ms、t_memory_ms、bottleneck 等）。
        """
        plan = TilePlan()
        lmem_budget = self._get_lmem_budget()

        for op in dist_model.get_compute_ops():
            if not op.local_shape:
                continue

            cache_key = self._build_cache_key(op)
            cached = self._tile_cache.get(cache_key)
            if cached is None:
                # 尝试持久化缓存
                persistent_hit = self._try_persistent_get(op)
                if persistent_hit is not None:
                    tile_config, tile_meta = persistent_hit
                else:
                    t0 = time.monotonic()
                    evaluator = self.registry.select(op)
                    if evaluator is not None:
                        result = evaluator.select_tile(
                            op, lmem_budget, l4_scorer=self.l4_evaluator
                        )
                    else:
                        result = None
                    if result is None:
                        tile_config, tile_meta = self._fallback_tile(op)
                    else:
                        tile_config, tile_meta = result
                    search_ms = (time.monotonic() - t0) * 1000
                    # 写入持久化缓存（附带 tile_config 维度以便恢复）
                    meta_with_tile = dict(tile_meta)
                    meta_with_tile["tile_m"] = tile_config.tile_m
                    meta_with_tile["tile_n"] = tile_config.tile_n
                    meta_with_tile["tile_k"] = tile_config.tile_k
                    self._try_persistent_put(op, meta_with_tile, search_ms)
                self._tile_cache[cache_key] = (tile_config, tile_meta)
            else:
                tile_config, tile_meta = cached

            plan.tile_configs[op.op_id] = tile_config
            plan.kernel_configs[op.op_id] = {
                "kernel": "default",
                "traffic": str(tile_meta.get("traffic", 0)),
                "lmem_bytes": str(tile_meta.get("lmem_bytes", 0)),
                "t_compute_ms": str(tile_meta.get("t_compute_ms", "")),
                "t_memory_ms": str(tile_meta.get("t_memory_ms", "")),
                "bottleneck": str(tile_meta.get("bottleneck", "")),
                "best_loop_order": str(tile_meta.get("best_loop_order", "")),
                "arch_urate": str(tile_meta.get("arch_urate", "")),
                "active_cores": str(tile_meta.get("active_cores", "")),
                "overlap_rate": str(tile_meta.get("overlap_rate", "")),
            }

            if self._needs_intra_reduce(op.local_shape, tile_config):
                plan.intra_chip_comms.append(self._build_intra_reduce(op, tile_config))

        return plan

    def _get_lmem_budget(self) -> int:
        """获取片内 LMEM 预算（按 sram_utilization * LMEM / core_count）"""
        try:
            lmem = self.chip.memory_hierarchy.get_level("lmem")
            utilization = lmem.sram_utilization
            per_core = lmem.capacity_bytes / self.chip.core_count if self.chip.core_count else 0
            return int(per_core * utilization)
        except KeyError:
            total = self.chip.get_total_sram()
            utilization = self.chip.sram_utilization
            per_core = total / self.chip.core_count if self.chip.core_count else total
            return int(per_core * utilization) if per_core else 0

    def _fallback_tile(self, op: DistributedOp) -> tuple[TileConfig, dict[str, int]]:
        """生成回退 tile（当无法搜索时）"""
        cube_m = self.chip.cube_m
        cube_n = self.chip.cube_n
        cube_k = self.chip.cube_k
        if not cube_m or not cube_n or not cube_k:
            raise ValueError(
                f"Cannot create fallback tile: invalid chip cube dims (m={cube_m}, n={cube_n}, k={cube_k})"
            )
        tile_config = TileConfig(tile_m=cube_m, tile_n=cube_n, tile_k=cube_k)
        return tile_config, {"traffic": 0, "lmem_bytes": 0}

    def get_cache_stats(self) -> dict:
        """获取缓存统计（包含内存缓存和持久化缓存）"""
        stats = {"memory_cache_entries": len(self._tile_cache)}
        if self._persistent_cache is not None:
            stats.update(self._persistent_cache.get_cache_stats())
        return stats

    def _try_persistent_get(self, op: DistributedOp) -> Optional[tuple[TileConfig, dict]]:
        """尝试从持久化缓存获取"""
        if self._persistent_cache is None:
            return None
        dtypes = self._build_dtype_tuple(op)
        cached_meta = self._persistent_cache.get(
            op.op_type, op.local_shape, dtypes,
        )
        if cached_meta is None:
            return None
        # 从 cached_meta 重建 TileConfig
        tile_m = cached_meta.get("tile_m")
        tile_n = cached_meta.get("tile_n")
        tile_k = cached_meta.get("tile_k")
        tc = TileConfig(
            tile_m=int(tile_m) if tile_m is not None and str(tile_m) != "" else None,
            tile_n=int(tile_n) if tile_n is not None and str(tile_n) != "" else None,
            tile_k=int(tile_k) if tile_k is not None and str(tile_k) != "" else None,
        )
        return tc, cached_meta

    def _try_persistent_put(self, op: DistributedOp, tile_meta: dict,
                            search_time_ms: float) -> None:
        """写入持久化缓存"""
        if self._persistent_cache is None:
            return
        dtypes = self._build_dtype_tuple(op)
        try:
            self._persistent_cache.put(
                op.op_type, op.local_shape, dtypes,
                tile_meta=tile_meta, search_time_ms=search_time_ms,
            )
        except Exception as e:
            logger.debug("Failed to save to persistent cache: %s", e)

    def _build_dtype_tuple(self, op: DistributedOp) -> tuple:
        return (
            op.attrs.get("input_dtype_bytes"),
            op.attrs.get("weight_dtype_bytes"),
            op.attrs.get("output_dtype_bytes"),
            op.attrs.get("accum_dtype_bytes"),
        )

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
