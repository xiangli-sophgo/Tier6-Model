"""Tiling evaluators for different op types."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

from math_model.L2_arch.chip import ChipSpecImpl
from math_model.L3_mapping.plan.distributed_model import DistributedOp
from math_model.L1_workload.specs import TileConfig


class TilingEvaluator(Protocol):
    """Op-specific tiling evaluator interface."""

    def supports(self, op: DistributedOp) -> bool: ...

    def select_tile(
        self,
        op: DistributedOp,
        lmem_budget: int,
        l4_scorer: L4TileEvaluator
        | Callable[[DistributedOp, TileConfig, ChipSpecImpl], dict[str, int]]
        | None = None,
    ) -> tuple[TileConfig, dict[str, int]] | None: ...


class L4TileEvaluator(Protocol):
    """L4 tile-level evaluator interface."""

    def evaluate_tile(
        self, op: DistributedOp, tile: TileConfig, chip: ChipSpecImpl
    ) -> dict[str, int]: ...


@dataclass
class TilingEvaluatorRegistry:
    """Registry for op-specific evaluators."""

    evaluators: list[TilingEvaluator]

    def select(self, op: DistributedOp) -> TilingEvaluator | None:
        for evaluator in self.evaluators:
            if evaluator.supports(op):
                return evaluator
        return None


class MatmulTilingEvaluator:
    """Matmul tiling evaluator (DS_TPU-style search)."""

    def __init__(self, chip: ChipSpecImpl) -> None:
        self.chip = chip

    def supports(self, op: DistributedOp) -> bool:
        return op.op_type in {"matmul", "gemm"}

    def select_tile(
        self,
        op: DistributedOp,
        lmem_budget: int,
        l4_scorer: L4TileEvaluator
        | Callable[[DistributedOp, TileConfig, ChipSpecImpl], dict[str, int]]
        | None = None,
    ) -> tuple[TileConfig, dict[str, int]] | None:
        shape = op.local_shape
        if not shape:
            return None

        g = int(shape.get("G", 1))
        m = int(shape.get("M", 0))
        n = int(shape.get("N", 0))
        k = int(shape.get("K", 0))
        a_bytes, b_bytes, c_bytes, accum_bytes = self._resolve_dtype_bytes(op)
        cube_m, cube_n, cube_k, align_bytes, lane_num = self._get_tile_arch_params()

        best_tile: tuple[int, int, int] | None = None
        best_score: float | None = None
        best_meta: dict[str, int | float | str] | None = None

        partitions = self._valid_partitions(self.chip.core_count)
        for p_g, p_m, p_n, p_k in partitions:
            g_blk = math.ceil(g / p_g)
            m_blk = math.ceil(m / p_m) if m else 0
            n_blk = math.ceil(n / p_n) if n else 0
            k_blk = math.ceil(k / p_k) if k else 0
            if min(m_blk, n_blk, k_blk) == 0:
                continue

            candidates = self._legal_tiles(
                m_blk,
                n_blk,
                k_blk,
                lmem_budget,
                cube_m,
                cube_n,
                cube_k,
                align_bytes,
                lane_num,
                a_bytes,
                b_bytes,
                c_bytes,
            )
            for m_t, n_t, k_t in candidates:
                if l4_scorer is not None:
                    # L4 精评估打分：内部已枚举 loop-order
                    tile_cfg = TileConfig(tile_m=m_t, tile_n=n_t, tile_k=k_t)
                    if hasattr(l4_scorer, "evaluate_tile"):
                        l4_result = l4_scorer.evaluate_tile(op, tile_cfg, self.chip)
                    else:
                        l4_result = l4_scorer(op, tile_cfg, self.chip)
                    t_compute = float(l4_result.get("t_compute_ms", 0))
                    t_memory = float(l4_result.get("t_memory_ms", 0))
                    score = max(t_compute, t_memory) * max(1, g_blk)
                    if best_score is None or score < best_score:
                        best_tile = (m_t, n_t, k_t)
                        best_score = score
                        best_meta = {
                            "traffic": int(l4_result.get("traffic", 0)),
                            "lmem_bytes": int(l4_result.get("lmem_bytes", 0)),
                            "t_compute_ms": t_compute,
                            "t_memory_ms": t_memory,
                            "bottleneck": l4_result.get("bottleneck", "unknown"),
                            "best_loop_order": l4_result.get("best_loop_order", ""),
                        }
                else:
                    # 简化 traffic 估算（无 L4 时的回退）
                    for order in ("mnk", "nkm", "mkn"):
                        traffic = self._estimate_traffic(
                            order,
                            m_blk,
                            n_blk,
                            k_blk,
                            m_t,
                            n_t,
                            k_t,
                            a_bytes,
                            b_bytes,
                            c_bytes,
                            accum_bytes,
                        )
                        traffic = traffic * max(1, g_blk)
                        score = float(traffic)
                        if best_score is None or score < best_score:
                            best_tile = (m_t, n_t, k_t)
                            best_score = score
                            best_meta = {
                                "traffic": traffic,
                                "lmem_bytes": self._estimate_lmem(
                                    m_t, n_t, k_t, a_bytes, b_bytes, c_bytes
                                ),
                            }

        if best_tile is None:
            fallback = TileConfig(
                tile_m=min(m or 1, cube_m),
                tile_n=min(n or 1, cube_n),
                tile_k=min(k or 1, cube_k),
            )
            return fallback, {"traffic": 0, "lmem_bytes": 0}

        tile_config = TileConfig(tile_m=best_tile[0], tile_n=best_tile[1], tile_k=best_tile[2])
        return tile_config, best_meta or {"traffic": 0, "lmem_bytes": 0}

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return int(math.ceil(value / alignment) * alignment)

    def _ceil_div(self, value: int, divisor: int) -> int:
        if divisor <= 0:
            return 0
        return int(math.ceil(value / divisor))

    def _estimate_lmem(
        self, m_t: int, n_t: int, k_t: int, a_bytes: int, b_bytes: int, c_bytes: int
    ) -> int:
        return (m_t * k_t) * a_bytes + (k_t * n_t) * b_bytes + (m_t * n_t) * c_bytes

    def _estimate_traffic(
        self,
        order: str,
        m: int,
        n: int,
        k: int,
        m_t: int,
        n_t: int,
        k_t: int,
        a_bytes: int,
        b_bytes: int,
        c_bytes: int,
        accum_bytes: int,
    ) -> int:
        if min(m, n, k) == 0 or min(m_t, n_t, k_t) == 0:
            return 0
        tiles_m = self._ceil_div(m, m_t)
        tiles_n = self._ceil_div(n, n_t)
        tiles_k = self._ceil_div(k, k_t)
        if order == "mnk":
            return int(
                (m * k) * a_bytes * tiles_n
                + (n * k) * b_bytes * tiles_m
                + (m * n) * c_bytes
            )
        if order == "nkm":
            return int(
                (n * k) * b_bytes
                + (m * k) * a_bytes * tiles_n
                + (m * n) * accum_bytes * 2 * max(tiles_k - 1, 0)
                + (m * n) * c_bytes
            )
        return int(
            (m * k) * a_bytes
            + (n * k) * b_bytes * tiles_m
            + (m * n) * accum_bytes * 2 * max(tiles_k - 1, 0)
            + (m * n) * c_bytes
        )

    def _valid_partitions(self, core_count: int) -> list[tuple[int, int, int, int]]:
        partitions: list[tuple[int, int, int, int]] = []
        for p_g in range(1, core_count + 1):
            if core_count % p_g:
                continue
            rem_m = core_count // p_g
            for p_m in range(1, rem_m + 1):
                if rem_m % p_m:
                    continue
                rem_n = rem_m // p_m
                for p_n in range(1, rem_n + 1):
                    if rem_n % p_n:
                        continue
                    p_k = rem_n // p_n
                    partitions.append((p_g, p_m, p_n, p_k))
        return partitions

    def _get_tile_arch_params(self) -> tuple[int, int, int, int, int]:
        """获取芯片微架构参数（必需）"""
        cube_m = self.chip.cube_m
        cube_n = self.chip.cube_n
        cube_k = self.chip.cube_k
        align_bytes = self.chip.align_bytes
        lane_num = self.chip.lane_per_core

        # 校验参数有效性
        if not cube_m or not cube_n or not cube_k:
            raise ValueError(
                f"Invalid chip cube dimensions: m={cube_m}, n={cube_n}, k={cube_k}. "
                "Check compute_units.cube config."
            )
        if not lane_num:
            raise ValueError("Invalid chip lane_per_core: must be > 0")

        return cube_m, cube_n, cube_k, align_bytes or 32, lane_num

    def _legal_tiles(
        self,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        lmem_budget: int,
        cube_m: int,
        cube_n: int,
        cube_k: int,
        align_bytes: int,
        lane_num: int,
        a_bytes: int,
        b_bytes: int,
        c_bytes: int,
    ) -> list[tuple[int, int, int]]:
        if m_blk * n_blk * k_blk == 0:
            return [(0, 0, 0)]

        all_tiles: list[tuple[int, int, int]] = []
        sram_limit = lmem_budget if lmem_budget > 0 else None
        align_row = lambda r: self._align_up(r, lane_num)
        align_col = lambda c, elem_bytes: self._align_up(c * elem_bytes, align_bytes)

        for m_t in range(self._align_up(m_blk, cube_m), 0, -cube_m):
            align_row_m = align_row(m_t)
            for n_t in range(self._align_up(n_blk, cube_n), 0, -cube_n):
                if sram_limit is None:
                    k_t = self._align_up(k_blk, cube_k)
                else:
                    # C tile shape = (m_t, n_t), 行对齐到 lane_num, 列字节对齐到 align_bytes
                    align_col_n = align_col(n_t, c_bytes)
                    c_tile_bytes = align_row_m * align_col_n
                    avail = sram_limit - c_tile_bytes
                    if avail <= 0:
                        continue
                    # A tile 每增加 1 个 K: align_up(m_t, lane_num) * a_bytes
                    # B tile 每增加 1 个 K: align_up(n_t, lane_num) * b_bytes
                    align_row_n = align_row(n_t)
                    denom = align_row_m * a_bytes + align_row_n * b_bytes
                    max_k = int(avail / denom) if denom > 0 else 0
                    if max_k == 0:
                        continue
                    align_k = self._align_up(min(k_blk, max_k), cube_k)
                    if align_k < cube_k:
                        k_t = max_k
                    else:
                        k_t = (align_k - cube_k) if align_k > max_k else align_k
                        if k_t == 0:
                            continue
                if self._is_pareto_max_tiles(all_tiles, m_t, n_t, k_t):
                    all_tiles.append((m_t, n_t, k_t))
        return all_tiles

    def _is_pareto_max_tiles(
        self, conds: list[tuple[int, int, int]], m_t: int, n_t: int, k_t: int
    ) -> bool:
        if len(conds) == 0:
            return True
        for m0, n0, k0 in conds:
            if m0 >= m_t and n0 >= n_t and k0 >= k_t:
                return False
        return True

    def _resolve_dtype_bytes(self, op: DistributedOp) -> tuple[int, int, int, int]:
        attrs = op.attrs or {}

        def _get_int(key: str) -> int | None:
            value = attrs.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        input_bytes = _get_int("input_dtype_bytes")
        weight_bytes = _get_int("weight_dtype_bytes")
        output_bytes = _get_int("output_dtype_bytes")
        accum_bytes = _get_int("accum_dtype_bytes")

        if input_bytes is None and output_bytes is not None:
            input_bytes = output_bytes
        if weight_bytes is None and input_bytes is not None:
            weight_bytes = input_bytes
        if output_bytes is None and input_bytes is not None:
            output_bytes = input_bytes

        input_bytes = input_bytes or 2
        weight_bytes = weight_bytes or input_bytes
        output_bytes = output_bytes or input_bytes
        accum_bytes = accum_bytes or 4

        return input_bytes, weight_bytes, output_bytes, accum_bytes


class ElementwiseTilingEvaluator:
    """Elementwise tiling evaluator placeholder."""

    def __init__(self, chip: ChipSpecImpl) -> None:
        self.chip = chip

    def supports(self, op: DistributedOp) -> bool:
        return op.op_type in {"elementwise", "relu", "gelu", "silu", "add", "mul"}

    def select_tile(
        self,
        op: DistributedOp,
        lmem_budget: int,
        l4_scorer: L4TileEvaluator
        | Callable[[DistributedOp, TileConfig, ChipSpecImpl], dict[str, int]]
        | None = None,
    ) -> tuple[TileConfig, dict[str, int]] | None:
        shape = op.local_shape
        if not shape:
            return None

        m = int(shape.get("M", 0))
        n = int(shape.get("N", 0))
        if m == 0 or n == 0:
            b = int(shape.get("B", 0) or shape.get("batch", 0))
            s = int(shape.get("S", 0) or shape.get("seq_len", 0))
            h = int(shape.get("H", 0) or shape.get("hidden", 0))
            if b and s and h:
                m = b * s
                n = h
        if m == 0 or n == 0:
            return None

        a_bytes, _, c_bytes, _ = MatmulTilingEvaluator(self.chip)._resolve_dtype_bytes(op)
        cube_m, cube_n, _, _, lane_num = MatmulTilingEvaluator(self.chip)._get_tile_arch_params()

        best_tile: tuple[int, int] | None = None
        best_score: float | None = None
        best_meta: dict[str, int | float | str] | None = None

        for m_t in self._tile_candidates(m, cube_m, lane_num):
            for n_t in self._tile_candidates(n, cube_n, lane_num):
                lmem = (m_t * n_t) * (a_bytes + c_bytes)
                if lmem_budget and lmem > lmem_budget:
                    continue
                if l4_scorer is not None:
                    tile_cfg = TileConfig(tile_m=m_t, tile_n=n_t, tile_k=None)
                    if hasattr(l4_scorer, "evaluate_tile"):
                        l4_result = l4_scorer.evaluate_tile(op, tile_cfg, self.chip)
                    else:
                        l4_result = l4_scorer(op, tile_cfg, self.chip)
                    t_compute = float(l4_result.get("t_compute_ms", 0))
                    t_memory = float(l4_result.get("t_memory_ms", 0))
                    score = max(t_compute, t_memory)
                    if best_score is None or score < best_score:
                        best_tile = (m_t, n_t)
                        best_score = score
                        best_meta = {
                            "traffic": int(l4_result.get("traffic", 0)),
                            "lmem_bytes": int(l4_result.get("lmem_bytes", 0)),
                            "t_compute_ms": t_compute,
                            "t_memory_ms": t_memory,
                            "bottleneck": l4_result.get("bottleneck", "unknown"),
                        }
                else:
                    traffic = self._estimate_traffic(m, n, m_t, n_t, a_bytes, c_bytes)
                    score = float(traffic)
                    if best_score is None or score < best_score:
                        best_tile = (m_t, n_t)
                        best_score = score
                        best_meta = {"traffic": traffic, "lmem_bytes": lmem}

        if best_tile is None:
            return None

        tile_config = TileConfig(tile_m=best_tile[0], tile_n=best_tile[1], tile_k=None)
        return tile_config, best_meta or {"traffic": 0, "lmem_bytes": 0}

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return int(math.ceil(value / alignment) * alignment)

    def _tile_candidates(self, size: int, cube: int, lane_num: int) -> list[int]:
        candidates: list[int] = []
        for factor in (1, 2, 4, 8, 16):
            base = max(1, size // factor)
            aligned = self._align_up(base, max(1, min(cube, lane_num)))
            aligned = min(size, aligned)
            if aligned not in candidates and aligned > 0:
                candidates.append(aligned)
        return candidates

    def _estimate_traffic(
        self, m: int, n: int, m_t: int, n_t: int, a_bytes: int, c_bytes: int
    ) -> int:
        tiles_m = int(math.ceil(m / m_t))
        tiles_n = int(math.ceil(n / n_t))
        tile_elems = m_t * n_t
        return int((a_bytes + c_bytes) * tile_elems * tiles_m * tiles_n)


class FA2TilingEvaluator:
    """FA2 tiling evaluator placeholder."""

    def __init__(self, chip: ChipSpecImpl) -> None:
        self.chip = chip

    def supports(self, op: DistributedOp) -> bool:
        return op.op_type in {"fa2", "flash_attention"}

    def select_tile(
        self,
        op: DistributedOp,
        lmem_budget: int,
        l4_scorer: L4TileEvaluator
        | Callable[[DistributedOp, TileConfig, ChipSpecImpl], dict[str, int]]
        | None = None,
    ) -> tuple[TileConfig, dict[str, int]] | None:
        shape = op.local_shape
        if not shape:
            return None

        b = int(shape.get("B", 0) or shape.get("batch", 0))
        qs = int(shape.get("QS", 0) or shape.get("Q", 0) or shape.get("q_seq_len", 0))
        ks = int(shape.get("KS", 0) or shape.get("K", 0) or shape.get("kv_seq_len", 0))
        qd = int(shape.get("QD", 0) or shape.get("D", 0) or shape.get("q_dim", 0))
        vd = int(shape.get("VD", 0) or shape.get("v_dim", 0))
        if min(b, qs, ks, qd, vd) == 0:
            return None

        a_bytes, _, c_bytes, accum_bytes = MatmulTilingEvaluator(self.chip)._resolve_dtype_bytes(op)
        cube_m, cube_n, cube_k, _, _ = MatmulTilingEvaluator(self.chip)._get_tile_arch_params()

        best_tile: tuple[int, int] | None = None
        best_score: float | None = None
        best_meta: dict[str, int | float | str] | None = None

        for p_b in self._valid_partitions(self.chip.core_count):
            b_blk = math.ceil(b / p_b)
            for q_t, k_t in self._legal_tiles(qs, ks, qd, vd, lmem_budget, cube_m, cube_n, cube_k, a_bytes, c_bytes):
                if l4_scorer is not None:
                    tile_cfg = TileConfig(tile_m=q_t, tile_n=k_t, tile_k=None)
                    if hasattr(l4_scorer, "evaluate_tile"):
                        l4_result = l4_scorer.evaluate_tile(op, tile_cfg, self.chip)
                    else:
                        l4_result = l4_scorer(op, tile_cfg, self.chip)
                    t_compute = float(l4_result.get("t_compute_ms", 0))
                    t_memory = float(l4_result.get("t_memory_ms", 0))
                    score = max(t_compute, t_memory) * max(1, b_blk)
                    if best_score is None or score < best_score:
                        best_tile = (q_t, k_t)
                        best_score = score
                        best_meta = {
                            "traffic": int(l4_result.get("traffic", 0)),
                            "lmem_bytes": int(l4_result.get("lmem_bytes", 0)),
                            "t_compute_ms": t_compute,
                            "t_memory_ms": t_memory,
                            "bottleneck": l4_result.get("bottleneck", "unknown"),
                        }
                else:
                    traffic = self._estimate_traffic(qs, ks, qd, vd, q_t, k_t, a_bytes, c_bytes, accum_bytes)
                    traffic = traffic * max(1, b_blk)
                    score = float(traffic)
                    if best_score is None or score < best_score:
                        best_tile = (q_t, k_t)
                        best_score = score
                        best_meta = {
                            "traffic": traffic,
                            "lmem_bytes": self._estimate_lmem(qd, vd, q_t, k_t, a_bytes, c_bytes),
                        }

        if best_tile is None:
            return None

        tile_config = TileConfig(tile_m=best_tile[0], tile_n=best_tile[1], tile_k=None)
        return tile_config, best_meta or {"traffic": 0, "lmem_bytes": 0}

    def _valid_partitions(self, core_count: int) -> list[int]:
        return [p_b for p_b in range(1, core_count + 1) if core_count % p_b == 0]

    def _legal_tiles(
        self,
        qs: int,
        ks: int,
        qd: int,
        vd: int,
        lmem_budget: int,
        cube_m: int,
        cube_n: int,
        cube_k: int,
        a_bytes: int,
        c_bytes: int,
    ) -> list[tuple[int, int]]:
        if qs * ks * qd * vd == 0:
            return [(0, 0)]

        all_tiles: list[tuple[int, int]] = []
        sram_limit = lmem_budget if lmem_budget > 0 else None
        max_cube = max(cube_n, cube_k)

        for q_t in range(self._align_up(qs, cube_m), 0, -cube_m):
            for k_t in range(self._align_up(ks, max_cube), 0, -max_cube):
                if sram_limit is not None:
                    occupied = self._estimate_lmem(qd, vd, q_t, k_t, a_bytes, c_bytes)
                    if occupied > sram_limit:
                        continue
                if self._is_pareto_max_tiles(all_tiles, q_t, k_t):
                    all_tiles.append((q_t, k_t))
        return all_tiles

    def _estimate_lmem(self, qd: int, vd: int, q_t: int, k_t: int, a_bytes: int, c_bytes: int) -> int:
        # occupied = q + k + v + 2*p + 4*o
        q_buf = q_t * qd * a_bytes
        k_buf = k_t * qd * a_bytes
        v_buf = k_t * vd * a_bytes
        p_buf = q_t * k_t * a_bytes
        o_buf = q_t * vd * c_bytes
        return q_buf + k_buf + v_buf + 2 * p_buf + 4 * o_buf

    def _estimate_traffic(
        self,
        qs: int,
        ks: int,
        qd: int,
        vd: int,
        q_t: int,
        k_t: int,
        a_bytes: int,
        c_bytes: int,
        accum_bytes: int,
    ) -> int:
        tile_num_q = int(math.ceil(qs / q_t))
        load_k = ks * qd * a_bytes * tile_num_q
        load_v = ks * vd * a_bytes * tile_num_q
        store_o = qs * vd * c_bytes
        acc = qs * vd * accum_bytes
        return int(load_k + load_v + store_o + acc)

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return int(math.ceil(value / alignment) * alignment)

    def _is_pareto_max_tiles(self, conds: list[tuple[int, int]], q_t: int, k_t: int) -> bool:
        if len(conds) == 0:
            return True
        for q0, k0 in conds:
            if q0 >= q_t and k0 >= k_t:
                return False
        return True
