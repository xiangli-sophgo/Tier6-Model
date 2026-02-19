"""指令发射 (G5 指令级仿真模式)

从 tiling 结果生成 TIU + DMA + SDMA + HAU 指令序列。

支持的 op 类型:
    - matmul: TIU MatMul (含 double buffering), 若 attrs.moe_gating 则追加 HAU Top-K
    - dispatch/all2all (reason含dispatch): SDMA SCATTER
    - combine: SDMA GATHER
    - allreduce: SDMA TENSOR (简化为单次传输)

循环结构 (MatMul, MNK, K 为内层):
    Prologue: Load A[0,0,0] -> A0, Load B[0,0,0] -> B0
    Main loop for (m, n, k):
        TIU compute(A{cur}, B{cur}) -> C{cur}
        if not last: Load A_next -> A{nxt}, Load B_next -> B{nxt}
        if k == last_k: Store C{cur} -> DDR

依赖跟踪:
    last_tiu_using_buf[0/1]: 最后一个使用 buffer set 的 TIU cmd_id
    last_dma_loading_buf[0/1]: 最后一个加载到 buffer set 的 DMA cmd_id

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

from typing import Any

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.instruction_tiler import TilingResult, tile_matmul
from perf_model.L3_mapping.g5.program import (
    CoreInstructions,
    CoreProgram,
    DMACommand,
    DMADirection,
    HAUCommand,
    HAUMsgAction,
    HAUOpType,
    SDMACommand,
    SDMACommandType,
    TIUCommand,
    TIUOpType,
)


def _precision_from_dtype_bytes(dtype_bytes: int) -> str:
    """dtype_bytes -> precision 字符串"""
    mapping = {1: "INT8", 2: "BF16", 4: "FP32"}
    if dtype_bytes not in mapping:
        raise ValueError(f"Unsupported dtype_bytes={dtype_bytes}")
    return mapping[dtype_bytes]


class G5InstructionEmitter:
    """G5 指令发射器

    将 DistributedOp 转换为单核 CoreProgram。
    支持: MatMul (MM2_NN), MoE dispatch/combine, AllReduce, HAU Top-K。
    """

    def __init__(self, chip: ChipSpecImpl) -> None:
        self._chip = chip

    def emit(self, ops: list[Any]) -> CoreProgram:
        """生成 CoreProgram

        Args:
            ops: DistributedOp 列表 (或含 local_shape/attrs 的类似对象)

        Returns:
            CoreProgram
        """
        all_tiu: list[TIUCommand] = []
        all_dma: list[DMACommand] = []
        all_sdma: list[SDMACommand] = []
        all_hau: list[HAUCommand] = []
        metadata: dict[str, Any] = {}

        tiu_id = 0
        dma_id = 0
        sdma_id = 0
        hau_id = 0

        for op in ops:
            op_type = op.op_type if hasattr(op, "op_type") else op.get("op_type", "")
            op_id = op.op_id if hasattr(op, "op_id") else op.get("op_id", "unknown")
            attrs = op.attrs if hasattr(op, "attrs") else op.get("attrs", {})

            # 真实 DistributedOp COMM 节点适配:
            # role=NodeRole.COMM, comm_type=CommType.ALL2ALL/ALLREDUCE 等
            role = getattr(op, "role", None)
            comm_type_val = getattr(op, "comm_type", None)
            if role is not None and hasattr(role, "name") and role.name == "COMM" and comm_type_val is not None:
                comm_type_name = comm_type_val.name if hasattr(comm_type_val, "name") else str(comm_type_val)
                comm_bytes = self._get_comm_bytes(op)
                reason = getattr(op, "reason", "") or ""

                if comm_type_name == "ALL2ALL":
                    if "dispatch" in reason:
                        sdma_cmds, sdma_id = self._emit_sdma_dispatch(
                            op_id, comm_bytes, hau_id, sdma_id,
                        )
                    else:
                        sdma_cmds, sdma_id = self._emit_sdma_combine(
                            op_id, comm_bytes, tiu_id, sdma_id,
                        )
                    all_sdma.extend(sdma_cmds)
                elif comm_type_name == "ALLREDUCE":
                    sdma_cmds, sdma_id = self._emit_allreduce(
                        op_id, comm_bytes, tiu_id, sdma_id,
                    )
                    all_sdma.extend(sdma_cmds)
                elif comm_type_name in ("ALLGATHER", "REDUCE_SCATTER", "P2P"):
                    # 简化: 统一映射为 SDMA TENSOR 传输
                    sdma_cmds, sdma_id = self._emit_allreduce(
                        op_id, comm_bytes, tiu_id, sdma_id,
                    )
                    all_sdma.extend(sdma_cmds)
                continue

            if op_type == "matmul":
                shape = op.local_shape if hasattr(op, "local_shape") else op.get("local_shape", {})
                M = shape["M"]
                N = shape["N"]
                K = shape["K"]
                dtype_bytes = int(attrs.get("input_dtype_bytes", 2))

                tiling = tile_matmul(M, N, K, dtype_bytes, self._chip)
                precision = _precision_from_dtype_bytes(dtype_bytes)

                tiu_cmds, dma_cmds, tiu_id, dma_id = self._emit_matmul_double_buffered(
                    tiling, precision, dtype_bytes, op_id, tiu_id, dma_id,
                    M, N, K,
                )

                all_tiu.extend(tiu_cmds)
                all_dma.extend(dma_cmds)

                metadata[op_id] = {
                    "tile_m": tiling.tile_m,
                    "tile_n": tiling.tile_n,
                    "tile_k": tiling.tile_k,
                    "m_tiles": tiling.m_tiles,
                    "n_tiles": tiling.n_tiles,
                    "k_tiles": tiling.k_tiles,
                    "lmem_total_bytes": tiling.layout.total_bytes,
                }

                # MoE gating -> 自动追加 HAU Top-K
                if attrs.get("moe_gating") == "true":
                    hau_cmds, hau_id = self._emit_moe_topk(
                        op_id, attrs, tiu_id, hau_id,
                    )
                    all_hau.extend(hau_cmds)

            elif op_type == "dispatch":
                comm_bytes = self._get_comm_bytes(op)
                sdma_cmds, sdma_id = self._emit_sdma_dispatch(
                    op_id, comm_bytes, hau_id, sdma_id,
                )
                all_sdma.extend(sdma_cmds)

            elif op_type == "combine":
                comm_bytes = self._get_comm_bytes(op)
                sdma_cmds, sdma_id = self._emit_sdma_combine(
                    op_id, comm_bytes, tiu_id, sdma_id,
                )
                all_sdma.extend(sdma_cmds)

            elif op_type == "allreduce":
                comm_bytes = self._get_comm_bytes(op)
                sdma_cmds, sdma_id = self._emit_allreduce(
                    op_id, comm_bytes, tiu_id, sdma_id,
                )
                all_sdma.extend(sdma_cmds)

            elif op_type in ("all2all",):
                # all2all 根据 reason 判断 dispatch/combine
                reason = attrs.get("reason", "")
                comm_bytes = self._get_comm_bytes(op)
                if "dispatch" in reason:
                    sdma_cmds, sdma_id = self._emit_sdma_dispatch(
                        op_id, comm_bytes, hau_id, sdma_id,
                    )
                else:
                    sdma_cmds, sdma_id = self._emit_sdma_combine(
                        op_id, comm_bytes, tiu_id, sdma_id,
                    )
                all_sdma.extend(sdma_cmds)

        core = CoreInstructions(
            core_id=0,
            tiu_cmds=all_tiu,
            dma_cmds=all_dma,
            sdma_cmds=all_sdma,
            hau_cmds=all_hau,
        )
        return CoreProgram(cores=[core], metadata=metadata)

    def _get_comm_bytes(self, op: Any) -> int:
        """从 op 中获取通信数据量"""
        if hasattr(op, "comm_bytes"):
            return int(op.comm_bytes)
        attrs = op.attrs if hasattr(op, "attrs") else op.get("attrs", {})
        return int(attrs.get("comm_bytes", 0))

    def _emit_moe_topk(
        self,
        source_op_id: str,
        attrs: dict[str, Any],
        last_tiu_id: int,
        hau_id_start: int,
    ) -> tuple[list[HAUCommand], int]:
        """生成 HAU Top-K 指令 (MoE gating 后自动插入)"""
        hau_id = hau_id_start
        num_experts = int(attrs.get("num_experts", 256))
        top_k = int(attrs.get("top_k", 8))

        hau_id += 1
        cmd = HAUCommand(
            cmd_id=hau_id,
            cmd_id_dep=last_tiu_id,
            dep_engine="tiu",
            op_type=HAUOpType.TOP_K,
            src_addr=0,
            dst_addr=1024,
            num_elements=num_experts,
            top_k=top_k,
            data_format="BF16",
            msg_action=HAUMsgAction.SEND,
            msg_id=hau_id,
            source_op_id=f"{source_op_id}_topk",
        )
        return [cmd], hau_id

    def _emit_sdma_dispatch(
        self,
        source_op_id: str,
        comm_bytes: int,
        last_hau_id: int,
        sdma_id_start: int,
    ) -> tuple[list[SDMACommand], int]:
        """生成 SDMA SCATTER 指令 (MoE dispatch)"""
        sdma_id = sdma_id_start + 1
        cmd = SDMACommand(
            cmd_id=sdma_id,
            cmd_id_dep=last_hau_id,
            dep_engine="hau",
            cmd_type=SDMACommandType.SCATTER,
            src_addr=0,
            dst_addr=0,
            data_bytes=max(comm_bytes, 1),
            elem_bytes=2,
            src_core_id=0,
            dst_core_id=0,
            source_op_id=source_op_id,
        )
        return [cmd], sdma_id

    def _emit_sdma_combine(
        self,
        source_op_id: str,
        comm_bytes: int,
        last_tiu_id: int,
        sdma_id_start: int,
    ) -> tuple[list[SDMACommand], int]:
        """生成 SDMA GATHER 指令 (MoE combine)"""
        sdma_id = sdma_id_start + 1
        cmd = SDMACommand(
            cmd_id=sdma_id,
            cmd_id_dep=last_tiu_id,
            dep_engine="tiu",
            cmd_type=SDMACommandType.GATHER,
            src_addr=0,
            dst_addr=0,
            data_bytes=max(comm_bytes, 1),
            elem_bytes=2,
            src_core_id=0,
            dst_core_id=0,
            source_op_id=source_op_id,
        )
        return [cmd], sdma_id

    def _emit_allreduce(
        self,
        source_op_id: str,
        comm_bytes: int,
        last_tiu_id: int,
        sdma_id_start: int,
    ) -> tuple[list[SDMACommand], int]:
        """生成 AllReduce SDMA 指令 (简化为单次 SDMA 传输)"""
        sdma_id = sdma_id_start + 1
        cmd = SDMACommand(
            cmd_id=sdma_id,
            cmd_id_dep=last_tiu_id,
            dep_engine="tiu",
            cmd_type=SDMACommandType.TENSOR,
            src_addr=0,
            dst_addr=0,
            data_bytes=max(comm_bytes, 1),
            elem_bytes=2,
            src_core_id=0,
            dst_core_id=0,
            source_op_id=source_op_id,
        )
        return [cmd], sdma_id

    def _emit_matmul_double_buffered(
        self,
        tiling: TilingResult,
        precision: str,
        dtype_bytes: int,
        source_op_id: str,
        tiu_id_start: int,
        dma_id_start: int,
        M: int, N: int, K: int,
    ) -> tuple[list[TIUCommand], list[DMACommand], int, int]:
        """生成 MatMul double buffering 指令序列"""
        layout = tiling.layout
        tiu_cmds: list[TIUCommand] = []
        dma_cmds: list[DMACommand] = []

        tiu_id = tiu_id_start
        dma_id = dma_id_start

        # 依赖跟踪
        last_tiu_using_buf = [0, 0]   # 最后使用 buf[i] 的 TIU cmd_id
        last_dma_loading_buf = [0, 0]  # 最后加载到 buf[i] 的 DMA cmd_id

        # 构建迭代序列: (mi, ni, ki)
        iterations: list[tuple[int, int, int]] = []
        for mi in range(tiling.m_tiles):
            for ni in range(tiling.n_tiles):
                for ki in range(tiling.k_tiles):
                    iterations.append((mi, ni, ki))

        if not iterations:
            return tiu_cmds, dma_cmds, tiu_id, dma_id

        # Prologue: 加载第一组数据到 buf0
        dma_id += 1
        dma_cmds.append(DMACommand(
            cmd_id=dma_id, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0,  # DDR 地址 (简化)
            dst_addr=layout.a_addrs[0],
            data_bytes=layout.a_tile_bytes,
            elem_bytes=dtype_bytes,
            source_op_id=source_op_id,
        ))
        last_dma_loading_buf[0] = dma_id

        dma_id += 1
        dma_cmds.append(DMACommand(
            cmd_id=dma_id, cmd_id_dep=0,
            direction=DMADirection.DDR_TO_LMEM,
            src_addr=0,
            dst_addr=layout.b_addrs[0],
            data_bytes=layout.b_tile_bytes,
            elem_bytes=dtype_bytes,
            source_op_id=source_op_id,
        ))
        last_dma_loading_buf[0] = dma_id

        # Main loop
        for idx, (mi, ni, ki) in enumerate(iterations):
            buf_cur = idx % 2
            buf_nxt = 1 - buf_cur
            is_last = (idx == len(iterations) - 1)
            is_last_k = (ki == tiling.k_tiles - 1)

            # 实际 tile 尺寸 (边界处理)
            actual_tm = min(tiling.tile_m, M - mi * tiling.tile_m)
            actual_tn = min(tiling.tile_n, N - ni * tiling.tile_n)
            actual_tk = min(tiling.tile_k, K - ki * tiling.tile_k)

            # TIU: compute A{cur} * B{cur} -> C{cur}
            tiu_id += 1
            tiu_cmds.append(TIUCommand(
                cmd_id=tiu_id,
                cmd_id_dep=last_dma_loading_buf[buf_cur],
                op_type=TIUOpType.MM2_NN,
                result_addr=layout.c_addrs[buf_cur],
                operand_addrs=[layout.a_addrs[buf_cur], layout.b_addrs[buf_cur]],
                tile_m=actual_tm,
                tile_n=actual_tn,
                tile_k=actual_tk,
                precision=precision,
                source_op_id=source_op_id,
            ))
            last_tiu_using_buf[buf_cur] = tiu_id

            # Prefetch next tile (if not last iteration)
            if not is_last:
                # Load A_next -> A{nxt}
                dma_id += 1
                dma_cmds.append(DMACommand(
                    cmd_id=dma_id,
                    cmd_id_dep=last_tiu_using_buf[buf_nxt],
                    direction=DMADirection.DDR_TO_LMEM,
                    src_addr=0,
                    dst_addr=layout.a_addrs[buf_nxt],
                    data_bytes=layout.a_tile_bytes,
                    elem_bytes=dtype_bytes,
                    source_op_id=source_op_id,
                ))
                last_dma_loading_buf[buf_nxt] = dma_id

                # Load B_next -> B{nxt}
                dma_id += 1
                dma_cmds.append(DMACommand(
                    cmd_id=dma_id,
                    cmd_id_dep=last_tiu_using_buf[buf_nxt],
                    direction=DMADirection.DDR_TO_LMEM,
                    src_addr=0,
                    dst_addr=layout.b_addrs[buf_nxt],
                    data_bytes=layout.b_tile_bytes,
                    elem_bytes=dtype_bytes,
                    source_op_id=source_op_id,
                ))
                last_dma_loading_buf[buf_nxt] = dma_id

            # Store C{cur} -> DDR at last K tile for this (m, n) pair
            if is_last_k:
                dma_id += 1
                c_bytes = dtype_bytes  # 输出精度与输入相同
                actual_c_bytes = actual_tm * actual_tn * c_bytes
                dma_cmds.append(DMACommand(
                    cmd_id=dma_id,
                    cmd_id_dep=tiu_id,  # 等当前 TIU 完成
                    direction=DMADirection.LMEM_TO_DDR,
                    src_addr=layout.c_addrs[buf_cur],
                    dst_addr=0,
                    data_bytes=actual_c_bytes,
                    elem_bytes=c_bytes,
                    source_op_id=source_op_id,
                ))

        return tiu_cmds, dma_cmds, tiu_id, dma_id
