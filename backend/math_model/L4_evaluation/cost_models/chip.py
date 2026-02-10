"""Chip 级代价模型.

芯片当黑盒，所需参数以 TFLOPS/HBM 带宽为主。
特点：粗/快，适用于快速筛选 TP/PP/DP 组合。
"""

from __future__ import annotations

from dataclasses import dataclass

from math_model.L4_evaluation.cost_models.base import BaseCostModel


@dataclass
class RooflineResult:
    """Roofline 分析结果

    Attributes:
        arithmetic_intensity: 算术强度 (FLOPS/Byte)
        ridge_point: 拐点 (FLOPS/Byte)
        is_compute_bound: 是否计算瓶颈
        is_memory_bound: 是否内存瓶颈
    """

    arithmetic_intensity: float
    ridge_point: float
    is_compute_bound: bool
    is_memory_bound: bool


class ChipCostModel(BaseCostModel):
    """Chip 级代价模型

    使用 Roofline 模型估算，芯片视为黑盒。

    所需硬件参数:
        - compute_tflops: 峰值算力（TFLOPS）
        - memory_bandwidth_gbps: 显存带宽（GB/s）
        - intra_board_bw_gbps: 板内互联带宽（GB/s）
        - inter_board_bw_gbps: 板间互联带宽（GB/s）
        - inter_node_bw_gbps: 节点间互联带宽（GB/s）
    """

    def required_fields(self) -> set[str]:
        """返回所需字段"""
        return {
            "compute_tflops",
            "memory_bandwidth_gbps",
            "intra_board_bw_gbps",
            "inter_board_bw_gbps",
            "inter_node_bw_gbps",
        }

    def estimate_compute(
        self,
        op_type: str,
        local_shape: dict[str, int],
        hardware: dict[str, float],
    ) -> float:
        """Roofline 模型估算计算时间

        输入:
            - op_type: Op 类型
            - local_shape: 切分后的 shape（单位: elements）
            - hardware: 硬件参数
        输出:
            - 计算时间（ms）
        关键步骤:
            - 计算 FLOPs 和访存量
            - t_compute = max(FLOPs / TFLOPS, bytes / BW)
        """
        flops = self.estimate_flops(op_type, local_shape)
        bytes_read, bytes_write = self.estimate_bytes(op_type, local_shape)
        total_bytes = bytes_read + bytes_write

        # 硬件参数必需
        if "compute_tflops" not in hardware:
            raise ValueError("Missing 'compute_tflops' in hardware spec")
        if "memory_bandwidth_gbps" not in hardware:
            raise ValueError("Missing 'memory_bandwidth_gbps' in hardware spec")

        compute_tflops = hardware["compute_tflops"]
        memory_bw_gbps = hardware["memory_bandwidth_gbps"]

        # FLOPs -> ms: FLOPs / (TFLOPS * 1e12) * 1e3 = FLOPs / (TFLOPS * 1e9)
        t_compute_bound = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0

        # Bytes -> ms: bytes / (GB/s * 1e9) * 1e3 = bytes / (GB/s * 1e6)
        t_memory_bound = (
            total_bytes / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0
        )

        # Roofline: 取较大值
        return max(t_compute_bound, t_memory_bound)

    def estimate_comm(
        self,
        comm_bytes: int,
        path_key: str,
        participants: int,
        hardware: dict[str, float],
    ) -> float:
        """估算通信时间

        输入:
            - comm_bytes: 通信数据量（bytes）
            - path_key: 路径键（intra_board/inter_board/inter_node）
            - participants: 参与者数量
            - hardware: 硬件参数
        输出:
            - 通信时间（ms）
        关键步骤:
            - 根据 path_key 选择带宽
            - t_comm = bytes / BW（Ring AllReduce 需乘以 2*(n-1)/n 系数）
        """
        # 根据 path_key 选择带宽
        if path_key in ("intra_board", "intra_noc"):
            bw_gbps = hardware.get("intra_board_bw_gbps", 400.0)
        elif path_key == "inter_board":
            bw_gbps = hardware.get("inter_board_bw_gbps", 200.0)
        elif path_key == "inter_node":
            bw_gbps = hardware.get("inter_node_bw_gbps", 100.0)
        else:
            # 默认使用最小带宽
            bw_gbps = hardware.get("inter_node_bw_gbps", 100.0)

        if bw_gbps <= 0:
            return 0.0

        # Ring AllReduce 系数: 2 * (n-1) / n
        if participants > 1:
            ring_factor = 2 * (participants - 1) / participants
        else:
            ring_factor = 1.0

        # bytes -> ms: bytes / (GB/s * 1e9) * 1e3 * ring_factor
        t_comm = comm_bytes * ring_factor / (bw_gbps * 1e6)
        return t_comm


class MatMulCostModel(ChipCostModel):
    """MatMul 专用代价模型"""

    pass


class AttentionCostModel(ChipCostModel):
    """Attention 专用代价模型"""

    pass
