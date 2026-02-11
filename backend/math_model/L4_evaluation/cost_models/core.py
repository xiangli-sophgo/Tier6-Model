"""Core 级代价模型.

考虑多核并行与片内路径键（如 NoC）。
所需参数：cores/SRAM/NoC 带宽。
特点：中/中，适用于优化 Tile/Kernel 相关决策（MVP 主力）。
"""

from __future__ import annotations

from math_model.L4_evaluation.cost_models.base import BaseCostModel


class CoreCostModel(BaseCostModel):
    """Core 级代价模型

    考虑多核并行与片内存储层级。

    所需硬件参数:
        - compute_tflops: 峰值算力（TFLOPS）
        - memory_bandwidth_gbps: 显存带宽（GB/s）
        - num_cores: Core 数量
        - sram_per_core_kb: 每 Core SRAM 容量（KB）
        - noc_bandwidth_gbps: 片内 NoC 带宽（GB/s）
        - c2c_bandwidth_gbps: C2C 互联带宽（GB/s）
        - b2b_bandwidth_gbps: B2B 互联带宽（GB/s）
        - r2r_bandwidth_gbps: R2R 互联带宽（GB/s）
        - p2p_bandwidth_gbps: P2P 互联带宽（GB/s）
    """

    def required_fields(self) -> set[str]:
        """返回所需字段"""
        return {
            "compute_tflops",
            "memory_bandwidth_gbps",
            "num_cores",
            "sram_per_core_kb",
            "noc_bandwidth_gbps",
            "c2c_bandwidth_gbps",
            "b2b_bandwidth_gbps",
            "r2r_bandwidth_gbps",
            "p2p_bandwidth_gbps",
        }

    def estimate_compute(
        self,
        op_type: str,
        local_shape: dict[str, int],
        hardware: dict[str, float],
    ) -> float:
        """考虑多核并行的计算时间估算

        输入:
            - op_type: Op 类型
            - local_shape: 切分后的 shape（单位: elements）
            - hardware: 硬件参数
        输出:
            - 计算时间（ms）
        关键步骤:
            - 计算 FLOPs 和访存量
            - 考虑多核并行效率
            - t_compute = max(FLOPs / (TFLOPS * efficiency), bytes / BW)
        """
        flops = self.estimate_flops(op_type, local_shape)
        bytes_read, bytes_write = self.estimate_bytes(op_type, local_shape)
        total_bytes = bytes_read + bytes_write

        compute_tflops = hardware["compute_tflops"]
        memory_bw_gbps = hardware["memory_bandwidth_gbps"]
        num_cores = int(hardware["num_cores"])
        sram_per_core_kb = hardware["sram_per_core_kb"]

        # 计算多核并行效率（基于数据量与 SRAM 容量的匹配度）
        total_sram_bytes = num_cores * sram_per_core_kb * 1024
        data_to_sram_ratio = (
            total_bytes / total_sram_bytes if total_sram_bytes > 0 else 1.0
        )

        # 效率模型：数据量超过 SRAM 时效率下降
        if data_to_sram_ratio <= 1:
            efficiency = 0.9  # 数据能完全放进 SRAM
        elif data_to_sram_ratio <= 2:
            efficiency = 0.7  # 需要一定的数据搬运
        else:
            efficiency = 0.5  # 大量数据搬运

        # FLOPs -> ms
        t_compute_bound = (
            flops / (compute_tflops * 1e9 * efficiency) if compute_tflops > 0 else 0
        )

        # Bytes -> ms（考虑多核并行访存）
        t_memory_bound = (
            total_bytes / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0
        )

        return max(t_compute_bound, t_memory_bound)

    def estimate_comm(
        self,
        comm_bytes: int,
        path_key: str,
        participants: int,
        hardware: dict[str, float],
    ) -> float:
        """估算通信时间（包含片内 NoC）

        输入:
            - comm_bytes: 通信数据量（bytes）
            - path_key: 路径键
            - participants: 参与者数量
            - hardware: 硬件参数
        输出:
            - 通信时间（ms）
        关键步骤:
            - 区分片内（NoC）和片间通信
            - 根据 path_key 选择带宽
        """
        # 片内通信使用 NoC 带宽 (无默认值，缺失时 KeyError)
        if path_key in ("intra_chip", "intra_noc", "intra_core"):
            bw_gbps = hardware["noc_bandwidth_gbps"]
            ring_factor = 1.0
        elif path_key == "c2c":
            bw_gbps = hardware["c2c_bandwidth_gbps"]
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        elif path_key == "b2b":
            bw_gbps = hardware["b2b_bandwidth_gbps"]
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        elif path_key == "r2r":
            bw_gbps = hardware["r2r_bandwidth_gbps"]
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        elif path_key == "p2p":
            bw_gbps = hardware["p2p_bandwidth_gbps"]
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        else:
            bw_gbps = hardware["p2p_bandwidth_gbps"]
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )

        if bw_gbps <= 0:
            return 0.0

        # bytes -> ms
        t_comm = comm_bytes * ring_factor / (bw_gbps * 1e6)
        return t_comm
