"""Core 级代价模型.

考虑多核并行与片内路径键（如 NoC）。
所需参数：cores/SRAM/NoC 带宽。
特点：中/中，适用于优化 Tile/Kernel 相关决策（MVP 主力）。
"""

from __future__ import annotations

from tier6.L4_evaluation.cost_models.base import BaseCostModel


class CoreCostModel(BaseCostModel):
    """Core 级代价模型

    考虑多核并行与片内存储层级。

    所需硬件参数:
        - compute_tflops: 峰值算力（TFLOPS）
        - memory_bandwidth_gbps: 显存带宽（GB/s）
        - num_cores: Core 数量
        - sram_per_core_kb: 每 Core SRAM 容量（KB）
        - noc_bandwidth_gbps: 片内 NoC 带宽（GB/s）
        - intra_board_bw_gbps: 板内互联带宽（GB/s）
        - inter_board_bw_gbps: 板间互联带宽（GB/s）
        - inter_node_bw_gbps: 节点间互联带宽（GB/s）
    """

    def required_fields(self) -> set[str]:
        """返回所需字段"""
        return {
            "compute_tflops",
            "memory_bandwidth_gbps",
            "num_cores",
            "sram_per_core_kb",
            "noc_bandwidth_gbps",
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

        compute_tflops = hardware.get("compute_tflops", 125.0)
        memory_bw_gbps = hardware.get("memory_bandwidth_gbps", 2000.0)
        num_cores = int(hardware.get("num_cores", 64))
        sram_per_core_kb = hardware.get("sram_per_core_kb", 512.0)

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
        # 片内通信使用 NoC 带宽
        if path_key in ("intra_chip", "intra_noc", "intra_core"):
            bw_gbps = hardware.get("noc_bandwidth_gbps", 1000.0)
            # 片内通信较简单，不需要 Ring 系数
            ring_factor = 1.0
        elif path_key in ("intra_board",):
            bw_gbps = hardware.get("intra_board_bw_gbps", 400.0)
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        elif path_key == "inter_board":
            bw_gbps = hardware.get("inter_board_bw_gbps", 200.0)
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        elif path_key == "inter_node":
            bw_gbps = hardware.get("inter_node_bw_gbps", 100.0)
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )
        else:
            bw_gbps = hardware.get("inter_node_bw_gbps", 100.0)
            ring_factor = (
                2 * (participants - 1) / participants if participants > 1 else 1.0
            )

        if bw_gbps <= 0:
            return 0.0

        # bytes -> ms
        t_comm = comm_bytes * ring_factor / (bw_gbps * 1e6)
        return t_comm
