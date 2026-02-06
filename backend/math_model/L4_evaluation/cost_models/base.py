"""代价模型基类."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCostModel(ABC):
    """代价模型基类

    定义代价模型的接口，子类需实现具体的估算逻辑。
    """

    @abstractmethod
    def required_fields(self) -> set[str]:
        """返回该模型所需的最小字段集合

        Returns:
            字段名称集合，如 {"compute_tflops", "memory_bandwidth_gbps"}
        """
        ...

    @abstractmethod
    def estimate_compute(
        self,
        op_type: str,
        local_shape: dict[str, int],
        hardware: dict[str, float],
    ) -> float:
        """估算计算时间

        Args:
            op_type: Op 类型（如 "matmul", "softmax"）
            local_shape: 切分后的 shape（如 {"M": 1024, "N": 1024, "K": 4096}）
            hardware: 硬件参数（如 {"compute_tflops": 125, "memory_bandwidth_gbps": 2000}）

        Returns:
            计算时间（ms）
        """
        ...

    @abstractmethod
    def estimate_comm(
        self,
        comm_bytes: int,
        path_key: str,
        participants: int,
        hardware: dict[str, float],
    ) -> float:
        """估算通信时间

        Args:
            comm_bytes: 通信数据量（bytes）
            path_key: 路径键（如 "intra_board", "inter_board", "inter_node"）
            participants: 参与者数量
            hardware: 硬件参数

        Returns:
            通信时间（ms）
        """
        ...

    def estimate_flops(self, op_type: str, local_shape: dict[str, int]) -> int:
        """估算浮点运算量

        Args:
            op_type: Op 类型
            local_shape: 切分后的 shape

        Returns:
            FLOPs 数量
        """
        m = local_shape.get("M", 0)
        n = local_shape.get("N", 0)
        k = local_shape.get("K", 0)
        g = local_shape.get("G", 1)

        if op_type in ("matmul", "linear", "gemm"):
            # MatMul: 2 * M * N * K
            return 2 * g * m * n * k
        if op_type in ("softmax", "layernorm", "rmsnorm"):
            # Elementwise: ~5 ops per element
            b = local_shape.get("B", 1)
            s = local_shape.get("S", 1)
            h = local_shape.get("H", 1)
            return 5 * b * s * h
        return 0

    def estimate_bytes(
        self, op_type: str, local_shape: dict[str, int], dtype_bytes: int = 2
    ) -> tuple[int, int]:
        """估算访存量

        Args:
            op_type: Op 类型
            local_shape: 切分后的 shape
            dtype_bytes: 数据类型字节数

        Returns:
            (bytes_read, bytes_write)
        """
        m = local_shape.get("M", 0)
        n = local_shape.get("N", 0)
        k = local_shape.get("K", 0)
        g = local_shape.get("G", 1)

        if op_type in ("matmul", "linear", "gemm"):
            # MatMul: read A[M,K] + B[K,N], write C[M,N]
            bytes_read = g * (m * k + k * n) * dtype_bytes
            bytes_write = g * m * n * dtype_bytes
            return bytes_read, bytes_write

        # 默认：输入输出 shape 相同
        b = local_shape.get("B", 1)
        s = local_shape.get("S", 1)
        h = local_shape.get("H", 1)
        elements = b * s * h
        return elements * dtype_bytes, elements * dtype_bytes
