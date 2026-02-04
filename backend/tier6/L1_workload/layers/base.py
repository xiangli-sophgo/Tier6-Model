"""层基类模块

定义 LayerBase 抽象基类和 LayerRole 枚举。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from tier6.core.types import DataType
from tier6.L1_workload.graph import GraphEdge, GraphNode, NodeKind, NodeRole
from tier6.L1_workload.layer import Layer
from tier6.L1_workload.specs import (
    CommSpec,
    TileableDim,
    TileConfig,
    TiledMemoryInfo,
)
from tier6.L1_workload.tensor import TensorDesc
from tier6.L1_workload.dtype_utils import resolve_layer_dtypes

if TYPE_CHECKING:
    from tier6.L1_workload.operators.base import OpBase


class LayerRole(Enum):
    """层角色"""

    COMPUTE = "compute"  # 计算密集型（MatMul/Conv）
    MEMORY = "memory"  # 内存密集型（Reshape/Transpose）
    COMM = "comm"  # 通信型（AllReduce/AllGather）


class LayerBase(ABC):
    """层基类

    定义层的计算与内存特征接口。

    设计原则:
        - 层是性能建模的基本单元
        - 每个层应能独立计算 FLOPs / Memory / Comm
        - 层可包含多个 Op（如 FFN = Linear + Activation + Linear）

    子类需实现:
        - op_type: 层类型
        - get_inputs: 获取输入张量描述
        - get_outputs: 获取输出张量描述
        - compute_flops: 计算 FLOPs
        - compute_memory: 计算内存占用

    Example:
        >>> @layer_registry.register("attention")
        ... class AttentionLayer(LayerBase):
        ...     @property
        ...     def op_type(self) -> str:
        ...         return "attention"
        ...
        ...     def compute_flops(self) -> int:
        ...         b, s, h = self._config["batch"], self._config["seq_len"], self._config["hidden_size"]
        ...         return b * s * (4 * h * h + 2 * s * h + h * h)
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        *,
        dtype: DataType = DataType.FP16,
    ):
        """初始化层

        Args:
            name: 层名称（唯一标识）
            config: 层配置（hidden_size/intermediate_size 等）
            dtype: 数据类型
        """
        self._name = name
        self._config = config
        dtypes = resolve_layer_dtypes(config, default=dtype)
        self._activation_dtype = dtypes["activation"]
        self._weight_dtype = dtypes["weight"]
        self._output_dtype = dtypes["output"]
        self._accum_dtype = dtypes["accum"]
        self._dtype = self._activation_dtype
        self._ops: list[OpBase] = []

    @property
    def name(self) -> str:
        """层名称"""
        return self._name

    @property
    @abstractmethod
    def op_type(self) -> str:
        """层类型，如 'attention' / 'ffn' / 'layernorm'"""
        ...

    @property
    def role(self) -> LayerRole:
        """层角色，默认为计算型，子类可覆盖"""
        return LayerRole.COMPUTE

    @property
    def activation_dtype(self) -> DataType:
        return self._activation_dtype

    @property
    def weight_dtype(self) -> DataType:
        return self._weight_dtype

    @property
    def output_dtype(self) -> DataType:
        return self._output_dtype

    @property
    def accum_dtype(self) -> DataType:
        return self._accum_dtype

    def _tensor(
        self,
        name: str,
        shape: list[int],
        *,
        is_weight: bool = False,
        is_output: bool = False,
        dtype: DataType | None = None,
        layout: str | None = None,
        producer_id: str | None = None,
        consumer_id: str | None = None,
        layout_signature: Any | None = None,
    ) -> TensorDesc:
        if dtype is None:
            if is_weight:
                dtype = self._weight_dtype
            elif is_output:
                dtype = self._output_dtype
            else:
                dtype = self._activation_dtype
        return TensorDesc(
            name=name,
            shape=shape,
            dtype=dtype,
            is_weight=is_weight,
            layout=layout,
            producer_id=producer_id,
            consumer_id=consumer_id,
            layout_signature=layout_signature,
        )

    def _matmul_op(self, name: str, params: dict[str, int]) -> "OpBase":
        from tier6.L1_workload.operators.compute.matmul import MatMulOp

        return MatMulOp(
            name,
            params,
            dtype=self._activation_dtype,
            weight_dtype=self._weight_dtype,
            output_dtype=self._output_dtype,
            accum_dtype=self._accum_dtype,
        )

    @abstractmethod
    def get_inputs(self) -> list[TensorDesc]:
        """获取输入张量描述"""
        ...

    @abstractmethod
    def get_outputs(self) -> list[TensorDesc]:
        """获取输出张量描述"""
        ...

    @abstractmethod
    def compute_flops(self) -> int:
        """计算 FLOPs

        Returns:
            int: 浮点运算次数
        """
        ...

    @abstractmethod
    def compute_memory(self) -> tuple[int, int]:
        """计算内存占用

        Returns:
            tuple[int, int]: (weight_bytes, activation_bytes)
        """
        ...

    def get_comm_spec(self) -> CommSpec | None:
        """获取通信规格，默认无通信"""
        return None

    def build_intra_graph(
        self, layer_node_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge], list[str], list[str]]:
        """构建层内 OP 级计算图（默认空实现）

        计算流程:
            1. 生成 OP 级节点列表
            2. 生成 OP 级依赖边列表
            3. 标记层内入口/出口 OP 节点

        参数说明:
            - layer_node_id: 模型级 Layer 节点 ID，用于生成唯一的 OP 节点前缀

        Returns:
            tuple: (nodes, edges, entry_nodes, exit_nodes)
                - nodes: OP 级节点列表
                - edges: OP 级边列表
                - entry_nodes: 层内入口 OP 节点 ID 列表
                - exit_nodes: 层内出口 OP 节点 ID 列表

        说明:
            - L1 仅负责结构表达，不负责通信插入与时延建模（由 L3/L4 处理）
            - 若子类未覆盖此方法，则仅保留 Layer 级节点与层间连边
        """
        return ([], [], [], [])

    # =========================================================================
    # Tile-Aware 接口（供 L3 Layer Group 搜索使用）
    # =========================================================================

    def get_tileable_dims(self) -> list[TileableDim]:
        """返回可 tiling 的维度列表

        子类应覆盖此方法，提供具体的可分块维度信息。

        Returns:
            list[TileableDim]: 可 tiling 的维度描述列表

        Example:
            >>> # MatMul: C[M,N] = A[M,K] @ B[K,N]
            >>> return [
            ...     TileableDim(name="m", size=batch*seq, min_tile=64, alignment=64),
            ...     TileableDim(name="n", size=hidden, min_tile=64, alignment=64),
            ...     TileableDim(name="k", size=hidden, min_tile=256, is_reduction=True),
            ... ]
        """
        return []

    def get_tiled_memory(self, tile_config: TileConfig) -> TiledMemoryInfo:
        """给定 tiling 配置，计算实际内存需求

        子类应覆盖此方法，提供精确的 tiling 内存计算。

        Args:
            tile_config: Tiling 配置（tile_m, tile_n, tile_k 等）

        Returns:
            TiledMemoryInfo: 包含各类 buffer 大小和峰值 LMEM 需求

        Example:
            >>> # MatMul: C[M,N] = A[M,K] @ B[K,N]
            >>> tile_m = tile_config.tile_m or M
            >>> tile_n = tile_config.tile_n or N
            >>> tile_k = tile_config.tile_k or K
            >>> return TiledMemoryInfo(
            ...     input_buffer=tile_m * tile_k * dtype_size,
            ...     weight_buffer=tile_k * tile_n * dtype_size,
            ...     output_buffer=tile_m * tile_n * dtype_size,
            ...     peak_lmem=...,
            ... )
        """
        # 默认实现：返回未分块的内存需求
        weight_bytes, activation_bytes = self.compute_memory()
        return TiledMemoryInfo(
            input_buffer=activation_bytes // 2,  # 粗略估计
            output_buffer=activation_bytes // 2,
            weight_buffer=weight_bytes,
            peak_lmem=weight_bytes + activation_bytes,
        )

    def get_recompute_cost(self, tile_config: TileConfig) -> int:
        """给定 tiling 配置，计算重计算开销

        当 K 维度被 tiling 时，可能需要多次读取输入或累加结果。

        Args:
            tile_config: Tiling 配置

        Returns:
            int: 重计算导致的额外 FLOPs（不含原始 FLOPs）
        """
        # 默认实现：检查 reduction 维度是否被 tiling
        tileable_dims = self.get_tileable_dims()
        base_flops = self.compute_flops()

        for dim in tileable_dims:
            if dim.is_reduction:
                tile_size = tile_config.get(dim.name)
                if tile_size and tile_size < dim.size:
                    # K 维度被 tiling，需要多次累加
                    num_k_tiles = (dim.size + tile_size - 1) // tile_size
                    # 累加开销通常较小，这里简化处理
                    return 0  # 子类可覆盖提供更精确的估计

        return 0

    def can_fuse_with(self, next_layer: "LayerBase") -> bool:
        """判断是否可以与下一层融合

        融合条件（默认实现）：
        - 输出 shape 与下一层输入 shape 兼容
        - 中间结果可以留在 LMEM

        Args:
            next_layer: 下一层

        Returns:
            bool: 是否可融合
        """
        # 默认：检查输出输入 shape 是否匹配
        my_outputs = self.get_outputs()
        next_inputs = next_layer.get_inputs()

        if not my_outputs or not next_inputs:
            return False

        # 简单检查：第一个输出的 shape 与第一个输入匹配
        return my_outputs[0].shape == next_inputs[0].shape

    def get_ops(self) -> list["OpBase"]:
        """获取层内算子列表"""
        if not self._ops:
            self._ops = self._build_ops()
        return self._ops

    def _build_ops(self) -> list["OpBase"]:
        """构建层内算子，子类可覆盖"""
        return []

    def get_info(self) -> dict[str, Any]:
        """获取层汇总信息

        Returns:
            dict: 包含 name/op_type/flops/weight_bytes/activation_bytes/comm_hint_bytes
        """
        weight_bytes, activation_bytes = self.compute_memory()
        comm = self.get_comm_spec()
        return {
            "name": self.name,
            "op_type": self.op_type,
            "role": self.role.value,
            "flops": self.compute_flops(),
            "weight_bytes": weight_bytes,
            "activation_bytes": activation_bytes,
            "comm_hint_bytes": comm.bytes if comm else 0,
        }

    def to_layer(self) -> Layer:
        """转换为 Layer 数据结构"""
        return Layer(
            name=self.name,
            op_type=self.op_type,
            inputs=self.get_inputs(),
            outputs=self.get_outputs(),
            params=self._config,
            ops=[op.to_op() for op in self.get_ops()],
            comm=self.get_comm_spec(),
            attrs={"role": self.role.value},
        )
