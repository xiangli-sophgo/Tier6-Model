"""
Dispatch 通信算子

用于 MoE 中分发 token 到各专家
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import CommunicationOperator, CommOpType


@dataclass
class DispatchOperator(CommunicationOperator):
    """
    MoE Dispatch 通信算子

    parallel_params 必须包含:
        - moe_tp: int, MoE 张量并行度
        - ep: int, 专家并行度
        - comm_size: int, 通信数据量 (bytes)
        - batch_size: int, 批次大小
        - comm_protocol: int, 通信协议 (1/2/3)
        - is_prefill: bool, 是否为 prefill 阶段
    """
    name: str = ""
    op_type: CommOpType = CommOpType.DISPATCH
    comm_kind: str = "dispatch"
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后设置通信数据量"""
        self.comm_size = self.parallel_params.get('comm_size', 0)

    @property
    def moe_tp(self) -> int:
        return self.parallel_params.get('moe_tp', 1)

    @property
    def ep(self) -> int:
        return self.parallel_params.get('ep', 1)

    @property
    def batch_size(self) -> int:
        return self.parallel_params.get('batch_size', 1)

    @property
    def comm_protocol(self) -> int:
        return self.parallel_params.get('comm_protocol', 1)

    @property
    def is_prefill(self) -> bool:
        return self.parallel_params.get('is_prefill', False)
