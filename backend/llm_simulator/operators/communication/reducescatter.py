"""
ReduceScatter 通信算子

用于张量并行中的规约-散射
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import CommunicationOperator, CommOpType


@dataclass
class ReduceScatterOperator(CommunicationOperator):
    """
    ReduceScatter 通信算子

    parallel_params 必须包含:
        - tp: int, 张量并行度
        - comm_size: int, 通信数据量 (bytes)
        - comm_protocol: int, 通信协议 (1/2/3)
    """
    name: str = ""
    op_type: CommOpType = CommOpType.REDUCESCATTER
    comm_kind: str = "reducescatter"
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后设置通信数据量"""
        self.comm_size = self.parallel_params.get('comm_size', 0)

    @property
    def tp(self) -> int:
        return self.parallel_params.get('tp', 1)

    @property
    def comm_protocol(self) -> int:
        return self.parallel_params.get('comm_protocol', 1)
