"""
算子基类定义

ComputeOperator: 计算算子基类 (MatMul, FA2, Softmax, RMSNorm)
CommunicationOperator: 通信算子基类 (AllReduce, AllGather, Dispatch, Combine)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from enum import Enum, auto


class ComputeOpType(Enum):
    """计算算子类型枚举"""
    MATMUL = auto()
    FA2 = auto()
    MHA = auto()      # Multi-Head Attention (用于 MLA/MLAv32)
    MQA = auto()      # Multi-Query Attention (用于 MLAAbsorb/MLAAbsorbv32)
    SOFTMAX = auto()
    RMSNORM = auto()


class CommOpType(Enum):
    """通信算子类型枚举"""
    ALLREDUCE = auto()
    ALLGATHER = auto()
    REDUCESCATTER = auto()
    DISPATCH = auto()
    COMBINE = auto()


@dataclass
class ComputeOperator:
    """
    计算算子基类

    存储算子的输入参数和评估结果
    """
    name: str
    op_type: ComputeOpType
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    # 评估结果 (由 Evaluator 填充)
    param: int = 0                    # 权重参数量
    flops: int = 0                    # 浮点操作数
    dram_occupy: int = 0              # DRAM 占用 (bytes)
    dram_traffic: int = 0             # DRAM 流量 (bytes)
    elapse: float = 0.0               # 总执行时间 (us)
    comp_elapse: float = 0.0          # 计算时间 (us)
    dma_elapse: float = 0.0           # DMA 传输时间 (us)
    urate: float = 0.0                # 硬件利用率 (0-1)

    # 最优配置 (由 Evaluator 填充)
    best_tile: Optional[Dict[str, Any]] = None
    best_partition: Optional[Dict[str, Any]] = None

    @property
    def operator_id(self) -> str:
        return f"{self.name}_id"

    @property
    def operator_type(self) -> str:
        return self.__class__.__name__

    def get_cache_key(self) -> Tuple:
        """生成缓存键"""
        return (self.__class__.__name__, tuple(sorted(self.parallel_params.items())))

    def get_info(self) -> Dict[str, Any]:
        """返回算子信息字典 (用于 JSON 输出)"""
        info = {
            'operator_id': self.operator_id,
            'operator_type': self.operator_type,
            'name': self.name,
            'param': self.param,
            'flops': self.flops,
            'dram_occupy': self.dram_occupy,
            'dram_traffic': self.dram_traffic,
            'elapse': self.elapse,
            'comp_elapse': self.comp_elapse,
            'dma_elapse': self.dma_elapse,
            'urate': self.urate,
            'parallel_params': self.parallel_params,
        }
        if self.best_tile is not None:
            info['best_tile'] = self.best_tile
        if self.best_partition is not None:
            info['best_partition'] = self.best_partition
        return info

    def apply_result(self, result: Dict[str, Any]):
        """从评估结果字典应用到算子属性"""
        for key, value in result.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class CommunicationOperator:
    """
    通信算子基类

    存储通信算子的参数和评估结果
    """
    name: str
    op_type: CommOpType
    comm_kind: str  # 'allreduce', 'allgather', 'dispatch', 'combine', 'reducescatter'
    parallel_params: Dict[str, Any] = field(default_factory=dict)

    # 评估结果 (由 Evaluator 填充)
    comm_size: int = 0                # 通信数据量 (bytes)
    comm_elapse: float = 0.0          # 通信延迟 (us)

    @property
    def operator_id(self) -> str:
        return f"{self.name}_id"

    @property
    def operator_type(self) -> str:
        return self.__class__.__name__

    def get_cache_key(self) -> Tuple:
        """生成缓存键"""
        return (self.__class__.__name__, tuple(sorted(self.parallel_params.items())))

    def get_info(self) -> Dict[str, Any]:
        """返回算子信息字典 (用于 JSON 输出)"""
        return {
            'operator_id': self.operator_id,
            'operator_type': self.operator_type,
            'comm_kind': self.comm_kind,
            'name': self.name,
            'comm_size': self.comm_size,
            'comm_elapse': self.comm_elapse,
            'parallel_params': self.parallel_params,
        }

    def apply_result(self, result: Dict[str, Any]):
        """从评估结果字典应用到算子属性"""
        for key, value in result.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
