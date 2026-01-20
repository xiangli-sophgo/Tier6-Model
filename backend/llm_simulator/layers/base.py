"""
层基类定义

BaseLayer 提供:
- 算子注册和管理
- 性能指标聚合
- 信息导出接口
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..operators.base import ComputeOperator, CommunicationOperator


@dataclass
class BaseLayer:
    """
    层基类

    每个 Layer 包含多个计算算子和通信算子
    """
    name: str = ""
    layer_type: str = "BaseLayer"

    # 算子列表
    comp_ops: List[ComputeOperator] = field(default_factory=list)
    comm_ops: List[CommunicationOperator] = field(default_factory=list)

    # 按类型分组的算子 (用于 Model.operator_map)
    operator_categories: Dict[str, List] = field(default_factory=dict)

    # 聚合后的性能指标
    param: int = 0                    # 参数量
    flops: int = 0                    # 浮点操作数
    dram_occupy: int = 0              # DRAM 占用
    dram_traffic: int = 0             # DRAM 流量
    elapse: float = 0.0               # 总执行时间 (us)
    comp_elapse: float = 0.0          # 计算时间 (us)
    dma_elapse: float = 0.0           # DMA 时间 (us)
    comm_elapse: float = 0.0          # 通信时间 (us)
    comm_size: int = 0                # 通信数据量

    def add_operator(self, operator):
        """添加算子到层"""
        if isinstance(operator, ComputeOperator):
            self.comp_ops.append(operator)
            op_type = operator.operator_type
        elif isinstance(operator, CommunicationOperator):
            self.comm_ops.append(operator)
            op_type = operator.comm_kind
        else:
            raise TypeError(f"Unknown operator type: {type(operator)}")

        # 注册到分类字典
        if op_type not in self.operator_categories:
            self.operator_categories[op_type] = []
        self.operator_categories[op_type].append(operator)

    def aggregate_metrics(self) -> Dict[str, float]:
        """聚合所有算子的性能指标"""
        # 重置
        self.param = 0
        self.flops = 0
        self.dram_occupy = 0
        self.dram_traffic = 0
        self.elapse = 0.0
        self.comp_elapse = 0.0
        self.dma_elapse = 0.0
        self.comm_elapse = 0.0
        self.comm_size = 0

        # 聚合计算算子
        for op in self.comp_ops:
            self.param += op.param
            self.flops += op.flops
            self.dram_occupy += op.dram_occupy
            self.dram_traffic += op.dram_traffic
            self.elapse += op.elapse
            self.comp_elapse += op.comp_elapse
            self.dma_elapse += op.dma_elapse

        # 聚合通信算子
        for op in self.comm_ops:
            self.comm_elapse += op.comm_elapse
            self.comm_size += op.comm_size
            self.elapse += op.comm_elapse

        return {
            'param': self.param,
            'flops': self.flops,
            'dram_occupy': self.dram_occupy,
            'dram_traffic': self.dram_traffic,
            'elapse': self.elapse,
            'comp_elapse': self.comp_elapse,
            'dma_elapse': self.dma_elapse,
            'comm_elapse': self.comm_elapse,
            'comm_size': self.comm_size,
        }

    def get_info(self) -> Dict[str, Any]:
        """返回层信息字典 (用于 JSON 输出)"""
        return {
            'name': self.name,
            'layer_type': self.layer_type,
            'total_operators': len(self.comp_ops) + len(self.comm_ops),
            'perf': {
                'param': self.param,
                'flops': self.flops,
                'dram_occupy': self.dram_occupy,
                'dram_traffic': self.dram_traffic,
                'elapse': self.elapse,
                'comp_elapse': self.comp_elapse,
                'dma_elapse': self.dma_elapse,
                'comm_elapse': self.comm_elapse,
                'comm_size': self.comm_size,
            },
            'comp_operators': [op.get_info() for op in self.comp_ops],
            'comm_operators': [op.get_info() for op in self.comm_ops],
        }
