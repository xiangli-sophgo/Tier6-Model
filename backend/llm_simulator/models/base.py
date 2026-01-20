"""
模型基类定义

BaseModel 提供:
- Layer 管理
- operator_map 构建
- 性能指标聚合
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

from ..layers.base import BaseLayer


@dataclass
class BaseModel:
    """
    模型基类

    每个 Model 包含多个 Layer，并提供:
    - operator_map: 按算子类型分组的所有算子
    - 性能指标聚合方法
    """
    name: str = ""
    model_type: str = "BaseModel"

    # 层列表
    layers: List[BaseLayer] = field(default_factory=list)

    # 层数映射 (layer_name -> count)
    layer_counts: Dict[str, int] = field(default_factory=dict)

    # 按算子类型分组 (用于 Analyzer 调度)
    operator_map: Dict[str, List] = field(default_factory=dict)
    operator_types: set = field(default_factory=set)

    # 聚合后的性能指标
    total_param: int = 0
    total_flops: int = 0
    total_dram_occupy: int = 0
    total_elapse: float = 0.0
    total_comm_elapse: float = 0.0

    def add_layer(self, layer: BaseLayer, count: int = 1):
        """添加层到模型"""
        self.layers.append(layer)
        self.layer_counts[layer.name] = count

        # 将层的算子注册到 operator_map
        for op_type, ops in layer.operator_categories.items():
            self.operator_types.add(op_type)
            if op_type not in self.operator_map:
                self.operator_map[op_type] = []
            self.operator_map[op_type].extend(ops)

    def build_operator_map(self):
        """重建 operator_map (在所有层添加完成后调用)"""
        self.operator_map = {}
        self.operator_types = set()

        for layer in self.layers:
            for op_type, ops in layer.operator_categories.items():
                self.operator_types.add(op_type)
                if op_type not in self.operator_map:
                    self.operator_map[op_type] = []
                self.operator_map[op_type].extend(ops)

    def aggregate_metrics(self) -> Dict[str, Any]:
        """聚合所有层的性能指标"""
        self.total_param = 0
        self.total_flops = 0
        self.total_dram_occupy = 0
        self.total_elapse = 0.0
        self.total_comm_elapse = 0.0

        for layer in self.layers:
            # 先聚合层内指标
            layer.aggregate_metrics()

            # 乘以层数
            count = self.layer_counts.get(layer.name, 1)
            self.total_param += layer.param * count
            self.total_flops += layer.flops * count
            self.total_dram_occupy += layer.dram_occupy * count
            self.total_elapse += layer.elapse * count
            self.total_comm_elapse += layer.comm_elapse * count

        return {
            'total_param': self.total_param,
            'total_flops': self.total_flops,
            'total_dram_occupy': self.total_dram_occupy,
            'total_elapse_us': self.total_elapse,
            'total_comm_elapse_us': self.total_comm_elapse,
        }

    def analyze_performance(self, batch_size: int = 1, seq_len: int = 1,
                           tpu_flops: float = 0) -> Dict[str, Any]:
        """
        生成性能分析摘要

        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            tpu_flops: TPU 峰值算力 (FLOPS)

        Returns:
            性能分析字典
        """
        metrics = self.aggregate_metrics()

        total_elapse_us = metrics['total_elapse_us']
        total_flops = metrics['total_flops']

        # 计算吞吐量
        tokens = batch_size * seq_len
        tps = 0.0
        if total_elapse_us > 0:
            tps = tokens / (total_elapse_us * 1e-6)

        # 计算 MFU
        mfu = None
        if total_elapse_us > 0 and total_flops > 0 and tpu_flops > 0:
            achieved_flops = total_flops / (total_elapse_us * 1e-6)
            mfu = achieved_flops / tpu_flops

        # 构建层信息
        layers_info = {}
        for layer in self.layers:
            count = self.layer_counts.get(layer.name, 1)
            layer_info = layer.get_info()
            layer_info['count'] = count
            layers_info[layer.name] = layer_info

        return {
            'total_elapse_us': total_elapse_us,
            'total_elapse_ms': total_elapse_us / 1000,
            'comm_elapse_us': metrics['total_comm_elapse_us'],
            'total_param': metrics['total_param'],
            'total_flops': total_flops,
            'dram_occupy': metrics['total_dram_occupy'],
            'tps': tps,
            'tps_per_batch': tps / batch_size if batch_size > 0 else 0,
            'mfu': mfu,
            'layers': layers_info,
        }

    def get_info(self) -> Dict[str, Any]:
        """返回模型信息字典"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'num_layers': len(self.layers),
            'layer_counts': self.layer_counts,
            'operator_types': list(self.operator_types),
        }
