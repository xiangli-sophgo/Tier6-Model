"""
性能分析器 - DS_TPU 风格

PerformanceAnalyzer 负责:
- 调度各类评估器评估算子
- 管理缓存提高效率
- 聚合性能指标
- 生成分析报告
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from .models.base import BaseModel
from .operators.base import ComputeOperator, CommunicationOperator
from .evaluators import (
    get_arch_preset,
    AcceleratorMicroArch,
    GEMMEvaluator,
    FA2Evaluator,
    RMSNormEvaluator,
    AllReduceEval,
    AllGatherEval,
    ReduceScatterEval,
    DispatchEval,
    CombineEval,
)
from .types import ProtocolConfig, NetworkInfraConfig


class PerformanceAnalyzer:
    """
    性能分析器

    调度评估器对模型中的所有算子进行评估，并聚合性能指标

    Args:
        model: 模型实例
        arch: 加速器微架构配置
        global_cache: 全局缓存 (跨多次分析复用)
    """

    # 需要从评估结果提取的属性
    _COMPUTE_RESULT_ATTRS = (
        'elapse', 'comp_elapse', 'dma_elapse',
        'dram_traffic', 'urate', 'best_tile', 'best_partition'
    )
    _COMM_RESULT_ATTRS = ('comm_elapse',)

    def __init__(
        self,
        model: BaseModel,
        arch: AcceleratorMicroArch,
        global_cache: Optional[Dict] = None,
        protocol_config: Optional[ProtocolConfig] = None,
        network_config: Optional[NetworkInfraConfig] = None,
        moe_topk: int = 8,
        prefill_factor: float = 8 / 128,
    ):
        self.model = model
        self.arch = arch
        self.global_cache = global_cache or {}

        # 初始化评估器 (传递配置参数)
        self.evaluators = {
            'MatMulOperator': GEMMEvaluator(arch),
            'FA2Operator': FA2Evaluator(arch),
            'MHAOperator': FA2Evaluator(arch),  # MHA 复用 FA2 评估器
            'MQAOperator': FA2Evaluator(arch),  # MQA 复用 FA2 评估器
            'RMSNormOperator': RMSNormEvaluator(arch),
            'allreduce': AllReduceEval(arch, protocol_config, network_config),
            'allgather': AllGatherEval(arch, protocol_config, network_config),
            'reducescatter': ReduceScatterEval(arch, protocol_config, network_config),
            'dispatch': DispatchEval(arch, protocol_config, network_config, moe_topk, prefill_factor),
            'combine': CombineEval(arch, protocol_config, network_config, moe_topk, prefill_factor),
        }

        # 执行分析
        self._analyze_model()

    def _analyze_model(self):
        """遍历所有算子并评估"""
        print(f"开始分析模型: {self.model.name}")
        print(f"算子类型: {list(self.model.operator_types)}")

        # 按算子类型分组评估
        for op_type, operators in self.model.operator_map.items():
            evaluator = self._get_evaluator(op_type)
            if evaluator is None:
                print(f"  警告: 未找到 {op_type} 的评估器，跳过")
                continue

            self._evaluate_operators(op_type, evaluator, operators)

        # 更新层级指标
        self._update_layer_metrics()

    def _get_evaluator(self, op_type: str):
        """获取算子对应的评估器"""
        # 直接匹配
        if op_type in self.evaluators:
            return self.evaluators[op_type]

        # 尝试通信算子类型
        comm_types = ['allreduce', 'allgather', 'reducescatter', 'dispatch', 'combine']
        for ct in comm_types:
            if ct in op_type.lower():
                return self.evaluators.get(ct)

        return None

    def _evaluate_operators(self, op_type: str, evaluator, operators: List):
        """批量评估算子"""
        cache_hits = 0
        total_ops = len(operators)

        for operator in operators:
            cache_key = operator.get_cache_key()

            if cache_key in self.global_cache:
                # 复用全局缓存
                self._apply_cached_result(operator, self.global_cache[cache_key])
                cache_hits += 1
            else:
                # 计算新结果
                result = self._evaluate_single(operator, evaluator)
                self.global_cache[cache_key] = result

        hit_rate = cache_hits / total_ops * 100 if total_ops > 0 else 0
        print(f"  {op_type}: {total_ops} 算子, {cache_hits} 缓存命中 ({hit_rate:.1f}%)")

    def _evaluate_single(self, operator, evaluator) -> Dict[str, Any]:
        """评估单个算子"""
        if isinstance(operator, ComputeOperator):
            return self._evaluate_compute_op(operator, evaluator)
        elif isinstance(operator, CommunicationOperator):
            return self._evaluate_comm_op(operator, evaluator)
        else:
            return {}

    def _evaluate_compute_op(self, operator: ComputeOperator, evaluator) -> Dict[str, Any]:
        """评估计算算子"""
        params = operator.parallel_params
        result = {}

        if operator.operator_type == 'MatMulOperator':
            # GEMM 评估
            gemm_result = evaluator.evaluate(
                G=params.get('G', 1),
                M=params.get('M', 1),
                K=params.get('K', 1),
                N=params.get('N', 1),
                input_dtype=params.get('input_dtype', 'bf16'),
                output_dtype=params.get('output_dtype', 'bf16'),
            )
            result = {
                'elapse': gemm_result.latency_us,
                'comp_elapse': gemm_result.compute_time_us,
                'dma_elapse': gemm_result.memory_time_us,
                'dram_traffic': gemm_result.dram_traffic_bytes,
                'urate': gemm_result.effective_utilization,
                'best_tile': {
                    'M': gemm_result.best_tile[0],
                    'N': gemm_result.best_tile[1],
                    'K': gemm_result.best_tile[2],
                },
                'best_partition': {
                    'P_G': gemm_result.best_partition[0],
                    'P_M': gemm_result.best_partition[1],
                    'P_N': gemm_result.best_partition[2],
                    'P_K': gemm_result.best_partition[3],
                },
            }

        elif operator.operator_type == 'FA2Operator':
            # Flash Attention 评估
            fa2_result = evaluator.evaluate(
                B=params.get('B', 1),
                QS=params.get('QS', 1),
                KS=params.get('KS', 1),
                QD=params.get('QD', 1),
                VD=params.get('VD', 1),
            )
            result = {
                'elapse': fa2_result.latency_us,
                'comp_elapse': fa2_result.compute_time_us,
                'dma_elapse': fa2_result.memory_time_us,
                'dram_traffic': fa2_result.dram_traffic_bytes,
                'urate': fa2_result.effective_utilization,
                'best_tile': {
                    'Q': fa2_result.best_tile[0],
                    'K': fa2_result.best_tile[1],
                },
                'best_partition': {
                    'P_B': fa2_result.best_partition,
                },
            }

        elif operator.operator_type == 'MHAOperator':
            # MHA 评估 (Multi-Head Attention)
            # MHA 有 B 和 H 两个维度，等效 B_eff = B * H
            B = params.get('B', 1)
            H = params.get('H', 1)
            B_eff = B * H  # 等效 batch
            fa2_result = evaluator.evaluate(
                B=B_eff,
                QS=params.get('QS', 1),
                KS=params.get('KS', 1),
                QD=params.get('QD', 1),
                VD=params.get('VD', 1),
            )
            result = {
                'elapse': fa2_result.latency_us,
                'comp_elapse': fa2_result.compute_time_us,
                'dma_elapse': fa2_result.memory_time_us,
                'dram_traffic': fa2_result.dram_traffic_bytes,
                'urate': fa2_result.effective_utilization,
                'best_tile': {
                    'Q': fa2_result.best_tile[0],
                    'K': fa2_result.best_tile[1],
                },
                'best_partition': {
                    'P_B': fa2_result.best_partition,
                },
            }

        elif operator.operator_type == 'MQAOperator':
            # MQA 评估 (Multi-Query Attention)
            # MQA 的参数格式与 FA2 兼容
            fa2_result = evaluator.evaluate(
                B=params.get('B', 1),
                QS=params.get('QS', 1),
                KS=params.get('KS', 1),
                QD=params.get('QD', 1),
                VD=params.get('VD', 1),
            )
            result = {
                'elapse': fa2_result.latency_us,
                'comp_elapse': fa2_result.compute_time_us,
                'dma_elapse': fa2_result.memory_time_us,
                'dram_traffic': fa2_result.dram_traffic_bytes,
                'urate': fa2_result.effective_utilization,
                'best_tile': {
                    'Q': fa2_result.best_tile[0],
                    'K': fa2_result.best_tile[1],
                },
                'best_partition': {
                    'P_B': fa2_result.best_partition,
                },
            }

        elif operator.operator_type == 'RMSNormOperator':
            # RMSNorm 评估 - 内存带宽受限
            batch_size = params.get('batch_size', 1)
            hidden_dim = params.get('hidden_dim', 1)
            dtype_bytes = 2  # bf16

            # RMSNorm 需要读写输入数据
            data_bytes = batch_size * hidden_dim * dtype_bytes * 2  # 读 + 写
            latency_us = (data_bytes / self.arch.dram_bandwidth_bytes) * 1e6

            # 使用 RMSNorm 评估器获取利用率
            rmsnorm_result = evaluator.evaluate(
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                has_scale=params.get('has_scale', True),
                has_bias=params.get('has_bias', False),
            )
            result = {
                'elapse': latency_us,
                'comp_elapse': latency_us * 0.1,  # 计算占比很小
                'dma_elapse': latency_us * 0.9,
                'dram_traffic': data_bytes,
                'urate': rmsnorm_result.utilization,
            }

        else:
            # 其他计算算子 (Softmax 等) - 简化评估
            # 使用内存带宽模型
            data_bytes = params.get('batch_size', 1) * params.get('hidden_dim', 1) * 2
            latency_us = (data_bytes / self.arch.dram_bandwidth_bytes) * 1e6
            result = {
                'elapse': latency_us,
                'comp_elapse': 0,
                'dma_elapse': latency_us,
                'dram_traffic': data_bytes,
                'urate': 0.0,
            }

        # 应用结果到算子
        operator.apply_result(result)
        return result

    def _evaluate_comm_op(self, operator: CommunicationOperator, evaluator) -> Dict[str, Any]:
        """评估通信算子"""
        params = operator.parallel_params
        result = {}

        comm_kind = operator.comm_kind
        tp = params.get('tp', params.get('moe_tp', 1))
        comm_size = params.get('comm_size', 0)
        comm_protocol = params.get('comm_protocol', 1)

        if comm_kind == 'allreduce':
            comm_result = evaluator.evaluate(tp, comm_size, comm_protocol)
            latency_us = comm_result.latency_us
        elif comm_kind == 'allgather':
            comm_result = evaluator.evaluate(tp, comm_size, comm_protocol)
            latency_us = comm_result.latency_us
        elif comm_kind == 'reducescatter':
            comm_result = evaluator.evaluate(tp, comm_size, comm_protocol)
            latency_us = comm_result.latency_us
        elif comm_kind in ('dispatch', 'combine'):
            # Dispatch/Combine 近似为 AllToAll
            ep = params.get('ep', 1)
            # 简化: 使用 AllReduce 评估器
            comm_result = evaluator.evaluate(ep, comm_size, comm_protocol)
            latency_us = comm_result.latency_us
        else:
            latency_us = 0.0

        result = {'comm_elapse': latency_us}
        operator.apply_result(result)
        return result

    def _apply_cached_result(self, operator, cached_result: Dict[str, Any]):
        """应用缓存的评估结果"""
        operator.apply_result(cached_result)

    def _update_layer_metrics(self):
        """更新层级性能指标"""
        for layer in self.model.layers:
            layer.aggregate_metrics()

    def get_summary(self, batch_size: int = 1, seq_len: int = 1) -> Dict[str, Any]:
        """
        生成性能分析摘要

        Returns:
            完整的性能分析报告
        """
        # 计算 TPU 峰值算力
        tpu_flops = self.arch.num_cores * self.arch.cube_m * self.arch.cube_n * self.arch.cube_k * 2 * self.arch.freq_ghz * 1e9

        # 聚合模型性能
        perf = self.model.analyze_performance(
            batch_size=batch_size,
            seq_len=seq_len,
            tpu_flops=tpu_flops,
        )

        return {
            'run_info': {
                'timestamp': datetime.now().isoformat(),
            },
            'config': {
                'model': self.model.get_info(),
                'arch': {
                    'name': getattr(self.arch, 'name', 'unknown'),
                    'cores': self.arch.num_cores,
                    'flops': tpu_flops,
                    'dram_bw': self.arch.dram_bandwidth_bytes,
                },
                'batch_size': batch_size,
                'seq_len': seq_len,
            },
            'performance': perf,
        }


def analyze_model(model: BaseModel, arch_name: str = 'SG2260E',
                  batch_size: int = 1, seq_len: int = 1,
                  global_cache: Optional[Dict] = None) -> Dict[str, Any]:
    """
    分析模型性能的便捷函数

    Args:
        model: 模型实例
        arch_name: 加速器架构名称
        batch_size: 批次大小
        seq_len: 序列长度
        global_cache: 全局缓存

    Returns:
        性能分析报告
    """
    arch = get_arch_preset(arch_name)
    analyzer = PerformanceAnalyzer(model, arch, global_cache)
    return analyzer.get_summary(batch_size, seq_len)
