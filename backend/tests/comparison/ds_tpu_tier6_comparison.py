#!/usr/bin/env python3
"""
DS_TPU vs Tier6 评估结果对比脚本

功能：
1. 使用统一的配置参数调用两边的评估
2. 提取并对比层级、算子级别的性能指标
3. 生成对比报告

使用方式：
    python ds_tpu_tier6_comparison.py --mode single_layer  # 单层对比
    python ds_tpu_tier6_comparison.py --mode full_model    # 完整模型对比
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
DS_TPU_PATH = Path(r"c:\Users\DELL\Documents\code\DS_TPU_1209")
TIER6_PATH = Path(r"C:\Users\DELL\Documents\code\Tier6-Model\backend")

sys.path.insert(0, str(DS_TPU_PATH))
sys.path.insert(0, str(TIER6_PATH))


# ============================================
# 统一配置定义
# ============================================

@dataclass
class UnifiedConfig:
    """统一配置参数"""
    # 模型参数
    model_name: str = "DeepSeek-V3"
    hidden_dim: int = 7168
    num_layers: int = 1  # 单层对比
    num_heads: int = 128
    vocab_size: int = 129280
    inter_dim: int = 18432

    # MLA 参数
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE 参数
    num_experts: int = 256
    num_activated_experts: int = 8
    num_shared_experts: int = 1
    moe_inter_dim: int = 2048

    # 部署参数
    batch_size: int = 4
    q_seq_len: int = 1  # Decode 模式
    kv_seq_len: int = 4096
    tp: int = 1
    dp: int = 32
    moe_tp: int = 1
    ep: int = 32
    is_prefill: bool = False
    enable_tp_sp: bool = True
    comm_protocol: int = 1

    # 硬件参数 (SG2260E)
    tpu_cores: int = 64

    # 通信延迟参数 (与 DS_TPU 对齐)
    chip_to_chip_us: float = 0.2
    memory_read_latency_us: float = 0.15
    memory_write_latency_us: float = 0.01
    noc_latency_us: float = 0.05
    die_to_die_latency_us: float = 0.04
    switch_delay_us: float = 1.0
    cable_delay_us: float = 0.025
    rtt_tp_us: float = 0.35
    rtt_ep_us: float = 0.85
    bandwidth_utilization: float = 0.95


# ============================================
# DS_TPU 评估器
# ============================================

class DsTPUEvaluator:
    """DS_TPU 评估封装"""

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._import_modules()

    def _import_modules(self):
        """动态导入 DS_TPU 模块"""
        from config.config_loader import load_model_config
        from config.deployment_config import DeploymentConfig
        from top.simulator import TPUSimulator

        self.load_model_config = load_model_config
        self.DeploymentConfig = DeploymentConfig
        self.TPUSimulator = TPUSimulator

    def evaluate(self) -> Dict[str, Any]:
        """运行 DS_TPU 评估"""
        # 加载模型配置
        model_config = self.load_model_config('deepseek-v3.2')

        # 覆盖层数 (用于单层测试)
        if self.config.num_layers != 61:
            model_config['n_layers'] = self.config.num_layers
            model_config['n_moe_layers'] = max(0, self.config.num_layers - 3)
            model_config['n_dense_layers'] = min(3, self.config.num_layers)

        # 创建部署配置
        deploy_config = self.DeploymentConfig(
            batch_size=self.config.batch_size,
            q_seq_len=self.config.q_seq_len,
            kv_seq_len=self.config.kv_seq_len,
            tp=self.config.tp,
            dp=self.config.dp,
            moe_tp=self.config.moe_tp,
            ep=self.config.ep,
            is_prefill=self.config.is_prefill,
            enable_tp_sp=self.config.enable_tp_sp,
            comm_protocol=self.config.comm_protocol,
        )

        # TPU 参数
        tpu_kwargs = {'core': self.config.tpu_cores}

        # 运行模拟
        simulator = self.TPUSimulator(verbose=False)
        results = simulator.run_simulation(
            model_cfg=model_config,
            deploy_cfg=deploy_config,
            tpu_kwargs=tpu_kwargs,
            model_version='v3.2',
            global_cache={},
        )

        return results


# ============================================
# Tier6 评估器
# ============================================

class Tier6Evaluator:
    """Tier6 评估封装"""

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._import_modules()

    def _import_modules(self):
        """动态导入 Tier6 模块"""
        from llm_simulator.evaluators import get_arch_preset
        from llm_simulator.analyzer import PerformanceAnalyzer
        from llm_simulator.models.deepseek import DeepSeekModel
        from llm_simulator.types import ProtocolConfig, NetworkInfraConfig

        self.get_arch_preset = get_arch_preset
        self.PerformanceAnalyzer = PerformanceAnalyzer
        self.DeepSeekModel = DeepSeekModel
        self.ProtocolConfig = ProtocolConfig
        self.NetworkInfraConfig = NetworkInfraConfig

    def _build_model(self):
        """构建 Tier6 DeepSeek 模型"""
        cfg = self.config

        # 计算 dense/moe 层数
        n_layers = cfg.num_layers
        n_dense_layers = min(3, n_layers)
        n_moe_layers = max(0, n_layers - 3)

        # 模型配置
        model_config = {
            # 模型结构
            'hidden_dim': cfg.hidden_dim,
            'inter_dim': cfg.inter_dim,
            'vocab_size': cfg.vocab_size,
            'n_layers': n_layers,
            'n_dense_layers': n_dense_layers,
            'n_moe_layers': n_moe_layers,
            'num_heads': cfg.num_heads,

            # MLA 参数 (对齐 DS_TPU V3.2)
            'qk_nope_dim': cfg.qk_nope_head_dim,
            'qk_rope_dim': cfg.qk_rope_head_dim,
            'v_head_dim': cfg.v_head_dim,
            'kv_lora_rank': cfg.kv_lora_rank,
            'q_lora_rank': cfg.q_lora_rank,
            'mla_type': 'mla_absorb',  # 使用 absorbed 变体 (对齐 DS_TPU v3.2)
            'enable_tp_sp': cfg.enable_tp_sp,

            # MoE 参数
            'num_experts': cfg.num_experts,
            'num_activated_experts': cfg.num_activated_experts,
            'num_shared_experts': cfg.num_shared_experts,
            'expert_inter_dim': cfg.moe_inter_dim,

            # 部署参数
            'batch_size': cfg.batch_size,
            'seq_len': cfg.q_seq_len,
            'kv_seq_len': cfg.kv_seq_len,
            'tp': cfg.tp,
            'moe_tp': cfg.moe_tp,
            'ep': cfg.ep,
            'comm_protocol': cfg.comm_protocol,
            'is_prefill': cfg.is_prefill,
        }

        return self.DeepSeekModel(name="deepseek-v3", config=model_config)

    def evaluate(self) -> Dict[str, Any]:
        """运行 Tier6 评估"""
        # 获取架构
        arch = self.get_arch_preset('SG2260E')

        # 创建协议配置
        protocol_config = self.ProtocolConfig(
            rtt_tp_us=self.config.rtt_tp_us,
            rtt_ep_us=self.config.rtt_ep_us,
            bandwidth_utilization=self.config.bandwidth_utilization,
        )

        # 创建网络配置
        network_config = self.NetworkInfraConfig(
            switch_delay_us=self.config.switch_delay_us,
            cable_delay_us=self.config.cable_delay_us,
        )

        # 构建模型
        model = self._build_model()

        # 运行分析
        analyzer = self.PerformanceAnalyzer(
            model=model,
            arch=arch,
            global_cache={},
            protocol_config=protocol_config,
            network_config=network_config,
        )

        # 获取摘要 (包含详细的层级和算子信息)
        results = analyzer.get_summary(
            batch_size=self.config.batch_size,
            seq_len=self.config.q_seq_len,
        )

        return results


# ============================================
# 结果对比器
# ============================================

@dataclass
class ComparisonResult:
    """对比结果"""
    metric_name: str
    ds_tpu_value: float
    tier6_value: float
    absolute_diff: float
    relative_diff_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultComparator:
    """结果对比器"""

    def __init__(self, ds_tpu_results: Dict, tier6_results: Dict):
        self.ds_tpu = ds_tpu_results
        self.tier6 = tier6_results
        self.comparisons: List[ComparisonResult] = []

    def compare_total_metrics(self) -> List[ComparisonResult]:
        """对比总体指标"""
        comparisons = []

        # DS_TPU 总体指标
        ds_perf = self.ds_tpu.get('performance', {})
        ds_total_us = ds_perf.get('total_elapse_us', 0)
        ds_comm_us = ds_perf.get('comm_elapse_us', 0)
        ds_mfu = ds_perf.get('mfu', 0)

        # Tier6 总体指标
        t6_perf = self.tier6.get('performance', {})
        t6_total_us = t6_perf.get('total_elapse_us', 0)
        t6_comm_us = t6_perf.get('comm_elapse_us', 0)
        t6_mfu = t6_perf.get('mfu', 0)

        # 对比
        comparisons.append(self._compare('total_elapse_us', ds_total_us, t6_total_us))
        comparisons.append(self._compare('comm_elapse_us', ds_comm_us, t6_comm_us))
        comparisons.append(self._compare('mfu', ds_mfu, t6_mfu))

        self.comparisons.extend(comparisons)
        return comparisons

    def compare_layer_metrics(self) -> List[ComparisonResult]:
        """对比层级指标"""
        comparisons = []

        # DS_TPU 层级
        ds_layers = self.ds_tpu.get('performance', {}).get('layers', {})

        # Tier6 层级
        t6_layers = self.tier6.get('performance', {}).get('layers', {})

        # 对比每一层
        for layer_name, ds_layer in ds_layers.items():
            t6_layer = t6_layers.get(layer_name, {})

            ds_elapse = ds_layer.get('perf', {}).get('elapse', 0)
            t6_elapse = t6_layer.get('perf', {}).get('elapse', 0)

            comparisons.append(self._compare(
                f'layer:{layer_name}:elapse_us',
                ds_elapse,
                t6_elapse
            ))

        self.comparisons.extend(comparisons)
        return comparisons

    def compare_operator_metrics(self) -> List[ComparisonResult]:
        """对比算子级指标"""
        comparisons = []

        # DS_TPU 层级
        ds_layers = self.ds_tpu.get('performance', {}).get('layers', {})

        # Tier6 层级
        t6_layers = self.tier6.get('performance', {}).get('layers', {})

        # 对比每层的算子
        for layer_name, ds_layer in ds_layers.items():
            t6_layer = t6_layers.get(layer_name, {})

            # 计算算子
            ds_comp_ops = ds_layer.get('comp_operators', [])
            t6_comp_ops = t6_layer.get('comp_operators', [])

            for i, ds_op in enumerate(ds_comp_ops):
                t6_op = t6_comp_ops[i] if i < len(t6_comp_ops) else {}
                op_name = ds_op.get('name', f'op_{i}')

                ds_elapse = ds_op.get('elapsed', 0)
                t6_elapse = t6_op.get('elapsed', 0)

                comparisons.append(self._compare(
                    f'op:{layer_name}:{op_name}:elapsed_us',
                    ds_elapse,
                    t6_elapse
                ))

            # 通信算子
            ds_comm_ops = ds_layer.get('comm_operators', [])
            t6_comm_ops = t6_layer.get('comm_operators', [])

            for i, ds_op in enumerate(ds_comm_ops):
                t6_op = t6_comm_ops[i] if i < len(t6_comm_ops) else {}
                op_name = ds_op.get('name', f'comm_{i}')

                ds_elapse = ds_op.get('comm_elapse', 0)
                t6_elapse = t6_op.get('comm_elapse', 0)

                comparisons.append(self._compare(
                    f'comm:{layer_name}:{op_name}:elapsed_us',
                    ds_elapse,
                    t6_elapse
                ))

        self.comparisons.extend(comparisons)
        return comparisons

    def _compare(self, name: str, ds_val: float, t6_val: float) -> ComparisonResult:
        """计算对比结果"""
        abs_diff = t6_val - ds_val
        rel_diff = (abs_diff / ds_val * 100) if ds_val != 0 else 0

        return ComparisonResult(
            metric_name=name,
            ds_tpu_value=ds_val,
            tier6_value=t6_val,
            absolute_diff=abs_diff,
            relative_diff_percent=rel_diff,
        )

    def generate_report(self) -> str:
        """生成对比报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("DS_TPU vs Tier6 评估结果对比报告")
        lines.append(f"生成时间: {datetime.now().isoformat()}")
        lines.append("=" * 80)
        lines.append("")

        # 总体指标
        lines.append("## 总体指标对比")
        lines.append("-" * 80)
        lines.append(f"{'指标名称':<30} {'DS_TPU':<15} {'Tier6':<15} {'差异':<15} {'误差%':<10}")
        lines.append("-" * 80)

        for cmp in self.comparisons:
            if not cmp.metric_name.startswith(('layer:', 'op:', 'comm:')):
                lines.append(
                    f"{cmp.metric_name:<30} "
                    f"{cmp.ds_tpu_value:<15.4f} "
                    f"{cmp.tier6_value:<15.4f} "
                    f"{cmp.absolute_diff:<15.4f} "
                    f"{cmp.relative_diff_percent:<10.2f}%"
                )

        lines.append("")

        # 层级指标
        layer_cmps = [c for c in self.comparisons if c.metric_name.startswith('layer:')]
        if layer_cmps:
            lines.append("## 层级指标对比")
            lines.append("-" * 80)
            for cmp in layer_cmps:
                lines.append(
                    f"{cmp.metric_name:<40} "
                    f"{cmp.ds_tpu_value:<12.4f} "
                    f"{cmp.tier6_value:<12.4f} "
                    f"{cmp.relative_diff_percent:<8.2f}%"
                )
            lines.append("")

        # 算子指标
        op_cmps = [c for c in self.comparisons if c.metric_name.startswith(('op:', 'comm:'))]
        if op_cmps:
            lines.append("## 算子级指标对比")
            lines.append("-" * 80)
            for cmp in op_cmps:
                lines.append(
                    f"{cmp.metric_name:<50} "
                    f"{cmp.ds_tpu_value:<10.4f} "
                    f"{cmp.tier6_value:<10.4f} "
                    f"{cmp.relative_diff_percent:<8.2f}%"
                )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================
# 主函数
# ============================================

def run_comparison(config: UnifiedConfig, output_dir: Path) -> Dict[str, Any]:
    """运行对比"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DS_TPU vs Tier6 评估对比")
    print("=" * 60)
    print(f"\n配置: batch={config.batch_size}, tp={config.tp}, ep={config.ep}, "
          f"layers={config.num_layers}, prefill={config.is_prefill}")
    print("")

    # Step 1: 运行 DS_TPU 评估
    print("[1/3] 运行 DS_TPU 评估...")
    try:
        ds_evaluator = DsTPUEvaluator(config)
        ds_results = ds_evaluator.evaluate()
        print(f"  ✓ DS_TPU 完成: {ds_results.get('performance', {}).get('total_elapse_us', 0):.2f} μs")
    except Exception as e:
        print(f"  ✗ DS_TPU 失败: {e}")
        ds_results = {}

    # Step 2: 运行 Tier6 评估
    print("[2/3] 运行 Tier6 评估...")
    try:
        t6_evaluator = Tier6Evaluator(config)
        t6_results = t6_evaluator.evaluate()
        print(f"  ✓ Tier6 完成: {t6_results.get('performance', {}).get('total_elapse_us', 0):.2f} μs")
    except Exception as e:
        print(f"  ✗ Tier6 失败: {e}")
        t6_results = {}

    # Step 3: 对比结果
    print("[3/3] 对比结果...")
    comparator = ResultComparator(ds_results, t6_results)
    comparator.compare_total_metrics()
    comparator.compare_layer_metrics()
    comparator.compare_operator_metrics()

    report = comparator.generate_report()
    print(report)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 DS_TPU 结果
    with open(output_dir / f"ds_tpu_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(ds_results, f, indent=2, ensure_ascii=False, default=str)

    # 保存 Tier6 结果
    with open(output_dir / f"tier6_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(t6_results, f, indent=2, ensure_ascii=False, default=str)

    # 保存对比报告
    with open(output_dir / f"comparison_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存对比数据
    comparison_data = {
        'config': asdict(config),
        'comparisons': [c.to_dict() for c in comparator.comparisons],
        'timestamp': timestamp,
    }
    with open(output_dir / f"comparison_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_dir}")

    return comparison_data


def main():
    parser = argparse.ArgumentParser(description='DS_TPU vs Tier6 评估对比')
    parser.add_argument('--mode', choices=['single_layer', 'full_model'],
                        default='single_layer', help='对比模式')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-layers', type=int, default=1, help='层数')
    parser.add_argument('--prefill', action='store_true', help='Prefill 模式')
    parser.add_argument('--output', type=str, default='comparison_results', help='输出目录')

    args = parser.parse_args()

    # 创建配置
    config = UnifiedConfig(
        batch_size=args.batch_size,
        num_layers=args.num_layers if args.mode == 'single_layer' else 61,
        is_prefill=args.prefill,
        q_seq_len=4096 if args.prefill else 1,
    )

    output_dir = Path(args.output)
    run_comparison(config, output_dir)


if __name__ == '__main__':
    main()
