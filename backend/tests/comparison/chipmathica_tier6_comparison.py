#!/usr/bin/env python3
"""
CHIPMathica vs Tier6-Model 评估结果对比脚本

功能：
1. 使用统一的配置参数调用两边的评估
2. 提取并对比性能指标 (MFU, TPS, 延迟等)
3. 生成对比报告

使用方式：
    # 单次对比
    python chipmathica_tier6_comparison.py --batch-size 2048 --tp 2 --ep 16

    # 批量扫参对比
    python chipmathica_tier6_comparison.py --sweep --batch-sizes 1024,2048,4096
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
CHIPMATHICA_PATH = Path(r"c:\Users\DELL\Documents\code\CHIPMathica")
TIER6_PATH = Path(r"C:\Users\DELL\Documents\code\Tier6-Model\backend")

sys.path.insert(0, str(CHIPMATHICA_PATH))
sys.path.insert(0, str(TIER6_PATH))


# ============================================
# 统一配置定义
# ============================================

@dataclass
class UnifiedConfig:
    """统一配置参数（对齐 CHIPMathica 和 Tier6-Model）"""
    # 模型参数
    model_name: str = "DeepSeek-V3"
    hidden_size: int = 7168
    num_layers: int = 61
    num_dense_layers: int = 3
    num_moe_layers: int = 58
    num_heads: int = 128
    vocab_size: int = 129280
    intermediate_size: int = 18432  # Dense FFN 维度

    # MLA 参数
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE 参数
    num_routed_experts: int = 256
    num_activated_experts: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 2048

    # 推理参数
    batch_size: int = 2048
    seq_len: int = 1
    kv_seq_len: int = 4096
    q_seq_len: int = 1  # Decode 模式默认为 1
    is_prefill: bool = False

    # 并行策略
    tp: int = 2
    pp: int = 1
    dp: int = 16
    ep: int = 16
    moe_tp: int = 2
    enable_tp_sp: bool = True
    comm_protocol: int = 1
    embed_tp: int = 1
    lmhead_tp: int = 2

    # 硬件参数 (SG2262)
    num_chips: int = 32
    chip_memory_gb: int = 16
    inter_chip_bw_gbps: float = 400.0

    # 通信延迟参数
    rtt_tp_us: float = 0.35
    rtt_ep_us: float = 0.85
    bandwidth_utilization: float = 0.95


# ============================================
# CHIPMathica 评估器
# ============================================

class CHIPMathicaEvaluator:
    """CHIPMathica L4 评估封装"""

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._import_modules()

    def _import_modules(self):
        """动态导入 CHIPMathica 模块"""
        try:
            from chipmathica.arch.chips.sg2262 import SG2262Chip
            from chipmathica.arch.topology import TopologySpec
            from chipmathica.core.utils import safe_load_yaml
            from chipmathica.evaluation import (
                CommProtocolSpec,
                HardwareSpec,
                EvaluationEngine,
                Granularity,
                PreciseTileEvaluator,
                merge_specs,
            )
            from chipmathica.mapping.parallelism.planner import (
                BoardSpec,
                DeploymentSpec,
                ParallelismPlanner,
            )
            from chipmathica.mapping.scheduling import Scheduler
            from chipmathica.mapping.tiling.planner import TilingPlanner
            from chipmathica.reporting import OutputConfig, ReportingEngine
            from chipmathica.workload.models.llm.deepseek import DeepSeekV3Model

            self.SG2262Chip = SG2262Chip
            self.TopologySpec = TopologySpec
            self.safe_load_yaml = safe_load_yaml
            self.CommProtocolSpec = CommProtocolSpec
            self.HardwareSpec = HardwareSpec
            self.EvaluationEngine = EvaluationEngine
            self.Granularity = Granularity
            self.PreciseTileEvaluator = PreciseTileEvaluator
            self.merge_specs = merge_specs
            self.BoardSpec = BoardSpec
            self.DeploymentSpec = DeploymentSpec
            self.ParallelismPlanner = ParallelismPlanner
            self.Scheduler = Scheduler
            self.TilingPlanner = TilingPlanner
            self.OutputConfig = OutputConfig
            self.ReportingEngine = ReportingEngine
            self.DeepSeekV3Model = DeepSeekV3Model
        except ImportError as e:
            raise ImportError(
                f"Failed to import CHIPMathica modules. "
                f"Make sure CHIPMathica is installed at {CHIPMATHICA_PATH}. Error: {e}"
            )

    def _build_model_ir(self):
        """构建 DeepSeek-V3 模型 IR"""
        cfg = self.config
        model_config = {
            "dtype": "bf16",
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_dense_layers": cfg.num_dense_layers,
            "num_moe_layers": cfg.num_moe_layers,
            "num_heads": cfg.num_heads,
            "vocab_size": cfg.vocab_size,
            "q_lora_rank": cfg.q_lora_rank,
            "kv_lora_rank": cfg.kv_lora_rank,
            "qk_nope_head_dim": cfg.qk_nope_head_dim,
            "qk_rope_head_dim": cfg.qk_rope_head_dim,
            "v_head_dim": cfg.v_head_dim,
            "n_routed_experts": cfg.num_routed_experts,
            "n_shared_experts": cfg.num_shared_experts,
            "n_activated_experts": cfg.num_activated_experts,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "intermediate_size": cfg.intermediate_size,
            # 运行态参数
            "seq_len": cfg.seq_len,
            "kv_seq_len": cfg.kv_seq_len,
            "q_seq_len": cfg.q_seq_len,
            "batch": cfg.batch_size,
            "is_prefill": cfg.is_prefill,
        }

        model = self.DeepSeekV3Model(model_config)
        return model.to_ir(), model_config

    def _build_hardware_spec(self) -> dict:
        """构建硬件规格参数"""
        chip = self.SG2262Chip.with_chip_id(0)
        compute_tflops = chip.get_peak_flops("BF16", "cube") / 1e12
        memory_bw_gbps = chip.get_gmem_bandwidth()
        sram_per_core_kb = chip.get_total_sram() / max(1, chip.core_count) / 1024

        hardware_spec = self.HardwareSpec(
            compute_tflops=compute_tflops,
            memory_bandwidth_gbps=memory_bw_gbps,
            num_cores=chip.core_count,
            sram_per_core_kb=sram_per_core_kb,
            noc_bandwidth_gbps=chip.interconnect.noc_bandwidth_gbps,
        )

        topology_spec = self.TopologySpec(
            intra_board_bw_gbps=self.config.inter_chip_bw_gbps,
            inter_board_bw_gbps=self.config.inter_chip_bw_gbps,
            inter_node_bw_gbps=self.config.inter_chip_bw_gbps,
            c2c_lat_us=0.15,
            ddr_r_lat_us=0.15,
            ddr_w_lat_us=0.01,
            noc_lat_us=0.05,
            d2d_lat_us=0.04,
            link_delay_us=0.0,
            switch_delay_us=0.25,
            cable_delay_us=0.025,
        )

        comm_spec = self.CommProtocolSpec(
            rtt_tp_us=self.config.rtt_tp_us,
            rtt_ep_us=self.config.rtt_ep_us,
            sync_lat_us=0.0,
            bw_utilization=self.config.bandwidth_utilization,
            cpu_fetch_delay_us=0.0,
            moe_topk=float(self.config.num_activated_experts),
            prefill_topk_factor=8 / 128,
        )

        hardware = self.merge_specs(hardware_spec, topology_spec, comm_spec)
        hardware["compute_efficiency"] = 0.9
        return hardware

    def evaluate(self) -> Dict[str, Any]:
        """运行 CHIPMathica L4 评估"""
        print("  [CHIPMathica] Building model IR...")
        ir, model_config = self._build_model_ir()

        print("  [CHIPMathica] Building deployment spec...")
        deployment = self.DeploymentSpec(
            tp=self.config.tp,
            pp=self.config.pp,
            ep=self.config.ep,
            dp=self.config.dp,
            moe_tp=self.config.moe_tp,
            seq_len=self.config.seq_len,
            batch_size=self.config.batch_size,
            enable_tp_sp=self.config.enable_tp_sp,
            embed_tp=self.config.embed_tp,
            lmhead_tp=self.config.lmhead_tp,
            comm_protocol=self.config.comm_protocol,
            kv_cache_rate=0.0,
            is_prefill=self.config.is_prefill,
        )

        board = self.BoardSpec(
            num_chips=self.config.num_chips,
            chip_memory_gb=self.config.chip_memory_gb,
            inter_chip_bw_gbps=self.config.inter_chip_bw_gbps,
        )

        print("  [CHIPMathica] Running ParallelismPlanner...")
        dist_model = self.ParallelismPlanner(deployment, board).plan(ir)

        print("  [CHIPMathica] Running TilingPlanner...")
        chip = self.SG2262Chip.with_chip_id(0)
        precise_evaluator = self.PreciseTileEvaluator(
            compute_tflops=chip.get_peak_flops("BF16", "cube") / 1e12,
            memory_bandwidth_gbps=chip.get_gmem_bandwidth(),
        )
        tile_plan = self.TilingPlanner(chip, l4_evaluator=precise_evaluator).plan(dist_model)

        print("  [CHIPMathica] Running Scheduler...")
        exec_plan = self.Scheduler().plan(dist_model, tile_plan)

        print("  [CHIPMathica] Running L4 EvaluationEngine...")
        l4_result = self.EvaluationEngine().evaluate(
            exec_plan=exec_plan,
            distributed_model=dist_model,
            hardware=self._build_hardware_spec(),
            granularity=self.Granularity.CHIP,
            output_tokens=deployment.batch_size,
        )

        print("  [CHIPMathica] Running L5 ReportingEngine...")
        reporting_engine = self.ReportingEngine()
        run_config = {
            "model": {
                "name": "DeepSeek-V3",
                "batch": model_config.get("batch"),
                "q_seq_len": model_config.get("q_seq_len"),
                "kv_seq_len": model_config.get("kv_seq_len"),
            },
            "deployment": {
                "tp": deployment.tp,
                "dp": deployment.dp,
                "moe_tp": deployment.moe_tp,
                "ep": deployment.ep,
                "comm_protocol": deployment.comm_protocol,
                "batch_size": deployment.batch_size,
            },
            "board": {
                "num_chips": board.num_chips,
                "inter_chip_bw_gbps": board.inter_chip_bw_gbps,
            },
        }
        report = reporting_engine.run(engine_result=l4_result, config=run_config)

        # 转换为统一格式
        return self._convert_to_standard_format(report, run_config)

    def _convert_to_standard_format(self, report, run_config: dict) -> Dict[str, Any]:
        """将 CHIPMathica 报告转换为标准格式"""
        rpt = report.performance
        batch_size = int(run_config.get("deployment", {}).get("batch_size", 0))
        num_chips = int(run_config.get("board", {}).get("num_chips", 0))

        return {
            "performance": {
                "total_elapse_us": rpt.total_time_ms * 1000.0,
                "total_elapse_ms": rpt.total_time_ms,
                "comm_elapse_us": rpt.comm_time_ms * 1000.0,
                "flops": float(rpt.total_flops),
                "dram_occupy": rpt.memory_peak_mb * 1024.0 * 1024.0,
                "mfu": rpt.mfu,
                "tps": rpt.tps,
                "tps_per_batch": (rpt.tps / batch_size) if batch_size > 0 else 0.0,
                "tps_per_chip": (rpt.tps / num_chips) if num_chips > 0 else 0.0,
            },
            "schema_version": report.schema_version,
            "granularity": str(report.granularity),
        }


# ============================================
# Tier6 评估器
# ============================================

class Tier6Evaluator:
    """Tier6-Model 评估封装"""

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._import_modules()

    def _import_modules(self):
        """动态导入 Tier6 模块"""
        try:
            from llm_simulator.core.simulator import run_simulation

            self.run_simulation = run_simulation
        except ImportError as e:
            raise ImportError(
                f"Failed to import Tier6 modules. Error: {e}"
            )

    def evaluate(self) -> Dict[str, Any]:
        """运行 Tier6 评估"""
        cfg = self.config

        print("  [Tier6] Building configurations...")

        # 模型配置字典
        model_dict = {
            "model_name": cfg.model_name,
            "model_type": "moe" if cfg.num_routed_experts > 0 else "dense",
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_heads,  # DeepSeek 使用 MQA
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "dtype": "bf16",
            "max_seq_length": cfg.kv_seq_len,
            "attention_type": "mla",
            # MLA 配置
            "mla_config": {
                "enabled": True,
                "kv_lora_rank": cfg.kv_lora_rank,
                "q_lora_rank": cfg.q_lora_rank,
                "qk_rope_head_dim": cfg.qk_rope_head_dim,
                "qk_nope_head_dim": cfg.qk_nope_head_dim,
                "v_head_dim": cfg.v_head_dim,
            },
            # MoE 配置
            "moe_config": {
                "enabled": True,
                "num_experts": cfg.num_routed_experts,
                "num_experts_per_tok": cfg.num_activated_experts,
                "num_shared_experts": cfg.num_shared_experts,
                "expert_intermediate_size": cfg.moe_intermediate_size,
                "expert_capacity_factor": 1.0,
                "first_k_dense_replace": 0,
                "moe_tp": cfg.moe_tp,
            },
        }

        # 推理配置字典
        inference_dict = {
            "batch_size": cfg.batch_size,
            "input_seq_length": cfg.kv_seq_len if cfg.is_prefill else cfg.kv_seq_len - cfg.seq_len,
            "output_seq_length": cfg.seq_len,
            "max_seq_length": cfg.kv_seq_len,
        }

        # 并行配置字典
        parallelism_dict = {
            "tp": cfg.tp,
            "pp": cfg.pp,
            "dp": cfg.dp,
            "ep": cfg.ep,
            "sp": 1,
            "moe_tp": cfg.moe_tp,
        }

        # 拓扑配置（使用 pods 格式，包含硬件参数）
        total_chips = cfg.tp * cfg.ep * cfg.dp * cfg.pp

        # 构建芯片列表
        chip_params = {
            "name": "SG2262",
            "num_cores": 64,
            "compute_tflops_fp8": 768.0,
            "compute_tflops_bf16": 384.0,
            "memory_capacity_gb": 64.0,
            "memory_bandwidth_gbps": 12000.0,
            "memory_bandwidth_utilization": 0.85,
            "lmem_capacity_mb": 64,
            "lmem_bandwidth_gbps": 6400.0,
            "cube_m": 16,
            "cube_k": 32,
            "cube_n": 8,
            "sram_size_kb": 2048,
            "sram_utilization": 0.45,
            "lane_num": 16,
            "align_bytes": 32,
            "compute_dma_overlap_rate": 0.8,
        }

        chips_list = []
        for chip_idx in range(total_chips):
            chips_list.append({
                "id": f"pod_0/rack_0/board_0/chip_{chip_idx}",
                "type": "chip",
                "position": [chip_idx % 4, chip_idx // 4],
                **chip_params,
            })

        topology_dict = {
            "name": f"Tier6-Comparison-{total_chips}chips",
            "pods": [
                {
                    "id": "pod_0",
                    "racks": [
                        {
                            "id": "rack_0",
                            "boards": [
                                {
                                    "id": "board_0",
                                    "chips": chips_list,
                                }
                            ]
                        }
                    ]
                }
            ],
            "hardware_params": {
                "chips": {
                    "SG2262": chip_params,
                },
                "interconnect": {
                    "c2c": {
                        "bandwidth_gbps": cfg.inter_chip_bw_gbps,
                        "latency_us": 0.2,
                    },
                    "b2b": {
                        "bandwidth_gbps": cfg.inter_chip_bw_gbps,
                        "latency_us": 0.35,
                    },
                    "r2r": {
                        "bandwidth_gbps": cfg.inter_chip_bw_gbps,
                        "latency_us": 2.0,
                    },
                    "p2p": {
                        "bandwidth_gbps": cfg.inter_chip_bw_gbps,
                        "latency_us": 5.0,
                    },
                },
                "comm_latency_config": {
                    "allreduce_algorithm": "ring",
                    "alltoall_algorithm": "pairwise",
                    "enable_compute_comm_overlap": True,
                    "network_efficiency": cfg.bandwidth_utilization,
                },
            },
        }

        # 硬件配置（必须包含 hardware_params.chips，用于验证）
        hardware_dict = {
            "hardware_params": {
                "chips": topology_dict["hardware_params"]["chips"]
            }
        }

        # 配置字典（可选）
        config_dict = {
            "max_simulated_tokens": 4,
            "enable_tile_search": True,
        }

        print("  [Tier6] Running simulator...")
        result = self.run_simulation(
            topology_dict=topology_dict,
            model_dict=model_dict,
            inference_dict=inference_dict,
            parallelism_dict=parallelism_dict,
            hardware_dict=hardware_dict,
            config_dict=config_dict,
        )

        # 转换为标准格式
        return self._convert_to_standard_format(result, cfg)

    def _convert_to_standard_format(self, result: dict, config: UnifiedConfig) -> Dict[str, Any]:
        """将 Tier6 结果转换为标准格式"""
        # result 是字典格式，使用驼峰命名（前端JSON格式）
        stats = result.get('stats', {})
        throughput = result.get('throughput', {})

        total_chips = config.tp * config.ep * config.dp * config.pp

        # 计算 TPS (throughput per second)
        # totalRunTime 单位是微秒 (us)
        # simulatedTokens 是仿真的 token 数量
        total_run_time_us = stats.get('totalRunTime', 0)
        simulated_tokens = stats.get('simulatedTokens', 1)
        tps = (simulated_tokens * 1_000_000 / total_run_time_us) if total_run_time_us > 0 else 0

        # 计算总 FLOPs (需要从 throughput 中获取)
        total_flops = throughput.get('totalFlops', 0)

        # 内存占用（假设从芯片参数计算）
        memory_per_chip_gb = 64.0  # SG2262
        dram_occupy_bytes = total_chips * memory_per_chip_gb * 1024 * 1024 * 1024

        # 通信时间 (prefill + decode)
        prefill_comm_us = stats.get('prefill', {}).get('commTime', 0)
        decode_comm_us = stats.get('decode', {}).get('commTime', 0)
        total_comm_us = prefill_comm_us + decode_comm_us

        return {
            "performance": {
                "total_elapse_us": total_run_time_us,
                "total_elapse_ms": total_run_time_us / 1000.0,
                "comm_elapse_us": total_comm_us,
                "flops": total_flops,
                "dram_occupy": dram_occupy_bytes,
                "mfu": stats.get('dynamicMfu', 0),
                "tps": tps * config.batch_size,  # 总 TPS = 每token TPS × batch size
                "tps_per_batch": tps,
                "tps_per_chip": (tps * config.batch_size) / total_chips if total_chips > 0 else 0.0,
                # Tier6 独有指标
                "ttft_ms": stats.get('ttft', 0) / 1000.0,  # us to ms
                "tpot_ms": stats.get('avgTpot', 0) / 1000.0,  # us to ms
                "mbu": stats.get('dynamicMbu', 0),
                "bottleneck_type": "unknown",  # 前端格式中没有这个字段
            },
        }


# ============================================
# 结果映射函数
# ============================================

def map_chipmathica_to_standard(chipmathica_result: Dict) -> Dict:
    """将 CHIPMathica 结果映射到标准格式（已在 evaluator 中处理）"""
    return chipmathica_result.get('performance', {})


def map_tier6_to_standard(tier6_result: Dict) -> Dict:
    """将 Tier6 结果映射到标准格式（已在 evaluator 中处理）"""
    return tier6_result.get('performance', {})


# ============================================
# 结果对比器
# ============================================

@dataclass
class ComparisonResult:
    """对比结果"""
    metric_name: str
    chipmathica_value: float
    tier6_value: float
    absolute_diff: float
    relative_diff_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossSystemComparator:
    """跨系统对比器"""

    def __init__(self, chipmathica_result: Dict, tier6_result: Dict):
        self.cm_result = map_chipmathica_to_standard(chipmathica_result)
        self.t6_result = map_tier6_to_standard(tier6_result)
        self.comparisons: List[ComparisonResult] = []

    def compare_metrics(self) -> List[ComparisonResult]:
        """对比共同指标"""
        # 共同指标列表
        common_metrics = [
            'total_elapse_ms',
            'total_elapse_us',
            'comm_elapse_us',
            'mfu',
            'tps',
            'tps_per_chip',
            'tps_per_batch',
            'flops',
            'dram_occupy',
        ]

        comparisons = []
        for metric in common_metrics:
            cm_val = self.cm_result.get(metric, 0.0)
            t6_val = self.t6_result.get(metric, 0.0)
            comparisons.append(self._compare(metric, cm_val, t6_val))

        self.comparisons.extend(comparisons)
        return comparisons

    def _compare(self, name: str, cm_val: float, t6_val: float) -> ComparisonResult:
        """计算对比结果"""
        abs_diff = t6_val - cm_val
        rel_diff = (abs_diff / cm_val * 100) if cm_val != 0 else 0.0

        return ComparisonResult(
            metric_name=name,
            chipmathica_value=cm_val,
            tier6_value=t6_val,
            absolute_diff=abs_diff,
            relative_diff_percent=rel_diff,
        )

    def generate_report(self, config: UnifiedConfig) -> str:
        """生成对比报告"""
        lines = []
        lines.append("=" * 100)
        lines.append("CHIPMathica vs Tier6-Model 对比报告")
        lines.append(f"生成时间: {datetime.now().isoformat()}")
        lines.append("=" * 100)
        lines.append("")

        # 配置信息
        lines.append("配置:")
        lines.append(f"  模型: {config.model_name} ({config.num_layers}层)")
        lines.append(f"  批次: {config.batch_size}, TP: {config.tp}, EP: {config.ep}, 芯片数: {config.num_chips}")
        lines.append(f"  序列长度: q={config.q_seq_len}, kv={config.kv_seq_len}")
        lines.append("")

        # 总体指标对比
        lines.append("总体指标对比")
        lines.append("-" * 100)
        lines.append(f"{'指标名称':<25} {'CHIPMathica':<20} {'Tier6-Model':<20} {'差异':<15} {'误差%':<10}")
        lines.append("-" * 100)

        for cmp in self.comparisons:
            lines.append(
                f"{cmp.metric_name:<25} "
                f"{cmp.chipmathica_value:<20.6f} "
                f"{cmp.tier6_value:<20.6f} "
                f"{cmp.absolute_diff:<15.6f} "
                f"{cmp.relative_diff_percent:+10.2f}%"
            )

        lines.append("")

        # Tier6 独有指标
        tier6_only = ['ttft_ms', 'tpot_ms', 'mbu', 'bottleneck_type']
        has_tier6_only = any(self.t6_result.get(m) is not None for m in tier6_only)

        if has_tier6_only:
            lines.append("Tier6 独有指标")
            lines.append("-" * 100)
            for metric in tier6_only:
                val = self.t6_result.get(metric)
                if val is not None:
                    if isinstance(val, str):
                        lines.append(f"{metric:<25} {val}")
                    else:
                        lines.append(f"{metric:<25} {val:.6f}")
            lines.append("")

        lines.append("=" * 100)

        return "\n".join(lines)


# ============================================
# 主函数
# ============================================

def run_comparison(config: UnifiedConfig, output_dir: Path) -> Dict[str, Any]:
    """运行对比"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CHIPMathica vs Tier6-Model 评估对比")
    print("=" * 80)
    print(f"\n配置: batch={config.batch_size}, tp={config.tp}, ep={config.ep}, "
          f"chips={config.num_chips}, prefill={config.is_prefill}")
    print("")

    # Step 1: 运行 CHIPMathica 评估
    print("[1/3] 运行 CHIPMathica 评估...")
    try:
        cm_evaluator = CHIPMathicaEvaluator(config)
        cm_results = cm_evaluator.evaluate()
        cm_perf = cm_results.get('performance', {})
        print(f"  [OK] CHIPMathica 完成: {cm_perf.get('total_elapse_ms', 0):.2f} ms, "
              f"TPS: {cm_perf.get('tps', 0):.2f}, MFU: {cm_perf.get('mfu', 0):.2%}")
    except Exception as e:
        print(f"  [FAIL] CHIPMathica 失败: {e}")
        import traceback
        traceback.print_exc()
        cm_results = {}

    # Step 2: 运行 Tier6 评估
    print("[2/3] 运行 Tier6 评估...")
    try:
        t6_evaluator = Tier6Evaluator(config)
        t6_results = t6_evaluator.evaluate()
        t6_perf = t6_results.get('performance', {})
        print(f"  [OK] Tier6 完成: {t6_perf.get('total_elapse_ms', 0):.2f} ms, "
              f"TPS: {t6_perf.get('tps', 0):.2f}, MFU: {t6_perf.get('mfu', 0):.2%}")
    except Exception as e:
        print(f"  [FAIL] Tier6 失败: {e}")
        import traceback
        traceback.print_exc()
        t6_results = {}

    # Step 3: 对比结果
    print("[3/3] 对比结果...")
    comparator = CrossSystemComparator(cm_results, t6_results)
    comparator.compare_metrics()

    report = comparator.generate_report(config)
    print(report)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 CHIPMathica 结果
    with open(output_dir / f"chipmathica_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(cm_results, f, indent=2, ensure_ascii=False, default=str)

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


def run_sweep(
    batch_sizes: List[int],
    tp_values: List[int],
    ep_values: List[int],
    output_dir: Path
):
    """批量扫参对比"""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for batch_size in batch_sizes:
        for tp in tp_values:
            for ep in ep_values:
                print(f"\n{'='*80}")
                print(f"扫参配置: batch={batch_size}, tp={tp}, ep={ep}")
                print(f"{'='*80}")

                config = UnifiedConfig(
                    batch_size=batch_size,
                    tp=tp,
                    ep=ep,
                    num_chips=tp * ep,  # 简化假设
                )

                try:
                    result = run_comparison(config, output_dir / f"batch{batch_size}_tp{tp}_ep{ep}")
                    all_results.append(result)
                except Exception as e:
                    print(f"[WARN] 配置 batch={batch_size}, tp={tp}, ep={ep} 失败: {e}")

    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"sweep_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n扫参完成！汇总结果保存到: {output_dir / f'sweep_summary_{timestamp}.json'}")


def main():
    parser = argparse.ArgumentParser(description='CHIPMathica vs Tier6-Model 评估对比')
    parser.add_argument('--batch-size', type=int, default=2048, help='批次大小')
    parser.add_argument('--tp', type=int, default=2, help='Tensor 并行度')
    parser.add_argument('--ep', type=int, default=16, help='Expert 并行度')
    parser.add_argument('--num-chips', type=int, default=32, help='芯片数量')
    parser.add_argument('--prefill', action='store_true', help='Prefill 模式')
    parser.add_argument('--output', type=str, default='comparison_results', help='输出目录')

    # 扫参模式
    parser.add_argument('--sweep', action='store_true', help='批量扫参模式')
    parser.add_argument('--batch-sizes', type=str, help='批次大小列表（逗号分隔），例如: 1024,2048,4096')
    parser.add_argument('--tp-values', type=str, help='TP 值列表（逗号分隔），例如: 1,2,4,8')
    parser.add_argument('--ep-values', type=str, help='EP 值列表（逗号分隔），例如: 8,16,32')

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.sweep:
        # 批量扫参模式
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')] if args.batch_sizes else [1024, 2048, 4096]
        tp_values = [int(x) for x in args.tp_values.split(',')] if args.tp_values else [1, 2, 4]
        ep_values = [int(x) for x in args.ep_values.split(',')] if args.ep_values else [8, 16]

        run_sweep(batch_sizes, tp_values, ep_values, output_dir)
    else:
        # 单次对比模式
        config = UnifiedConfig(
            batch_size=args.batch_size,
            tp=args.tp,
            ep=args.ep,
            num_chips=args.num_chips,
            is_prefill=args.prefill,
            q_seq_len=4096 if args.prefill else 1,
        )

        run_comparison(config, output_dir)


if __name__ == '__main__':
    main()
