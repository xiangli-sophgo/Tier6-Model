"""兼容层模块

将 tier6 的评估结果转换为 llm_simulator 兼容的格式，
使前端的性能分析可视化组件能够正常工作。
"""

from __future__ import annotations

import re
from typing import Any


# ============================================
# 任务类型映射 - 与前端 ganttDataUtils.ts 保持一致
# ============================================

# 计算类型任务及其颜色
COMPUTE_TASK_TYPES = {
    # 通用计算
    "compute": {"type": "compute", "color": "#52c41a"},
    "linear": {"type": "compute", "color": "#52c41a"},
    # Embedding
    "embedding": {"type": "embedding", "color": "#73d13d"},
    "embed": {"type": "embedding", "color": "#73d13d"},
    # LayerNorm / RMSNorm
    "layernorm": {"type": "layernorm", "color": "#95de64"},
    "rmsnorm": {"type": "layernorm", "color": "#95de64"},
    "rmsnorm_q": {"type": "rmsnorm_q_lora", "color": "#95de64"},
    "rmsnorm_kv": {"type": "rmsnorm_kv_lora", "color": "#95de64"},
    # Attention 相关
    "attention": {"type": "attention_qkv", "color": "#389e0d"},
    "attn": {"type": "attention_qkv", "color": "#389e0d"},
    "qkv": {"type": "attention_qkv", "color": "#389e0d"},
    "attn_qkv": {"type": "attention_qkv", "color": "#389e0d"},
    "attn_score": {"type": "attention_score", "color": "#52c41a"},
    "attn_softmax": {"type": "attention_softmax", "color": "#a0d911"},
    "attn_output": {"type": "attention_output", "color": "#73d13d"},
    "attn_fc": {"type": "attn_fc", "color": "#389e0d"},
    "bmm_qk": {"type": "bmm_qk", "color": "#52c41a"},
    "bmm_sv": {"type": "bmm_sv", "color": "#52c41a"},
    # MLA 相关
    "mla": {"type": "attention_qkv", "color": "#13c2c2"},
    "mla_qkv": {"type": "attention_qkv", "color": "#13c2c2"},
    "mla_q_proj": {"type": "mm_q_lora_a", "color": "#13c2c2"},
    "mla_kv_proj": {"type": "mm_kv_lora_a", "color": "#36cfc9"},
    "q_lora_a": {"type": "mm_q_lora_a", "color": "#13c2c2"},
    "q_lora_b": {"type": "mm_q_lora_b", "color": "#13c2c2"},
    "kv_lora_a": {"type": "mm_kv_lora_a", "color": "#36cfc9"},
    # FFN / MLP 相关
    "ffn": {"type": "ffn_gate", "color": "#237804"},
    "mlp": {"type": "ffn_gate", "color": "#237804"},
    "ffn_gate": {"type": "ffn_gate", "color": "#237804"},
    "ffn_up": {"type": "ffn_up", "color": "#389e0d"},
    "ffn_down": {"type": "ffn_down", "color": "#52c41a"},
    "gate_proj": {"type": "ffn_gate", "color": "#237804"},
    "up_proj": {"type": "ffn_up", "color": "#389e0d"},
    "down_proj": {"type": "ffn_down", "color": "#52c41a"},
    # MoE 相关
    "moe": {"type": "moe_expert", "color": "#eb2f96"},
    "moe_gate": {"type": "moe_gate", "color": "#f759ab"},
    "moe_expert": {"type": "moe_expert", "color": "#eb2f96"},
    "moe_shared": {"type": "moe_shared_expert", "color": "#c41d7f"},
    "shared_expert": {"type": "moe_shared_expert", "color": "#c41d7f"},
    "routed_expert": {"type": "moe_expert", "color": "#eb2f96"},
    # LM Head
    "lm_head": {"type": "lm_head", "color": "#135200"},
    "lmhead": {"type": "lm_head", "color": "#135200"},
}

# 访存类型任务
MEMORY_TASK_TYPES = {
    "pcie_h2d": {"type": "pcie_h2d", "color": "#faad14"},
    "pcie_d2h": {"type": "pcie_d2h", "color": "#d48806"},
    "hbm_write": {"type": "hbm_write", "color": "#fa8c16"},
    "hbm_read": {"type": "hbm_read", "color": "#ffc53d"},
    "weight_load": {"type": "weight_load", "color": "#fa8c16"},
    "kv_cache_read": {"type": "kv_cache_read", "color": "#ffd666"},
    "kv_cache_write": {"type": "kv_cache_write", "color": "#ffec3d"},
}

# 通信类型任务
COMM_TASK_TYPES = {
    # TP 通信
    "tp_comm": {"type": "tp_comm", "color": "#1890ff"},
    "allreduce": {"type": "tp_comm", "color": "#1890ff"},
    "tp_allreduce": {"type": "tp_comm", "color": "#1890ff"},
    # PP 通信
    "pp_comm": {"type": "pp_comm", "color": "#722ed1"},
    "p2p": {"type": "pp_comm", "color": "#722ed1"},
    "send_recv": {"type": "pp_comm", "color": "#722ed1"},
    # EP 通信
    "ep_comm": {"type": "ep_comm", "color": "#eb2f96"},
    "alltoall": {"type": "ep_comm", "color": "#eb2f96"},
    "ep_dispatch": {"type": "ep_dispatch", "color": "#f759ab"},
    "ep_combine": {"type": "ep_combine", "color": "#c41d7f"},
    "dispatch": {"type": "ep_dispatch", "color": "#f759ab"},
    "combine": {"type": "ep_combine", "color": "#c41d7f"},
    # SP 通信
    "sp_allgather": {"type": "sp_allgather", "color": "#52c41a"},
    "sp_reduce_scatter": {"type": "sp_reduce_scatter", "color": "#73d13d"},
    "allgather": {"type": "sp_allgather", "color": "#52c41a"},
    "reduce_scatter": {"type": "sp_reduce_scatter", "color": "#73d13d"},
}

# 其他类型
OTHER_TASK_TYPES = {
    "wait": {"type": "idle", "color": "#d9d9d9"},
    "idle": {"type": "idle", "color": "#d9d9d9"},
    "bubble": {"type": "idle", "color": "#d9d9d9"},
}

# 合并所有任务类型映射
ALL_TASK_TYPES = {
    **COMPUTE_TASK_TYPES,
    **MEMORY_TASK_TYPES,
    **COMM_TASK_TYPES,
    **OTHER_TASK_TYPES,
}


def convert_to_gantt_chart(
    step_metrics: list[dict[str, Any]],
    parallelism: dict[str, Any],
    aggregates: dict[str, Any] | None = None,
    topology_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 step_metrics 转换为前端兼容的 gantt_chart 格式

    Args:
        step_metrics: tier6 的 step 级指标列表
        parallelism: 并行配置 {tp, pp, dp, ep, ...}
        aggregates: 聚合指标（可选，用于获取总时间）
        topology_config: 拓扑配置（可选，用于生成链路流量）

    Returns:
        前端兼容的 gantt_chart 格式:
        {
            "resources": [{ id, name, ppStage, type }, ...],
            "tasks": [{ id, name, resource, start, end, type, phase, ... }, ...],
            "timeRange": { start, end },
            "phaseTransition": number | null,
        }
    """
    pp = parallelism.get("pp", 1)

    # 1. 创建 resources（基于 PP 阶段）
    resources = []
    for pp_stage in range(pp):
        resources.append({
            "id": f"stage{pp_stage}_compute",
            "name": f"PP{pp_stage} Compute",
            "ppStage": pp_stage,
            "type": "compute",
        })
        resources.append({
            "id": f"stage{pp_stage}_network",
            "name": f"PP{pp_stage} Network",
            "ppStage": pp_stage,
            "type": "network",
        })

    # 2. 从 step_metrics 生成 tasks
    tasks = []
    current_time = 0.0  # 当前时间（ms）

    for i, step in enumerate(step_metrics):
        op_id = step.get("op_id", f"op_{i}")

        # 解析 op_id 获取层号和类型
        layer_index = _extract_layer_index(op_id)
        op_type_key = _extract_op_type(op_id)

        # 获取任务类型信息（从合并的映射表中查找）
        type_info = ALL_TASK_TYPES.get(op_type_key, COMPUTE_TASK_TYPES["compute"])

        # 获取时间（ms）
        t_compute_ms = step.get("t_compute_ms", step.get("t_compute", 0.0))
        t_comm_ms = step.get("t_comm_ms", step.get("t_comm", 0.0))
        t_wait_ms = step.get("t_wait_ms", step.get("t_wait", 0.0))
        t_total_ms = step.get("t_total_ms", step.get("t_total", 0.0))

        if t_total_ms == 0.0:
            t_total_ms = t_compute_ms + t_comm_ms + t_wait_ms

        # 判断 phase（简单策略：根据 op_id 判断）
        phase = _determine_phase(op_id, i, len(step_metrics))

        # 获取额外元数据
        flops = step.get("flops", 0)
        bytes_read = step.get("bytes_read", 0)
        bytes_write = step.get("bytes_write", 0)
        bottleneck = step.get("bottleneck_tag", step.get("bottleneck", "unknown"))

        # 计算任务（如果有计算时间）
        if t_compute_ms > 0:
            start_us = current_time * 1000
            end_us = (current_time + t_compute_ms) * 1000
            tasks.append({
                "id": f"task_{i}_compute",
                "name": op_id,
                "resourceId": "stage0_compute",
                "start": start_us,
                "end": end_us,
                "type": type_info["type"],
                "phase": phase,
                "chipId": "chip_0",
                "ppStage": 0,
                "layer": layer_index,
                "tokenIndex": None,
                "color": type_info["color"],
                # 扩展信息（用于图表详情）
                "flops": flops,
                "compute_time_us": t_compute_ms * 1000,
                "dram_traffic_bytes": bytes_read + bytes_write,
                "bytes_read": bytes_read,
                "bytes_write": bytes_write,
                "bottleneck": bottleneck,
            })
            current_time += t_compute_ms

        # 通信任务（如果有通信时间）
        if t_comm_ms > 0:
            start_us = current_time * 1000
            end_us = (current_time + t_comm_ms) * 1000

            # 智能判断通信类型
            comm_type_info = _get_comm_type_info(op_id, parallelism)

            tasks.append({
                "id": f"task_{i}_comm",
                "name": f"{op_id}_comm",
                "resourceId": "stage0_network",
                "start": start_us,
                "end": end_us,
                "type": comm_type_info["type"],
                "phase": phase,
                "chipId": "chip_0",
                "ppStage": 0,
                "layer": layer_index,
                "tokenIndex": None,
                "color": comm_type_info["color"],
                # 扩展信息
                "comm_time_us": t_comm_ms * 1000,
                "comm_size_bytes": bytes_read + bytes_write,  # 通信数据量估计
            })
            current_time += t_comm_ms

        # 等待/气泡（如果有等待时间且大于阈值）
        if t_wait_ms > 0.001:  # 忽略非常小的等待时间
            start_us = current_time * 1000
            end_us = (current_time + t_wait_ms) * 1000
            tasks.append({
                "id": f"task_{i}_wait",
                "name": f"{op_id}_wait",
                "resourceId": "stage0_compute",
                "start": start_us,
                "end": end_us,
                "type": "idle",
                "phase": phase,
                "chipId": "chip_0",
                "ppStage": 0,
                "layer": layer_index,
                "tokenIndex": None,
                "color": "#d9d9d9",
            })
            current_time += t_wait_ms

    # 3. 计算时间范围
    if tasks:
        time_start = min(t["start"] for t in tasks)
        time_end = max(t["end"] for t in tasks)
    else:
        time_start = 0
        time_end = 0

    # 4. 计算 phase transition（简化：无法准确判断时返回 None）
    phase_transition = None
    if aggregates:
        ttft_ms = aggregates.get("ttft_ms", aggregates.get("ttft", 0))
        if ttft_ms > 0:
            phase_transition = ttft_ms * 1000  # ms -> us

    return {
        "resources": resources,
        "tasks": tasks,
        "timeRange": {
            "start": time_start,
            "end": time_end,
        },
        "phaseTransition": phase_transition,
    }


def convert_to_stats(
    aggregates: dict[str, Any],
    step_metrics: list[dict[str, Any]] | None = None,
    inference_config: dict[str, Any] | None = None,
    parallelism: dict[str, Any] | None = None,
    topology_config: dict[str, Any] | None = None,
    link_traffic_stats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """将 tier6 aggregates 转换为前端兼容的 stats 格式

    Args:
        aggregates: tier6 的聚合指标
        step_metrics: step 级指标（可选，用于计算事件数）
        inference_config: 推理配置（可选，用于获取 token 数）
        parallelism: 并行配置（可选）
        topology_config: 拓扑配置（可选）
        link_traffic_stats: 预计算的链路流量统计（来自 TrafficAnalyzer）

    Returns:
        前端兼容的 stats 格式
    """
    # 获取各项指标
    total_time_ms = aggregates.get("total_time_ms", aggregates.get("total_time", 0))
    compute_time_ms = aggregates.get("compute_time_ms", aggregates.get("total_compute_time", 0))
    comm_time_ms = aggregates.get("comm_time_ms", aggregates.get("total_comm_time", 0))
    wait_time_ms = aggregates.get("wait_time_ms", aggregates.get("total_wait_time", 0))
    ttft_ms = aggregates.get("ttft_ms", aggregates.get("ttft", 0))
    tpot_ms = aggregates.get("tpot_ms", aggregates.get("tpot", 0))
    mfu = aggregates.get("mfu", 0)
    mbu = aggregates.get("mbu", 0)
    total_flops = aggregates.get("total_flops", 0)
    num_ops = aggregates.get("num_ops", aggregates.get("num_steps", 0))

    if step_metrics and num_ops == 0:
        num_ops = len(step_metrics)

    # 获取 token 数量
    simulated_tokens = 1
    if inference_config:
        simulated_tokens = inference_config.get("output_seq_length", 1)

    # 计算效率（简化：使用 MFU）
    compute_efficiency = mfu if mfu > 0 else 0

    # 假设 prefill 时间约等于 TTFT，其余为 decode 时间
    prefill_time = ttft_ms if ttft_ms > 0 else total_time_ms * 0.3
    decode_time = max(0, total_time_ms - prefill_time)

    # 按比例分配计算/通信时间
    if total_time_ms > 0:
        prefill_ratio = prefill_time / total_time_ms
        decode_ratio = decode_time / total_time_ms
    else:
        prefill_ratio = 0.3
        decode_ratio = 0.7

    return {
        "prefill": {
            "computeTime": compute_time_ms * prefill_ratio,
            "commTime": comm_time_ms * prefill_ratio,
            "bubbleTime": wait_time_ms * prefill_ratio,
            "overlapTime": 0,
            "totalTime": prefill_time,
            "computeEfficiency": compute_efficiency,
        },
        "decode": {
            "computeTime": compute_time_ms * decode_ratio,
            "commTime": comm_time_ms * decode_ratio,
            "bubbleTime": wait_time_ms * decode_ratio,
            "overlapTime": 0,
            "totalTime": decode_time,
            "computeEfficiency": compute_efficiency,
        },
        "totalRunTime": total_time_ms,
        "simulatedTokens": simulated_tokens,
        "ttft": ttft_ms,
        "avgTpot": tpot_ms,
        "dynamicMfu": mfu,
        "dynamicMbu": mbu,
        "maxPpBubbleRatio": 0,
        "totalEvents": num_ops,
        "prefillFlops": int(total_flops * prefill_ratio) if total_flops else 0,
        "linkTrafficStats": link_traffic_stats or [],
    }


def _extract_layer_index(op_id: str) -> int | None:
    """从 op_id 提取层索引

    Args:
        op_id: 操作 ID，支持多种格式:
            - "layers.5.mla" (tier6 标准格式)
            - "layer_0_attn_qkv"
            - "L5.MLA.QKV"

    Returns:
        层索引，如果无法提取则返回 None
    """
    # 匹配常见格式（按优先级排序）
    patterns = [
        r"layers\.(\d+)\.",   # layers.5.mla (tier6 标准格式)
        r"layer[_\-]?(\d+)",  # layer_0, layer-1
        r"L(\d+)\.",          # L5.MLA
        r"^(\d+)[_\-]",       # 0_attn
    ]

    for pattern in patterns:
        match = re.search(pattern, op_id, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def _extract_op_type(op_id: str) -> str:
    """从 op_id 提取操作类型

    Args:
        op_id: 操作 ID

    Returns:
        操作类型关键字（用于在 ALL_TASK_TYPES 中查找）
    """
    op_lower = op_id.lower()

    # 按优先级检查（更具体的关键字优先）
    type_keywords = [
        # MLA 细分
        "q_lora_a", "q_lora_b", "kv_lora_a",
        "mla_q_proj", "mla_kv_proj", "mla_qkv",
        "rmsnorm_q", "rmsnorm_kv",
        # MoE 细分
        "moe_gate", "moe_expert", "moe_shared", "shared_expert", "routed_expert",
        # Attention 细分
        "attn_qkv", "attn_score", "attn_softmax", "attn_output", "attn_fc",
        "bmm_qk", "bmm_sv",
        # FFN 细分
        "ffn_gate", "ffn_up", "ffn_down",
        "gate_proj", "up_proj", "down_proj",
        # 通信细分
        "ep_dispatch", "ep_combine", "dispatch", "combine",
        "tp_allreduce", "allreduce", "alltoall", "p2p",
        "sp_allgather", "sp_reduce_scatter", "allgather", "reduce_scatter",
        # 访存
        "kv_cache_read", "kv_cache_write", "weight_load",
        "hbm_read", "hbm_write", "pcie_h2d", "pcie_d2h",
        # 通用类型
        "mla", "moe", "attention", "attn", "ffn", "mlp",
        "embedding", "embed", "layernorm", "rmsnorm",
        "lm_head", "lmhead", "linear",
    ]

    for keyword in type_keywords:
        if keyword in op_lower:
            return keyword

    return "compute"


def _get_comm_type_info(op_id: str, parallelism: dict[str, Any]) -> dict[str, str]:
    """根据 op_id 和并行配置判断通信类型

    Args:
        op_id: 操作 ID
        parallelism: 并行配置 {tp, pp, dp, ep, ...}

    Returns:
        {"type": 通信类型, "color": 颜色}
    """
    op_lower = op_id.lower()
    ep = parallelism.get("ep", 1)
    pp = parallelism.get("pp", 1)

    # 根据 op_id 中的关键字判断
    if "alltoall" in op_lower or "dispatch" in op_lower or "combine" in op_lower:
        if "dispatch" in op_lower:
            return COMM_TASK_TYPES["ep_dispatch"]
        elif "combine" in op_lower:
            return COMM_TASK_TYPES["ep_combine"]
        return COMM_TASK_TYPES["ep_comm"]

    if "p2p" in op_lower or "send_recv" in op_lower or "pp_" in op_lower:
        return COMM_TASK_TYPES["pp_comm"]

    if "allgather" in op_lower:
        return COMM_TASK_TYPES["sp_allgather"]

    if "reduce_scatter" in op_lower:
        return COMM_TASK_TYPES["sp_reduce_scatter"]

    # 根据并行配置推断
    # MoE 层的通信通常是 EP 通信
    if ep > 1 and ("moe" in op_lower or "expert" in op_lower):
        return COMM_TASK_TYPES["ep_comm"]

    # PP > 1 时，层间通信可能是 PP 通信
    if pp > 1 and ("layer" in op_lower or "stage" in op_lower):
        return COMM_TASK_TYPES["pp_comm"]

    # 默认为 TP 通信（AllReduce）
    return COMM_TASK_TYPES["tp_comm"]


def _determine_phase(op_id: str, index: int, total: int) -> str:
    """确定操作所属的推理阶段

    Args:
        op_id: 操作 ID
        index: 操作在列表中的索引
        total: 操作总数

    Returns:
        "prefill" 或 "decode"
    """
    op_lower = op_id.lower()

    # 根据 op_id 判断
    if "prefill" in op_lower:
        return "prefill"
    if "decode" in op_lower:
        return "decode"

    # 根据位置判断（简单策略：前 30% 为 prefill）
    if total > 0 and index < total * 0.3:
        return "prefill"

    return "decode"


def convert_traffic_report_to_stats(
    report: Any,
    topology_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """将 TrafficReport 转换为前端 linkTrafficStats 格式

    Args:
        report: TrafficReport 对象 (来自 L5 TrafficAnalyzer)
        topology_config: 拓扑配置 (用于获取链路带宽/延迟)

    Returns:
        前端兼容的 linkTrafficStats 列表
    """
    if "interconnect" not in topology_config:
        raise ValueError(
            "topology_config 中缺少 'interconnect' 字段"
        )
    interconnect = topology_config["interconnect"]
    if "links" not in interconnect:
        raise ValueError(
            "topology_config.interconnect 中缺少 'links' 字段"
        )
    links_config = interconnect["links"]

    stats: list[dict[str, Any]] = []
    for link in report.links:
        link_type_key = link.link_type.value
        if link_type_key not in links_config:
            raise ValueError(
                f"links_config 中缺少 '{link_type_key}' 链路配置"
            )
        link_config = links_config[link_type_key]
        if "bandwidth_gbps" not in link_config:
            raise ValueError(
                f"links_config.{link_type_key} 中缺少 'bandwidth_gbps' 字段"
            )
        bandwidth_gbps = link_config["bandwidth_gbps"]
        if "latency_us" not in link_config:
            raise ValueError(
                f"links_config.{link_type_key} 中缺少 'latency_us' 字段"
            )
        latency_us = link_config["latency_us"]

        stats.append({
            "source": link.src,
            "target": link.dst,
            "trafficMb": link.total_bytes / (1024 * 1024),
            "bandwidthGbps": bandwidth_gbps,
            "latencyUs": latency_us,
            "utilizationPercent": link.utilization * 100,
            "linkType": link.link_type.value,
            "contributingTasks": list(link.comm_breakdown.keys()),
            "taskTypeBreakdown": {
                k: v / (1024 * 1024)
                for k, v in link.comm_breakdown.items()
            },
        })

    return stats
