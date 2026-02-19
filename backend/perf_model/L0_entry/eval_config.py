"""EvalConfig -- 管线配置入口

单一类型化配置对象，在 API 边界一次性转换，全管线传递。
解决:
1. topology_overrides/comm_overrides 数据丢失
2. 静默默认值
3. 不必要的扁平化
4. 无类型 dict 传递
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ============================================
# 子配置 dataclass
# ============================================


@dataclass
class MLAConfig:
    """MLA (Multi-head Latent Attention) 配置"""

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    mla_mode: str  # standard / absorb / auto


@dataclass
class MoEConfig:
    """MoE (Mixture of Experts) 配置"""

    num_routed_experts: int
    num_shared_experts: int
    num_activated_experts: int
    intermediate_size: int


@dataclass
class ModelConfig:
    """结构化模型配置 -- 镜像 YAML 嵌套结构"""

    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    intermediate_size: int  # Dense FFN intermediate
    num_dense_layers: int
    num_moe_layers: int
    mla: MLAConfig
    moe: MoEConfig
    # 运行时参数 (从 deployment/inference 注入)
    weight_dtype: str
    activation_dtype: str
    seq_len: int
    kv_seq_len: int
    q_seq_len: int
    batch: int
    is_prefill: bool


@dataclass
class TopologyOverrides:
    """拓扑带宽/延迟参数 -- 全部必填，来自 interconnect.links + comm_params"""

    c2c_bandwidth_gbps: float
    c2c_latency_us: float
    b2b_bandwidth_gbps: float
    b2b_latency_us: float
    r2r_bandwidth_gbps: float
    r2r_latency_us: float
    p2p_bandwidth_gbps: float
    p2p_latency_us: float
    switch_latency_us: float
    cable_latency_us: float
    memory_read_latency_us: float
    memory_write_latency_us: float
    noc_latency_us: float
    die_to_die_latency_us: float


@dataclass
class CommOverrides:
    """通信协议参数 -- 来自 interconnect.comm_params"""

    bw_utilization: float
    sync_lat_us: float


@dataclass
class DeploymentConfig:
    """部署/并行配置"""

    tp: int
    pp: int
    dp: int
    ep: int
    moe_tp: int
    seq_len: int
    batch_size: int
    enable_tp_sp: bool
    enable_ring_attention: bool
    enable_zigzag: bool
    enable_tbo: bool
    embed_tp: int
    lmhead_tp: int
    comm_protocol: int
    kv_cache_rate: float
    is_prefill: bool
    q_seq_len: int
    kv_seq_len: int


@dataclass
class BoardConfig:
    """板卡规格"""

    num_chips: int
    chip_memory_gb: int
    inter_chip_bw_gbps: float


@dataclass
class InferenceConfig:
    """推理配置"""

    batch_size: int
    input_seq_length: int
    output_seq_length: int
    weight_dtype: str
    activation_dtype: str


@dataclass
class EvalConfig:
    """管线配置入口 -- 全管线单一 source of truth"""

    model: ModelConfig
    chip_config: dict[str, Any]  # 保留 raw dict 给 ChipSpecImpl.from_config
    topology: TopologyOverrides
    comm: CommOverrides
    deployment: DeploymentConfig
    board: BoardConfig
    inference: InferenceConfig
    raw_model_config: dict[str, Any]  # 报告/快照用
    raw_topology_config: dict[str, Any]  # 报告/快照用
    mode: str = "math"  # "math" (代数模型) | "g5" (指令级仿真)


# ============================================
# dict -> EvalConfig 转换
# ============================================


def _require(cfg: dict[str, Any], key: str, source: str) -> Any:
    """从 dict 中获取必需字段，缺失时报错"""
    if key not in cfg:
        raise ValueError(f"Missing required field '{key}' in {source}")
    val = cfg[key]
    if val is None:
        raise ValueError(f"Field '{key}' is None in {source}")
    return val


def _build_topology_overrides(
    topology_config: dict[str, Any],
) -> TopologyOverrides:
    """从 topology_config 构建 TopologyOverrides

    带宽/延迟从 interconnect.links，延迟参数从 interconnect.comm_params
    """
    ic = topology_config.get("interconnect")
    if not ic:
        raise ValueError("Missing 'interconnect' in topology_config")

    links = ic.get("links")
    if not links:
        raise ValueError("Missing 'interconnect.links' in topology_config")

    comm_params = ic.get("comm_params")
    if not comm_params:
        raise ValueError("Missing 'interconnect.comm_params' in topology_config")

    # 从 links 提取 4 层带宽和延迟
    c2c = _require(links, "c2c", "interconnect.links")
    b2b = _require(links, "b2b", "interconnect.links")
    r2r = _require(links, "r2r", "interconnect.links")
    p2p = _require(links, "p2p", "interconnect.links")

    return TopologyOverrides(
        c2c_bandwidth_gbps=float(_require(c2c, "bandwidth_gbps", "interconnect.links.c2c")),
        c2c_latency_us=float(_require(c2c, "latency_us", "interconnect.links.c2c")),
        b2b_bandwidth_gbps=float(_require(b2b, "bandwidth_gbps", "interconnect.links.b2b")),
        b2b_latency_us=float(_require(b2b, "latency_us", "interconnect.links.b2b")),
        r2r_bandwidth_gbps=float(_require(r2r, "bandwidth_gbps", "interconnect.links.r2r")),
        r2r_latency_us=float(_require(r2r, "latency_us", "interconnect.links.r2r")),
        p2p_bandwidth_gbps=float(_require(p2p, "bandwidth_gbps", "interconnect.links.p2p")),
        p2p_latency_us=float(_require(p2p, "latency_us", "interconnect.links.p2p")),
        switch_latency_us=float(_require(comm_params, "switch_latency_us", "interconnect.comm_params")),
        cable_latency_us=float(_require(comm_params, "cable_latency_us", "interconnect.comm_params")),
        memory_read_latency_us=float(_require(comm_params, "memory_read_latency_us", "interconnect.comm_params")),
        memory_write_latency_us=float(_require(comm_params, "memory_write_latency_us", "interconnect.comm_params")),
        noc_latency_us=float(_require(comm_params, "noc_latency_us", "interconnect.comm_params")),
        die_to_die_latency_us=float(_require(comm_params, "die_to_die_latency_us", "interconnect.comm_params")),
    )


def _build_comm_overrides(topology_config: dict[str, Any]) -> CommOverrides:
    """从 topology_config.interconnect.comm_params 构建 CommOverrides"""
    ic = topology_config.get("interconnect", {})
    comm_params = ic.get("comm_params")
    if not comm_params:
        raise ValueError("Missing 'interconnect.comm_params' in topology_config")

    return CommOverrides(
        bw_utilization=float(_require(comm_params, "bandwidth_utilization", "interconnect.comm_params")),
        sync_lat_us=float(_require(comm_params, "sync_latency_us", "interconnect.comm_params")),
    )


def _build_model_config(
    model_cfg: dict[str, Any],
    deployment_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
) -> ModelConfig:
    """从嵌套 YAML 结构 + 运行时参数构建 ModelConfig"""
    # MLA
    mla_raw = model_cfg.get("MLA")
    if mla_raw is None:
        raise ValueError("Missing 'MLA' section in model config")
    mla = MLAConfig(
        q_lora_rank=int(_require(mla_raw, "q_lora_rank", "model.MLA")),
        kv_lora_rank=int(_require(mla_raw, "kv_lora_rank", "model.MLA")),
        qk_nope_head_dim=int(_require(mla_raw, "qk_nope_head_dim", "model.MLA")),
        qk_rope_head_dim=int(_require(mla_raw, "qk_rope_head_dim", "model.MLA")),
        v_head_dim=int(_require(mla_raw, "v_head_dim", "model.MLA")),
        mla_mode=str(_require(mla_raw, "mla_mode", "model.MLA")),
    )

    # MoE
    moe_raw = model_cfg.get("MoE")
    if moe_raw is None:
        raise ValueError("Missing 'MoE' section in model config")
    moe = MoEConfig(
        num_routed_experts=int(_require(moe_raw, "num_routed_experts", "model.MoE")),
        num_shared_experts=int(_require(moe_raw, "num_shared_experts", "model.MoE")),
        num_activated_experts=int(_require(moe_raw, "num_activated_experts", "model.MoE")),
        intermediate_size=int(_require(moe_raw, "intermediate_size", "model.MoE")),
    )

    # 精度配置: 优先从 inference_config 获取，其次 deployment_config
    if "weight_dtype" in inference_cfg:
        weight_dtype = inference_cfg["weight_dtype"]
    elif "weight_dtype" in deployment_cfg:
        weight_dtype = deployment_cfg["weight_dtype"]
    else:
        raise ValueError("Missing 'weight_dtype' in inference config or deployment config")

    if "activation_dtype" in inference_cfg:
        activation_dtype = inference_cfg["activation_dtype"]
    elif "activation_dtype" in deployment_cfg:
        activation_dtype = deployment_cfg["activation_dtype"]
    else:
        raise ValueError("Missing 'activation_dtype' in inference config or deployment config")

    # 运行时参数
    seq_len = int(_require(deployment_cfg, "seq_len", "deployment config"))
    kv_seq_len = int(_require(deployment_cfg, "kv_seq_len", "deployment config"))
    q_seq_len = int(_require(deployment_cfg, "q_seq_len", "deployment config"))
    batch = int(_require(deployment_cfg, "batch_size", "deployment config"))
    is_prefill = bool(_require(deployment_cfg, "is_prefill", "deployment config"))

    return ModelConfig(
        name=str(_require(model_cfg, "name", "model config")),
        hidden_size=int(_require(model_cfg, "hidden_size", "model config")),
        num_layers=int(_require(model_cfg, "num_layers", "model config")),
        num_attention_heads=int(_require(model_cfg, "num_attention_heads", "model config")),
        vocab_size=int(_require(model_cfg, "vocab_size", "model config")),
        intermediate_size=int(_require(model_cfg, "intermediate_size", "model config")),
        num_dense_layers=int(_require(model_cfg, "num_dense_layers", "model config")),
        num_moe_layers=int(_require(model_cfg, "num_moe_layers", "model config")),
        mla=mla,
        moe=moe,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        seq_len=seq_len,
        kv_seq_len=kv_seq_len,
        q_seq_len=q_seq_len,
        batch=batch,
        is_prefill=is_prefill,
    )


def _build_deployment_config(
    manual_parallelism: dict[str, Any],
    inference_cfg: dict[str, Any],
) -> DeploymentConfig:
    """从手动并行配置 + 推理配置构建 DeploymentConfig"""
    input_seq_length = int(_require(inference_cfg, "input_seq_length", "inference config"))
    is_prefill = bool(_require(manual_parallelism, "is_prefill", "manual_parallelism"))

    return DeploymentConfig(
        tp=int(_require(manual_parallelism, "tp", "manual_parallelism")),
        pp=int(_require(manual_parallelism, "pp", "manual_parallelism")),
        dp=int(_require(manual_parallelism, "dp", "manual_parallelism")),
        ep=int(_require(manual_parallelism, "ep", "manual_parallelism")),
        moe_tp=int(_require(manual_parallelism, "moe_tp", "manual_parallelism")),
        seq_len=int(_require(manual_parallelism, "seq_len", "manual_parallelism")),
        batch_size=int(_require(inference_cfg, "batch_size", "inference config")),
        enable_tp_sp=bool(_require(manual_parallelism, "enable_tp_sp", "manual_parallelism")),
        enable_ring_attention=bool(_require(manual_parallelism, "enable_ring_attention", "manual_parallelism")),
        enable_zigzag=bool(_require(manual_parallelism, "enable_zigzag", "manual_parallelism")),
        enable_tbo=bool(manual_parallelism.get("enable_tbo", False)),
        embed_tp=int(_require(manual_parallelism, "embed_tp", "manual_parallelism")),
        lmhead_tp=int(_require(manual_parallelism, "lmhead_tp", "manual_parallelism")),
        comm_protocol=int(_require(manual_parallelism, "comm_protocol", "manual_parallelism")),
        kv_cache_rate=float(_require(manual_parallelism, "kv_cache_rate", "manual_parallelism")),
        is_prefill=is_prefill,
        q_seq_len=input_seq_length if is_prefill else 1,
        kv_seq_len=input_seq_length,
    )


def _build_board_config(topology_config: dict[str, Any]) -> BoardConfig:
    """从拓扑配置提取 BoardConfig"""
    from perf_model.L0_entry.topology_format import count_chips

    num_chips = count_chips(topology_config)

    # 从 interconnect.links.c2c 获取芯片间带宽
    ic = topology_config.get("interconnect", {})
    links = ic.get("links", {})
    if not links:
        raise ValueError("Missing 'interconnect.links' in topology_config")
    c2c = _require(links, "c2c", "topology config interconnect.links")
    inter_chip_bw = float(_require(c2c, "bandwidth_gbps", "topology config interconnect.links.c2c"))

    # 从 chips 获取芯片内存
    chips = topology_config.get("chips")
    if not chips:
        raise ValueError("Missing 'chips' in topology_config")
    first_chip = next(iter(chips.values()))
    memory = _require(first_chip, "memory", "chip config")
    gmem = _require(memory, "gmem", "chip config memory")
    chip_memory = int(_require(gmem, "capacity_gb", "chip config memory.gmem"))

    return BoardConfig(
        num_chips=num_chips,
        chip_memory_gb=chip_memory,
        inter_chip_bw_gbps=inter_chip_bw,
    )


def _build_inference_config(inference_cfg: dict[str, Any]) -> InferenceConfig:
    """构建 InferenceConfig"""
    # 精度配置
    if "weight_dtype" not in inference_cfg:
        raise ValueError("Missing 'weight_dtype' in inference config")
    if "activation_dtype" not in inference_cfg:
        raise ValueError("Missing 'activation_dtype' in inference config")

    return InferenceConfig(
        batch_size=int(_require(inference_cfg, "batch_size", "inference config")),
        input_seq_length=int(_require(inference_cfg, "input_seq_length", "inference config")),
        output_seq_length=int(_require(inference_cfg, "output_seq_length", "inference config")),
        weight_dtype=inference_cfg["weight_dtype"],
        activation_dtype=inference_cfg["activation_dtype"],
    )


def _extract_first_chip_config(chips_dict: dict[str, Any]) -> dict[str, Any]:
    """从芯片字典中提取第一个芯片配置"""
    if not chips_dict:
        raise ValueError("No chip configuration found in topology_config.chips")
    first_chip_name = next(iter(chips_dict))
    first_chip = chips_dict[first_chip_name]
    if "name" not in first_chip:
        first_chip["name"] = first_chip_name
    return first_chip


def build_eval_config(
    chip_config: dict[str, Any],
    model_config: dict[str, Any],
    topology_config: dict[str, Any],
    manual_parallelism: dict[str, Any],
    inference_config: dict[str, Any],
    eval_mode: str = "math",
) -> EvalConfig:
    """唯一的 dict -> EvalConfig 转换点

    Args:
        chip_config: 芯片配置 (raw dict，含 name/cores/memory/...)
        model_config: 模型配置 (嵌套 YAML 格式，含 MLA/MoE)
        topology_config: 完整拓扑配置 (含 interconnect.links + comm_params)
        manual_parallelism: 手动并行配置
        inference_config: 推理配置

    Returns:
        EvalConfig: 类型化配置对象
    """
    # 构建 DeploymentConfig (需要先构建，因为 ModelConfig 依赖其运行时参数)
    deployment = _build_deployment_config(manual_parallelism, inference_config)

    # 将 DeploymentConfig 转为 dict 供 _build_model_config 使用
    deployment_dict = {
        "seq_len": deployment.seq_len,
        "kv_seq_len": deployment.kv_seq_len,
        "q_seq_len": deployment.q_seq_len,
        "batch_size": deployment.batch_size,
        "is_prefill": deployment.is_prefill,
    }

    return EvalConfig(
        model=_build_model_config(model_config, deployment_dict, inference_config),
        chip_config=chip_config,
        topology=_build_topology_overrides(topology_config),
        comm=_build_comm_overrides(topology_config),
        deployment=deployment,
        board=_build_board_config(topology_config),
        inference=_build_inference_config(inference_config),
        raw_model_config=model_config,
        raw_topology_config=topology_config,
        mode=eval_mode,
    )
