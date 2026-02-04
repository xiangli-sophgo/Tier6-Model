"""配置 Schema 定义模块

定义配置验证和转换规则，以及与前端对齐的 Pydantic 请求/响应模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================
# 芯片配置 Schema
# ============================================

CHIP_REQUIRED_FIELDS = [
    "name",
    "compute_tflops_bf16",
    "memory_capacity_gb",
    "memory_bandwidth_gbps",
]

CHIP_OPTIONAL_FIELDS = {
    "num_cores": 64,
    "compute_tflops_fp8": None,  # 默认为 bf16 * 2
    "memory_bandwidth_utilization": 0.85,
    "lmem_capacity_mb": 128,
    "lmem_bandwidth_gbps": 6400,
    "cube_m": 16,
    "cube_k": 32,
    "cube_n": 8,
    "sram_size_kb": 2048,
    "sram_utilization": 0.45,
    "lane_num": 16,
    "align_bytes": 32,
    "compute_dma_overlap_rate": 0.8,
}


def validate_chip_config(config: dict[str, Any]) -> list[str]:
    """验证芯片配置

    Args:
        config: 芯片配置字典

    Returns:
        list[str]: 错误列表 (空列表表示验证通过)
    """
    errors = []

    for field in CHIP_REQUIRED_FIELDS:
        if field not in config or config[field] is None:
            errors.append(f"Missing required field: {field}")

    # 数值验证
    if config.get("compute_tflops_bf16", 0) <= 0:
        errors.append("compute_tflops_bf16 must be positive")
    if config.get("memory_capacity_gb", 0) <= 0:
        errors.append("memory_capacity_gb must be positive")
    if config.get("memory_bandwidth_gbps", 0) <= 0:
        errors.append("memory_bandwidth_gbps must be positive")

    return errors


def normalize_chip_config(config: dict[str, Any]) -> dict[str, Any]:
    """标准化芯片配置

    填充缺失的可选字段为默认值。

    Args:
        config: 原始配置

    Returns:
        dict: 标准化后的配置
    """
    result = dict(config)

    # 填充可选字段
    for field, default in CHIP_OPTIONAL_FIELDS.items():
        if field not in result or result[field] is None:
            if field == "compute_tflops_fp8" and "compute_tflops_bf16" in result:
                # FP8 默认为 BF16 的 2 倍
                result[field] = result["compute_tflops_bf16"] * 2
            else:
                result[field] = default

    return result


# ============================================
# 模型配置 Schema
# ============================================

MODEL_REQUIRED_FIELDS = [
    "model_name",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
]

MODEL_OPTIONAL_FIELDS = {
    "model_type": "dense",
    "num_kv_heads": None,  # 默认等于 num_attention_heads
    "intermediate_size": None,  # 默认为 hidden_size * 4
    "vocab_size": 32000,
    "weight_dtype": "bf16",
    "activation_dtype": "bf16",
    "max_seq_length": 4096,
    "norm_type": "rmsnorm",
    "attention_type": "gqa",
}


def validate_model_config(config: dict[str, Any]) -> list[str]:
    """验证模型配置

    Args:
        config: 模型配置字典

    Returns:
        list[str]: 错误列表
    """
    errors = []

    for field in MODEL_REQUIRED_FIELDS:
        if field not in config or config[field] is None:
            errors.append(f"Missing required field: {field}")

    # 数值验证
    if config.get("hidden_size", 0) <= 0:
        errors.append("hidden_size must be positive")
    if config.get("num_layers", 0) <= 0:
        errors.append("num_layers must be positive")
    if config.get("num_attention_heads", 0) <= 0:
        errors.append("num_attention_heads must be positive")

    # MoE 验证
    if config.get("model_type") == "moe":
        moe_config = config.get("moe_config", {})
        if not moe_config.get("num_experts"):
            errors.append("MoE model requires moe_config.num_experts")

    # MLA 验证
    if config.get("attention_type") == "mla":
        mla_config = config.get("mla_config", {})
        if not mla_config.get("kv_lora_rank"):
            errors.append("MLA attention requires mla_config.kv_lora_rank")

    return errors


def normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """标准化模型配置

    Args:
        config: 原始配置

    Returns:
        dict: 标准化后的配置
    """
    result = dict(config)

    # 填充可选字段
    for field, default in MODEL_OPTIONAL_FIELDS.items():
        if field not in result or result[field] is None:
            if field == "num_kv_heads" and "num_attention_heads" in result:
                result[field] = result["num_attention_heads"]
            elif field == "intermediate_size" and "hidden_size" in result:
                result[field] = result["hidden_size"] * 4
            else:
                result[field] = default

    # 标准化 MoE 配置
    if result.get("model_type") == "moe" and "moe_config" in result:
        moe = result["moe_config"]
        moe.setdefault("num_experts_per_tok", 8)
        moe.setdefault("expert_capacity_factor", 1.0)
        moe.setdefault("num_shared_experts", 1)

    # 标准化 MLA 配置
    if result.get("attention_type") == "mla" and "mla_config" in result:
        mla = result["mla_config"]
        mla.setdefault("q_lora_rank", mla.get("kv_lora_rank", 512) * 3)
        mla.setdefault("qk_nope_head_dim", 128)
        mla.setdefault("qk_rope_head_dim", 64)
        mla.setdefault("v_head_dim", 128)

    return result


# ============================================
# 拓扑配置 Schema
# ============================================

TOPOLOGY_REQUIRED_FIELDS = [
    "name",
    "pod_count",
]

INTERCONNECT_DEFAULTS = {
    "c2c": {"bandwidth_gbps": 448, "latency_us": 0.2},
    "b2b": {"bandwidth_gbps": 400, "latency_us": 2.0},
    "r2r": {"bandwidth_gbps": 200, "latency_us": 3.0},
    "p2p": {"bandwidth_gbps": 100, "latency_us": 5.0},
}


def validate_topology_config(config: dict[str, Any]) -> list[str]:
    """验证拓扑配置

    Args:
        config: 拓扑配置字典

    Returns:
        list[str]: 错误列表
    """
    errors = []

    for field in TOPOLOGY_REQUIRED_FIELDS:
        if field not in config or config[field] is None:
            errors.append(f"Missing required field: {field}")

    # 验证芯片数量
    if config.get("pod_count", 0) <= 0:
        errors.append("pod_count must be positive")

    # 验证 hardware_params
    hw = config.get("hardware_params", {})
    if not hw.get("chips"):
        errors.append("hardware_params.chips is required")

    return errors


def normalize_topology_config(config: dict[str, Any]) -> dict[str, Any]:
    """标准化拓扑配置

    Args:
        config: 原始配置

    Returns:
        dict: 标准化后的配置
    """
    result = dict(config)

    # 确保 hardware_params 存在
    if "hardware_params" not in result:
        result["hardware_params"] = {}

    hw = result["hardware_params"]

    # 确保 interconnect 存在并填充默认值
    if "interconnect" not in hw:
        hw["interconnect"] = {}

    for level, defaults in INTERCONNECT_DEFAULTS.items():
        if level not in hw["interconnect"]:
            hw["interconnect"][level] = dict(defaults)
        else:
            for key, val in defaults.items():
                hw["interconnect"][level].setdefault(key, val)

    # 确保 racks_per_pod 存在
    result.setdefault("racks_per_pod", 1)

    return result


# ============================================
# 并行配置 Schema
# ============================================

PARALLELISM_DEFAULTS = {
    "tp": 1,
    "pp": 1,
    "dp": 1,
    "ep": 1,
    "sp": 1,
    "moe_tp": 1,
}


def validate_parallelism_config(config: dict[str, Any]) -> list[str]:
    """验证并行配置

    Args:
        config: 并行配置字典

    Returns:
        list[str]: 错误列表
    """
    errors = []

    for field in ["tp", "pp", "dp", "ep"]:
        val = config.get(field, 1)
        if val < 1:
            errors.append(f"{field} must be >= 1")

    return errors


def normalize_parallelism_config(config: dict[str, Any]) -> dict[str, Any]:
    """标准化并行配置

    Args:
        config: 原始配置

    Returns:
        dict: 标准化后的配置
    """
    result = dict(config)

    for field, default in PARALLELISM_DEFAULTS.items():
        result.setdefault(field, default)

    return result


# ============================================
# 推理配置 Schema
# ============================================

INFERENCE_DEFAULTS = {
    "batch_size": 1,
    "input_seq_length": 1024,
    "output_seq_length": 128,
    "max_seq_length": 4096,
    "num_micro_batches": 1,
}


def validate_inference_config(config: dict[str, Any]) -> list[str]:
    """验证推理配置

    Args:
        config: 推理配置字典

    Returns:
        list[str]: 错误列表
    """
    errors = []

    if config.get("batch_size", 0) < 1:
        errors.append("batch_size must be >= 1")
    if config.get("input_seq_length", 0) < 1:
        errors.append("input_seq_length must be >= 1")
    if config.get("output_seq_length", 0) < 1:
        errors.append("output_seq_length must be >= 1")

    return errors


def normalize_inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """标准化推理配置

    Args:
        config: 原始配置

    Returns:
        dict: 标准化后的配置
    """
    result = dict(config)

    for field, default in INFERENCE_DEFAULTS.items():
        result.setdefault(field, default)

    return result


# ============================================
# 完整配置验证
# ============================================

@dataclass
class ValidationResult:
    """验证结果

    Attributes:
        valid: 是否有效
        errors: 错误列表
        warnings: 警告列表
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_evaluation_config(config: dict[str, Any]) -> ValidationResult:
    """验证完整评估配置

    Args:
        config: 评估配置字典

    Returns:
        ValidationResult: 验证结果
    """
    result = ValidationResult()

    # 验证各子配置
    if "chip_config" in config:
        errors = validate_chip_config(config["chip_config"])
        result.errors.extend([f"chip_config: {e}" for e in errors])

    if "model_config" in config:
        errors = validate_model_config(config["model_config"])
        result.errors.extend([f"model_config: {e}" for e in errors])

    if "topology_config" in config:
        errors = validate_topology_config(config["topology_config"])
        result.errors.extend([f"topology_config: {e}" for e in errors])

    if "parallelism" in config:
        errors = validate_parallelism_config(config["parallelism"])
        result.errors.extend([f"parallelism: {e}" for e in errors])

    if "inference" in config:
        errors = validate_inference_config(config["inference"])
        result.errors.extend([f"inference: {e}" for e in errors])

    # 检查是否使用预设
    if not config.get("chip_config") and not config.get("chip_preset"):
        result.warnings.append("No chip configuration or preset specified")

    if not config.get("model_config") and not config.get("model_preset"):
        result.warnings.append("No model configuration or preset specified")

    result.valid = len(result.errors) == 0

    return result


def normalize_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    """标准化完整评估配置

    Args:
        config: 原始配置

    Returns:
        dict: 标准化后的配置
    """
    result = dict(config)

    if "chip_config" in result:
        result["chip_config"] = normalize_chip_config(result["chip_config"])

    if "model_config" in result:
        result["model_config"] = normalize_model_config(result["model_config"])

    if "topology_config" in result:
        result["topology_config"] = normalize_topology_config(result["topology_config"])

    if "parallelism" in result:
        result["parallelism"] = normalize_parallelism_config(result["parallelism"])
    else:
        result["parallelism"] = dict(PARALLELISM_DEFAULTS)

    if "inference" in result:
        result["inference"] = normalize_inference_config(result["inference"])
    else:
        result["inference"] = dict(INFERENCE_DEFAULTS)

    return result


# ============================================
# Pydantic 请求/响应模型 (与前端对齐)
# ============================================


class ManualParallelism(BaseModel):
    """手动并行配置 - 与前端 ManualParallelism 接口对齐"""

    tp: int = Field(1, ge=1, description="Tensor Parallelism 并行度")
    pp: int = Field(1, ge=1, description="Pipeline Parallelism 并行度")
    dp: int = Field(1, ge=1, description="Data Parallelism 并行度")
    ep: int = Field(1, ge=1, description="Expert Parallelism 并行度")
    moe_tp: int = Field(1, ge=1, description="MoE Expert 内部 TP 并行度")
    seq_len: int = Field(1, ge=1, description="序列长度分片")
    batch_size: int = Field(1, ge=1, description="批次大小")
    enable_tp_sp: bool = Field(False, description="是否启用 TP sequence parallelism")
    embed_tp: int = Field(1, ge=1, description="Embedding 层 TP 并行度")
    lmhead_tp: int = Field(1, ge=1, description="LMHead 层 TP 并行度")
    comm_protocol: int = Field(1, ge=0, description="通信协议 (0=基础, 1=优化)")
    kv_cache_rate: float = Field(0.0, ge=0.0, le=1.0, description="KV Cache 比例")
    is_prefill: bool = Field(False, description="是否为 Prefill 阶段")


class SearchConstraints(BaseModel):
    """搜索约束配置"""

    min_tp: int = Field(1, ge=1)
    max_tp: int = Field(32, ge=1)
    min_pp: int = Field(1, ge=1)
    max_pp: int = Field(16, ge=1)
    min_dp: int = Field(1, ge=1)
    max_dp: int = Field(64, ge=1)
    min_ep: int = Field(1, ge=1)
    max_ep: int = Field(256, ge=1)
    target_latency_ms: Optional[float] = Field(None, description="目标延迟约束 (ms)")
    target_throughput_tps: Optional[float] = Field(None, description="目标吞吐约束 (tokens/s)")
    memory_budget_gb: Optional[float] = Field(None, description="显存预算约束 (GB)")


class EvaluationRequest(BaseModel):
    """评估请求 - 与前端 EvaluationRequest 接口对齐"""

    experiment_name: str = Field(..., description="实验名称")
    description: str = Field("", description="实验描述 (简短)")
    experiment_description: str = Field("", description="实验描述 (详细)")
    benchmark_name: str = Field(..., description="Benchmark 名称")
    topology_config_name: str = Field(..., description="拓扑配置名称")
    benchmark_config: Dict[str, Any] = Field(..., description="Benchmark 配置 { model, inference }")
    topology_config: Dict[str, Any] = Field(..., description="完整拓扑配置 (含 comm_latency_config)")
    search_mode: Literal["manual", "auto", "sweep"] = Field("manual", description="搜索模式")
    manual_parallelism: Optional[ManualParallelism] = Field(None, description="手动并行配置")
    search_constraints: Optional[SearchConstraints] = Field(None, description="自动搜索约束")
    max_workers: int = Field(4, ge=1, le=32, description="最大并行任务数")
    enable_tile_search: bool = Field(True, description="是否启用 Tile 搜索")
    enable_partition_search: bool = Field(True, description="是否启用分区搜索")
    max_simulated_tokens: int = Field(4, ge=1, description="最大仿真 Token 数")


class SimulateRequest(BaseModel):
    """同步仿真请求"""

    chip_preset: Optional[str] = Field(None, description="芯片预设名称")
    model_preset: Optional[str] = Field(None, description="模型预设名称")
    topology_preset: Optional[str] = Field(None, description="拓扑预设名称")
    chip_config: Optional[Dict[str, Any]] = Field(None, description="芯片配置 (内联)")
    model_params: Optional[Dict[str, Any]] = Field(None, description="模型配置 (内联)", alias="model_config")
    topology_config: Optional[Dict[str, Any]] = Field(None, description="拓扑配置 (内联)")
    parallelism: ManualParallelism = Field(default_factory=ManualParallelism)
    inference: Dict[str, Any] = Field(default_factory=dict)


class ValidateRequest(BaseModel):
    """配置验证请求"""

    chip_config: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = Field(None, alias="model_config")
    topology_config: Optional[Dict[str, Any]] = None
    parallelism: Optional[Dict[str, Any]] = None
    inference: Optional[Dict[str, Any]] = None


class CalculateParamsRequest(BaseModel):
    """计算模型参数量请求"""

    model_params: Dict[str, Any] = Field(..., description="模型配置", alias="model_config")


class BenchmarkCreateRequest(BaseModel):
    """创建 Benchmark 请求"""

    id: str = Field(..., description="Benchmark ID")
    name: str = Field(..., description="Benchmark 名称")
    model: Dict[str, Any] = Field(..., description="模型配置")
    inference: Dict[str, Any] = Field(..., description="推理配置")


class BenchmarkUpdateRequest(BaseModel):
    """更新 Benchmark 请求"""

    name: Optional[str] = Field(None, description="Benchmark 名称")
    model: Optional[Dict[str, Any]] = Field(None, description="模型配置")
    inference: Optional[Dict[str, Any]] = Field(None, description="推理配置")


class TopologyCreateRequest(BaseModel):
    """创建拓扑请求 - 兼容前端 SavedConfig 格式"""
    model_config = {"extra": "allow"}  # 允许额外字段

    name: str = Field(..., description="拓扑名称")
    description: Optional[str] = Field(None, description="描述")
    pod_count: Optional[int] = Field(None, description="Pod 数量")
    racks_per_pod: Optional[int] = Field(None, description="每 Pod 的 Rack 数量")
    rack_config: Optional[Dict[str, Any]] = Field(None, description="Rack 配置")
    switch_config: Optional[Dict[str, Any]] = Field(None, description="Switch 配置")
    manual_connections: Optional[Dict[str, Any]] = Field(None, description="手动连接配置")
    generated_topology: Optional[Dict[str, Any]] = Field(None, description="生成的拓扑")
    hardware_params: Optional[Dict[str, Any]] = Field(None, description="硬件参数")


class TopologyUpdateRequest(BaseModel):
    """更新拓扑请求 - 兼容前端 SavedConfig 格式"""
    model_config = {"extra": "allow"}  # 允许额外字段

    name: Optional[str] = None
    description: Optional[str] = None
    pod_count: Optional[int] = None
    racks_per_pod: Optional[int] = None
    rack_config: Optional[Dict[str, Any]] = None
    switch_config: Optional[Dict[str, Any]] = None
    manual_connections: Optional[Dict[str, Any]] = None
    generated_topology: Optional[Dict[str, Any]] = None
    hardware_params: Optional[Dict[str, Any]] = None


class ExecutorConfigRequest(BaseModel):
    """执行器配置请求"""

    max_workers: int = Field(4, ge=1, le=32, description="最大并行任务数")
    max_queued: int = Field(100, ge=1, le=1000, description="最大排队任务数")


class ExperimentUpdateRequest(BaseModel):
    """更新实验请求"""

    name: Optional[str] = Field(None, description="实验名称")
    description: Optional[str] = Field(None, description="实验描述")


class BatchDeleteRequest(BaseModel):
    """批量删除请求"""

    ids: List[str] = Field(..., description="要删除的 ID 列表")


class ImportCheckResponse(BaseModel):
    """导入检查响应"""

    temp_file_id: str = Field(..., description="临时文件 ID")
    experiments: List[Dict[str, Any]] = Field(..., description="实验预览列表")
    conflicts: List[str] = Field(default_factory=list, description="冲突的实验名称")


class ImportExecuteRequest(BaseModel):
    """执行导入请求"""

    temp_file_id: str = Field(..., description="临时文件 ID")
    conflict_strategy: Literal["skip", "overwrite", "rename"] = Field(
        "skip", description="冲突解决策略"
    )


# ============================================
# 响应模型
# ============================================


class PresetInfo(BaseModel):
    """预设信息"""

    name: str
    config: Dict[str, Any]


class PresetListResponse(BaseModel):
    """预设列表响应"""

    presets: List[PresetInfo]


class BenchmarkInfo(BaseModel):
    """Benchmark 信息"""

    id: str
    name: str
    format: str = "yaml"
    filename: str


class BenchmarkListResponse(BaseModel):
    """Benchmark 列表响应"""

    benchmarks: List[BenchmarkInfo]


class TopologyInfo(BaseModel):
    """拓扑信息"""

    name: str
    chip_count: int = 0
    topology_type: str = ""


class TopologyListResponse(BaseModel):
    """拓扑列表响应"""

    topologies: List[TopologyInfo]


class TaskStatusResponse(BaseModel):
    """任务状态响应"""

    task_id: str
    status: str
    progress: float = 0.0
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""

    tasks: List[TaskStatusResponse]
    total: int


class ExperimentInfo(BaseModel):
    """实验信息"""

    id: str
    name: str
    description: str = ""
    created_at: str
    updated_at: Optional[str] = None
    task_count: int = 0
    status: str = "pending"


class ExperimentListResponse(BaseModel):
    """实验列表响应"""

    experiments: List[ExperimentInfo]
    total: int


class ValidationResponse(BaseModel):
    """验证响应"""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CalculateParamsResponse(BaseModel):
    """计算模型参数量响应"""

    total_params: int = Field(..., description="总参数量")
    total_params_b: float = Field(..., description="总参数量 (B)")
    active_params: int = Field(0, description="激活参数量 (MoE)")
    active_params_b: float = Field(0.0, description="激活参数量 (B)")
    weight_size_bytes: int = Field(..., description="权重大小 (bytes)")
    weight_size_gb: float = Field(..., description="权重大小 (GB)")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="参数分解")
