"""配置 Schema 定义模块

定义配置验证规则和 Pydantic 请求/响应模型。
所有配置加载禁止使用默认值 - 缺失必需字段必须报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================
# 芯片配置 Schema
# ============================================

# 芯片配置必须存在的路径
CHIP_REQUIRED_PATHS = [
    "name",
    "frequency_ghz",
    ("cores", "count"),
    ("cores", "lanes_per_core"),
    ("memory", "gmem", "capacity_gb"),
    ("memory", "gmem", "bandwidth_gbps"),
    ("memory", "lmem", "capacity_mb"),
    ("memory", "lmem", "bandwidth_gbps"),
    ("dma_engines", "gdma", "bandwidth_gbps"),
    "align_bytes",
    "compute_dma_overlap_rate",
]


def _check_nested_path(config: dict, path: tuple | str, config_source: str = "") -> str | None:
    """检查嵌套路径是否存在

    Returns:
        错误信息或 None
    """
    if isinstance(path, str):
        path = (path,)

    current = config
    for i, key in enumerate(path):
        if not isinstance(current, dict) or key not in current:
            full_path = ".".join(path[:i + 1])
            source_info = f" in {config_source}" if config_source else ""
            return f"Missing required field '{full_path}'{source_info}"
        current = current[key]

    if current is None:
        full_path = ".".join(path)
        source_info = f" in {config_source}" if config_source else ""
        return f"Field '{full_path}' is None{source_info}"

    return None


def validate_chip_config(config: dict[str, Any], source: str = "") -> list[str]:
    """验证芯片配置

    Args:
        config: 芯片配置字典
        source: 配置来源 (用于错误信息)

    Returns:
        错误列表 (空列表表示验证通过)
    """
    errors = []

    for path in CHIP_REQUIRED_PATHS:
        error = _check_nested_path(config, path, source)
        if error:
            errors.append(error)

    # 计算单元至少需要一种 dtype 的 mac_per_lane
    cube = config.get("compute_units", {}).get("cube", {})
    mac_per_lane = cube.get("mac_per_lane", {})
    if not mac_per_lane:
        errors.append(f"Missing 'compute_units.cube.mac_per_lane' (at least one dtype required)")

    # 数值验证
    cores = config.get("cores", {})
    if isinstance(cores, dict):
        count = cores.get("count")
        if count is not None and count <= 0:
            errors.append("cores.count must be positive")

    gmem = config.get("memory", {}).get("gmem", {})
    if isinstance(gmem, dict):
        cap = gmem.get("capacity_gb")
        if cap is not None and cap <= 0:
            errors.append("memory.gmem.capacity_gb must be positive")
        bw = gmem.get("bandwidth_gbps")
        if bw is not None and bw <= 0:
            errors.append("memory.gmem.bandwidth_gbps must be positive")

    return errors


# ============================================
# 模型配置 Schema
# ============================================

MODEL_REQUIRED_FIELDS = [
    "name",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "vocab_size",
]


def validate_model_config(config: dict[str, Any], source: str = "") -> list[str]:
    """验证模型配置

    Args:
        config: 模型配置字典
        source: 配置来源

    Returns:
        错误列表
    """
    errors = []

    # 支持 model: {} 嵌套格式
    check_config = config.get("model", config) if "model" in config else config

    for f in MODEL_REQUIRED_FIELDS:
        if f not in check_config or check_config[f] is None:
            source_info = f" in {source}" if source else ""
            errors.append(f"Missing required field '{f}'{source_info}")

    # 数值验证
    if check_config.get("hidden_size", 0) <= 0:
        errors.append("hidden_size must be positive")
    if check_config.get("num_layers", 0) <= 0:
        errors.append("num_layers must be positive")
    if check_config.get("num_attention_heads", 0) <= 0:
        errors.append("num_attention_heads must be positive")

    # MoE 验证
    moe = check_config.get("MoE", {})
    if moe:
        for moe_field in ["num_routed_experts", "num_activated_experts", "intermediate_size"]:
            if moe_field not in moe:
                errors.append(f"Missing 'moe.{moe_field}' in MoE model config")

    # MLA 验证
    mla = check_config.get("MLA", {})
    if mla:
        for mla_field in ["q_lora_rank", "kv_lora_rank"]:
            if mla_field not in mla:
                errors.append(f"Missing 'mla.{mla_field}' in MLA config")

    return errors


# ============================================
# 拓扑配置 Schema
# ============================================

TOPOLOGY_REQUIRED_FIELDS = [
    "name",
]


def validate_topology_config(config: dict[str, Any], source: str = "") -> list[str]:
    """验证拓扑配置 (grouped_pods 格式)

    Args:
        config: 拓扑配置字典
        source: 配置来源

    Returns:
        错误列表
    """
    errors = []

    for f in TOPOLOGY_REQUIRED_FIELDS:
        if f not in config or config[f] is None:
            source_info = f" in {source}" if source else ""
            errors.append(f"Missing required field '{f}'{source_info}")

    # pods 结构验证
    pods = config.get("pods")
    if not pods:
        source_info = f" in {source}" if source else ""
        errors.append(f"Missing required field 'pods'{source_info}")
    elif isinstance(pods, list):
        for i, pod_group in enumerate(pods):
            if not isinstance(pod_group, dict):
                errors.append(f"pods[{i}] must be a dict")
                continue
            racks = pod_group.get("racks")
            if not racks:
                errors.append(f"pods[{i}] missing 'racks'")
                continue
            for j, rack_group in enumerate(racks):
                boards = rack_group.get("boards")
                if not boards:
                    errors.append(f"pods[{i}].racks[{j}] missing 'boards'")
                    continue
                for k, board in enumerate(boards):
                    chips = board.get("chips")
                    if not chips:
                        errors.append(f"pods[{i}].racks[{j}].boards[{k}] missing 'chips'")

    # interconnect.links 必需
    ic = config.get("interconnect", {})
    links = ic.get("links", {})
    for level in ["c2c", "b2b", "r2r", "p2p"]:
        if level not in links:
            errors.append(f"Missing 'interconnect.links.{level}'")
        else:
            level_config = links[level]
            if "bandwidth_gbps" not in level_config:
                errors.append(f"Missing 'interconnect.links.{level}.bandwidth_gbps'")
            if "latency_us" not in level_config:
                errors.append(f"Missing 'interconnect.links.{level}.latency_us'")

    return errors


# ============================================
# 并行配置 Schema
# ============================================


def validate_parallelism_config(config: dict[str, Any]) -> list[str]:
    """验证并行配置"""
    errors = []
    for f in ["tp", "pp", "dp", "ep"]:
        val = config.get(f, 1)
        if val < 1:
            errors.append(f"{f} must be >= 1")
    return errors


# ============================================
# 推理配置 Schema
# ============================================


def validate_inference_config(config: dict[str, Any]) -> list[str]:
    """验证推理配置"""
    errors = []
    if config.get("batch_size", 0) < 1:
        errors.append("batch_size must be >= 1")
    if config.get("input_seq_length", 0) < 1:
        errors.append("input_seq_length must be >= 1")
    if config.get("output_seq_length", 0) < 1:
        errors.append("output_seq_length must be >= 1")
    return errors


# ============================================
# 完整配置验证
# ============================================

@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_evaluation_config(config: dict[str, Any]) -> ValidationResult:
    """验证完整评估配置"""
    result = ValidationResult()

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

    if not config.get("chip_config") and not config.get("chip_preset"):
        result.warnings.append("No chip configuration or preset specified")

    if not config.get("model_config") and not config.get("model_preset"):
        result.warnings.append("No model configuration or preset specified")

    result.valid = len(result.errors) == 0
    return result


# ============================================
# Pydantic 请求/响应模型 (与前端对齐)
# ============================================


class ManualParallelism(BaseModel):
    """手动并行配置"""
    tp: int = Field(1, ge=1)
    pp: int = Field(1, ge=1)
    dp: int = Field(1, ge=1)
    ep: int = Field(1, ge=1)
    moe_tp: int = Field(1, ge=1)
    seq_len: int = Field(1, ge=1)
    batch_size: int = Field(1, ge=1)
    enable_tp_sp: bool = Field(False)
    embed_tp: int = Field(1, ge=1)
    lmhead_tp: int = Field(1, ge=1)
    comm_protocol: int = Field(1, ge=0)
    kv_cache_rate: float = Field(0.0, ge=0.0, le=1.0)
    is_prefill: bool = Field(False)
    enable_zigzag: bool = Field(False)
    enable_ring_attention: bool = Field(False)


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
    target_latency_ms: Optional[float] = None
    target_throughput_tps: Optional[float] = None
    memory_budget_gb: Optional[float] = None


class EvaluationRequest(BaseModel):
    """评估请求"""
    experiment_name: str = Field(...)
    description: str = Field("")
    experiment_description: str = Field("")
    benchmark_name: str = Field(...)
    topology_config_name: str = Field(...)
    benchmark_config: Dict[str, Any] = Field(...)
    topology_config: Dict[str, Any] = Field(...)
    search_mode: Literal["manual", "auto", "sweep"] = Field("manual")
    manual_parallelism: Optional[ManualParallelism] = None
    search_constraints: Optional[SearchConstraints] = None
    max_workers: int = Field(4, ge=1, le=32)


class SimulateRequest(BaseModel):
    """同步仿真请求"""
    chip_preset: Optional[str] = None
    model_preset: Optional[str] = None
    topology_preset: Optional[str] = None
    chip_config: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = Field(None, alias="model_config")
    topology_config: Optional[Dict[str, Any]] = None
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
    model_params: Dict[str, Any] = Field(..., alias="model_config")


class BenchmarkCreateRequest(BaseModel):
    """创建 Benchmark 请求"""
    id: str = Field(...)
    name: str = Field(...)
    model: Any = Field(...)  # 字符串引用 (如 "deepseek-v3") 或完整配置 dict
    topology: Optional[str] = None  # 拓扑引用名称 (如 "P1-R1-B1-C8")
    inference: Dict[str, Any] = Field(...)


class BenchmarkUpdateRequest(BaseModel):
    """更新 Benchmark 请求"""
    name: Optional[str] = None
    model: Optional[Any] = None  # 字符串引用或完整配置 dict
    topology: Optional[str] = None  # 拓扑引用名称
    inference: Optional[Dict[str, Any]] = None


class TopologyCreateRequest(BaseModel):
    """创建拓扑请求 (grouped_pods 格式)"""
    model_config = {"extra": "allow"}

    name: str = Field(...)
    description: Optional[str] = None
    pods: List[Dict[str, Any]] = Field(...)
    chips: Optional[Dict[str, Any]] = None
    interconnect: Optional[Dict[str, Any]] = None
    switch_config: Optional[Dict[str, Any]] = None
    manual_connections: Optional[Dict[str, Any]] = None
    generated_topology: Optional[Dict[str, Any]] = None


class TopologyUpdateRequest(BaseModel):
    """更新拓扑请求 (grouped_pods 格式)"""
    model_config = {"extra": "allow"}

    name: Optional[str] = None
    description: Optional[str] = None
    pods: Optional[List[Dict[str, Any]]] = None
    chips: Optional[Dict[str, Any]] = None
    interconnect: Optional[Dict[str, Any]] = None
    switch_config: Optional[Dict[str, Any]] = None
    manual_connections: Optional[Dict[str, Any]] = None
    generated_topology: Optional[Dict[str, Any]] = None


# ============================================
# 新增: 模型预设 CRUD 请求
# ============================================


class ModelPresetCreateRequest(BaseModel):
    """创建模型预设请求"""
    name: str = Field(...)
    config: Dict[str, Any] = Field(...)


class ModelPresetUpdateRequest(BaseModel):
    """更新模型预设请求"""
    config: Dict[str, Any] = Field(...)


class ChipPresetUpdateRequest(BaseModel):
    """更新芯片预设请求"""
    config: Dict[str, Any] = Field(...)


# ============================================
# 其他请求/响应模型
# ============================================


class ExecutorConfigRequest(BaseModel):
    """执行器配置请求"""
    max_workers: int = Field(4, ge=1, le=32)
    max_queued: int = Field(100, ge=1, le=1000)


class ExperimentUpdateRequest(BaseModel):
    """更新实验请求"""
    name: Optional[str] = None
    description: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    """批量删除请求 - 兼容多种字段名"""
    ids: Optional[List[int]] = None
    experiment_ids: Optional[List[int]] = None
    result_ids: Optional[List[int]] = None

    @property
    def resolved_ids(self) -> List[int]:
        """统一解析出实际的 ID 列表（兼容 ids / experiment_ids / result_ids）"""
        if self.ids is not None:
            return self.ids
        if self.experiment_ids is not None:
            return self.experiment_ids
        if self.result_ids is not None:
            return self.result_ids
        return []


class ImportCheckResponse(BaseModel):
    """导入检查响应"""
    temp_file_id: str = Field(...)
    experiments: List[Dict[str, Any]] = Field(...)
    conflicts: List[str] = Field(default_factory=list)


class ImportExecuteRequest(BaseModel):
    """执行导入请求"""
    temp_file_id: str = Field(...)
    conflict_strategy: Literal["skip", "overwrite", "rename"] = Field("skip")


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
    topology: Optional[str] = None
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
    total_params: int = Field(...)
    total_params_b: float = Field(...)
    active_params: int = Field(0)
    active_params_b: float = Field(0.0)
    weight_size_bytes: int = Field(...)
    weight_size_gb: float = Field(...)
    breakdown: Dict[str, Any] = Field(default_factory=dict)
