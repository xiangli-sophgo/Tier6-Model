"""
Pydantic 模型定义

所有 API 请求/响应的数据模型集中定义在这里
"""

from pydantic import BaseModel, Field
from typing import Any, Optional


# ============================================
# 硬件配置相关
# ============================================

class ChipHardwareConfigRequest(BaseModel):
    """芯片硬件配置请求 - 所有核心字段必须明确指定"""
    chip_type: str = Field(..., description="芯片型号，如 SG2260E, A100, H100")
    num_cores: int = Field(..., gt=0, description="计算核心数")
    compute_tflops_fp8: float = Field(..., gt=0, description="FP8 算力（TFLOPS）")
    compute_tflops_bf16: float = Field(..., gt=0, description="BF16 算力（TFLOPS）")
    memory_capacity_gb: float = Field(..., gt=0, description="显存容量（GB）")
    memory_bandwidth_gbps: float = Field(..., gt=0, description="显存带宽（GB/s）")
    memory_bandwidth_utilization: float = Field(..., gt=0, le=1, description="显存带宽利用率")
    lmem_capacity_mb: float = Field(..., gt=0, description="LMEM/SRAM 缓存容量（MB）")
    lmem_bandwidth_gbps: float = Field(..., gt=0, description="LMEM 缓存带宽（GB/s）")
    c2c_bandwidth_gbps: float = Field(..., gt=0, description="C2C 芯片间互联带宽（GB/s）")
    c2c_latency_us: float = Field(..., gt=0, description="C2C 芯片间互联延迟（微秒）")
    cost_per_hour: float = Field(0.0, ge=0, description="每小时成本")

    # 微架构参数（可选，用于精确 GEMM 评估）
    cube_m: Optional[int] = Field(None, description="矩阵单元 M 维度")
    cube_k: Optional[int] = Field(None, description="矩阵单元 K 维度（累加维度）")
    cube_n: Optional[int] = Field(None, description="矩阵单元 N 维度")
    sram_size_kb: Optional[float] = Field(None, description="每核 SRAM 大小（KB）")
    sram_utilization: Optional[float] = Field(None, description="SRAM 可用比例（0-1）")
    lane_num: Optional[int] = Field(None, description="SIMD lane 数量")
    align_bytes: Optional[int] = Field(None, description="内存对齐字节数")
    compute_dma_overlap_rate: Optional[float] = Field(None, description="计算-搬运重叠率（0-1）")


class BoardConfigRequest(BaseModel):
    """Board 配置请求 (板级互联)"""
    chips_per_board: int = Field(..., gt=0, description="每板芯片数")
    b2b_bandwidth_gbps: float = Field(..., gt=0, description="Board-to-Board 互联带宽（GB/s）")
    b2b_latency_us: float = Field(..., gt=0, description="Board-to-Board 互联延迟（微秒）")


class RackConfigRequest(BaseModel):
    """Rack 配置请求 (机架互联)"""
    boards_per_rack: int = Field(..., gt=0, description="每 Rack 的 Board 数量")
    r2r_bandwidth_gbps: float = Field(..., gt=0, description="Rack-to-Rack 互联带宽（GB/s）")
    r2r_latency_us: float = Field(..., gt=0, description="Rack-to-Rack 互联延迟（微秒）")


class PodConfigRequest(BaseModel):
    """Pod 配置请求 (集群互联)"""
    racks_per_pod: int = Field(..., gt=0, description="每 Pod 的 Rack 数量")
    p2p_bandwidth_gbps: float = Field(..., gt=0, description="Pod-to-Pod 互联带宽（GB/s）")
    p2p_latency_us: float = Field(..., gt=0, description="Pod-to-Pod 互联延迟（微秒）")


class HardwareConfigRequest(BaseModel):
    """硬件配置请求 (四层结构: Chip → Board → Rack → Pod)"""
    chip: ChipHardwareConfigRequest = Field(..., description="芯片配置")
    board: Optional[BoardConfigRequest] = Field(None, description="Board 配置")
    rack: Optional[RackConfigRequest] = Field(None, description="Rack 配置")
    pod: Optional[PodConfigRequest] = Field(None, description="Pod 配置")


# ============================================
# 模型配置相关
# ============================================

class ModelConfigRequest(BaseModel):
    """模型配置请求 - 所有核心字段必须明确指定"""
    model_name: str = Field(..., description="模型名称")
    model_type: str = Field(..., description="模型类型: dense 或 moe")
    hidden_size: int = Field(..., gt=0, description="隐藏层维度")
    num_layers: int = Field(..., gt=0, description="层数")
    num_attention_heads: int = Field(..., gt=0, description="注意力头数")
    num_kv_heads: int = Field(..., gt=0, description="KV 头数（GQA）")
    intermediate_size: int = Field(..., gt=0, description="FFN 中间层维度")
    vocab_size: int = Field(..., gt=0, description="词表大小")
    dtype: str = Field(..., description="数据类型: fp32, fp16, bf16, int8, int4")
    max_seq_length: int = Field(..., gt=0, description="最大序列长度")

    # 可选配置
    attention_type: str = Field("gqa", description="注意力类型: mha, gqa, mqa, mla")
    norm_type: str = Field("rmsnorm", description="归一化类型: layernorm, rmsnorm")
    moe_config: Optional[dict[str, Any]] = Field(None, description="MoE 配置")
    mla_config: Optional[dict[str, Any]] = Field(None, description="MLA 配置（DeepSeek）")


# ============================================
# 推理配置相关
# ============================================

class InferenceConfigRequest(BaseModel):
    """推理配置请求 - 所有核心字段必须明确指定"""
    batch_size: int = Field(..., gt=0, description="批次大小")
    input_seq_length: int = Field(..., gt=0, description="输入序列长度")
    output_seq_length: int = Field(..., gt=0, description="输出序列长度")
    max_seq_length: int = Field(..., gt=0, description="最大序列长度")
    num_micro_batches: int = Field(1, gt=0, description="微批次数量（Pipeline Parallelism）")


class ParallelismConfigRequest(BaseModel):
    """并行策略配置请求 - manual 模式下所有并行度必须明确指定"""
    dp: int = Field(..., ge=1, description="数据并行度")
    tp: int = Field(..., ge=1, description="张量并行度")
    pp: int = Field(..., ge=1, description="流水线并行度")
    ep: int = Field(..., ge=1, description="专家并行度")
    sp: int = Field(1, ge=1, description="序列并行度")
    moe_tp: int = Field(1, ge=1, description="MoE 专家内张量并行度")


# ============================================
# 模拟请求/响应
# ============================================

class SimulationRequest(BaseModel):
    """模拟请求 - 使用严格类型"""
    topology: dict[str, Any]  # 拓扑配置保持灵活，因为结构复杂
    model: ModelConfigRequest
    inference: InferenceConfigRequest
    parallelism: ParallelismConfigRequest
    hardware: HardwareConfigRequest
    config: dict[str, Any] | None = None  # protocol_config 和 network_config 保持灵活


class SimulationResponse(BaseModel):
    """模拟响应"""
    ganttChart: dict[str, Any]
    stats: dict[str, Any]
    timestamp: float


# ============================================
# Benchmark 相关
# ============================================

class BenchmarkConfig(BaseModel):
    """Benchmark 配置"""
    id: str
    name: str
    model: dict[str, Any]
    inference: dict[str, Any]


# ============================================
# 评估任务相关
# ============================================

class EvaluationRequest(BaseModel):
    """评估请求 - 保持向后兼容（使用 dict 类型）"""
    experiment_name: str
    description: str = ""

    # 配置文件引用（追溯来源）
    benchmark_name: Optional[str] = None
    topology_config_name: Optional[str] = None

    # 完整配置数据（保持灵活的 dict 类型，兼容现有前端）
    topology: dict[str, Any]
    model: dict[str, Any]  # 暂时保持 dict，避免破坏前端
    hardware: dict[str, Any]  # 暂时保持 dict
    inference: dict[str, Any]  # 暂时保持 dict

    # 搜索配置
    search_mode: str  # 'manual' or 'auto'
    manual_parallelism: Optional[dict[str, Any]] = None
    search_constraints: Optional[dict[str, Any]] = None

    # 任务并发配置
    max_workers: int = Field(4, ge=1, le=32, description="本任务的最大并发数")

    # GEMM评估配置
    enable_tile_search: bool = Field(True, description="是否启用Tile搜索（关闭可提升评估速度）")
    enable_partition_search: bool = Field(True, description="是否启用分区搜索")

    # 模拟配置
    max_simulated_tokens: int = Field(4, ge=1, le=16, description="最大模拟token数（Decode阶段）")


class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    task_id: str
    message: str


class ExecutorConfigResponse(BaseModel):
    """全局资源池配置响应"""
    max_workers: int  # 全局资源池最大 worker 数量
    running_tasks: int  # 当前已分配的 worker 总数
    active_tasks: int  # 活跃任务数量
    note: str = "修改 max_workers 需要重启服务后生效"


class ExecutorConfigUpdateRequest(BaseModel):
    """全局资源池配置更新请求"""
    max_workers: int  # 全局资源池最大 worker 数量（1-32）


# ============================================
# 实验管理相关
# ============================================

class ExperimentUpdateRequest(BaseModel):
    """实验更新请求 - 用于编辑实验信息"""
    name: Optional[str] = None
    description: Optional[str] = None


class BatchDeleteExperimentsRequest(BaseModel):
    """批量删除实验请求"""
    experiment_ids: list[int] = Field(..., min_length=1, description="要删除的实验 ID 列表")


class BatchDeleteResultsRequest(BaseModel):
    """批量删除结果请求"""
    result_ids: list[int]


# ============================================
# 导入/导出相关
# ============================================

class ExperimentExportData(BaseModel):
    """实验导出数据"""
    id: int
    name: str
    description: Optional[str]
    total_tasks: int
    completed_tasks: int
    tasks: list[dict[str, Any]]


class ExportInfo(BaseModel):
    """导出信息"""
    version: str = "1.0"
    export_time: str
    experiments: list[dict[str, Any]]


class CheckImportResult(BaseModel):
    """导入包检查结果"""
    valid: bool
    error: Optional[str] = None
    experiments: Optional[list[dict[str, Any]]] = None
    temp_file_id: Optional[str] = None


class ImportConfigItem(BaseModel):
    """导入配置项"""
    original_id: Optional[int] = None
    original_name: str
    action: str = Field(..., description="rename, overwrite, or skip")
    new_name: Optional[str] = None


class ImportExecuteRequest(BaseModel):
    """执行导入请求"""
    temp_file_id: str
    configs: list[ImportConfigItem]


class ImportResult(BaseModel):
    """导入结果"""
    success: bool
    imported_count: int
    skipped_count: int
    overwritten_count: int
    message: str


# ============================================
# 拓扑配置相关
# ============================================

class TopologyConfigRequest(BaseModel):
    """拓扑配置请求"""
    name: str = Field(..., description="配置名称（唯一标识）")
    description: Optional[str] = Field(None, description="配置描述")
    pod_count: int = Field(1, description="Pod 数量")
    racks_per_pod: int = Field(1, description="每个 Pod 的 Rack 数量")
    board_configs: Optional[dict] = Field(None, description="Board 配置")
    rack_config: Optional[dict] = Field(None, description="Rack 配置")
    switch_config: Optional[dict] = Field(None, description="交换机配置")
    manual_connections: Optional[dict] = Field(None, description="手动连接配置")
    generated_topology: Optional[dict] = Field(None, description="生成的完整拓扑数据")
    chip_configs: Optional[list] = Field(None, description="芯片硬件配置列表")
    network_config: Optional[dict] = Field(None, description="网络配置")
    comm_latency_config: Optional[dict] = Field(None, description="通信延迟配置")
