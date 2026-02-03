"""
LLM 推理模拟器 - 类型定义

定义所有数据结构，包括：
- 拓扑配置
- 硬件配置
- 模型配置
- 模拟结果
- 甘特图数据
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================
# 枚举类型
# ============================================

class GanttTaskType(str, Enum):
    """甘特图任务类型"""
    # 计算任务
    COMPUTE = "compute"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_SCORE = "attention_score"
    ATTENTION_SOFTMAX = "attention_softmax"
    ATTENTION_OUTPUT = "attention_output"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    LM_HEAD = "lm_head"

    # 数据搬运
    PCIE_H2D = "pcie_h2d"
    PCIE_D2H = "pcie_d2h"
    HBM_WRITE = "hbm_write"
    HBM_READ = "hbm_read"
    WEIGHT_LOAD = "weight_load"
    KV_CACHE_READ = "kv_cache_read"
    KV_CACHE_WRITE = "kv_cache_write"

    # 通信
    TP_COMM = "tp_comm"
    PP_COMM = "pp_comm"
    EP_COMM = "ep_comm"

    # SP 通信 (序列并行)
    SP_ALLGATHER = "sp_allgather"
    SP_REDUCE_SCATTER = "sp_reduce_scatter"

    # DP 通信 (数据并行梯度同步)
    DP_GRADIENT_SYNC = "dp_gradient_sync"

    # MLA细粒度 (DeepSeek特有)
    RMSNORM_Q_LORA = "rmsnorm_q_lora"
    RMSNORM_KV_LORA = "rmsnorm_kv_lora"
    MM_Q_LORA_A = "mm_q_lora_a"
    MM_Q_LORA_B = "mm_q_lora_b"
    MM_KV_LORA_A = "mm_kv_lora_a"
    ATTN_FC = "attn_fc"
    BMM_QK = "bmm_qk"
    BMM_SV = "bmm_sv"

    # MoE (DeepSeek, Qwen-MoE, Mixtral等)
    MOE_GATE = "moe_gate"
    MOE_EXPERT = "moe_expert"
    MOE_SHARED_EXPERT = "moe_shared_expert"
    EP_DISPATCH = "ep_dispatch"
    EP_COMBINE = "ep_combine"

    # 其他
    BUBBLE = "bubble"
    IDLE = "idle"


class InferencePhase(str, Enum):
    """推理阶段"""
    PREFILL = "prefill"
    DECODE = "decode"


class BottleneckType(str, Enum):
    """瓶颈类型"""
    COMPUTE = "compute"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    PIPELINE_BUBBLE = "pipeline_bubble"


class AllReduceAlgorithm(str, Enum):
    """AllReduce算法类型

    - RING: Ring AllReduce, 适合N<=32的场景
    - DOUBLE_BINARY_TREE: 双二叉树, NCCL多机默认, 适合N>32
    - HALVING_DOUBLING: Halving-Doubling, 适合Fat-Tree拓扑
    - REDUCE_BROADCAST: Reduce+Broadcast, 适合N<8且Full-Mesh拓扑
    """
    RING = "ring"
    DOUBLE_BINARY_TREE = "double_binary_tree"
    HALVING_DOUBLING = "halving_doubling"
    REDUCE_BROADCAST = "reduce_broadcast"


class AllToAllAlgorithm(str, Enum):
    """All-to-All算法类型

    - PAIRWISE: 两两交换, 适合小规模(N<8)
    - RING: Ring-based, 适合大规模, 低带宽要求
    - BRUCK: Bruck算法, 适合中规模(8<=N<=32), 低延迟
    """
    PAIRWISE = "pairwise"
    RING = "ring"
    BRUCK = "bruck"


# ============================================
# 拓扑配置 (拓扑结构 + 硬件参数 合并)
# ============================================

@dataclass
class ChipConfig:
    """芯片配置 (拓扑 + 硬件参数)"""
    # === 拓扑信息 ===
    id: str
    type: str = "chip"
    position: tuple[int, int] = (0, 0)
    label: str = ""
    # === 硬件参数 ===
    num_cores: int = 0  # 计算核心数
    compute_tflops_fp8: float = 0.0  # FP8 算力 (TFLOPS)
    compute_tflops_bf16: float = 0.0  # BF16 算力 (TFLOPS)
    memory_capacity_gb: float = 0.0  # 显存容量 (GB)
    memory_bandwidth_gbps: float = 0.0  # 显存带宽 (GB/s)
    memory_bandwidth_utilization: float = 0.85  # 显存带宽利用率 (0-1)
    lmem_capacity_mb: float = 0.0  # LMEM/SRAM 片上缓存容量 (MB)
    lmem_bandwidth_gbps: float = 0.0  # LMEM 缓存带宽 (GB/s)
    # === 微架构参数 (可选，用于精确 GEMM 评估) ===
    cube_m: Optional[int] = None
    cube_k: Optional[int] = None
    cube_n: Optional[int] = None
    sram_size_kb: Optional[float] = None
    sram_utilization: Optional[float] = None
    lane_num: Optional[int] = None
    align_bytes: Optional[int] = None
    compute_dma_overlap_rate: Optional[float] = None


@dataclass
class BoardConfig:
    """板卡配置 (拓扑 + 互联参数)"""
    # === 拓扑信息 ===
    id: str
    u_position: int
    u_height: int
    label: str
    chips: list[ChipConfig] = field(default_factory=list)
    # === 互联参数 (B2B: Board-to-Board) ===
    b2b_bandwidth_gbps: float = 0.0  # Board 间互联带宽 (GB/s)
    b2b_latency_us: float = 0.0  # Board 间互联延迟 (us)


@dataclass
class RackConfig:
    """机柜配置 (拓扑 + 互联参数)"""
    # === 拓扑信息 ===
    id: str
    position: tuple[int, int]
    label: str
    total_u: int = 42
    boards: list[BoardConfig] = field(default_factory=list)
    # === 互联参数 (R2R: Rack-to-Rack) ===
    r2r_bandwidth_gbps: float = 0.0  # Rack 间互联带宽 (GB/s)
    r2r_latency_us: float = 0.0  # Rack 间互联延迟 (us)


@dataclass
class PodConfig:
    """Pod配置 (拓扑 + 互联参数)"""
    # === 拓扑信息 ===
    id: str
    label: str
    grid_size: tuple[int, int]
    racks: list[RackConfig] = field(default_factory=list)
    # === 互联参数 (P2P: Pod-to-Pod) ===
    p2p_bandwidth_gbps: float = 0.0  # Pod 间互联带宽 (GB/s)
    p2p_latency_us: float = 0.0  # Pod 间互联延迟 (us)


@dataclass
class ConnectionConfig:
    """连接配置"""
    source: str
    target: str
    type: str  # 'intra' | 'inter' | 'switch' | 'manual'
    bandwidth: float = 0.0  # GB/s (与 presets.ts 保持一致)
    latency: float = 0.0  # us (微秒，与其他配置保持一致)


@dataclass
class HierarchicalTopology:
    """层级拓扑配置"""
    pods: list[PodConfig] = field(default_factory=list)
    connections: list[ConnectionConfig] = field(default_factory=list)
    # Switch 配置（可选，Phase 3）
    # 格式: {"switches": [...], "switch_types": {...}}
    switches: list[dict] = field(default_factory=list)
    switch_types: dict[str, dict] = field(default_factory=dict)


# ============================================
# 模型配置
# ============================================

@dataclass
class MoEConfig:
    """MoE配置"""
    num_experts: int
    num_experts_per_tok: int
    expert_capacity_factor: float = 1.0
    num_shared_experts: int = 0
    expert_intermediate_size: int = 0
    first_k_dense_replace: int = 0  # 前K层使用Dense FFN（DeepSeek V3 = 3）
    # EP+TP组合配置
    moe_tp: int = 1  # MoE专家内的TP切分度
    ep_tp_strategy: str = "scatter_gather"  # 'scatter_gather' 或 'group_alltoall'


@dataclass
class MLAConfig:
    """MLA (Multi-head Latent Attention) 配置 - DeepSeek V3/R1 专用"""
    kv_lora_rank: int        # KV 压缩后的隐维度
    q_lora_rank: int         # Q 的 LoRA rank
    qk_nope_head_dim: int    # 非 RoPE 头维度
    qk_rope_head_dim: int    # RoPE 头维度
    v_head_dim: int          # V 的头维度
    variant: str = "mla"     # MLA 变体: mla | mla_v32 | mla_absorb | mla_absorb_v32
    mla_tp: int = 0          # MLA 张量并行度 (0 表示使用全局 tp)
    mla_dp: int = 0          # MLA 数据并行度 (0 表示使用全局 dp)


@dataclass
class LLMModelConfig:
    """LLM模型配置"""
    model_name: str
    model_type: str  # 'dense' | 'moe'
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    dtype: str  # 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4'
    max_seq_length: int
    moe_config: Optional[MoEConfig] = None
    mla_config: Optional[MLAConfig] = None  # DeepSeek V3/R1 MLA 配置
    attention_type: str = "gqa"  # 'mha' | 'gqa' | 'mqa' | 'mla'
    norm_type: str = "rmsnorm"  # 'layernorm' | 'rmsnorm'


# ============================================
# 运行时配置
# ============================================

@dataclass
class ProtocolConfig:
    """通信协议运行时配置

    这些参数与通信协议栈 (如 NCCL/RCCL) 的实现相关，
    不同的软件版本或配置可能有不同的开销。

    所有延迟单位：微秒 (us)
    """
    rtt_tp_us: float = 0.35
    """TP 通信 RTT (Round-Trip Time)

    张量并行通信的往返延迟，主要来自通信协议开销。
    典型值：0.3-0.5 us (高速互联如 NVLink)
    """

    rtt_ep_us: float = 0.85
    """EP 通信 RTT (Round-Trip Time)

    专家并行通信的往返延迟，通常比 TP 通信稍高。
    典型值：0.7-1.0 us
    """

    bandwidth_utilization: float = 0.95
    """带宽利用率 (0-1)

    实际带宽与峰值带宽的比值，取决于通信模式、消息大小等。
    典型值：0.90-0.98 (大消息), 0.5-0.8 (小消息)
    """

    sync_latency_us: float = 0.0
    """同步延迟

    集合通信后的同步开销。
    典型值：0-1 us
    """


@dataclass
class NetworkInfraConfig:
    """网络基础设施配置

    这些参数取决于数据中心的网络设备（交换机型号、线缆类型等）。

    所有延迟单位：微秒 (us)
    """
    switch_delay_us: float = 1.0
    """交换机转发延迟

    单个交换机的数据包处理和转发延迟。
    典型值：
    - 高端 InfiniBand 交换机: 0.1-0.3 us
    - 以太网交换机: 0.5-2.0 us
    """

    cable_delay_us: float = 0.025
    """线缆传输延迟

    信号在线缆中传输的延迟（约 5 ns/m）。
    典型值：
    - 机架内 (1-2m): 0.01-0.02 us
    - 机架间 (5-10m): 0.025-0.05 us
    """

    @property
    def link_delay_us(self) -> float:
        """端到端链路延迟 (包括两端交换机和线缆)"""
        return 2 * self.switch_delay_us + 2 * self.cable_delay_us


@dataclass
class SimulationConfig:
    """模拟运行时配置 (汇总所有可配置参数)"""
    protocol: ProtocolConfig = None  # type: ignore
    network: NetworkInfraConfig = None  # type: ignore

    def __post_init__(self):
        if self.protocol is None:
            self.protocol = ProtocolConfig()
        if self.network is None:
            self.network = NetworkInfraConfig()


# ============================================
# 推理配置
# ============================================

@dataclass
class InferenceConfig:
    """推理配置"""
    batch_size: int
    input_seq_length: int
    output_seq_length: int
    max_seq_length: int
    num_micro_batches: int = 1


@dataclass
class ParallelismStrategy:
    """并行策略"""
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    sp: int = 1
    moe_tp: int = 1  # MoE 专家内张量并行度


# ============================================
# 甘特图数据
# ============================================

@dataclass
class GanttTask:
    """甘特图任务（增强版，包含详细性能数据）"""
    id: str
    name: str
    resource: str
    start: float  # us (微秒)
    end: float  # us (微秒)
    type: GanttTaskType
    phase: InferencePhase
    chip_id: str
    pp_stage: int
    layer_index: Optional[int] = None
    token_index: Optional[int] = None
    color: Optional[str] = None

    # ========== 计算详细信息 ==========
    flops: Optional[float] = None                       # 浮点运算数
    params_bytes: Optional[float] = None                # 参数量（字节）
    dram_occupy_bytes: Optional[float] = None           # 显存占用（字节）
    dram_traffic_bytes: Optional[float] = None          # 显存流量（字节）

    # ========== 时间分解 ==========
    compute_time_us: Optional[float] = None             # 纯计算时间（微秒）
    memory_time_us: Optional[float] = None              # 内存搬运时间（微秒）
    comm_time_us: Optional[float] = None                # 通信时间（微秒）

    # ========== GEMM 优化结果 ==========
    best_tile: Optional[dict] = None                    # 最优 Tile 大小 {"M": 128, "N": 384, "K": 448, "order": "mnk"}
    best_partition: Optional[dict] = None               # 最优分块策略 {"dims": {...}, "procs": {...}}
    gemm_shape: Optional[dict] = None                   # GEMM 形状 {"G": 1, "M": 128, "K": 7168, "N": 1536}

    # ========== 利用率 ==========
    arch_utilization: Optional[float] = None            # 架构利用率（硬件峰值比）
    effective_utilization: Optional[float] = None       # 有效利用率（考虑内存墙）

    # ========== 通信详细信息 ==========
    comm_size_bytes: Optional[float] = None             # 通信数据量（字节）
    comm_algorithm: Optional[str] = None                # 通信算法 ("ring"/"tree"/"halving_doubling"/etc)
    comm_group_size: Optional[int] = None               # 通信组大小（TP/DP/EP 度）

    # ========== 并行配置（算子级别） ==========
    parallel_config: Optional[dict] = None              # 并行参数 {"tp": 8, "dp": 16, "ep": 1, "sp": 1}


@dataclass
class GanttResource:
    """甘特图资源行"""
    id: str
    name: str
    pp_stage: int
    type: str  # 'compute' | 'network' | 'pcie' | 'hbm'


@dataclass
class GanttChartData:
    """甘特图数据"""
    resources: list[GanttResource]
    tasks: list[GanttTask]
    time_range: tuple[float, float]
    phase_transition: Optional[float] = None


# ============================================
# 模拟统计
# ============================================

@dataclass
class PhaseTimeStats:
    """阶段时间统计"""
    compute_time: float = 0.0
    comm_time: float = 0.0
    bubble_time: float = 0.0
    overlap_time: float = 0.0
    total_time: float = 0.0
    compute_efficiency: float = 0.0


@dataclass
class SimulationStats:
    """模拟统计"""
    prefill: PhaseTimeStats = field(default_factory=PhaseTimeStats)
    decode: PhaseTimeStats = field(default_factory=PhaseTimeStats)
    total_run_time: float = 0.0
    simulated_tokens: int = 0
    ttft: float = 0.0
    avg_tpot: float = 0.0
    dynamic_mfu: float = 0.0
    dynamic_mbu: float = 0.0
    max_pp_bubble_ratio: float = 0.0
    total_events: int = 0
    prefill_flops: float = 0.0  # Prefill 阶段计算量 (FLOPs)


# ============================================
# 模拟结果
# ============================================

@dataclass
class SimulationResult:
    """模拟结果"""
    gantt_chart: GanttChartData
    stats: SimulationStats
    timestamp: float = 0.0


# ============================================
# 芯片互联图
# ============================================

@dataclass
class ChipNode:
    """芯片节点"""
    chip_id: str
    pod_id: str
    rack_id: str
    board_id: str
    position: tuple[int, int]  # 在board上的位置


@dataclass
class ChipLink:
    """芯片间链路"""
    source: str
    target: str
    bandwidth_gbps: float
    latency_us: float
    link_type: str  # 'nvlink' | 'pcie' | 'ib' | 'ethernet'


@dataclass
class InterconnectGraph:
    """芯片互联图"""
    nodes: list[ChipNode]
    links: list[ChipLink]

    def get_link_params(self, src: str, dst: str) -> tuple[float, float]:
        """获取两个芯片间的链路参数 (带宽, 延迟)"""
        for link in self.links:
            if (link.source == src and link.target == dst) or \
               (link.source == dst and link.target == src):
                return link.bandwidth_gbps, link.latency_us
        raise ValueError(f"链路不存在: {src} -> {dst}")


# ============================================
# 并行组分配
# ============================================

@dataclass
class ChipAssignment:
    """芯片并行组分配"""
    chip_id: str
    global_rank: int
    dp_rank: int
    tp_rank: int
    pp_rank: int
    ep_rank: int
    sp_rank: int


@dataclass
class ParallelGroupAssignment:
    """并行组分配结果"""
    assignments: list[ChipAssignment]
    tp_groups: list[list[str]]  # 每个TP组的芯片ID列表
    pp_groups: list[list[str]]  # 每个PP组的芯片ID列表
    dp_groups: list[list[str]]  # 每个DP组的芯片ID列表
    ep_groups: list[list[str]]  # 每个EP组的芯片ID列表


# ============================================
# 辅助函数
# ============================================

BYTES_PER_DTYPE = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'int8': 1,
    'int4': 0.5,
}


def get_bytes_per_element(dtype: str) -> float:
    """获取数据类型的字节数"""
    return BYTES_PER_DTYPE.get(dtype, 2)


# ============================================
# 配置验证函数
# ============================================

def validate_mla_config(mla_dict: dict) -> MLAConfig:
    """
    验证并解析 MLA 配置

    Args:
        mla_dict: MLA 配置字典

    Returns:
        MLAConfig 对象

    Raises:
        ValueError: 配置无效时抛出
    """
    required_fields = ["kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"]

    missing = [f for f in required_fields if f not in mla_dict]
    if missing:
        raise ValueError(f"MLA 配置缺少必填字段: {missing}")

    for field in required_fields:
        value = mla_dict[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"MLA 配置 {field} 必须为正整数，当前值: {value}")

    return MLAConfig(
        kv_lora_rank=mla_dict["kv_lora_rank"],
        q_lora_rank=mla_dict["q_lora_rank"],
        qk_nope_head_dim=mla_dict["qk_nope_head_dim"],
        qk_rope_head_dim=mla_dict["qk_rope_head_dim"],
        v_head_dim=mla_dict["v_head_dim"],
    )


def validate_moe_config(moe_dict: dict) -> MoEConfig:
    """
    验证并解析 MoE 配置

    Args:
        moe_dict: MoE 配置字典

    Returns:
        MoEConfig 对象

    Raises:
        ValueError: 配置无效时抛出
    """
    required_fields = ["num_experts", "num_experts_per_tok"]

    missing = [f for f in required_fields if f not in moe_dict]
    if missing:
        raise ValueError(f"MoE 配置缺少必填字段: {missing}")

    num_experts = moe_dict["num_experts"]
    num_experts_per_tok = moe_dict["num_experts_per_tok"]

    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError(f"MoE num_experts 必须为正整数，当前值: {num_experts}")
    if not isinstance(num_experts_per_tok, int) or num_experts_per_tok <= 0:
        raise ValueError(f"MoE num_experts_per_tok 必须为正整数，当前值: {num_experts_per_tok}")
    if num_experts_per_tok > num_experts:
        raise ValueError(f"MoE num_experts_per_tok ({num_experts_per_tok}) 不能大于 num_experts ({num_experts})")

    expert_intermediate_size = moe_dict.get("expert_intermediate_size", 0)
    if expert_intermediate_size <= 0:
        raise ValueError(f"MoE 配置必须指定 expert_intermediate_size (> 0)，当前值: {expert_intermediate_size}")

    return MoEConfig(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        expert_capacity_factor=moe_dict.get("expert_capacity_factor", 1.0),
        num_shared_experts=moe_dict.get("num_shared_experts", 0),
        expert_intermediate_size=expert_intermediate_size,
        first_k_dense_replace=moe_dict.get("first_k_dense_replace", 0),
        moe_tp=moe_dict.get("moe_tp", 1),
        ep_tp_strategy=moe_dict.get("ep_tp_strategy", "scatter_gather"),
    )


def validate_model_config(model_dict: dict) -> None:
    """
    验证模型配置的有效性

    Args:
        model_dict: 模型配置字典

    Raises:
        ValueError: 配置无效时抛出
    """
    required_fields = ["hidden_size", "num_layers", "num_attention_heads", "intermediate_size"]

    missing = [f for f in required_fields if f not in model_dict]
    if missing:
        raise ValueError(f"模型配置缺少必填字段: {missing}")

    hidden_size = model_dict["hidden_size"]
    num_heads = model_dict["num_attention_heads"]

    if hidden_size <= 0:
        raise ValueError(f"hidden_size 必须为正数，当前值: {hidden_size}")
    if num_heads <= 0:
        raise ValueError(f"num_attention_heads 必须为正数，当前值: {num_heads}")
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_attention_heads ({num_heads}) 整除")


def validate_hardware_config(hardware_dict: dict) -> None:
    """
    验证硬件配置的有效性（仅支持新格式 v2.1.0+）

    Args:
        hardware_dict: 硬件配置字典

    Raises:
        ValueError: 配置无效或缺少必需字段时抛出
    """
    logger.warning(f"[DEBUG] validate_hardware_config: hardware_dict keys = {hardware_dict.keys()}")
    logger.warning(f"[DEBUG] validate_hardware_config: hardware_dict content = {hardware_dict}")

    # 检查新格式: hardware_params.chips
    if "hardware_params" not in hardware_dict:
        logger.error("[DEBUG] hardware_dict缺少hardware_params字段!")
        raise ValueError("硬件配置缺少 'hardware_params' 字段")

    hardware_params = hardware_dict["hardware_params"]
    logger.warning(f"[DEBUG] validate_hardware_config: hardware_params keys = {hardware_params.keys()}")

    if "chips" not in hardware_params:
        logger.error("[DEBUG] hardware_params缺少chips字段!")
        raise ValueError("硬件配置中 'hardware_params' 缺少 'chips' 字段")

    chips_dict = hardware_params["chips"]
    logger.warning(f"[DEBUG] validate_hardware_config: chips_dict = {chips_dict}")

    # 验证至少有一个芯片配置
    if not chips_dict:
        logger.error("[DEBUG] chips_dict为空!")
        raise ValueError("硬件配置中 'chips' 字典为空，至少需要一个芯片配置")

    # 验证每个芯片的必需字段
    for chip_name, chip_config in chips_dict.items():
        logger.warning(f"[DEBUG] validate_hardware_config: validating chip '{chip_name}', config keys = {chip_config.keys() if isinstance(chip_config, dict) else type(chip_config)}")
        _validate_chip_fields(chip_config, chip_name)


def _validate_chip_fields(chip_config: dict, chip_name: str) -> None:
    """
    验证单个芯片配置的必需字段

    Args:
        chip_config: 芯片配置字典
        chip_name: 芯片名称（用于错误消息）

    Raises:
        ValueError: 缺少必需字段或字段值无效时抛出
    """
    required_fields = {
        "name": "芯片型号",
        "num_cores": "核心数",
        "compute_tflops_bf16": "BF16 算力",
        "memory_capacity_gb": "显存容量",
        "memory_bandwidth_gbps": "显存带宽"
    }

    missing_fields = []
    for field, description in required_fields.items():
        if field not in chip_config or chip_config[field] is None:
            missing_fields.append(f"{description} ({field})")

    if missing_fields:
        raise ValueError(
            f"芯片配置 '{chip_name}' 缺少必需字段: {', '.join(missing_fields)}"
        )

    # 验证数值有效性
    compute_tflops = chip_config["compute_tflops_bf16"]
    memory_gb = chip_config["memory_capacity_gb"]
    memory_bw = chip_config["memory_bandwidth_gbps"]

    if compute_tflops <= 0:
        raise ValueError(f"芯片 '{chip_name}' 的 compute_tflops_bf16 必须为正数，当前值: {compute_tflops}")
    if memory_gb <= 0:
        raise ValueError(f"芯片 '{chip_name}' 的 memory_capacity_gb 必须为正数，当前值: {memory_gb}")
    if memory_bw <= 0:
        raise ValueError(f"芯片 '{chip_name}' 的 memory_bandwidth_gbps 必须为正数，当前值: {memory_bw}")


def validate_parallelism_config(parallelism_dict: dict, model_dict: Optional[dict] = None) -> None:
    """
    验证并行策略配置的有效性

    Args:
        parallelism_dict: 并行策略配置字典
        model_dict: 模型配置字典（可选，用于交叉验证）

    Raises:
        ValueError: 配置无效时抛出
    """
    for key in ["dp", "tp", "pp", "ep"]:
        value = parallelism_dict.get(key, 1)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"并行度 {key} 必须为正整数，当前值: {value}")

    if model_dict:
        pp = parallelism_dict.get("pp", 1)
        num_layers = model_dict.get("num_layers", 1)
        if pp > num_layers:
            raise ValueError(f"PP ({pp}) 不能大于模型层数 ({num_layers})")

        tp = parallelism_dict.get("tp", 1)
        num_heads = model_dict.get("num_attention_heads", 1)
        if tp > num_heads:
            raise ValueError(f"TP ({tp}) 不能大于注意力头数 ({num_heads})")

        moe_config = model_dict.get("moe_config")
        if moe_config:
            dp = parallelism_dict.get("dp", 1)
            ep = parallelism_dict.get("ep", 1)
            moe_tp = parallelism_dict.get("moe_tp", 1)

            attention_chips = dp * tp
            moe_chips = moe_tp * ep

            if attention_chips != moe_chips:
                raise ValueError(f"MoE 并行约束不满足: dp*tp ({dp}*{tp}={attention_chips}) 必须等于 moe_tp*ep ({moe_tp}*{ep}={moe_chips})")


# ============================================
# Switch 网络设备配置 (Phase 3)
# ============================================

class SwitchType(str, Enum):
    """Switch 类型"""
    LEAF = "leaf"        # 叶子交换机 (ToR Switch)
    SPINE = "spine"      # 脊交换机
    CORE = "core"        # 核心交换机


class SwitchLayer(str, Enum):
    """Switch 所属层级（对应互联层级）"""
    INTER_CHIP = "inter_chip"    # 芯片间（同板）
    INTER_BOARD = "inter_board"  # 板间（同 Rack）
    INTER_RACK = "inter_rack"    # Rack 间（同 Pod）
    INTER_POD = "inter_pod"      # Pod 间


@dataclass
class SwitchHardwareConfig:
    """Switch 硬件配置

    定义交换机的硬件特性，用于精确计算转发延迟。
    """
    # 基本信息
    name: str                           # Switch 型号名称
    switch_type: SwitchType             # 类型（leaf/spine/core）
    port_count: int                     # 端口数量（如 72, 128, 512）

    # 性能参数
    port_bandwidth_gbps: float          # 单端口带宽 (GB/s，如 100, 400)
    processing_delay_us: float = 0.5    # 固定处理延迟 (us，Store-and-Forward)
    cut_through_delay_us: float = 0.1   # Cut-Through 模式延迟 (us)

    # 缓冲区配置（用于高级排队模型，Phase 3 预留）
    buffer_size_mb: float = 32.0        # 总缓冲区大小 (MB)
    buffer_per_port_kb: float = 512.0   # 每端口缓冲区 (KB)

    # 转发模式
    forwarding_mode: str = "store_and_forward"  # 'store_and_forward' | 'cut_through'


@dataclass
class SwitchInstanceConfig:
    """Switch 实例配置

    表示拓扑中的一个具体 Switch 实例。
    """
    # 唯一标识
    switch_id: str                      # 全局唯一 ID（如 "leaf_0_rack_0"）

    # 硬件类型
    hardware_type: str                  # 引用 SwitchHardwareConfig.name

    # 拓扑位置
    layer: SwitchLayer                  # 所属互联层级
    pod_id: Optional[str] = None        # 所属 Pod ID
    rack_id: Optional[str] = None       # 所属 Rack ID

    # 连接信息（端口 → 设备映射）
    # 由 SwitchManager 动态填充
    port_to_device: dict = field(default_factory=dict)
    device_to_port: dict = field(default_factory=dict)


@dataclass
class SwitchPortState:
    """Switch 端口状态

    追踪单个端口的资源状态和统计信息。
    用于端口级别的排队建模。
    """
    # 标识
    port_id: str                        # "switch_id:port_num"
    switch_id: str
    port_number: int

    # 连接信息
    connected_device: str = ""          # 连接的设备 ID（Chip 或 Switch）
    link_bandwidth_gbps: float = 0.0    # 实际链路带宽

    # 统计信息（由仿真过程中更新）
    total_bytes_transferred: float = 0.0
    total_packets_forwarded: int = 0
    total_busy_time_us: float = 0.0
    total_idle_time_us: float = 0.0

    def get_utilization(self, total_time_us: float) -> float:
        """计算端口利用率"""
        if total_time_us <= 0:
            return 0.0
        return self.total_busy_time_us / total_time_us


@dataclass
class Packet:
    """数据包

    表示网络中传输的数据包（NCCL Chunk 级别）。
    用于包级仿真。
    """
    # 唯一标识
    packet_id: str                      # 全局唯一 ID
    flow_id: str                        # 所属通信流 ID

    # 数据信息
    size_bytes: float                   # 包大小（默认 512 KB）
    payload_type: str = "data"          # 'data' | 'ack' | 'control'

    # 路由信息
    src_chip: str = ""                  # 源芯片 ID
    dst_chip: str = ""                  # 目标芯片 ID
    current_location: str = ""          # 当前位置（chip_id 或 switch_id）

    # 路径（经过的 Switch 列表）
    route: list[str] = field(default_factory=list)  # ["leaf_0", "spine_0", "leaf_1"]
    hop_index: int = 0                  # 当前跳索引 (0 = 在源芯片, 1 = 第一个 Switch)

    # 时间戳
    created_at: float = 0.0             # 创建时间 (us)
    last_hop_time: float = 0.0          # 上一跳完成时间 (us)

    # 元数据（用于追踪通信上下文）
    layer_index: int = -1
    micro_batch: int = 0
    comm_type: str = ""                 # allreduce, p2p, etc.

    def get_next_hop(self) -> Optional[str]:
        """获取下一跳设备 ID"""
        if self.hop_index < len(self.route):
            return self.route[self.hop_index]
        return None

    def advance_hop(self) -> None:
        """前进到下一跳"""
        self.hop_index += 1

    def is_at_destination(self) -> bool:
        """检查是否已到达目标"""
        return self.hop_index >= len(self.route)


@dataclass
class SwitchGraph:
    """Switch 互联图

    描述 Switch 之间以及 Switch 与 Chip 之间的连接关系。
    """
    # Switch 节点列表
    switches: list[SwitchInstanceConfig] = field(default_factory=list)

    # Switch 硬件配置（按名称索引）
    hardware_configs: dict[str, SwitchHardwareConfig] = field(default_factory=dict)

    # 连接关系
    # key: (src_id, dst_id), value: (bandwidth_gbps, latency_us)
    connections: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)

    def get_switch(self, switch_id: str) -> Optional[SwitchInstanceConfig]:
        """获取 Switch 实例"""
        for switch in self.switches:
            if switch.switch_id == switch_id:
                return switch
        return None

    def get_hardware_config(self, hardware_type: str) -> Optional[SwitchHardwareConfig]:
        """获取 Switch 硬件配置"""
        return self.hardware_configs.get(hardware_type)

    def get_connection_params(
        self, src_id: str, dst_id: str
    ) -> tuple[float, float]:
        """获取两个设备之间的连接参数

        Returns:
            (bandwidth_gbps, latency_us)
        """
        # 尝试正向和反向查找
        if (src_id, dst_id) in self.connections:
            return self.connections[(src_id, dst_id)]
        if (dst_id, src_id) in self.connections:
            return self.connections[(dst_id, src_id)]
        return (0.0, 0.0)
