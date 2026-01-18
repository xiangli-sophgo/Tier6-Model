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
# 拓扑配置
# ============================================

@dataclass
class ChipConfig:
    """芯片配置"""
    id: str
    type: str = "chip"
    position: tuple[int, int] = (0, 0)
    label: str = ""


@dataclass
class BoardConfig:
    """板卡配置"""
    id: str
    u_position: int
    u_height: int
    label: str
    chips: list[ChipConfig] = field(default_factory=list)


@dataclass
class RackConfig:
    """机柜配置"""
    id: str
    position: tuple[int, int]
    label: str
    total_u: int = 42
    boards: list[BoardConfig] = field(default_factory=list)


@dataclass
class PodConfig:
    """Pod配置"""
    id: str
    label: str
    grid_size: tuple[int, int]
    racks: list[RackConfig] = field(default_factory=list)


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


# ============================================
# 硬件配置
# ============================================

@dataclass
class ChipHardwareConfig:
    """芯片硬件配置"""
    chip_type: str
    compute_tflops_fp16: float
    memory_gb: float
    memory_bandwidth_gbps: float
    compute_tops_int8: float = 0.0
    cost_per_hour: float = 0.0
    # 新增参数
    num_cores: int = 8  # 计算核心数
    memory_bandwidth_utilization: float = 0.9  # 显存带宽利用率
    l2_cache_mb: float = 16.0  # L2 缓存容量 (MB)
    l2_bandwidth_gbps: float = 512.0  # L2 缓存带宽 (GB/s)
    # 扩展配置
    pcie_bandwidth_gbps: float = 64.0  # PCIe Gen5 x16 默认
    pcie_latency_us: float = 1.0
    hbm_random_access_latency_ns: float = 100.0


@dataclass
class NodeConfig:
    """节点配置"""
    chips_per_node: int
    intra_node_bandwidth_gbps: float
    intra_node_latency_us: float
    # 新增参数
    bandwidth_utilization: float = 0.9  # 带宽利用率
    startup_latency_us: float = 1.0  # 通信启动延迟
    sync_latency_us: float = 1.0  # 同步延迟


@dataclass
class ClusterConfig:
    """集群配置"""
    num_nodes: int
    inter_node_bandwidth_gbps: float
    inter_node_latency_us: float


@dataclass
class HardwareConfig:
    """完整硬件配置"""
    chip: ChipHardwareConfig
    node: NodeConfig
    cluster: ClusterConfig


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


# ============================================
# 甘特图数据
# ============================================

@dataclass
class GanttTask:
    """甘特图任务"""
    id: str
    name: str
    resource: str
    start: float  # ms
    end: float  # ms
    type: GanttTaskType
    phase: InferencePhase
    chip_id: str
    pp_stage: int
    layer_index: Optional[int] = None
    token_index: Optional[int] = None
    color: Optional[str] = None


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
