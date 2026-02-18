/**
 * LLM 部署分析系统 - 类型定义
 *
 * 定义模型参数、硬件配置、部署方案、分析结果等核心类型
 */

import { ChipPreset } from '../../types/math_model';

// ============================================
// 数据类型
// ============================================

/** 数据精度类型 */
export type DataType = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4';

/** 模型类型 */
export type ModelType = 'dense' | 'moe';

/** Attention 类型 */
export type AttentionType = 'mha' | 'gqa' | 'mqa' | 'mla';

/** Norm 类型 */
export type NormType = 'layernorm' | 'rmsnorm';

/** 推理阶段 */
export type InferencePhase = 'prefill' | 'decode';

/** 瓶颈类型 */
export type BottleneckType = 'compute' | 'memory' | 'communication' | 'pipeline_bubble' | 'balanced';

/** AllReduce算法类型 */
export type AllReduceAlgorithm = 'ring' | 'double_binary_tree' | 'halving_doubling' | 'reduce_broadcast';

/** All-to-All算法类型 */
export type AllToAllAlgorithm = 'pairwise' | 'ring' | 'bruck';

/** 单阶段瓶颈分析 */
export interface PhaseBottleneckAnalysis {
  /** 阶段名称 */
  phase: 'prefill' | 'decode';
  /** 算术强度 (ops/byte) */
  arithmetic_intensity: number;
  /** 硬件临界点 (ops/byte) */
  hardware_ridge_point: number;
  /** 瓶颈类型 */
  bound_type: 'compute' | 'memory' | 'balanced';
  /** 计算延迟占比 (0-1) */
  compute_ratio: number;
  /** 访存延迟占比 (0-1) */
  memory_ratio: number;
  /** 通信延迟占比 (0-1) */
  comm_ratio: number;
  /** 利用率 (MFU for prefill, MBU for decode) */
  utilization: number;
  /** 理论最优延迟 (ms) */
  theoretical_latency_ms: number;
  /** 实际延迟 (ms) */
  actual_latency_ms: number;
  /** 效率损失原因 */
  efficiency_loss: string[];
}

/** 完整瓶颈分析结果 */
export interface BottleneckAnalysis {
  /** Prefill 阶段瓶颈分析 */
  prefill: PhaseBottleneckAnalysis;
  /** Decode 阶段瓶颈分析 */
  decode: PhaseBottleneckAnalysis;
  /** 主导阶段 */
  dominant_phase: 'prefill' | 'decode';
  /** 综合瓶颈类型 */
  overall_bottleneck: BottleneckType;
  /** 瓶颈严重程度 (0-1, 越高越严重) */
  severity: number;
  /** 优化潜力分析 */
  optimization_potential: {
    /** 增大batch的潜在提升 */
    batch_scaling: { current_ai: number; target_ai: number; potential_speedup: number };
    /** 量化的潜在提升 */
    quantization: { current_bytes: number; target_bytes: number; potential_speedup: number };
    /** 减少TP的潜在提升 (减少通信) */
    reduce_tp: { current_comm_ratio: number; potential_speedup: number };
  };
  /** 瓶颈摘要 */
  summary: string;
}

/** 优化目标 */
export type OptimizationTarget = 'latency' | 'throughput' | 'efficiency' | 'balanced';

// ============================================
// 模型配置
// ============================================

/** MoE 专用配置 */
export interface MoEConfig {
  /** 专家总数 */
  num_experts: number;
  /** 每token激活的专家数 */
  num_experts_per_tok: number;
  /** 共享专家数量 (DeepSeek-V2风格) */
  num_shared_experts?: number;
  /** 每个专家的FFN中间维度 (可选，不设置则使用 intermediate_size) */
  expert_intermediate_size?: number;
  /** 前K层使用Dense FFN（DeepSeek V3 = 3，即layer 0-2是Dense，layer 3+是MoE） */
  first_k_dense_replace?: number;
  /** MoE层出现频率 (1=每层都是MoE, 2=隔层MoE)，默认为1 */
  moe_layer_freq?: number;
  /** MoE专家内的TP切分度 */
  moe_tp?: number;
  /** EP+TP策略: 'scatter_gather' 或 'group_alltoall' */
  ep_tp_strategy?: 'scatter_gather' | 'group_alltoall';
}

/** MLA 变体类型 */
export type MLAVariant = 'mla' | 'mla_v32' | 'mla_absorb' | 'mla_absorb_v32';

/** MLA (Multi-head Latent Attention) 配置 - DeepSeek V3/R1 专用 */
export interface MLAConfig {
  /** KV 压缩后的隐维度 (kv_lora_rank) */
  kv_lora_rank: number;
  /** Q 的 LoRA rank */
  q_lora_rank: number;
  /** 非 RoPE 头维度 (qk_nope_head_dim) */
  qk_nope_head_dim: number;
  /** RoPE 头维度 (qk_rope_head_dim) */
  qk_rope_head_dim: number;
  /** V 的头维度 */
  v_head_dim: number;
  /** MLA 变体: mla | mla_v32 | mla_absorb | mla_absorb_v32 */
  variant?: MLAVariant;
  /** MLA 张量并行度 (可选，默认使用全局 tp) */
  mla_tp?: number;
  /** MLA 数据并行度 (可选，默认使用全局 dp) */
  mla_dp?: number;
}

/** LLM 模型配置 */
export interface LLMModelConfig {
  /** 模型名称 */
  model_name: string;
  /** 模型类型 */
  model_type: ModelType;
  /** 隐藏层维度 */
  hidden_size: number;
  /** Transformer 层数 */
  num_layers: number;
  /** 注意力头数 */
  num_attention_heads: number;
  /** KV 头数 (GQA/MQA) */
  num_kv_heads: number;
  /** FFN 中间层维度 */
  intermediate_size: number;
  /** 词表大小 */
  vocab_size: number;
  /** 最大序列长度 */
  max_seq_length: number;
  /** MoE 配置 (可选) */
  moe_config?: MoEConfig;
  /** MLA 配置 (可选，DeepSeek V3/R1 专用) */
  mla_config?: MLAConfig;
  /** Attention 类型 (可选，默认根据 num_kv_heads 推断) */
  attention_type?: AttentionType;
  /** Norm 类型 (可选，默认 rmsnorm) */
  norm_type?: NormType;
  /** 是否共享 embedding 和 LM Head 权重 (可选，默认 true) */
  tie_word_embeddings?: boolean;
}

// ============================================
// 推理配置
// ============================================

/** 推理配置 */
export interface InferenceConfig {
  /** 批次大小 */
  batch_size: number;
  /** 输入序列长度 */
  input_seq_length: number;
  /** 输出序列长度 */
  output_seq_length: number;
  /** 最大序列长度 (KV Cache预分配) */
  max_seq_length: number;
  /** micro batch 数量 (PP流水线) */
  num_micro_batches?: number;
  /** 权重精度 */
  weight_dtype: DataType;
  /** 激活精度 */
  activation_dtype: DataType;
}

// ============================================
// 硬件配置
// ============================================

/** 算力精度类型 */
export type FlopsDtype = 'BF16' | 'FP16' | 'FP8' | 'INT8';

/**
 * 芯片硬件配置 - 使用 ChipPreset 格式
 *
 * 参见 types/math_model.ts 中的 ChipPreset 接口定义
 */
export type ChipHardwareConfig = ChipPreset;

/** 互联参数 */
export interface InterconnectParams {
  /** 带宽 (GB/s) */
  bandwidth_gbps: number;
  /** 延迟 (us) */
  latency_us: number;
}

/** 互联配置（新格式 v2.2+） */
export interface InterconnectConfig {
  /** 层级互联链路 */
  links?: {
    /** Chip-to-Chip (Die间，板内) */
    c2c?: InterconnectParams;
    /** Board-to-Board (板间，机架内) */
    b2b?: InterconnectParams;
    /** Rack-to-Rack (机架间，Pod内) */
    r2r?: InterconnectParams;
    /** Pod-to-Pod (Pod间，跨Pod) */
    p2p?: InterconnectParams;
  };
  /** 通信参数 */
  comm_params?: {
    /** AllReduce算法 */
    allreduce_algorithm?: AllReduceAlgorithm;
    /** AllToAll算法 */
    alltoall_algorithm?: AllToAllAlgorithm;
    /** 计算通信重叠 */
    enable_compute_comm_overlap?: boolean;
    /** 网络效率 (0-1) */
    network_efficiency?: number;
  };
}

/** 硬件参数配置（新格式 v2.1.0+，v2.2+ 移除 hardware_params 包装） */
export interface HardwareParams {
  /** 芯片配置字典，key为芯片名称 */
  chips: Record<string, ChipHardwareConfig>;
  /** 互联配置 */
  interconnect?: {
    /** Chip-to-Chip (Die间，板内) */
    c2c?: InterconnectParams;
    /** Board-to-Board (板间，机架内) */
    b2b?: InterconnectParams;
    /** Rack-to-Rack (机架间，Pod内) */
    r2r?: InterconnectParams;
    /** Pod-to-Pod (Pod间，跨Pod) */
    p2p?: InterconnectParams;
  };
  /** 通信配置 */
  comm_latency_config?: {
    /** AllReduce算法 */
    allreduce_algorithm?: AllReduceAlgorithm;
    /** AllToAll算法 */
    alltoall_algorithm?: AllToAllAlgorithm;
    /** 计算通信重叠 */
    enable_compute_comm_overlap?: boolean;
    /** 网络效率 (0-1) */
    network_efficiency?: number;
  };
}

// ============================================
// 旧格式硬件配置接口（向后兼容，已废弃）
// ============================================

/** Board 层级配置（已废弃，仅用于向后兼容） */
export interface BoardConfig {
  /** 每个 Board 的芯片数量 */
  chips_per_board: number;
  /** Board 间互联带宽 (GB/s) */
  b2b_bandwidth_gbps: number;
  /** Board 间延迟 (us) */
  b2b_latency_us: number;
}

/** Rack 层级配置（已废弃，仅用于向后兼容） */
export interface RackConfig {
  /** 每个 Rack 的 Board 数量 */
  boards_per_rack: number;
  /** Rack 间互联带宽 (GB/s) */
  r2r_bandwidth_gbps: number;
  /** Rack 间延迟 (us) */
  r2r_latency_us: number;
}

/** Pod 层级配置（已废弃，仅用于向后兼容） */
export interface PodConfig {
  /** 每个 Pod 的 Rack 数量 */
  racks_per_pod: number;
  /** Pod 间互联带宽 (GB/s) */
  p2p_bandwidth_gbps: number;
  /** Pod 间延迟 (us) */
  p2p_latency_us: number;
}

// ============================================
// 硬件配置（v2.1.0+ 格式）
// ============================================

/** 完整硬件配置（v2.2+ 格式：chips 和 interconnect 直接在顶层） */
export interface HardwareConfig {
  /** 芯片配置字典，key为芯片名称 */
  chips: Record<string, ChipHardwareConfig>;
  /** 互联配置 */
  interconnect: InterconnectConfig;

  // 以下字段仅用于向后兼容旧代码（已废弃）
  /** @deprecated 使用 chips + interconnect 代替 */
  hardware_params?: HardwareParams;
  chip?: ChipHardwareConfig;
  board?: BoardConfig;
  rack?: RackConfig;
  pod?: PodConfig;
}

// ============================================
// 部署方案
// ============================================

/** 并行策略配置 */
export interface ParallelismStrategy {
  /** 数据并行度 */
  dp: number;
  /** 张量并行度 (Attention部分) */
  tp: number;
  /** 流水线并行度 */
  pp: number;
  /** 专家并行度 (MoE) */
  ep: number;
  /** 序列并行度 (与后端 ParallelismConfig.sp 对应) */
  sp: number;
  /** 启用 TP Sequence Parallelism (SP 跟随 TP，序列按 TP 度切分) */
  enable_tp_sp: boolean;
  /** 启用 Ring Attention overlap (Layer 级，计算与通信完全重叠) */
  enable_ring_attention?: boolean;
  /** 启用 TBO (Transport/Bandwidth Overlap，MoE dispatch/combine 与计算重叠) */
  enable_tbo?: boolean;
  /** MoE专家内张量并行度 (可选，默认=1，仅MoE模型使用) */
  moe_tp?: number;
  /** 是否为 Prefill 阶段 (false=Decode) */
  is_prefill?: boolean;
  /** 启用 Zigzag Reorder 优化 (仅 Prefill + Causal Attention 生效，减少约40% GEMM) */
  enable_zigzag?: boolean;
}

/** 部署方案 */
export interface DeploymentPlan {
  /** 方案ID */
  plan_id: string;
  /** 并行策略 */
  parallelism: ParallelismStrategy;
  /** 总芯片数 */
  total_chips: number;
}

// ============================================
// 分析结果
// ============================================

/** 显存分析结果 */
export interface MemoryAnalysis {
  /** 模型参数显存 (GB) */
  model_memory_gb: number;
  /** KV Cache 显存 (GB) */
  kv_cache_memory_gb: number;
  /** 激活值显存 (GB) */
  activation_memory_gb: number;
  /** 其他开销 (GB) */
  overhead_gb: number;
  /** 每芯片总显存需求 (GB) */
  total_per_chip_gb: number;
  /** 是否超出显存限制 */
  is_memory_sufficient: boolean;
  /** 显存利用率 (0-1) */
  memory_utilization: number;
}

/**
 * 判断内存是否充足
 * @param totalMemoryGB 总内存使用量（GB）
 * @param chipCapacityGB 芯片容量（GB，从配置中获取）
 * @returns 是否充足（利用率 <= 100%）
 */
export function isMemorySufficient(totalMemoryGB: number, chipCapacityGB: number): boolean {
  return totalMemoryGB <= chipCapacityGB;
}

/** 通信分析结果 */
export interface CommunicationAnalysis {
  /** TP 通信量 (GB) */
  tp_comm_volume_gb: number;
  /** PP 通信量 (GB) */
  pp_comm_volume_gb: number;
  /** EP 通信量 (GB) */
  ep_comm_volume_gb: number;
  /** SP 通信量 (GB) */
  sp_comm_volume_gb: number;
  /** 总通信量 (GB) */
  total_comm_volume_gb: number;
  /** TP 通信延迟 (ms) */
  tp_comm_latency_ms: number;
  /** PP 通信延迟 (ms) */
  pp_comm_latency_ms: number;
  /** 通信瓶颈描述 */
  bottleneck_description?: string;
}

/** 延迟分位数统计 */
export interface LatencyPercentiles {
  /** P50 中位数 (ms) */
  p50: number;
  /** P90 (ms) */
  p90: number;
  /** P99 (ms) */
  p99: number;
}

/** 延迟分析结果 */
export interface LatencyAnalysis {
  /** Prefill 计算延迟 (ms) */
  prefill_compute_latency_ms: number;
  /** Prefill 通信延迟 (ms) */
  prefill_comm_latency_ms: number;
  /** Prefill 总延迟 (ms) = TTFT */
  prefill_total_latency_ms: number;
  /** Prefill 计算量 (FLOPs) */
  prefill_flops?: number;
  /** Decode 单 token 计算延迟 (ms) */
  decode_compute_latency_ms: number;
  /** Decode 单 token 访存延迟 (ms) */
  decode_memory_latency_ms: number;
  /** Decode 单 token 通信延迟 (ms) */
  decode_comm_latency_ms: number;
  /** Decode 单 token 总延迟 (ms) = TPOT */
  decode_per_token_latency_ms: number;
  /** 端到端延迟 (ms) */
  end_to_end_latency_ms: number;
  /** 流水线气泡比 (0-1) */
  pipeline_bubble_ratio: number;
  /** 主要瓶颈类型 */
  bottleneck_type: BottleneckType;
  /** 瓶颈详情 */
  bottleneck_details: string;
  /** 详细瓶颈分析 (Roofline模型) */
  bottleneck_analysis?: BottleneckAnalysis;
  /** TTFT 分位数分布 */
  ttft_percentiles?: LatencyPercentiles;
  /** TPOT 分位数分布 */
  tpot_percentiles?: LatencyPercentiles;
}

/** 吞吐量分析结果 */
export interface ThroughputAnalysis {
  /** Token 吞吐量 (tokens/s) - 集群总吞吐 = TPS_chip × NumChips */
  tokens_per_second: number;
  /** TPS per Batch (tokens/s) = 1000 / T_decode(ms) - 用户体验指标，SLO约束 ≥10 */
  tps_per_batch: number;
  /** TPS per Chip (tokens/s) = B × 1000 / T_decode(ms) - 成本效益，优化目标 */
  tps_per_chip: number;
  /** 请求吞吐量 (requests/s) */
  requests_per_second: number;
  /** 有效算力利用率 MFU (0-1) - Prefill 阶段关键指标 */
  model_flops_utilization: number;
  /** 显存带宽利用率 MBU (0-1) - Decode 阶段关键指标 */
  memory_bandwidth_utilization: number;
  /** 理论峰值吞吐量 (tokens/s) */
  theoretical_max_throughput: number;
}

/** 成本分析结果 */
export interface CostAnalysis {
  /** 服务器总成本 ($) */
  server_cost: number;
  /** 互联总成本 ($) */
  interconnect_cost: number;
  /** 总成本 ($) */
  total_cost: number;
  /** 互联带宽需求 (Gbps) */
  bandwidth_gbps: number;
  /** 所需 lane 数量 */
  lanes: number;
  /** 单 lane 成本 ($/lane) */
  lane_cost: number;
  /** 单芯片摊派成本 ($) */
  cost_per_chip: number;
  /** DFOP: 每 TPS 成本 ($/TPS) */
  dfop: number;
  /** 模型大小 (GB) */
  model_size_gb: number;
}

/** 资源利用率分析 */
export interface UtilizationAnalysis {
  /** 计算利用率 (0-1) */
  compute_utilization: number;
  /** 显存利用率 (0-1) */
  memory_utilization: number;
  /** 网络利用率 (0-1) */
  network_utilization: number;
  /** 负载均衡得分 (0-1) */
  load_balance_score: number;
}

/** 综合评分 */
export interface OverallScore {
  /** 延迟评分 (0-100) */
  latency_score: number;
  /** 吞吐量评分 (0-100) */
  throughput_score: number;
  /** 效率评分 (0-100) */
  efficiency_score: number;
  /** 均衡评分 (0-100) */
  balance_score: number;
  /** 显存评分 (0-100) */
  memory_score: number;
  /** 通信评分 (0-100) */
  communication_score: number;
  /** 综合评分 (0-100) */
  overall_score: number;
}

/** 优化建议 */
export interface OptimizationSuggestion {
  /** 建议类型 */
  type: 'reduce_tp' | 'increase_pp' | 'increase_batch' | 'reduce_seq' | 'other';
  /** 建议内容 */
  description: string;
  /** 预期收益 */
  expected_improvement: string;
  /** 优先级 (1-5) */
  priority: number;
}

/** 单方案完整分析结果 */
export interface PlanAnalysisResult {
  /** 方案信息 */
  plan: DeploymentPlan;
  /** 显存分析 */
  memory: MemoryAnalysis;
  /** 通信分析 */
  communication: CommunicationAnalysis;
  /** 延迟分析 */
  latency: LatencyAnalysis;
  /** 吞吐量分析 */
  throughput: ThroughputAnalysis;
  /** 成本分析 */
  cost?: CostAnalysis;
  /** 资源利用率 */
  utilization: UtilizationAnalysis;
  /** 综合评分 */
  score: OverallScore;
  /** 优化建议 */
  suggestions: OptimizationSuggestion[];
  /** 是否可行 */
  is_feasible: boolean;
  /** 不可行原因 (如有) */
  infeasibility_reason?: string;
}

// ============================================
// 方案搜索
// ============================================

/** 评分权重配置 (六维) */
export interface ScoreWeights {
  /** 延迟权重 (0-1) */
  latency: number;
  /** 吞吐权重 (0-1) */
  throughput: number;
  /** 效率权重 (0-1) */
  efficiency: number;
  /** 均衡权重 (0-1) */
  balance: number;
  /** 显存权重 (0-1) */
  memory: number;
  /** 通信权重 (0-1) */
  communication: number;
}

/** 默认评分权重 (六维) */
export const DEFAULT_SCORE_WEIGHTS: ScoreWeights = {
  latency: 0.2,
  throughput: 0.2,
  efficiency: 0.2,
  balance: 0.1,
  memory: 0.15,
  communication: 0.15,
};

/** TPS per chip 优化权重 (文档推荐: max TPS per chip) */
export const TPS_OPTIMIZED_WEIGHTS: ScoreWeights = {
  latency: 0.1,       // 仅作为约束，不主导评分
  throughput: 0.55,   // TPS per chip 主导
  efficiency: 0.1,    // 资源利用率
  balance: 0.05,      // 负载均衡
  memory: 0.1,        // 显存利用
  communication: 0.1, // 通信开销
};

/** Batch Size 候选值 (包含非 2 幂次值) */
export const BATCH_SIZE_OPTIONS: number[] = [
  1, 2, 4, 8, 10, 12, 16, 32, 64, 128, 256, 512, 1024, 1280,
];

/** 搜索约束 */
export interface SearchConstraints {
  /** 最大芯片数 */
  max_chips?: number;
  /** 最大延迟 (ms) */
  max_latency_ms?: number;
  /** 最小吞吐量 (tokens/s) */
  min_throughput?: number;
  /** 最大内存占用比 (0-1) */
  max_memory_ratio?: number;
  /** TP 必须整除 attention_heads */
  tp_divides_heads?: boolean;
  /** PP 必须整除 layers */
  pp_divides_layers?: boolean;
  /** TP 应放在节点内 */
  tp_within_node?: boolean;
  /** 优化目标 */
  optimization_target?: OptimizationTarget;
  /** 自定义评分权重 */
  score_weights?: ScoreWeights;
}

/** 搜索统计 */
export interface SearchStats {
  /** 评估的方案数 */
  evaluated_count: number;
  /** 被剪枝的方案数 */
  pruned_count: number;
  /** 可行方案数 */
  feasible_count: number;
  /** 搜索耗时 (ms) */
  search_time_ms: number;
}

/** 方案搜索结果 */
export interface PlanSearchResult {
  /** 最优方案 */
  optimal_plan: PlanAnalysisResult;
  /** Top-K 方案 */
  top_k_plans: PlanAnalysisResult[];
  /** Pareto 前沿方案 */
  pareto_frontier: PlanAnalysisResult[];
  /** 搜索统计 */
  search_stats: SearchStats;
}

// ============================================
// 方案对比
// ============================================

/** 指标对比项 */
export interface MetricComparison {
  /** 指标名称 */
  metric_name: string;
  /** 指标单位 */
  unit: string;
  /** 各方案的值 */
  values: Record<string, number>;
  /** 最优值 */
  best_value: number;
  /** 最优方案 ID */
  best_plan_id: string;
}

/** 方案对比结果 */
export interface PlanComparisonResult {
  /** 对比的方案列表 */
  plans: PlanAnalysisResult[];
  /** 各指标对比 */
  metrics: MetricComparison[];
  /** 综合最优方案 */
  overall_best_plan_id: string;
  /** 延迟最优方案 */
  latency_best_plan_id: string;
  /** 吞吐量最优方案 */
  throughput_best_plan_id: string;
}

// ============================================
// Benchmark 配置
// ============================================

/** Benchmark 场景 */
export interface BenchmarkScenario {
  /** 场景名称 */
  name: string;
  /** 场景描述 */
  description: string;
  /** 推理配置 */
  inference_config: InferenceConfig;
  /** 优化目标 */
  optimization_target: OptimizationTarget;
  /** 成功标准 */
  success_criteria: {
    max_ttft_ms?: number;
    max_tpot_ms?: number;
    min_throughput?: number;
  };
}

/** Benchmark 预设 */
export interface BenchmarkPreset {
  /** 预设名称 */
  name: string;
  /** 模型列表 */
  models: string[];
  /** 批次大小列表 */
  batch_sizes: number[];
  /** 序列长度列表 */
  seq_lengths: number[];
  /** 评测指标 */
  metrics: string[];
}

// ============================================
// 辅助类型
// ============================================

/** 字节到各单位的换算 */
export const BYTES_PER = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
} as const;

/** 获取数据类型的字节数 */
export function getBytesPerElement(dtype: DataType): number {
  return BYTES_PER[dtype];
}


// ============================================
// 拓扑融合 - 芯片映射与流量分析
// ============================================

/** 集合通信操作类型 */
export type CollectiveOp = 'allreduce' | 'p2p' | 'alltoall' | 'allgather' | 'reduce_scatter';

/** 并行策略类型 */
export type ParallelismType = 'tp' | 'pp' | 'dp' | 'ep' | 'sp';

/** 芯片到并行组的映射 */
export interface ChipMapping {
  /** 芯片ID (来自拓扑) */
  chipId: string;
  /** 物理位置 */
  physicalLocation: {
    pod: string;
    rack: string;
    board: string;
  };
  /** 并行组内的rank */
  parallelismRank: {
    dp: number;
    tp: number;
    pp: number;
    ep: number;
    sp: number;
  };
  /** 全局rank (扁平化索引) */
  globalRank: number;
}

/** 通信组 */
export interface CommunicationGroup {
  /** 组ID */
  id: string;
  /** 并行类型 */
  type: ParallelismType;
  /** 组内成员芯片ID */
  members: string[];
  /** 集合操作类型 */
  collectiveOp: CollectiveOp;
  /** 消息大小 (MB) */
  messageSizeMb: number;
  /** 通信频率 (每次前向传播的次数) */
  frequency: number;
}

/** 链路流量 */
export interface LinkTraffic {
  /** 源节点ID */
  source: string;
  /** 目标节点ID */
  target: string;
  /** 流量大小 (MB) */
  trafficMb: number;
  /** 链路带宽 (Gbps) */
  bandwidthGbps: number;
  /** 带宽利用率 (0-1) */
  utilizationPercent: number;
  /** 贡献流量的通信组ID列表 */
  contributingGroups: string[];
}

/** 拓扑流量分析结果 */
export interface TopologyTrafficResult {
  /** 芯片映射 */
  chipMapping: ChipMapping[];
  /** 通信组列表 */
  communicationGroups: CommunicationGroup[];
  /** 链路流量 */
  linkTraffic: LinkTraffic[];
  /** 瓶颈链路 (利用率 > 80%) */
  bottleneckLinks: string[];
  /** 最大链路利用率 */
  maxUtilization: number;
  /** 平均链路利用率 */
  avgUtilization: number;
}

// ============================================
// 通信延迟配置 (统一配置)
// ============================================

/** 通信延迟配置 (统一所有延迟相关参数) */
export interface CommLatencyConfig {
  /** Bandwidth Utilization (0-1) */
  bandwidth_utilization: number;
  /** Sync Latency (us) */
  sync_latency_us: number;
  /** Switch Latency (us) */
  switch_latency_us: number;
  /** Cable Latency (us) */
  cable_latency_us: number;
  /** DDR Read Latency (us) */
  memory_read_latency_us: number;
  /** DDR Write Latency (us) */
  memory_write_latency_us: number;
  /** NoC Latency (us) */
  noc_latency_us: number;
  /** Die-to-Die Latency (us) */
  die_to_die_latency_us: number;
}

/** 默认通信延迟配置 */
export const DEFAULT_COMM_LATENCY_CONFIG: CommLatencyConfig = {
  bandwidth_utilization: 0.95,
  sync_latency_us: 0.0,
  switch_latency_us: 1.0,
  cable_latency_us: 0.025,
  memory_read_latency_us: 0.15,
  memory_write_latency_us: 0.01,
  noc_latency_us: 0.05,
  die_to_die_latency_us: 0.04,
};

// ============================================
// ============================================
// 模拟结果类型（从后端 API 返回）
// ============================================

/** Gantt 图任务类型 */
export type GanttTaskType = string;

/** Gantt 图资源 */
export interface GanttResource {
  id: string;
  name: string;
  type: string;
}

/** Gantt 图任务 */
export interface GanttTask {
  id: string;
  name: string;
  resourceId: string;
  start: number;
  end: number;
  type: GanttTaskType;
  phase: string;
  layer?: number;
  ppStage?: number;  // Pipeline stage (for PP parallelism)
}

/** Gantt 图数据 */
export interface GanttChartData {
  resources: GanttResource[];
  tasks: GanttTask[];
  timeRange: { start: number; end: number };
  phaseTransition?: number;
}

/** 阶段时间统计 */
export interface PhaseTimeStats {
  computeTime: number;
  commTime: number;
  bubbleTime: number;
  overlapTime: number;
  totalTime: number;
  computeEfficiency: number;
}

/** 模拟统计数据 */
/** 链路流量统计 */
export interface LinkTrafficStats {
  /** 源芯片ID */
  source: string;
  /** 目标芯片ID */
  target: string;
  /** 累计流量（MB） */
  trafficMb: number;
  /** 链路带宽（Gbps） */
  bandwidthGbps: number;
  /** 链路延迟（微秒） */
  latencyUs: number;
  /** 带宽利用率（0-100） */
  utilizationPercent: number;
  /** 链路类型 */
  linkType: string;
  /** 贡献流量的任务ID列表 */
  contributingTasks: string[];
  /** 按任务类型分解的流量 */
  taskTypeBreakdown: Record<string, number>;
}

export interface SimulationStats {
  prefill: PhaseTimeStats;
  decode: PhaseTimeStats;
  totalRunTime: number;
  simulatedTokens: number;
  ttft: number;
  avgTpot: number;
  dynamicMfu: number;
  dynamicMbu: number;
  maxPpBubbleRatio: number;
  totalEvents: number;
  prefillFlops?: number;
  totalChips?: number;
  /** 错误原因（仅当模拟失败时存在） */
  errorReason?: string;
  /** 链路流量统计 */
  linkTrafficStats?: LinkTrafficStats[];
}

/** 模拟结果 */
export interface SimulationResult {
  ganttChart: GanttChartData;
  stats: SimulationStats;
  /** 吞吐量分析（后端计算） */
  throughput?: ThroughputAnalysis;
  timestamp: number;
}

/** 模拟配置（前端发送到后端）*/
export interface SimulationConfig {
  maxSimulatedTokens?: number;
  enableDataTransferSimulation?: boolean;
  enableDetailedTransformerOps?: boolean;
  enableKVCacheAccessSimulation?: boolean;
  usePreciseEvaluator?: boolean;
  evaluationGranularity?: 'fine' | 'coarse';
}

/** 仿真评分结果 */
export interface SimulationScoreResult {
  latency_score: number;
  throughput_score: number;
  efficiency_score: number;
  balance_score: number;
  overall_score: number;
  raw: {
    ttft_ms: number;
    avg_tpot_ms: number;
    dynamic_mfu: number;
    dynamic_mbu: number;
    pp_bubble_ratio: number;
  };
}

/** 公式计算 vs 仿真结果对比 */
export interface FormulaVsSimComparison {
  formula: {
    ttft_ms: number;
    tpot_ms: number;
    mfu: number;
    mbu: number;
    score: number;
  };
  simulation: {
    ttft_ms: number;
    tpot_ms: number;
    mfu: number;
    mbu: number;
    score: number;
  };
  deviation: {
    ttft_pct: number;
    tpot_pct: number;
    mfu_pct: number;
    mbu_pct: number;
    score_pct: number;
  };
}

// ============================================
// 方案搜索结果（用于前端展示）
// ============================================

/** 不可行方案结果 */
export interface InfeasibleResult {
  /** 并行策略配置 */
  parallelism: ParallelismStrategy;
  /** 不可行原因 */
  reason: string;
}

// ============================================
// 层级与通信聚合类型
// ============================================

/** 层级时间分解 */
export interface LayerBreakdown {
  /** 层索引 */
  layerIndex: number;
  /** 推理阶段 */
  phase: 'prefill' | 'decode';
  /** 计算时间 (us) */
  computeTime: number;
  /** 访存时间 (us) */
  memoryTime: number;
  /** 通信时间分解 */
  commTime: {
    tp: number;
    pp: number;
    ep: number;
    sp: number;
  };
  /** 总时间 (us) */
  totalTime: number;
  /** 计算量 (FLOPs) */
  flops: number;
  /** DRAM 流量 (bytes) */
  dramTraffic: number;
  /** 任务数量 */
  taskCount: number;
}

/** 通信类型详情 */
export interface CommTypeDetail {
  /** 时间 (us) */
  time: number;
  /** 数据量 (bytes) */
  volume: number;
  /** 通信次数 */
  count: number;
  /** 使用的算法 */
  algorithm?: string;
}

/** 通信类型分解 */
export interface CommTypeBreakdown {
  /** 总通信时间 (us) */
  totalTime: number;
  /** 各类型通信详情 */
  breakdown: {
    tp: CommTypeDetail;
    pp: CommTypeDetail;
    ep: CommTypeDetail;
    sp: CommTypeDetail;
  };
  /** 通信瓶颈层 (时间最长的前3层) */
  bottleneckLayers: number[];
}

/** 扩展的 GanttTask (包含详细性能信息) */
export interface GanttTaskExtended extends GanttTask {
  /** 计算时间 (us) */
  compute_time_us?: number;
  /** 访存时间 (us) */
  memory_time_us?: number;
  /** 通信时间 (us) */
  comm_time_us?: number;
  /** 计算量 (FLOPs) */
  flops?: number;
  /** GEMM 形状 [M, N, K] */
  gemm_shape?: [number, number, number];
  /** 最优 tile 配置 */
  best_tile?: string;
  /** 最优分区配置 */
  best_partition?: string;
  /** 架构利用率 */
  arch_utilization?: number;
  /** 有效利用率 */
  effective_utilization?: number;
  /** DRAM 流量 (bytes) */
  dram_traffic_bytes?: number;
  /** DRAM 占用 (bytes) */
  dram_occupy_bytes?: number;
  /** 通信数据量 (bytes) */
  comm_size_bytes?: number;
  /** 通信算法 */
  comm_algorithm?: string;
  /** 通信组大小 */
  comm_group_size?: number;
}
