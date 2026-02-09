/**
 * Tier6 类型定义
 *
 * 与 Tier6 仿真引擎后端对应的类型定义
 */

// ==================== 预设类型 ====================

/** GMEM 内存配置 */
export interface GmemConfig {
  type?: string;           // 内存类型 (LPDDR5, HBM3, etc.)
  capacity_gb: number;     // 容量 (GB)
  bandwidth_gbps: number;  // 带宽 (GB/s)
  bandwidth_utilization?: number; // 带宽利用率 (0-1)
  latency_ns?: number;     // 延迟 (ns)
}

/** LMEM 内存配置 */
export interface LmemConfig {
  capacity_mb: number;     // 容量 (MB)
  bandwidth_gbps: number;  // 带宽 (GB/s)
  latency_ns?: number;     // 延迟 (ns)
  sram_utilization?: number; // SRAM 利用率 (0-1)
}

/** DMA 引擎配置 */
export interface DmaEngineConfig {
  bandwidth_gbps: number;       // 带宽 (GB/s)
  startup_latency_ns?: number;  // 启动延迟 (ns)
  efficiency?: number;          // 效率 (0-1)
}

/** 互联配置 */
export interface ChipInterconnectConfig {
  topology?: string;       // 拓扑类型 (mesh, ring, etc.)
  links?: number;          // 链路数
  bandwidth_gbps: number;  // 带宽 (GB/s)
  latency_ns?: number;     // 延迟 (ns)
}

/** MAC/Lane 按数据类型 */
export interface MacPerLaneConfig {
  INT8?: number;
  FP8?: number;
  BF16?: number;
  FP16?: number;
  TF32?: number;
  INT4?: number;
}

/** EU/Lane 按数据类型 */
export interface EuPerLaneConfig {
  INT8?: number;
  FP8?: number;
  INT16?: number;
  BF16?: number;
  FP16?: number;
  INT32?: number;
  FP32?: number;
}

/** 芯片预设 - Tier6 结构化格式 */
export interface ChipPreset {
  name: string;
  architecture?: string;   // 架构 (TPU_V7, etc.)
  process?: string;        // 工艺 (7nm, 5nm, etc.)
  frequency_ghz: number;   // 频率 (GHz)

  // 核心配置
  cores: {
    count: number;         // 核心数
    lanes_per_core: number; // 每核 Lane 数
  };

  // 计算单元
  compute_units: {
    cube: {
      m?: number;          // 矩阵计算 M 维度
      k?: number;          // 矩阵计算 K 维度
      n?: number;          // 矩阵计算 N 维度
      mac_per_lane: MacPerLaneConfig;
    };
    vector?: {
      eu_per_lane: EuPerLaneConfig;
    };
  };

  // 内存层级 (两级模型: gmem + lmem)
  memory: {
    gmem: GmemConfig;
    lmem: LmemConfig;
  };

  // DMA 引擎 (单 GDMA)
  dma_engines: {
    gdma: DmaEngineConfig;
  };

  // 其他参数
  align_bytes: number;
  compute_dma_overlap_rate: number;

  // 片上互联 (NoC)
  interconnect?: {
    noc?: ChipInterconnectConfig;
  };
}

/** MoE 配置 */
export interface MoEConfig {
  num_routed_experts: number;
  num_shared_experts?: number;
  num_activated_experts: number;
  intermediate_size: number;
  num_expert_groups?: number;
  num_limited_groups?: number;
  route_scale?: number;
  decoder_sparse_step?: number;
  norm_topk_prob?: boolean;
  router_aux_loss_coef?: number;
}

/** MLA (Multi-head Latent Attention) 配置 */
export interface MLAConfig {
  q_lora_rank: number;
  kv_lora_rank: number;
  qk_nope_head_dim: number;
  qk_rope_head_dim: number;
  v_head_dim: number;
}

/** DSA (Dynamic Sparse Attention) 配置 */
export interface DSAConfig {
  num_index_heads: number;
  index_head_dim: number;
  topk_index: number;
}

/** NSA (Native Sparse Attention) 配置 */
export interface NSAConfig {
  l: number;
  d: number;
  sl: number;
  sn: number;
  w: number;
}

/** RoPE 位置编码配置 */
export interface RoPEConfig {
  theta?: number;
  factor?: number;
  original_seq_len?: number;
  max_position_embeddings?: number;
  beta_fast?: number;
  beta_slow?: number;
  mscale?: number;
}

/** 模型预设 (混合结构: 核心参数扁平 + 特性模块嵌套) */
export interface ModelPreset {
  name: string;
  // 核心参数 (扁平)
  vocab_size: number;
  hidden_size: number;
  intermediate_size: number;
  num_layers: number;
  num_dense_layers?: number;
  num_moe_layers?: number;
  num_attention_heads: number;
  num_key_value_heads?: number;
  v_head_dim?: number;
  max_seq_len?: number;
  hidden_act?: string;
  rms_norm_eps?: number;
  attention_bias?: boolean;
  attention_dropout?: number;
  // 特性模块 (嵌套，可选 -- 省略 = 不使用该特性)
  MoE?: MoEConfig;
  MLA?: MLAConfig;
  DSA?: DSAConfig;
  NSA?: NSAConfig;
  RoPE?: RoPEConfig;
}

// ==================== Benchmark 类型 ====================

/** Benchmark 列表项 */
export interface BenchmarkListItem {
  id: string;
  name: string;
  model_name?: string;
  topology?: string;
  batch_size?: number;
  input_seq_length?: number;
  output_seq_length?: number;
}

/** Tier6 Benchmark 配置 */
export interface Tier6BenchmarkConfig {
  id?: string;
  name?: string;
  model: string | ModelPreset | Record<string, unknown>;
  model_preset_ref?: string;
  topology?: string | Tier6TopologyConfig | Record<string, unknown>;
  topology_preset_ref?: string;
  inference: Tier6InferenceConfig | Record<string, unknown>;
}

/** Tier6 推理配置 */
export interface Tier6InferenceConfig {
  batch_size: number;
  input_seq_length: number;
  output_seq_length: number;
  weight_dtype?: string;
  activation_dtype?: string;
}

// ==================== Topology 类型 ====================

/** Topology 列表项 */
export interface TopologyListItem {
  name: string;
  description?: string;
  chip_count?: number;
}

/** 芯片分组 (board 内) */
export interface TopologyChipGroup {
  name: string;
  preset_id?: string;
  count: number;
}

/** 板卡分组 (rack 内) */
export interface TopologyBoardGroup {
  id?: string;
  name?: string;
  u_height?: number;
  count: number;
  chips: TopologyChipGroup[];
}

/** Rack 分组 (pod 内) */
export interface TopologyRackGroup {
  count: number;
  total_u?: number;
  boards: TopologyBoardGroup[];
}

/** Pod 分组 (顶层) */
export interface TopologyPodGroup {
  count: number;
  racks: TopologyRackGroup[];
}

/** 互联链路配置 */
export interface InterconnectLinks {
  c2c?: { bandwidth_gbps: number; latency_us: number };
  b2b?: { bandwidth_gbps: number; latency_us: number };
  r2r?: { bandwidth_gbps: number; latency_us: number };
  p2p?: { bandwidth_gbps: number; latency_us: number };
}

/** 互联配置 (链路 + 通信参数) */
export interface InterconnectConfig {
  links?: InterconnectLinks;
  comm_params?: CommLatencyConfig | Record<string, unknown>;
}

/** Tier6 拓扑配置 (grouped_pods 格式) */
export interface Tier6TopologyConfig {
  name?: string;
  description?: string;
  pods?: TopologyPodGroup[];
  chips?: Record<string, ChipPreset>;
  interconnect?: InterconnectConfig;
  [key: string]: unknown;  // 允许额外字段
}

/** 通信延迟配置 */
export interface CommLatencyConfig {
  rtt_tp_us?: number;
  rtt_ep_us?: number;
  bandwidth_utilization?: number;
  sync_latency_us?: number;
  switch_delay_us?: number;
  cable_delay_us?: number;
  memory_read_latency_us?: number;
  memory_write_latency_us?: number;
  noc_latency_us?: number;
  die_to_die_latency_us?: number;
}

// ==================== 评估请求类型 ====================

/** Tier6 评估请求 */
export interface Tier6EvaluationRequest {
  experiment_name: string;
  description?: string;
  experiment_description?: string;
  benchmark_name: string;
  topology_config_name: string;
  benchmark_config: Tier6BenchmarkConfig | Record<string, unknown>;
  topology_config: Tier6TopologyConfig | Record<string, unknown>;
  search_mode: 'manual' | 'auto' | 'sweep';
  manual_parallelism?: Tier6ManualParallelism | Record<string, unknown>;
  search_constraints?: Record<string, unknown>;
  max_workers?: number;
  enable_tile_search?: boolean;
  enable_partition_search?: boolean;
  max_simulated_tokens?: number;
}

/** 手动并行策略 */
export interface Tier6ManualParallelism {
  tp?: number;
  pp?: number;
  dp?: number;
  ep?: number;
  moe_tp?: number;
  seq_len?: number;
  batch_size?: number;
  enable_tp_sp?: boolean;
  embed_tp?: number;
  lmhead_tp?: number;
  comm_protocol?: number;
  kv_cache_rate?: number;
  is_prefill?: boolean;
  [key: string]: unknown;  // 允许额外字段
}

// ==================== 任务与结果类型 ====================

/** 任务状态 */
export interface Tier6TaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message?: string;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

/** 任务结果 */
export interface Tier6TaskResults {
  top_k_plans: Tier6PlanResult[];
  infeasible_plans: Tier6InfeasiblePlan[];
  search_stats?: {
    total_plans: number;
    feasible_plans: number;
    infeasible_plans: number;
  };
  config_snapshot?: Record<string, unknown>;
}

/** 可行方案结果 */
export interface Tier6PlanResult {
  parallelism: Tier6ManualParallelism | Record<string, unknown>;
  tps: number;
  ttft: number;
  tpot: number;
  mfu: number;
  mbu: number;
  score: number;
  is_feasible: boolean;
  aggregates?: Record<string, unknown>;
  step_metrics?: unknown[];
  config?: Record<string, unknown>;
  [key: string]: unknown;  // 允许额外字段
}

/** 不可行方案 */
export interface Tier6InfeasiblePlan {
  parallelism: Tier6ManualParallelism | Record<string, unknown>;
  infeasible_reason: string;
  is_feasible: boolean;
  [key: string]: unknown;  // 允许额外字段
}

/** 评估结果 (保留兼容) */
export interface Tier6EvaluationResult {
  parallelism: Tier6ManualParallelism;
  metrics: {
    tps: number;
    ttft: number;
    tpot: number;
    mfu: number;
    mbu: number;
  };
  is_feasible: boolean;
  score?: number;
  full_result?: Record<string, unknown>;
}

// ==================== 实验类型 ====================

/** 实验 */
export interface Tier6Experiment {
  id: number;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  total_tasks: number;
  completed_tasks: number;
  tasks?: Tier6TaskStatus[];
}

// ==================== 其他类型 ====================

/** 仿真请求 */
export interface Tier6SimulateRequest {
  benchmark_config: Tier6BenchmarkConfig;
  topology_config: Tier6TopologyConfig;
  parallelism: Tier6ManualParallelism;
}

/** 仿真响应 */
export interface Tier6SimulateResponse {
  success: boolean;
  metrics?: {
    tps: number;
    ttft: number;
    tpot: number;
    mfu: number;
    mbu: number;
  };
  gantt_data?: Record<string, unknown>;
  error?: string;
}

/** 验证结果 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}
