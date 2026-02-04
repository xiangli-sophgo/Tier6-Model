/**
 * Tier6 类型定义
 *
 * 与 Tier6 仿真引擎后端对应的类型定义
 */

// ==================== 预设类型 ====================

/** 内存层级配置 */
export interface MemoryLevelConfig {
  type?: string;           // 内存类型 (LPDDR5, HBM3, etc.)
  channels?: number;       // 通道数
  rate_mt?: number;        // 传输速率 (MT/s)
  width_bits?: number;     // 位宽
  capacity_gb?: number;    // 容量 (GB) - gmem/l2m
  capacity_mb?: number;    // 容量 (MB) - lmem
  capacity_kb?: number;    // 容量 (KB) - smem
  bandwidth_gbps: number;  // 带宽 (GB/s)
  latency_ns?: number;     // 延迟 (ns)
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

/** 芯片预设 - 完整 Tier6 格式 */
export interface ChipPreset {
  name: string;
  architecture?: string;   // 架构 (TPU_V7, TPU_V7.1, etc.)
  process?: string;        // 工艺 (7nm, 5nm, etc.)
  frequency_ghz?: number;  // 频率 (GHz)

  // 核心配置
  cores?: {
    count: number;         // 核心数
    lanes_per_core: number; // 每核 Lane 数
  };

  // 计算单元
  compute_units?: {
    cube?: {
      // MAC 数量/Lane (按数据类型)
      mac_per_lane?: {
        INT8?: number;
        FP8?: number;
        BF16?: number;
        FP16?: number;
        TF32?: number;
        INT4?: number;
        [key: string]: number | undefined;
      };
    };
    vector?: {
      // EU 数量/Lane (按数据类型)
      eu_per_lane?: {
        INT8?: number;
        FP8?: number;
        INT16?: number;
        BF16?: number;
        FP16?: number;
        INT32?: number;
        FP32?: number;
        [key: string]: number | undefined;
      };
    };
  };

  // 内存层级
  memory?: {
    gmem?: MemoryLevelConfig;  // 全局内存 (DDR/HBM)
    l2m?: MemoryLevelConfig;   // L2 缓存
    lmem?: MemoryLevelConfig;  // 本地内存 (每核)
    smem?: MemoryLevelConfig;  // 共享内存
  };

  // DMA 引擎
  dma_engines?: {
    gdma?: DmaEngineConfig;    // Global DMA
    sdma?: DmaEngineConfig;    // Shared DMA
    cdma?: DmaEngineConfig;    // C2C DMA
  };

  // 片上互联 (NoC)
  interconnect?: {
    noc?: ChipInterconnectConfig;  // 片上网络
  };

  // 兼容 llm_simulator 旧格式字段
  num_cores?: number;              // 核心数 (兼容)
  compute_tflops_fp8?: number;     // FP8 算力
  compute_tflops_bf16?: number;    // BF16 算力
  memory_capacity_gb?: number;     // 显存容量
  memory_bandwidth_gbps?: number;  // 显存带宽
  memory_bandwidth_utilization?: number;
  lmem_capacity_mb?: number;
  lmem_bandwidth_gbps?: number;
  cube_m?: number;
  cube_k?: number;
  cube_n?: number;
  sram_size_kb?: number;
  sram_utilization?: number;
  lane_num?: number;
  align_bytes?: number;
  compute_dma_overlap_rate?: number;
  [key: string]: unknown;  // 允许额外字段
}

/** 模型预设 */
export interface ModelPreset {
  name: string;
  hidden_size: number;
  num_layers: number;
  num_dense_layers?: number;
  num_moe_layers?: number;
  num_heads: number;
  vocab_size: number;
  dtype: string;
  moe?: MoEConfig;
  mla?: MLAConfig;
  ffn?: FFNConfig;
}

/** MoE 配置 */
export interface MoEConfig {
  num_routed_experts: number;
  num_shared_experts: number;
  num_activated_experts: number;
  intermediate_size: number;
}

/** MLA 配置 */
export interface MLAConfig {
  q_lora_rank: number;
  kv_lora_rank: number;
  qk_nope_head_dim: number;
  qk_rope_head_dim: number;
  v_head_dim: number;
}

/** FFN 配置 */
export interface FFNConfig {
  intermediate_size: number;
}

// ==================== Benchmark 类型 ====================

/** Benchmark 列表项 */
export interface BenchmarkListItem {
  id: string;
  name: string;
  model_name?: string;
  batch_size?: number;
  input_seq_length?: number;
  output_seq_length?: number;
}

/** Tier6 Benchmark 配置 */
export interface Tier6BenchmarkConfig {
  id?: string;
  name?: string;
  model: Tier6ModelConfig | Record<string, unknown>;
  inference: Tier6InferenceConfig | Record<string, unknown>;
}

/** Tier6 模型配置 */
export interface Tier6ModelConfig {
  name: string;
  hidden_size: number;
  num_layers: number;
  num_dense_layers?: number;
  num_moe_layers?: number;
  num_heads: number;
  vocab_size: number;
  dtype: string;
  moe?: MoEConfig;
  mla?: MLAConfig;
  ffn?: FFNConfig;
}

/** Tier6 推理配置 */
export interface Tier6InferenceConfig {
  batch_size: number;
  input_seq_length: number;
  output_seq_length: number;
}

// ==================== Topology 类型 ====================

/** Topology 列表项 */
export interface TopologyListItem {
  name: string;
  description?: string;
  chip_count?: number;
}

/** Tier6 拓扑配置 - 宽松类型以兼容前端现有代码 */
export interface Tier6TopologyConfig {
  name?: string;
  description?: string;
  pod_count?: number;
  racks_per_pod?: number;
  rack_config?: Record<string, unknown>;
  topology?: Record<string, unknown>;
  chips?: Record<string, unknown>;
  hardware_params?: {
    chips?: Record<string, unknown>;
    interconnect?: Record<string, unknown>;
  };
  interconnect?: Record<string, unknown>;
  comm_latency_config?: CommLatencyConfig | Record<string, unknown>;
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

/** 任务结果 - 兼容 llm_simulator 格式 */
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
