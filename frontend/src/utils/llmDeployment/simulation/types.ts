/**
 * LLM 推理模拟系统 - 类型定义
 *
 * 定义事件驱动模拟器相关的所有类型，包括：
 * - 模拟配置
 * - 事件类型
 * - 通信Trace
 * - 甘特图数据
 * - 模拟结果统计
 */

import { CollectiveOp, InferencePhase } from '../types'

// ============================================
// 模拟配置
// ============================================

/** 模拟粒度 */
export type SimulationGranularity = 'token' | 'layer' | 'phase'

/** 模拟配置 */
export interface SimulationConfig {
  /** 模拟粒度 */
  granularity: SimulationGranularity
  /** 启用计算通信重叠 */
  enableOverlap: boolean
  /** 启用流水线模拟 */
  enablePipeline: boolean
  /** 延迟抖动因子 (0-1)，用于模拟真实场景的延迟波动 */
  jitterFactor: number
  /** Decode阶段最大模拟token数 */
  maxSimulatedTokens: number
}

/** 默认模拟配置 */
export const DEFAULT_SIMULATION_CONFIG: SimulationConfig = {
  granularity: 'layer',
  enableOverlap: true,
  enablePipeline: true,
  jitterFactor: 0.05,
  maxSimulatedTokens: 32,
}

// ============================================
// 事件类型
// ============================================

/** 事件类型 */
export type SimEventType =
  | 'compute_start'
  | 'compute_end'
  | 'comm_start'
  | 'comm_end'
  | 'layer_start'
  | 'layer_end'
  | 'phase_start'
  | 'phase_end'
  | 'token_start'
  | 'token_end'

/** 资源类型 */
export type ResourceType = 'compute' | 'memory' | 'network_tp' | 'network_pp' | 'network_ep'

/** 操作类型 */
export type OperationType =
  | 'attention'
  | 'ffn'
  | 'layernorm'
  | 'tp_allreduce'
  | 'pp_send'
  | 'pp_recv'
  | 'ep_alltoall'
  | 'embedding'
  | 'lm_head'

/** 模拟事件 */
export interface SimEvent {
  /** 事件唯一ID */
  id: string
  /** 事件类型 */
  type: SimEventType
  /** 时间戳 (ms) */
  timestamp: number
  /** 持续时间 (ms) */
  duration?: number
  /** 芯片ID */
  chipId: string
  /** PP阶段索引 */
  ppStage: number
  /** 推理阶段 */
  phase: InferencePhase
  /** 层索引 */
  layerIndex?: number
  /** Token索引 (Decode阶段) */
  tokenIndex?: number
  /** 操作类型 */
  operation: OperationType
  /** 资源类型 */
  resource: ResourceType
  /** 额外元数据 */
  metadata?: Record<string, unknown>
}

// ============================================
// 通信Trace
// ============================================

/** 并行类型 */
export type ParallelismTypeSimulation = 'tp' | 'pp' | 'ep' | 'sp'

/** 通信Trace项 */
export interface CommTraceItem {
  /** 唯一ID */
  id: string
  /** 开始时间 (ms) */
  startTime: number
  /** 结束时间 (ms) */
  endTime: number
  /** 源芯片ID */
  sourceChipId: string
  /** 目标芯片ID */
  targetChipId: string
  /** 并行类型 */
  parallelismType: ParallelismTypeSimulation
  /** 集合通信操作 */
  collectiveOp: CollectiveOp
  /** 数据量 (MB) */
  dataSizeMb: number
  /** 推理阶段 */
  phase: InferencePhase
  /** 层索引 */
  layerIndex: number
  /** 通信组ID */
  groupId: string
}

// ============================================
// 甘特图数据
// ============================================

/** 甘特图任务类型 */
export type GanttTaskType =
  // 计算任务
  | 'compute'
  | 'embedding'
  | 'layernorm'
  | 'attention_qkv'
  | 'attention_score'
  | 'attention_softmax'
  | 'attention_output'
  | 'ffn_gate'
  | 'ffn_up'
  | 'ffn_down'
  | 'lm_head'
  // 数据搬运
  | 'pcie_h2d'
  | 'pcie_d2h'
  | 'hbm_write'
  | 'hbm_read'
  | 'weight_load'
  | 'kv_cache_read'
  | 'kv_cache_write'
  // 通信
  | 'tp_comm'
  | 'pp_comm'
  | 'ep_comm'
  // SP 通信 (序列并行)
  | 'sp_allgather'
  | 'sp_reduce_scatter'
  // DP 通信 (数据并行梯度同步)
  | 'dp_gradient_sync'
  // MLA细粒度 (DeepSeek特有)
  | 'rmsnorm_q_lora'
  | 'rmsnorm_kv_lora'
  | 'mm_q_lora_a'
  | 'mm_q_lora_b'
  | 'mm_kv_lora_a'
  | 'attn_fc'
  | 'bmm_qk'
  | 'bmm_sv'
  // MoE (专家并行)
  | 'moe_gate'
  | 'moe_expert'
  | 'moe_shared_expert'
  | 'ep_dispatch'
  | 'ep_combine'
  // 其他
  | 'bubble'
  | 'idle'

/** 甘特图任务 */
export interface GanttTask {
  /** 唯一ID */
  id: string
  /** 任务名称 */
  name: string
  /** 资源行标识 (如 "Stage 0 - Compute") */
  resource: string
  /** 开始时间 (ms) */
  start: number
  /** 结束时间 (ms) */
  end: number
  /** 任务类型 */
  type: GanttTaskType
  /** 推理阶段 */
  phase: InferencePhase
  /** 芯片ID */
  chipId: string
  /** PP阶段 */
  ppStage: number
  /** 层索引 */
  layerIndex?: number
  /** Token索引 */
  tokenIndex?: number
  /** 颜色 (可选，用于自定义) */
  color?: string
}

/** 甘特图资源行 */
export interface GanttResource {
  /** 资源ID */
  id: string
  /** 显示名称 */
  name: string
  /** PP阶段 */
  ppStage: number
  /** 资源类型 */
  type: 'compute' | 'network'
}

/** 甘特图数据 */
export interface GanttChartData {
  /** 资源行列表 */
  resources: GanttResource[]
  /** 任务列表 */
  tasks: GanttTask[]
  /** 时间范围 */
  timeRange: {
    start: number
    end: number
  }
  /** Prefill/Decode 分界时间点 */
  phaseTransition?: number
}

// ============================================
// 模拟统计
// ============================================

/** 阶段时间统计 */
export interface PhaseTimeStats {
  /** 计算时间 (ms) */
  computeTime: number
  /** 通信时间 (ms) */
  commTime: number
  /** 气泡时间 (ms) */
  bubbleTime: number
  /** 重叠时间 (ms) - 计算和通信同时进行的时间 */
  overlapTime: number
  /** 总时间 (ms) */
  totalTime: number
  /** 计算效率 (computeTime / totalTime) */
  computeEfficiency: number
}

/** 延迟分布 */
export interface LatencyDistribution {
  /** 最小值 */
  min: number
  /** 最大值 */
  max: number
  /** 平均值 */
  mean: number
  /** 标准差 */
  stdDev: number
  /** P50 */
  p50: number
  /** P90 */
  p90: number
  /** P99 */
  p99: number
}

/** 模拟统计 */
export interface SimulationStats {
  /** Prefill阶段统计 */
  prefill: PhaseTimeStats
  /** Decode阶段统计 */
  decode: PhaseTimeStats
  /** 总运行时间 (ms) */
  totalRunTime: number
  /** 模拟的Token数 */
  simulatedTokens: number
  /** 首Token延迟 (TTFT) */
  ttft: number
  /** 平均每Token延迟 (TPOT) */
  avgTpot: number
  /** TPOT延迟分布 */
  tpotDistribution?: LatencyDistribution
  /** 动态计算的MFU (基于实际模拟) */
  dynamicMfu: number
  /** 动态计算的MBU (基于实际模拟) */
  dynamicMbu: number
  /** 最大PP气泡比 */
  maxPPBubbleRatio: number
  /** 事件总数 */
  totalEvents: number
  /** Prefill阶段计算量 (FLOPs) */
  prefillFlops: number
}

// ============================================
// 模拟结果
// ============================================

/** 模拟结果 */
export interface SimulationResult {
  /** 模拟配置 */
  config: SimulationConfig
  /** 所有事件 */
  events: SimEvent[]
  /** 通信Trace */
  commTrace: CommTraceItem[]
  /** 甘特图数据 */
  ganttChart: GanttChartData
  /** 统计信息 */
  stats: SimulationStats
  /** 模拟完成时间戳 */
  timestamp: number
}

// ============================================
// 芯片状态 (模拟器内部使用)
// ============================================

/** 芯片状态 */
export interface ChipState {
  /** 芯片ID */
  chipId: string
  /** PP阶段 */
  ppStage: number
  /** TP组内rank */
  tpRank: number
  /** EP组内rank */
  epRank: number
  /** 当前时间 (ms) */
  currentTime: number
  /** 计算资源空闲时间 */
  computeIdleAt: number
  /** TP网络空闲时间 */
  tpNetworkIdleAt: number
  /** PP网络空闲时间 */
  ppNetworkIdleAt: number
  /** EP网络空闲时间 */
  epNetworkIdleAt: number
  /** 已完成的层数 */
  completedLayers: number
  /** 已完成的Token数 */
  completedTokens: number
}

// ============================================
// 任务类型 (模拟器内部使用)
// ============================================

/** 待执行任务 */
export interface SimTask {
  /** 任务ID */
  id: string
  /** 任务类型 */
  type: 'compute' | 'comm'
  /** 操作 */
  operation: OperationType
  /** 芯片ID */
  chipId: string
  /** PP阶段 */
  ppStage: number
  /** 推理阶段 */
  phase: InferencePhase
  /** 层索引 */
  layerIndex?: number
  /** Token索引 */
  tokenIndex?: number
  /** 预计耗时 (ms) */
  duration: number
  /** 依赖的任务ID列表 */
  dependencies: string[]
  /** 最早开始时间 */
  earliestStart: number
}

// ============================================
// 仿真评分与对比
// ============================================

/** 基于仿真结果的评分 */
export interface SimulationScoreResult {
  /** 延迟评分 - 基于 TTFT (100ms=满分, 1000ms=0分) */
  latency_score: number
  /** 吞吐评分 - 基于 dynamic_mfu (50%=满分) */
  throughput_score: number
  /** 效率评分 - (mfu + mbu) / 2 */
  efficiency_score: number
  /** 均衡评分 - 1 - pp_bubble_ratio */
  balance_score: number
  /** 综合评分 */
  overall_score: number
  /** 原始仿真指标 */
  raw: {
    ttft_ms: number
    avg_tpot_ms: number
    dynamic_mfu: number
    dynamic_mbu: number
    pp_bubble_ratio: number
  }
}

/** 公式计算 vs 仿真结果对比 */
export interface FormulaVsSimComparison {
  /** 公式计算结果 */
  formula: {
    ttft_ms: number
    tpot_ms: number
    mfu: number
    mbu: number
    score: number
  }
  /** 仿真结果 */
  simulation: {
    ttft_ms: number
    tpot_ms: number
    mfu: number
    mbu: number
    score: number
  }
  /** 偏差百分比 (sim - formula) / formula * 100 */
  deviation: {
    ttft_pct: number
    tpot_pct: number
    mfu_pct: number
    mbu_pct: number
    score_pct: number
  }
}
