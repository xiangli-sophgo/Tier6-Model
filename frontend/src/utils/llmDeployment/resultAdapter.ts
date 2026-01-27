/**
 * 结果适配器
 *
 * 将后端 SimulationResult 转换为前端 PlanAnalysisResult 格式
 * 注意：所有评估计算均由后端完成，此适配器只做格式转换
 */

import {
  SimulationResult,
  PlanAnalysisResult,
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  MemoryAnalysis,
  CommunicationAnalysis,
  LatencyAnalysis,
  ThroughputAnalysis,
  UtilizationAnalysis,
  OverallScore,
  SimulationStats,
  ScoreWeights,
  DEFAULT_SCORE_WEIGHTS,
} from './types';

/**
 * 简化的评分计算（基于后端仿真结果）
 */
function calculateScoreFromStats(
  stats: SimulationStats,
  weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS
): OverallScore {
  // 延迟评分 (TTFT < 100ms 满分, > 1000ms 零分)
  const ttft = stats.ttft;
  const latencyScore = Math.max(0, Math.min(100, 100 - (ttft - 100) / 9));

  // 吞吐评分 (MFU > 50% 满分)
  const mfu = stats.dynamicMfu;
  const throughputScore = Math.min(100, mfu * 200);

  // 效率评分 (综合 MFU 和 MBU)
  const avgUtilization = (stats.dynamicMfu + stats.dynamicMbu) / 2;
  const efficiencyScore = avgUtilization * 100;

  // 均衡评分 (基于气泡比)
  const bubbleRatio = stats.maxPpBubbleRatio;
  const balanceScore = (1 - bubbleRatio) * 100;

  // 综合评分 (加权平均)
  const overallScore =
    latencyScore * weights.latency +
    throughputScore * weights.throughput +
    efficiencyScore * weights.efficiency +
    balanceScore * weights.balance;

  return {
    latency_score: latencyScore,
    throughput_score: throughputScore,
    efficiency_score: efficiencyScore,
    balance_score: balanceScore,
    overall_score: overallScore,
  };
}

/**
 * 将后端仿真结果适配为前端 PlanAnalysisResult 格式
 */
export function adaptSimulationResult(
  simulation: SimulationResult,
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): PlanAnalysisResult {
  const stats = simulation.stats;

  // 1. 显存分析（简化版，后端应提供详细数据）
  const memory: MemoryAnalysis = {
    model_memory_gb: 0,
    kv_cache_memory_gb: 0,
    activation_memory_gb: 0,
    overhead_gb: 0,
    total_per_chip_gb: 0,
    is_memory_sufficient: true,
    memory_utilization: 0,
  };

  // 2. 通信分析（简化版，后端应提供详细数据）
  const communication: CommunicationAnalysis = {
    tp_comm_volume_gb: 0,
    pp_comm_volume_gb: 0,
    ep_comm_volume_gb: 0,
    sp_comm_volume_gb: 0,
    total_comm_volume_gb: 0,
    tp_comm_latency_ms: 0,
    pp_comm_latency_ms: 0,
  };

  // 3. 延迟分析（从后端 stats 提取）
  const latency: LatencyAnalysis = {
    prefill_compute_latency_ms: stats.prefill.computeTime,
    prefill_comm_latency_ms: stats.prefill.commTime,
    prefill_total_latency_ms: stats.ttft,
    prefill_flops: stats.prefillFlops,
    decode_compute_latency_ms: stats.decode.computeTime,
    decode_memory_latency_ms: 0,
    decode_comm_latency_ms: stats.decode.commTime,
    decode_per_token_latency_ms: stats.avgTpot,
    end_to_end_latency_ms: stats.totalRunTime,
    pipeline_bubble_ratio: stats.maxPpBubbleRatio,
    bottleneck_type: 'compute',
    bottleneck_details: '基于后端仿真结果',
  };

  // 4. 吞吐量分析（从后端 stats 提取）
  const throughput: ThroughputAnalysis = {
    tokens_per_second: (stats.simulatedTokens / stats.totalRunTime) * 1000,
    tps_per_batch: stats.avgTpot > 0 ? 1000 / stats.avgTpot : 0,
    tps_per_chip: 0,
    requests_per_second: 0,
    model_flops_utilization: stats.dynamicMfu,
    memory_bandwidth_utilization: stats.dynamicMbu,
    theoretical_max_throughput: 0,
  };

  // 5. 利用率分析（从后端 stats 提取）
  const utilization: UtilizationAnalysis = {
    compute_utilization: stats.dynamicMfu,
    memory_utilization: memory.memory_utilization,
    network_utilization: 0,
    load_balance_score: 1 - stats.maxPpBubbleRatio,
  };

  // 6. 评分
  const score = calculateScoreFromStats(stats);

  // 7. 检查是否可行
  const is_feasible = stats.ttft > 0 && stats.ttft < Infinity;

  return {
    plan: {
      plan_id: `${parallelism.tp > 1 ? `tp${parallelism.tp}` : ''}${parallelism.pp > 1 ? `_pp${parallelism.pp}` : ''}${parallelism.ep > 1 ? `_ep${parallelism.ep}` : ''}${parallelism.dp > 1 ? `_dp${parallelism.dp}` : ''}`.replace(/^_/, '') || 'single',
      parallelism,
      total_chips: parallelism.dp * parallelism.tp,
    },
    memory,
    communication,
    latency,
    throughput,
    utilization,
    score,
    suggestions: [],
    is_feasible,
    infeasibility_reason: is_feasible ? undefined : '后端模拟失败',
  };
}
