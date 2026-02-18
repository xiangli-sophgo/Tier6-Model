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
  isMemorySufficient,
} from './types';
import { calculateScores, type ScoreInput } from './scoreCalculator';

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
  // 从 hardware.chips 中获取第一个芯片的容量
  const chips = hardware?.chips || {};
  const firstChipName = Object.keys(chips)[0];

  if (!firstChipName || !chips[firstChipName]?.memory?.gmem?.capacity_gb) {
    throw new Error('无法从硬件配置中获取芯片容量 (memory.gmem.capacity_gb)');
  }

  const chipCapacityGB = chips[firstChipName].memory.gmem.capacity_gb;
  const totalMemoryGB = 0; // TODO: 从后端获取实际内存数据
  const memory: MemoryAnalysis = {
    model_memory_gb: 0,
    kv_cache_memory_gb: 0,
    activation_memory_gb: 0,
    overhead_gb: 0,
    total_per_chip_gb: totalMemoryGB,
    is_memory_sufficient: isMemorySufficient(totalMemoryGB, chipCapacityGB),
    memory_utilization: totalMemoryGB / chipCapacityGB,
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
  // 注意：avgTpot 单位是微秒(us)，需要转换为毫秒(ms)
  const avgTpotMs = stats.avgTpot / 1000.0;
  const latency: LatencyAnalysis = {
    prefill_compute_latency_ms: stats.prefill.computeTime,
    prefill_comm_latency_ms: stats.prefill.commTime,
    prefill_total_latency_ms: stats.ttft,
    prefill_flops: stats.prefillFlops,
    decode_compute_latency_ms: stats.decode.computeTime,
    decode_memory_latency_ms: 0,
    decode_comm_latency_ms: stats.decode.commTime,
    decode_per_token_latency_ms: avgTpotMs,
    end_to_end_latency_ms: stats.totalRunTime,
    pipeline_bubble_ratio: stats.maxPpBubbleRatio,
    bottleneck_type: 'compute',
    bottleneck_details: '基于后端仿真结果',
  };

  // 4. 吞吐量分析（优先使用后端计算的 throughput，否则降级计算）
  const throughput: ThroughputAnalysis = simulation.throughput || {
    tokens_per_second: (stats.simulatedTokens / stats.totalRunTime) * 1000,
    tps_per_batch: avgTpotMs > 0 ? 1000 / avgTpotMs : 0,
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

  // 6. 评分 (使用统一的六维评分器)
  const scoreInput: ScoreInput = {
    ttft: stats.ttft,
    tpot: stats.avgTpot / 1000.0,
    tps: throughput.tokens_per_second,
    tpsPerChip: throughput.tps_per_chip,
    mfu: stats.dynamicMfu,
    mbu: stats.dynamicMbu,
    memoryUsedGB: totalMemoryGB,
    memoryCapacityGB: chipCapacityGB,
    prefillCommLatency: stats.prefill.commTime,
    prefillComputeLatency: stats.prefill.computeTime,
    decodeCommLatency: stats.decode.commTime,
    decodeComputeLatency: stats.decode.computeTime,
  };
  const scoreResult = calculateScores(scoreInput);
  const score: OverallScore = {
    latency_score: scoreResult.latencyScore,
    throughput_score: scoreResult.throughputScore,
    efficiency_score: scoreResult.efficiencyScore,
    balance_score: scoreResult.balanceScore,
    memory_score: scoreResult.memoryScore,
    communication_score: scoreResult.communicationScore,
    overall_score: scoreResult.overallScore,
  };

  // 7. 检查是否可行
  const is_feasible = stats.ttft > 0 && stats.ttft < Infinity;

  return {
    plan: {
      plan_id: `${parallelism.tp > 1 ? `tp${parallelism.tp}` : ''}${parallelism.pp > 1 ? `_pp${parallelism.pp}` : ''}${parallelism.ep > 1 ? `_ep${parallelism.ep}` : ''}${parallelism.dp > 1 ? `_dp${parallelism.dp}` : ''}`.replace(/^_/, '') || 'single',
      parallelism,
      total_chips: stats.totalChips || (parallelism.dp * parallelism.tp * parallelism.pp),
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
