/**
 * 结果适配器
 *
 * 将后端 SimulationResult 转换为前端 PlanAnalysisResult 格式
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
} from './types';
import { analyzeMemory } from './modelCalculator';
import { analyzeCommunication } from './commCalculator';
import { calculateSimulationScore } from './simulationScorer';

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

  // 1. 显存分析（使用前端计算器）
  const memory: MemoryAnalysis = analyzeMemory(
    model,
    inference,
    parallelism,
    hardware.chip.memory_gb
  );

  // 2. 通信分析（使用前端计算器）
  const communication: CommunicationAnalysis = analyzeCommunication(
    model,
    inference,
    parallelism,
    hardware
  );

  // 3. 延迟分析（从后端 stats 提取）
  const latency: LatencyAnalysis = {
    prefill_compute_latency_ms: stats.prefill.computeTime,
    prefill_comm_latency_ms: stats.prefill.commTime,
    prefill_total_latency_ms: stats.ttft,
    prefill_flops: stats.prefillFlops,
    decode_compute_latency_ms: stats.decode.computeTime,
    decode_memory_latency_ms: 0, // 后端不单独区分 memory
    decode_comm_latency_ms: stats.decode.commTime,
    decode_per_token_latency_ms: stats.avgTpot,
    end_to_end_latency_ms: stats.totalRunTime,
    pipeline_bubble_ratio: stats.maxPpBubbleRatio,
    bottleneck_type: 'compute', // 简化
    bottleneck_details: '基于后端仿真结果',
  };

  // 4. 吞吐量分析（从后端 stats 提取）
  const throughput: ThroughputAnalysis = {
    tokens_per_second: (stats.simulatedTokens / stats.totalRunTime) * 1000,
    tps_per_batch: stats.avgTpot > 0 ? 1000 / stats.avgTpot : 0,
    tps_per_chip: 0, // 暂不计算
    requests_per_second: 0, // 暂不计算
    model_flops_utilization: stats.dynamicMfu,
    memory_bandwidth_utilization: stats.dynamicMbu,
    theoretical_max_throughput: 0, // 暂不计算
  };

  // 5. 利用率分析（从后端 stats 提取）
  const utilization: UtilizationAnalysis = {
    compute_utilization: stats.dynamicMfu,
    memory_utilization: memory.memory_utilization,
    network_utilization: 0, // 暂不计算
    load_balance_score: 1 - stats.maxPpBubbleRatio,
  };

  // 6. 评分（使用仿真评分函数）
  const scoreResult = calculateSimulationScore(stats);
  const score: OverallScore = {
    latency_score: scoreResult.latency_score,
    throughput_score: scoreResult.throughput_score,
    efficiency_score: scoreResult.efficiency_score,
    balance_score: scoreResult.balance_score,
    overall_score: scoreResult.overall_score,
  };

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
    suggestions: [], // 暂不生成建议
    is_feasible,
    infeasibility_reason: is_feasible ? undefined : '后端模拟失败',
  };
}
