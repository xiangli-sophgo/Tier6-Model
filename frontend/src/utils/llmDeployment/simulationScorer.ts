/**
 * 仿真评分计算器
 *
 * 基于仿真结果计算评分，与 planAnalyzer.ts 使用相同的评分逻辑
 * 支持公式计算与仿真结果的对比分析
 */

import {
  SimulationStats,
  SimulationScoreResult,
  FormulaVsSimComparison,
} from './simulation/types'
import {
  PlanAnalysisResult,
  ScoreWeights,
  DEFAULT_SCORE_WEIGHTS,
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
} from './types'
import { estimatePrefillMFU } from './latencyEstimator'

/**
 * 基于仿真结果计算评分
 *
 * 评分逻辑与 planAnalyzer.calculateOverallScore 保持一致：
 * - 延迟评分: TTFT < 100ms 满分, > 1000ms 零分
 * - 吞吐评分: MFU > 50% 满分
 * - 效率评分: (MFU + MBU) / 2
 * - 均衡评分: 1 - 气泡比
 */
export function calculateSimulationScore(
  stats: SimulationStats,
  weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS
): SimulationScoreResult {
  // 延迟评分 (TTFT < 100ms 满分, > 1000ms 零分)
  const ttft = stats.ttft
  const latencyScore = Math.max(0, Math.min(100, 100 - (ttft - 100) / 9))

  // 吞吐评分 (MFU > 50% 满分)
  const mfu = stats.dynamicMfu
  const throughputScore = Math.min(100, mfu * 200)

  // 效率评分 (综合 MFU 和 MBU)
  const avgUtilization = (stats.dynamicMfu + stats.dynamicMbu) / 2
  const efficiencyScore = avgUtilization * 100

  // 均衡评分 (基于气泡比)
  const bubbleRatio = stats.maxPPBubbleRatio
  const balanceScore = (1 - bubbleRatio) * 100

  // 综合评分 (加权平均)
  const overallScore =
    latencyScore * weights.latency +
    throughputScore * weights.throughput +
    efficiencyScore * weights.efficiency +
    balanceScore * weights.balance

  return {
    latency_score: latencyScore,
    throughput_score: throughputScore,
    efficiency_score: efficiencyScore,
    balance_score: balanceScore,
    overall_score: overallScore,
    raw: {
      ttft_ms: stats.ttft,
      avg_tpot_ms: stats.avgTpot,
      dynamic_mfu: stats.dynamicMfu,
      dynamic_mbu: stats.dynamicMbu,
      pp_bubble_ratio: stats.maxPPBubbleRatio,
    },
  }
}

/**
 * 计算偏差百分比
 */
function calcDeviation(simValue: number, formulaValue: number): number {
  if (formulaValue === 0) {
    return simValue === 0 ? 0 : 100
  }
  return ((simValue - formulaValue) / formulaValue) * 100
}

/**
 * 对比公式计算与仿真结果
 *
 * 注意: MFU 对比使用 Prefill 阶段的值 (仿真计算的是 Prefill MFU)
 * 如果需要计算 Prefill MFU，需要传入模型和硬件配置
 */
export function compareFormulaAndSimulation(
  formulaResult: PlanAnalysisResult,
  simulationStats: SimulationStats,
  weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
  // 可选: 用于计算 Prefill MFU 的配置
  config?: {
    model: LLMModelConfig
    inference: InferenceConfig
    parallelism: ParallelismStrategy
    hardware: HardwareConfig
  }
): FormulaVsSimComparison {
  // 计算仿真评分
  const simScore = calculateSimulationScore(simulationStats, weights)

  // 计算公式的 Prefill MFU (与仿真同阶段对比)
  // 如果没有提供配置，则使用原来的 Decode MFU (会有较大偏差)
  let formulaMfu = formulaResult.throughput.model_flops_utilization
  if (config) {
    formulaMfu = estimatePrefillMFU(
      config.model,
      config.inference,
      config.parallelism,
      config.hardware
    )
  }

  // 从公式结果提取关键指标
  const formula = {
    ttft_ms: formulaResult.latency.prefill_total_latency_ms,
    tpot_ms: formulaResult.latency.decode_per_token_latency_ms,
    mfu: formulaMfu,  // 使用 Prefill MFU
    mbu: formulaResult.throughput.memory_bandwidth_utilization,
    score: formulaResult.score.overall_score,
  }

  // 仿真结果
  const simulation = {
    ttft_ms: simulationStats.ttft,
    tpot_ms: simulationStats.avgTpot,
    mfu: simulationStats.dynamicMfu,
    mbu: simulationStats.dynamicMbu,
    score: simScore.overall_score,
  }

  // 计算偏差
  const deviation = {
    ttft_pct: calcDeviation(simulation.ttft_ms, formula.ttft_ms),
    tpot_pct: calcDeviation(simulation.tpot_ms, formula.tpot_ms),
    mfu_pct: calcDeviation(simulation.mfu, formula.mfu),
    mbu_pct: calcDeviation(simulation.mbu, formula.mbu),
    score_pct: calcDeviation(simulation.score, formula.score),
  }

  return {
    formula,
    simulation,
    deviation,
  }
}

/**
 * 格式化偏差显示
 */
export function formatDeviation(pct: number): string {
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${pct.toFixed(1)}%`
}

/**
 * 判断偏差是否显著
 * @param pct 偏差百分比
 * @param threshold 阈值，默认 10%
 */
export function isSignificantDeviation(pct: number, threshold = 10): boolean {
  return Math.abs(pct) >= threshold
}
