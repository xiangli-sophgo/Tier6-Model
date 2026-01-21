/**
 * 仿真评分计算器
 *
 * 基于后端仿真结果计算评分
 */

import {
  SimulationStats,
  SimulationScoreResult,
  ScoreWeights,
  DEFAULT_SCORE_WEIGHTS,
} from './types';

/**
 * 基于仿真结果计算评分
 *
 * 评分逻辑：
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
    raw: {
      ttft_ms: stats.ttft,
      avg_tpot_ms: stats.avgTpot,
      dynamic_mfu: stats.dynamicMfu,
      dynamic_mbu: stats.dynamicMbu,
      pp_bubble_ratio: stats.maxPpBubbleRatio,
    },
  };
}

/**
 * 格式化偏差显示
 */
export function formatDeviation(pct: number): string {
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}%`;
}

/**
 * 判断偏差是否显著
 * @param pct 偏差百分比
 * @param threshold 阈值，默认 10%
 */
export function isSignificantDeviation(pct: number, threshold = 10): boolean {
  return Math.abs(pct) >= threshold;
}
