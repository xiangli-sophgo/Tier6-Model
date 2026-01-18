/**
 * LLM 部署分析系统 - 方案搜索器
 *
 * 自动搜索满足约束的最优部署方案
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  SearchConstraints,
  PlanAnalysisResult,
  PlanSearchResult,
  SearchStats,
  OptimizationTarget,
  ScoreWeights,
} from './types';
import { analyzePlan, quickAnalyze, checkFeasibility } from './planAnalyzer';

// ============================================
// 搜索空间生成
// ============================================

/**
 * 生成候选并行度值
 */
function generateCandidates(
  model: LLMModelConfig,
  hardware: HardwareConfig,
  constraints: SearchConstraints
): {
  dpCandidates: number[];
  tpCandidates: number[];
  ppCandidates: number[];
  epCandidates: number[];
  spCandidates: number[];
  moeTpCandidates: number[];
  mlaTpCandidates: number[];
  mlaDpCandidates: number[];
} {
  const maxChips = constraints.max_chips ?? hardware.node.chips_per_node * hardware.cluster.num_nodes;
  const chipsPerNode = hardware.node.chips_per_node;

  // TP 候选值: 1, 2, 4, 8... 且能整除 attention_heads
  const tpCandidates: number[] = [];
  for (let tp = 1; tp <= Math.min(maxChips, 16); tp *= 2) {
    if (model.num_attention_heads % tp === 0 && model.num_kv_heads % tp === 0) {
      // 如果约束 TP 在节点内
      if (constraints.tp_within_node && tp > chipsPerNode) continue;
      tpCandidates.push(tp);
    }
  }

  // PP 候选值: 1, 2, 4... 且能整除 layers
  const ppCandidates: number[] = [];
  for (let pp = 1; pp <= Math.min(maxChips, 16); pp *= 2) {
    if (model.num_layers % pp === 0) {
      ppCandidates.push(pp);
    }
  }

  // EP 候选值 (仅 MoE)
  const epCandidates: number[] = [1];
  if (model.model_type === 'moe' && model.moe_config) {
    for (let ep = 2; ep <= Math.min(model.moe_config.num_experts, 16); ep *= 2) {
      if (model.moe_config.num_experts % ep === 0) {
        epCandidates.push(ep);
      }
    }
  }

  // MoE TP 候选值 (仅 MoE，专家内张量并行度，通常 ≤ 8)
  const moeTpCandidates: number[] = [1];
  if (model.model_type === 'moe' && model.moe_config) {
    // moe_tp 通常较小 (1, 2, 4, 8)，且 moe_tp * ep 需等于 tp * dp
    for (let moeTp = 2; moeTp <= Math.min(8, chipsPerNode); moeTp *= 2) {
      moeTpCandidates.push(moeTp);
    }
  }

  // SP 候选值 (通常与 TP 相同或为 1)
  const spCandidates: number[] = [1];

  // DP 候选值: 根据剩余芯片数动态计算
  const dpCandidates: number[] = [];
  for (let dp = 1; dp <= maxChips; dp *= 2) {
    dpCandidates.push(dp);
  }

  // MLA TP/DP 候选值 (仅 MLA 模型，如 DeepSeek V3/R1)
  // 注意: mla_tp/mla_dp 放在 MLAConfig 中，默认使用全局 tp/dp
  // 这里生成候选值供手动配置时参考，搜索时默认使用全局值
  const mlaTpCandidates: number[] = [];
  const mlaDpCandidates: number[] = [];
  if (model.attention_type === 'mla' && model.mla_config) {
    // mla_tp 必须能整除 attention_heads
    for (let mlaTp = 1; mlaTp <= Math.min(16, chipsPerNode); mlaTp *= 2) {
      if (model.num_attention_heads % mlaTp === 0) {
        mlaTpCandidates.push(mlaTp);
      }
    }
    // mla_dp 基于芯片数约束
    for (let mlaDp = 1; mlaDp <= maxChips; mlaDp *= 2) {
      mlaDpCandidates.push(mlaDp);
    }
  }

  return { dpCandidates, tpCandidates, ppCandidates, epCandidates, spCandidates, moeTpCandidates, mlaTpCandidates, mlaDpCandidates };
}

/**
 * 生成所有候选方案
 */
function generateAllPlans(
  candidates: ReturnType<typeof generateCandidates>,
  maxChips: number,
  isMoE: boolean = false
): ParallelismStrategy[] {
  const plans: ParallelismStrategy[] = [];

  for (const dp of candidates.dpCandidates) {
    for (const tp of candidates.tpCandidates) {
      for (const pp of candidates.ppCandidates) {
        for (const ep of candidates.epCandidates) {
          for (const sp of candidates.spCandidates) {
            // 对于 MoE 模型，需要遍历 moe_tp
            const moeTpList = isMoE ? candidates.moeTpCandidates : [1];
            for (const moe_tp of moeTpList) {
              // Attention 部分芯片数: tp * dp
              // MoE 部分芯片数: moe_tp * ep
              // 两者需要一致才能正确部署
              const attnChips = tp * dp;
              const moeChips = moe_tp * ep;

              // MoE 模型: 检查 Attention 和 MoE 芯片数一致性
              if (isMoE && ep > 1 && attnChips !== moeChips) {
                continue;
              }

              const effectiveChips = attnChips * pp;
              if (effectiveChips <= maxChips && effectiveChips >= 1) {
                plans.push({ dp, tp, pp, ep, sp, moe_tp: isMoE ? moe_tp : undefined });
              }
            }
          }
        }
      }
    }
  }

  return plans;
}

// ============================================
// 早期剪枝
// ============================================

/**
 * 早期剪枝检查
 */
function shouldPrune(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  constraints: SearchConstraints
): { prune: boolean; reason?: string } {
  // 芯片数约束
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  if (constraints.max_chips && totalChips > constraints.max_chips) {
    return { prune: true, reason: '超出最大芯片数' };
  }

  // 基本可行性检查
  const feasibility = checkFeasibility(model, inference, parallelism, hardware);
  if (!feasibility.isFeasible) {
    return { prune: true, reason: feasibility.reason };
  }

  return { prune: false };
}

// ============================================
// Pareto 前沿
// ============================================

/**
 * 判断方案 A 是否 Pareto 支配方案 B
 * (在所有指标上不差，且至少一个指标更好)
 */
function dominates(a: PlanAnalysisResult, b: PlanAnalysisResult): boolean {
  // 比较维度: 延迟、吞吐、效率
  const aMetrics = [
    -a.latency.end_to_end_latency_ms, // 负值，因为越小越好
    a.throughput.tokens_per_second,
    a.throughput.model_flops_utilization,
  ];

  const bMetrics = [
    -b.latency.end_to_end_latency_ms,
    b.throughput.tokens_per_second,
    b.throughput.model_flops_utilization,
  ];

  let atLeastOneBetter = false;
  for (let i = 0; i < aMetrics.length; i++) {
    if (aMetrics[i] < bMetrics[i]) {
      return false; // A 在某指标上更差
    }
    if (aMetrics[i] > bMetrics[i]) {
      atLeastOneBetter = true;
    }
  }

  return atLeastOneBetter;
}

/**
 * 提取 Pareto 前沿
 */
function extractParetoFrontier(results: PlanAnalysisResult[]): PlanAnalysisResult[] {
  const frontier: PlanAnalysisResult[] = [];

  for (const candidate of results) {
    if (!candidate.is_feasible) continue;

    let dominated = false;
    const toRemove: number[] = [];

    for (let i = 0; i < frontier.length; i++) {
      if (dominates(frontier[i], candidate)) {
        dominated = true;
        break;
      }
      if (dominates(candidate, frontier[i])) {
        toRemove.push(i);
      }
    }

    if (!dominated) {
      // 移除被 candidate 支配的方案
      for (let i = toRemove.length - 1; i >= 0; i--) {
        frontier.splice(toRemove[i], 1);
      }
      frontier.push(candidate);
    }
  }

  return frontier;
}

// ============================================
// 排序和选择
// ============================================

/**
 * 根据优化目标排序
 */
function sortByTarget(
  results: PlanAnalysisResult[],
  target: OptimizationTarget
): PlanAnalysisResult[] {
  const sorted = [...results].filter(r => r.is_feasible);

  switch (target) {
    case 'latency':
      sorted.sort((a, b) => a.latency.end_to_end_latency_ms - b.latency.end_to_end_latency_ms);
      break;
    case 'throughput':
      sorted.sort((a, b) => b.throughput.tokens_per_second - a.throughput.tokens_per_second);
      break;
    case 'efficiency':
      sorted.sort((a, b) => b.throughput.model_flops_utilization - a.throughput.model_flops_utilization);
      break;
    case 'balanced':
    default:
      sorted.sort((a, b) => b.score.overall_score - a.score.overall_score);
      break;
  }

  return sorted;
}

/**
 * 应用约束过滤
 */
function applyConstraints(
  results: PlanAnalysisResult[],
  constraints: SearchConstraints
): PlanAnalysisResult[] {
  return results.filter(r => {
    if (!r.is_feasible) return false;

    if (constraints.max_latency_ms && r.latency.end_to_end_latency_ms > constraints.max_latency_ms) {
      return false;
    }

    if (constraints.min_throughput && r.throughput.tokens_per_second < constraints.min_throughput) {
      return false;
    }

    if (constraints.max_memory_ratio && r.memory.memory_utilization > constraints.max_memory_ratio) {
      return false;
    }

    return true;
  });
}

// ============================================
// 主搜索函数
// ============================================

/**
 * 搜索最优部署方案
 * @param weights 自定义评分权重，用于计算综合评分
 */
export function searchOptimalPlan(
  model: LLMModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  constraints: SearchConstraints = {},
  _target: OptimizationTarget = 'balanced',
  topK: number = 10,
  weights?: ScoreWeights
): PlanSearchResult {
  const startTime = performance.now();

  // 生成候选值
  const candidates = generateCandidates(model, hardware, constraints);

  // 生成所有方案组合
  const maxChips = constraints.max_chips ?? hardware.node.chips_per_node * hardware.cluster.num_nodes;
  const isMoE = model.model_type === 'moe';
  const allPlans = generateAllPlans(candidates, maxChips, isMoE);

  let evaluatedCount = 0;
  let prunedCount = 0;
  const results: PlanAnalysisResult[] = [];

  // 评估每个方案
  for (const parallelism of allPlans) {
    // 早期剪枝
    const pruneResult = shouldPrune(model, inference, parallelism, hardware, constraints);
    if (pruneResult.prune) {
      prunedCount++;
      continue;
    }

    // 完整分析 (传入自定义权重)
    evaluatedCount++;
    const result = analyzePlan(model, inference, parallelism, hardware, undefined, weights);
    results.push(result);
  }

  // 应用约束过滤
  const filteredResults = applyConstraints(results, constraints);

  // 排序 - 现在统一按 overall_score 排序（权重已在评分时应用）
  const sortedResults = sortByTarget(filteredResults, 'balanced');

  // 提取 Pareto 前沿
  const paretoFrontier = extractParetoFrontier(filteredResults);

  // 搜索统计
  const searchTimeMs = performance.now() - startTime;
  const stats: SearchStats = {
    evaluated_count: evaluatedCount,
    pruned_count: prunedCount,
    feasible_count: filteredResults.length,
    search_time_ms: searchTimeMs,
  };

  // 结果
  if (sortedResults.length === 0) {
    throw new Error('未找到可行的部署方案，请检查模型配置和硬件约束');
  }

  return {
    optimal_plan: sortedResults[0],
    top_k_plans: sortedResults.slice(0, topK),
    pareto_frontier: paretoFrontier,
    search_stats: stats,
  };
}

// ============================================
// 快速搜索
// ============================================

/**
 * 快速搜索 - 使用启发式减少搜索空间
 */
export function quickSearch(
  model: LLMModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  target: OptimizationTarget = 'balanced'
): PlanAnalysisResult {
  // 启发式: 常见的有效配置
  const heuristicPlans: ParallelismStrategy[] = [];

  // 单节点配置
  const singleNodeConfigs = [
    { dp: 1, tp: 8, pp: 1, ep: 1, sp: 1 },
    { dp: 1, tp: 4, pp: 2, ep: 1, sp: 1 },
    { dp: 2, tp: 4, pp: 1, ep: 1, sp: 1 },
    { dp: 1, tp: 2, pp: 4, ep: 1, sp: 1 },
    { dp: 4, tp: 2, pp: 1, ep: 1, sp: 1 },
  ];

  // 多节点配置
  const multiNodeConfigs = [
    { dp: 2, tp: 8, pp: 1, ep: 1, sp: 1 },
    { dp: 1, tp: 8, pp: 2, ep: 1, sp: 1 },
    { dp: 4, tp: 8, pp: 1, ep: 1, sp: 1 },
    { dp: 2, tp: 8, pp: 2, ep: 1, sp: 1 },
  ];

  // MoE 配置
  const moeConfigs = model.model_type === 'moe' && model.moe_config ? [
    { dp: 1, tp: 8, pp: 1, ep: 2, sp: 1 },
    { dp: 1, tp: 4, pp: 1, ep: 4, sp: 1 },
    { dp: 2, tp: 4, pp: 1, ep: 4, sp: 1 },
  ] : [];

  heuristicPlans.push(...singleNodeConfigs, ...multiNodeConfigs, ...moeConfigs);

  // 过滤并评估
  let bestResult: PlanAnalysisResult | null = null;
  let bestScore = -Infinity;

  for (const parallelism of heuristicPlans) {
    const quick = quickAnalyze(model, inference, parallelism, hardware);
    if (!quick.isFeasible) continue;

    const result = analyzePlan(model, inference, parallelism, hardware);
    if (!result.is_feasible) continue;

    let score: number;
    switch (target) {
      case 'latency':
        score = -result.latency.end_to_end_latency_ms;
        break;
      case 'throughput':
        score = result.throughput.tokens_per_second;
        break;
      case 'efficiency':
        score = result.throughput.model_flops_utilization;
        break;
      default:
        score = result.score.overall_score;
    }

    if (score > bestScore) {
      bestScore = score;
      bestResult = result;
    }
  }

  if (!bestResult) {
    throw new Error('快速搜索未找到可行方案，请使用完整搜索');
  }

  return bestResult;
}

// ============================================
// 指定芯片数搜索
// ============================================

/**
 * 给定最大芯片数，搜索最优配置
 * @param weights 自定义评分权重
 *
 * 注意：由于并行度参数（DP/TP/PP/EP）通常是 2 的幂次，
 * 实际使用的芯片数可能小于 targetChips，但会尽量接近。
 */
export function searchWithFixedChips(
  model: LLMModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  targetChips: number,
  target: OptimizationTarget = 'balanced',
  weights?: ScoreWeights
): PlanSearchResult {
  // 约束最大芯片数
  const constraints: SearchConstraints = {
    max_chips: targetChips,
  };

  // 搜索 (传入自定义权重)
  const result = searchOptimalPlan(model, inference, hardware, constraints, target, 10, weights);

  // 返回所有不超过 targetChips 的可行方案（已按评分排序）
  // 不再要求精确匹配芯片数，因为并行度参数组合可能无法精确达到目标值
  return result;
}

// ============================================
// 渐进式搜索
// ============================================

/**
 * 渐进式搜索 - 从最小芯片数开始，逐步增加直到满足性能要求
 */
export function progressiveSearch(
  model: LLMModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  performanceRequirement: {
    maxLatencyMs?: number;
    minThroughput?: number;
  }
): {
  minChipsRequired: number;
  optimalPlan: PlanAnalysisResult;
  chipScalingCurve: Array<{ chips: number; latency: number; throughput: number }>;
} {
  const maxChips = hardware.node.chips_per_node * hardware.cluster.num_nodes;
  const scalingCurve: Array<{ chips: number; latency: number; throughput: number }> = [];

  let minChipsRequired = maxChips;
  let optimalPlan: PlanAnalysisResult | null = null;

  // 按芯片数递增搜索
  for (let chips = 1; chips <= maxChips; chips *= 2) {
    try {
      const result = searchWithFixedChips(model, inference, hardware, chips, 'balanced');
      const best = result.optimal_plan;

      scalingCurve.push({
        chips,
        latency: best.latency.end_to_end_latency_ms,
        throughput: best.throughput.tokens_per_second,
      });

      // 检查是否满足性能要求
      const meetsLatency = !performanceRequirement.maxLatencyMs ||
        best.latency.end_to_end_latency_ms <= performanceRequirement.maxLatencyMs;
      const meetsThroughput = !performanceRequirement.minThroughput ||
        best.throughput.tokens_per_second >= performanceRequirement.minThroughput;

      if (meetsLatency && meetsThroughput && chips < minChipsRequired) {
        minChipsRequired = chips;
        optimalPlan = best;
      }
    } catch {
      // 该芯片数无可行方案，继续尝试更多芯片
      continue;
    }
  }

  if (!optimalPlan) {
    throw new Error('即使使用最大芯片数也无法满足性能要求');
  }

  return {
    minChipsRequired,
    optimalPlan,
    chipScalingCurve: scalingCurve,
  };
}
