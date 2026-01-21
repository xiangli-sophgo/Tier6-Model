/**
 * 方案搜索器 - 后端版本
 *
 * 生成候选并行策略，调用后端评估
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  SimulationResult,
} from './types';
import { batchSimulate } from './backendApi';

/**
 * 搜索选项
 */
export interface SearchOptions {
  /** 最大搜索方案数 */
  maxPlans?: number;
  /** 候选生成完成回调 */
  onCandidatesGenerated?: (totalCandidates: number) => void;
  /** 后端评估进度回调 */
  onProgress?: (current: number, total: number) => void;
  /** 单个结果完成回调（用于实时显示） */
  onResultReady?: (result: SearchResult | InfeasibleResult, index: number, isFeasible: boolean) => void;
  /** 取消信号 */
  abortSignal?: AbortSignal;
}

/**
 * 搜索结果（可行方案）
 */
export interface SearchResult {
  parallelism: ParallelismStrategy;
  simulation: SimulationResult;
  score: number;
}

/**
 * 不可行方案
 */
export interface InfeasibleResult {
  parallelism: ParallelismStrategy;
  reason: string;
}

/**
 * 完整搜索结果
 */
export interface FullSearchResult {
  /** 可行方案（按评分排序） */
  feasible: SearchResult[];
  /** 不可行方案 */
  infeasible: InfeasibleResult[];
}

/**
 * 获取一个数的所有因子（从小到大）
 */
function getDivisors(n: number, max: number): number[] {
  const divisors: number[] = [];
  for (let i = 1; i <= Math.min(n, max); i++) {
    if (n % i === 0) {
      divisors.push(i);
    }
  }
  return divisors;
}

/**
 * 生成候选并行策略
 *
 * @param maxChips - 最大芯片数，枚举所有 ≤ maxChips 的方案
 */
function generateCandidateStrategies(
  model: LLMModelConfig,
  hardware: HardwareConfig,
  maxChips: number
): ParallelismStrategy[] {
  const candidates: ParallelismStrategy[] = [];

  // 约束条件 (对齐 DS_TPU: 只有 dp, tp, ep, moe_tp)
  const maxTP = Math.min(128, model.num_attention_heads, hardware.node.chips_per_node);
  const maxEP = model.model_type === 'moe' && model.moe_config ? model.moe_config.num_experts : 1;

  // 获取所有有效的因子
  const tpCandidates = getDivisors(model.num_attention_heads, maxTP).filter(
    (tp) => model.num_kv_heads % tp === 0
  );

  // 处理 MoE 模型的 EP 和 moe_tp
  const isMoE = model.model_type === 'moe' && model.moe_config;
  const epCandidates = isMoE ? getDivisors(model.moe_config!.num_experts, maxEP) : [1];

  // 枚举所有可能的组合 (DS_TPU 模式: 无 PP)
  for (const tp of tpCandidates) {
    for (const ep of epCandidates) {
      // MoE 模型需要枚举 moe_tp
      const moeTPCandidates = isMoE ? getDivisors(model.num_attention_heads, maxTP) : [1];

      for (const moe_tp of moeTPCandidates) {
        // MoE 约束验证：dp * tp = moe_tp * ep
        if (isMoE) {
          // 从约束反推最小 dp: dp = (moe_tp * ep) / tp
          if ((moe_tp * ep) % tp !== 0) continue; // 无法整除，跳过

          const baseDp = (moe_tp * ep) / tp;
          const baseChips = baseDp * tp * ep;

          // 枚举所有 dp 的倍数，直到超过 maxChips
          for (let dpMultiplier = 1; baseChips * dpMultiplier <= maxChips; dpMultiplier++) {
            const dp = baseDp * dpMultiplier;
            const totalChips = dp * tp * ep;

            if (dp < 1) continue;
            if (totalChips > maxChips) break;

            candidates.push({
              dp,
              tp,
              pp: 1, // DS_TPU 无 PP
              ep,
              sp: 1, // DS_TPU 默认无 SP
              moe_tp,
            });
          }
        } else {
          // 非 MoE 模型：枚举所有 dp，使得 dp * tp ≤ maxChips
          for (let dp = 1; dp * tp <= maxChips; dp++) {
            candidates.push({
              dp,
              tp,
              pp: 1, // DS_TPU 无 PP
              ep: 1, // 非 MoE
              sp: 1, // DS_TPU 默认无 SP
            });
          }
        }
      }
    }
  }

  // 去重并排序（按芯片数升序）
  const unique = Array.from(
    new Map(candidates.map((c) => [JSON.stringify(c), c])).values()
  );

  unique.sort((a, b) => {
    const chipsA = a.dp * a.tp * a.pp * a.ep;
    const chipsB = b.dp * b.tp * b.pp * b.ep;
    return chipsA - chipsB;
  });

  return unique;
}

/**
 * 计算方案评分（对齐 DS_TPU）
 *
 * DS_TPU 的核心优化目标：TPS(CHIP) = 每芯片吞吐量
 * 这衡量的是单芯片效率，确保资源利用最优
 */
function calculateScore(
  stats: SimulationResult['stats'],
  parallelism: ParallelismStrategy
): number {
  // 如果模拟失败，返回 0 分
  if (stats.ttft === Infinity || stats.ttft <= 0) {
    return 0;
  }

  // 计算总芯片数
  const totalChips = parallelism.dp * parallelism.tp * parallelism.ep;
  if (totalChips <= 0) return 0;

  // 计算总吞吐量 (tokens/s)
  // avgTpot 是每 token 的延迟(ms)，所以 TPS = 1000 / avgTpot
  const totalTPS = stats.avgTpot > 0 ? (1000 / stats.avgTpot) * parallelism.dp : 0;

  // 主要优化目标：TPS per chip (单芯片吞吐量)
  const tpsPerChip = totalTPS / totalChips;

  // 次要优化目标：MFU (模型 FLOPs 利用率)
  // 当 TPS per chip 相近时，选择 MFU 更高的方案
  const mfu = stats.dynamicMfu;

  // 综合评分：TPS per chip 为主，MFU 为辅
  // TPS per chip 占 90%，MFU 占 10%
  const score = tpsPerChip * 0.9 + mfu * 1000 * 0.1;

  return score;
}

/**
 * 搜索最优方案（在最大芯片数约束下）
 *
 * 枚举所有使用 ≤ maxChips 个芯片的并行策略，按 TPS per Chip 排序
 * 返回完整结果，包括可行方案和不可行方案
 */
export async function searchWithFixedChips(
  topology: any,
  model: LLMModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  maxChips: number,
  options: SearchOptions = {}
): Promise<FullSearchResult> {
  const { maxPlans = 10, onCandidatesGenerated, onProgress, onResultReady, abortSignal } = options;

  // 生成候选策略（枚举所有 ≤ maxChips 的方案）
  const candidates = generateCandidateStrategies(model, hardware, maxChips);

  if (candidates.length === 0) {
    console.warn('未找到满足约束的并行策略');
    onCandidatesGenerated?.(0);
    return { feasible: [], infeasible: [] };
  }

  // 通知候选生成完成（评估所有候选，不再限制数量）
  onCandidatesGenerated?.(candidates.length);

  // 构建配置数组（评估所有候选方案）
  const configs = candidates.map((parallelism) => ({
    topology,
    model,
    inference,
    parallelism,
    hardware,
  }));

  // 收集结果（用于实时回调）
  const feasible: SearchResult[] = [];
  const infeasible: InfeasibleResult[] = [];

  // 批量模拟（支持取消和实时回调）
  await batchSimulate(
    configs,
    (current, total) => {
      onProgress?.(current, total);
    },
    {
      concurrency: 5,
      abortSignal,
      onProgress: (_completed, _total, result, index) => {
        // 实时处理每个结果
        const parallelism = candidates[index];
        const score = calculateScore(result.stats, parallelism);

        if (score > 0) {
          const searchResult: SearchResult = { parallelism, simulation: result, score };
          feasible.push(searchResult);
          onResultReady?.(searchResult, index, true);
        } else {
          const reason = result.stats.errorReason || '模拟失败（原因未知）';
          const infeasibleResult: InfeasibleResult = { parallelism, reason };
          infeasible.push(infeasibleResult);
          onResultReady?.(infeasibleResult, index, false);
        }
      },
    }
  );

  // 可行方案按评分降序排序
  feasible.sort((a, b) => b.score - a.score);

  // 返回完整结果
  return {
    feasible: feasible.slice(0, maxPlans),
    infeasible,
  };
}

/**
 * 渐进式搜索（已废弃）
 *
 * @deprecated 现在 searchWithFixedChips 会自动枚举所有 ≤ maxChips 的方案，无需渐进式搜索
 * 直接使用 searchWithFixedChips(topology, model, inference, hardware, maxChips) 即可
 */
export async function progressiveSearch(
  topology: any,
  model: LLMModelConfig,
  inference: InferenceConfig,
  hardware: HardwareConfig,
  options: SearchOptions = {}
): Promise<FullSearchResult> {
  const maxChips = hardware.node.chips_per_node * hardware.cluster.num_nodes;

  // 直接调用 searchWithFixedChips，它会枚举所有 ≤ maxChips 的方案
  return searchWithFixedChips(
    topology,
    model,
    inference,
    hardware,
    maxChips,
    options
  );
}
