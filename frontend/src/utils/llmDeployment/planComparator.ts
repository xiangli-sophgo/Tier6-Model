/**
 * LLM 部署分析系统 - 方案对比器
 *
 * 多方案对比分析
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  PlanAnalysisResult,
  PlanComparisonResult,
  MetricComparison,
} from './types';
import { analyzePlan } from './planAnalyzer';

// ============================================
// 指标提取
// ============================================

/**
 * 定义可对比的指标
 */
interface MetricDefinition {
  name: string;
  unit: string;
  extract: (result: PlanAnalysisResult) => number;
  higherIsBetter: boolean;
}

const METRIC_DEFINITIONS: MetricDefinition[] = [
  // 延迟指标
  {
    name: 'TTFT (首Token延迟)',
    unit: 'ms',
    extract: r => r.latency.prefill_total_latency_ms,
    higherIsBetter: false,
  },
  {
    name: 'TPOT (每Token延迟)',
    unit: 'ms',
    extract: r => r.latency.decode_per_token_latency_ms,
    higherIsBetter: false,
  },
  {
    name: '端到端延迟',
    unit: 'ms',
    extract: r => r.latency.end_to_end_latency_ms,
    higherIsBetter: false,
  },

  // 吞吐指标
  {
    name: 'Token吞吐量',
    unit: 'tokens/s',
    extract: r => r.throughput.tokens_per_second,
    higherIsBetter: true,
  },
  {
    name: '请求吞吐量',
    unit: 'req/s',
    extract: r => r.throughput.requests_per_second,
    higherIsBetter: true,
  },

  // 效率指标
  {
    name: 'MFU (算力利用率)',
    unit: '%',
    extract: r => r.throughput.model_flops_utilization * 100,
    higherIsBetter: true,
  },
  {
    name: '显存利用率',
    unit: '%',
    extract: r => r.memory.memory_utilization * 100,
    higherIsBetter: false, // 过高有 OOM 风险
  },

  // 资源指标
  {
    name: '芯片数',
    unit: '个',
    extract: r => r.plan.total_chips,
    higherIsBetter: false,
  },
  {
    name: '每芯片显存',
    unit: 'GB',
    extract: r => r.memory.total_per_chip_gb,
    higherIsBetter: false,
  },

  // 通信指标
  {
    name: '总通信量',
    unit: 'GB',
    extract: r => r.communication.total_comm_volume_gb,
    higherIsBetter: false,
  },
  {
    name: '流水线气泡比',
    unit: '%',
    extract: r => r.latency.pipeline_bubble_ratio * 100,
    higherIsBetter: false,
  },

  // 评分
  {
    name: '综合评分',
    unit: '分',
    extract: r => r.score.overall_score,
    higherIsBetter: true,
  },
];

// ============================================
// 对比分析
// ============================================

/**
 * 对比多个方案
 */
export function comparePlans(plans: PlanAnalysisResult[]): PlanComparisonResult {
  if (plans.length === 0) {
    throw new Error('至少需要一个方案进行对比');
  }

  // 过滤可行方案
  const feasiblePlans = plans.filter(p => p.is_feasible);
  if (feasiblePlans.length === 0) {
    throw new Error('没有可行的方案可供对比');
  }

  // 生成指标对比
  const metrics: MetricComparison[] = METRIC_DEFINITIONS.map(def => {
    const values: Record<string, number> = {};
    let bestValue = def.higherIsBetter ? -Infinity : Infinity;
    let bestPlanId = '';

    for (const plan of feasiblePlans) {
      const value = def.extract(plan);
      values[plan.plan.plan_id] = value;

      if (def.higherIsBetter) {
        if (value > bestValue) {
          bestValue = value;
          bestPlanId = plan.plan.plan_id;
        }
      } else {
        if (value < bestValue) {
          bestValue = value;
          bestPlanId = plan.plan.plan_id;
        }
      }
    }

    return {
      metric_name: def.name,
      unit: def.unit,
      values,
      best_value: bestValue,
      best_plan_id: bestPlanId,
    };
  });

  // 找出各维度最优
  const overallBest = feasiblePlans.reduce((a, b) =>
    a.score.overall_score > b.score.overall_score ? a : b
  );

  const latencyBest = feasiblePlans.reduce((a, b) =>
    a.latency.end_to_end_latency_ms < b.latency.end_to_end_latency_ms ? a : b
  );

  const throughputBest = feasiblePlans.reduce((a, b) =>
    a.throughput.tokens_per_second > b.throughput.tokens_per_second ? a : b
  );

  return {
    plans: feasiblePlans,
    metrics,
    overall_best_plan_id: overallBest.plan.plan_id,
    latency_best_plan_id: latencyBest.plan.plan_id,
    throughput_best_plan_id: throughputBest.plan.plan_id,
  };
}

/**
 * 分析并对比多个并行策略
 */
export function analyzeAndCompare(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelismStrategies: ParallelismStrategy[],
  hardware: HardwareConfig
): PlanComparisonResult {
  const results = parallelismStrategies.map(p =>
    analyzePlan(model, inference, p, hardware)
  );

  return comparePlans(results);
}

// ============================================
// 对比报告生成
// ============================================

/**
 * 生成对比报告 (文本格式)
 */
export function generateComparisonReport(comparison: PlanComparisonResult): string {
  const lines: string[] = [];

  lines.push('='.repeat(60));
  lines.push('LLM 部署方案对比报告');
  lines.push('='.repeat(60));
  lines.push('');

  // 方案概览
  lines.push('## 方案概览');
  lines.push('');
  for (const plan of comparison.plans) {
    const p = plan.plan.parallelism;
    lines.push(`- ${plan.plan.plan_id}: DP=${p.dp}, TP=${p.tp}, PP=${p.pp}, EP=${p.ep}`);
    lines.push(`  芯片数: ${plan.plan.total_chips}, 综合评分: ${plan.score.overall_score.toFixed(1)}`);
  }
  lines.push('');

  // 指标对比表
  lines.push('## 关键指标对比');
  lines.push('');

  // 表头
  const planIds = comparison.plans.map(p => p.plan.plan_id);
  const header = ['指标', ...planIds, '最优'].join(' | ');
  lines.push(header);
  lines.push('-'.repeat(header.length));

  // 表体 - 只显示关键指标
  const keyMetrics = ['TTFT (首Token延迟)', 'Token吞吐量', 'MFU (算力利用率)', '综合评分'];
  for (const metric of comparison.metrics) {
    if (!keyMetrics.includes(metric.metric_name)) continue;

    const values = planIds.map(id => {
      const val = metric.values[id];
      return val !== undefined ? val.toFixed(2) : '-';
    });
    const row = [
      `${metric.metric_name} (${metric.unit})`,
      ...values,
      metric.best_plan_id,
    ].join(' | ');
    lines.push(row);
  }
  lines.push('');

  // 结论
  lines.push('## 结论');
  lines.push('');
  lines.push(`- 综合最优方案: ${comparison.overall_best_plan_id}`);
  lines.push(`- 延迟最优方案: ${comparison.latency_best_plan_id}`);
  lines.push(`- 吞吐最优方案: ${comparison.throughput_best_plan_id}`);
  lines.push('');

  // 优化建议
  const bestPlan = comparison.plans.find(p => p.plan.plan_id === comparison.overall_best_plan_id);
  if (bestPlan && bestPlan.suggestions.length > 0) {
    lines.push('## 最优方案优化建议');
    lines.push('');
    for (const suggestion of bestPlan.suggestions.slice(0, 3)) {
      lines.push(`- [优先级 ${suggestion.priority}] ${suggestion.description}`);
      lines.push(`  预期收益: ${suggestion.expected_improvement}`);
    }
  }

  return lines.join('\n');
}

// ============================================
// 成本效益分析
// ============================================

/**
 * 计算每 token 成本 (相对值)
 */
export function calculateCostPerToken(plan: PlanAnalysisResult): number {
  // 成本 ∝ 芯片数 × 时间
  const chipsHours = plan.plan.total_chips / plan.throughput.tokens_per_second;
  return chipsHours * 1000; // 归一化
}

/**
 * 成本效益对比
 */
export function compareCostEfficiency(
  plans: PlanAnalysisResult[]
): Array<{
  planId: string;
  costPerToken: number;
  throughputPerChip: number;
  rank: number;
}> {
  const results = plans
    .filter(p => p.is_feasible)
    .map(p => ({
      planId: p.plan.plan_id,
      costPerToken: calculateCostPerToken(p),
      throughputPerChip: p.throughput.tokens_per_second / p.plan.total_chips,
    }))
    .sort((a, b) => a.costPerToken - b.costPerToken)
    .map((r, i) => ({ ...r, rank: i + 1 }));

  return results;
}

// ============================================
// 场景适配分析
// ============================================

/**
 * 场景适配得分
 */
export interface ScenarioFitScore {
  planId: string;
  lowLatencyFit: number;      // 低延迟交互场景适配度
  highThroughputFit: number;  // 高吞吐批处理场景适配度
  longContextFit: number;     // 长上下文场景适配度
  balancedFit: number;        // 均衡场景适配度
  bestScenario: string;
}

/**
 * 计算场景适配度
 */
export function analyzeScenarioFit(plans: PlanAnalysisResult[]): ScenarioFitScore[] {
  return plans.filter(p => p.is_feasible).map(plan => {
    // 低延迟: TTFT < 100ms 满分
    const ttft = plan.latency.prefill_total_latency_ms;
    const lowLatencyFit = Math.max(0, 100 - (ttft - 50) / 2);

    // 高吞吐: MFU 和吞吐量
    const mfu = plan.throughput.model_flops_utilization;
    const highThroughputFit = mfu * 100 * 1.5; // MFU 50% 得满分

    // 长上下文: 显存余量
    const memoryMargin = 1 - plan.memory.memory_utilization;
    const longContextFit = memoryMargin * 100 * 2; // 50% 余量得满分

    // 均衡
    const balancedFit = plan.score.overall_score;

    // 找最佳场景
    const scores = [
      { name: '低延迟交互', score: lowLatencyFit },
      { name: '高吞吐批处理', score: highThroughputFit },
      { name: '长上下文', score: longContextFit },
      { name: '均衡', score: balancedFit },
    ];
    const bestScenario = scores.reduce((a, b) => a.score > b.score ? a : b).name;

    return {
      planId: plan.plan.plan_id,
      lowLatencyFit: Math.min(100, lowLatencyFit),
      highThroughputFit: Math.min(100, highThroughputFit),
      longContextFit: Math.min(100, longContextFit),
      balancedFit,
      bestScenario,
    };
  });
}

// ============================================
// 敏感性分析
// ============================================

/**
 * 分析参数敏感性
 */
export function analyzeSensitivity(
  model: LLMModelConfig,
  baseInference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  parameterToVary: 'batch_size' | 'input_seq_length' | 'output_seq_length',
  variations: number[]
): Array<{
  paramValue: number;
  ttft: number;
  tpot: number;
  throughput: number;
  memoryUtilization: number;
}> {
  return variations.map(value => {
    const inference = { ...baseInference };

    switch (parameterToVary) {
      case 'batch_size':
        inference.batch_size = value;
        break;
      case 'input_seq_length':
        inference.input_seq_length = value;
        inference.max_seq_length = Math.max(inference.max_seq_length, value + inference.output_seq_length);
        break;
      case 'output_seq_length':
        inference.output_seq_length = value;
        inference.max_seq_length = Math.max(inference.max_seq_length, inference.input_seq_length + value);
        break;
    }

    const result = analyzePlan(model, inference, parallelism, hardware);

    return {
      paramValue: value,
      ttft: result.latency.prefill_total_latency_ms,
      tpot: result.latency.decode_per_token_latency_ms,
      throughput: result.throughput.tokens_per_second,
      memoryUtilization: result.memory.memory_utilization,
    };
  });
}

// ============================================
// 导出汇总数据
// ============================================

/**
 * 导出对比数据为 CSV 格式
 */
export function exportComparisonToCSV(comparison: PlanComparisonResult): string {
  const lines: string[] = [];

  // 表头
  const planIds = comparison.plans.map(p => p.plan.plan_id);
  lines.push(['Metric', 'Unit', ...planIds, 'Best Plan'].join(','));

  // 数据行
  for (const metric of comparison.metrics) {
    const values = planIds.map(id => {
      const val = metric.values[id];
      return val !== undefined ? val.toFixed(4) : '';
    });
    lines.push([
      `"${metric.metric_name}"`,
      metric.unit,
      ...values,
      metric.best_plan_id,
    ].join(','));
  }

  return lines.join('\n');
}

/**
 * 导出对比数据为 JSON 格式
 */
export function exportComparisonToJSON(comparison: PlanComparisonResult): string {
  return JSON.stringify(comparison, null, 2);
}
