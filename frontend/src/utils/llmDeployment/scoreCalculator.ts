/**
 * 六维评分计算器
 * 统一 Results 页面和雷达图组件的评分逻辑
 */

/** 评分输入参数 */
export interface ScoreInput {
  /** TTFT 时延 (ms) */
  ttft: number
  /** TPOT 时延 (ms) */
  tpot: number
  /** TPS 吞吐量 (tokens/s) */
  tps: number
  /** TPS per chip */
  tpsPerChip: number
  /** MFU 模型算力利用率 (0-1) */
  mfu: number
  /** MBU 显存带宽利用率 (0-1) */
  mbu: number
  /** 显存占用 (GB) */
  memoryUsedGB: number
  /** 芯片显存容量 (GB) */
  memoryCapacityGB: number
  /** Prefill 阶段通信延迟 (ms) */
  prefillCommLatency?: number
  /** Prefill 阶段计算延迟 (ms) */
  prefillComputeLatency?: number
  /** Decode 阶段通信延迟 (ms) */
  decodeCommLatency?: number
  /** Decode 阶段计算延迟 (ms) */
  decodeComputeLatency?: number
  /** 模型参数量 (B) */
  modelSizeB?: number
  /** 芯片数量 */
  chipCount?: number
}

/** 评分结果 */
export interface ScoreResult {
  /** 延迟评分 (0-100) */
  latencyScore: number
  /** 吞吐评分 (0-100) */
  throughputScore: number
  /** 效率评分 (0-100) */
  efficiencyScore: number
  /** 均衡评分 (0-100) */
  balanceScore: number
  /** 显存评分 (0-100) */
  memoryScore: number
  /** 通信评分 (0-100) */
  communicationScore: number
  /** 综合评分 (0-100) */
  overallScore: number
}

/** 评分规则说明 */
export const SCORE_RULES: Record<keyof Omit<ScoreResult, 'overallScore'>, {
  name: string
  rule: string
  tip: string
}> = {
  latencyScore: {
    name: '延迟评分',
    rule: 'TTFT < 50ms → 100分',
    tip: 'TTFT 越低越好，>500ms 则 10 分',
  },
  throughputScore: {
    name: '吞吐评分',
    rule: 'TPS/Chip 相对值',
    tip: '基于模型大小的相对吞吐',
  },
  efficiencyScore: {
    name: '效率评分',
    rule: '(MFU×0.6 + MBU×0.4)×100',
    tip: '综合算力和带宽利用率',
  },
  balanceScore: {
    name: '均衡评分',
    rule: '|MFU - MBU| 越小越好',
    tip: 'MFU 与 MBU 越接近越均衡',
  },
  memoryScore: {
    name: '显存评分',
    rule: '60-80% 利用率 → 100分',
    tip: '过高或过低都会扣分',
  },
  communicationScore: {
    name: '通信评分',
    rule: '(1 - 通信占比) × 100',
    tip: '通信开销越小越好',
  },
}

/**
 * 计算延迟评分
 * 分段函数：
 * - TTFT < 50ms → 100分
 * - 50-200ms → 线性递减 100 → 60
 * - 200-500ms → 线性递减 60 → 30
 * - > 500ms → 线性递减 30 → 10，最低10分
 */
export function calcLatencyScore(ttft: number): number {
  if (ttft <= 50) return 100
  if (ttft <= 200) return 100 - ((ttft - 50) / 150) * 40  // 100 → 60
  if (ttft <= 500) return 60 - ((ttft - 200) / 300) * 30   // 60 → 30
  if (ttft <= 1000) return 30 - ((ttft - 500) / 500) * 20  // 30 → 10
  return 10
}

/**
 * 计算吞吐评分
 * 基于 TPS/Chip 和模型大小的相对值
 *
 * 基准值（根据模型大小调整）：
 * - 小模型 (<10B): 期望 TPS/Chip > 100
 * - 中模型 (10-100B): 期望 TPS/Chip > 30
 * - 大模型 (100-500B): 期望 TPS/Chip > 10
 * - 超大模型 (>500B): 期望 TPS/Chip > 3
 */
export function calcThroughputScore(tpsPerChip: number, modelSizeB?: number): number {
  // 根据模型大小确定基准 TPS/Chip
  let baseTps = 30 // 默认中等模型
  if (modelSizeB) {
    if (modelSizeB < 10) baseTps = 100
    else if (modelSizeB < 100) baseTps = 30
    else if (modelSizeB < 500) baseTps = 10
    else baseTps = 3
  }

  // 计算相对得分
  const ratio = tpsPerChip / baseTps
  if (ratio >= 1) {
    // 超过基准，满分附近
    return Math.min(100, 80 + ratio * 20)
  } else {
    // 低于基准，线性衰减
    return Math.max(10, ratio * 80)
  }
}

/**
 * 计算效率评分
 * 综合 MFU 和 MBU，MFU 权重 0.6，MBU 权重 0.4
 */
export function calcEfficiencyScore(mfu: number, mbu: number): number {
  const combined = mfu * 0.6 + mbu * 0.4
  return Math.min(100, combined * 100)
}

/**
 * 计算均衡评分
 * 基于 MFU 和 MBU 的差值方差
 * - 差值 = 0 → 100分
 * - 差值 = 0.1 → 90分
 * - 差值 = 0.3 → 70分
 * - 差值 = 0.5 → 50分
 */
export function calcBalanceScore(mfu: number, mbu: number): number {
  const diff = Math.abs(mfu - mbu)
  // 使用平方根函数让小差值更敏感
  const score = 100 - diff * 100
  return Math.max(0, Math.min(100, score))
}

/**
 * 计算显存评分
 * 分段函数，60-80% 利用率最优：
 * - < 40%: 利用不足，50分起步
 * - 40-60%: 线性增长到 80分
 * - 60-80%: 满分区间 80-100
 * - 80-90%: 开始下降 100 → 80
 * - 90-95%: 快速下降 80 → 50
 * - > 95%: 危险区域 50 → 10
 */
export function calcMemoryScore(memoryUsedGB: number, memoryCapacityGB: number): number {
  if (memoryCapacityGB <= 0) return 0
  const utilization = memoryUsedGB / memoryCapacityGB

  if (utilization <= 0.4) {
    // 利用不足
    return 50 + utilization / 0.4 * 30  // 50 → 80
  }
  if (utilization <= 0.6) {
    // 接近最优区间
    return 80 + (utilization - 0.4) / 0.2 * 15  // 80 → 95
  }
  if (utilization <= 0.8) {
    // 最优区间
    return 95 + (utilization - 0.6) / 0.2 * 5  // 95 → 100
  }
  if (utilization <= 0.9) {
    // 开始偏高
    return 100 - (utilization - 0.8) / 0.1 * 20  // 100 → 80
  }
  if (utilization <= 0.95) {
    // 接近危险
    return 80 - (utilization - 0.9) / 0.05 * 30  // 80 → 50
  }
  // 危险区域
  return Math.max(10, 50 - (utilization - 0.95) / 0.05 * 40)  // 50 → 10
}

/**
 * 计算通信评分
 * 基于通信时间占总时间的比例
 * - 通信占比 0% → 100分
 * - 通信占比 10% → 90分
 * - 通信占比 30% → 70分
 * - 通信占比 50% → 50分
 * - 通信占比 > 70% → 30分以下
 */
export function calcCommunicationScore(
  prefillCommLatency: number = 0,
  prefillComputeLatency: number = 0,
  decodeCommLatency: number = 0,
  decodeComputeLatency: number = 0
): number {
  const totalComm = prefillCommLatency + decodeCommLatency
  const totalCompute = prefillComputeLatency + decodeComputeLatency
  const total = totalComm + totalCompute

  if (total <= 0) return 100 // 无数据时默认满分

  const commRatio = totalComm / total
  const score = (1 - commRatio) * 100
  return Math.max(10, Math.min(100, score))
}

/**
 * 计算综合评分
 * 各维度加权平均，权重可调整
 */
export function calcOverallScore(scores: Omit<ScoreResult, 'overallScore'>): number {
  const weights = {
    latencyScore: 0.2,       // 延迟占 20%
    throughputScore: 0.2,    // 吞吐占 20%
    efficiencyScore: 0.2,    // 效率占 20%
    balanceScore: 0.1,       // 均衡占 10%
    memoryScore: 0.15,       // 显存占 15%
    communicationScore: 0.15, // 通信占 15%
  }

  let weighted = 0
  let totalWeight = 0
  for (const [key, weight] of Object.entries(weights)) {
    const score = scores[key as keyof typeof scores]
    if (typeof score === 'number' && !isNaN(score)) {
      weighted += score * weight
      totalWeight += weight
    }
  }

  return totalWeight > 0 ? weighted / totalWeight : 0
}

/**
 * 计算完整的六维评分
 */
export function calculateScores(input: ScoreInput): ScoreResult {
  const latencyScore = calcLatencyScore(input.ttft)
  const throughputScore = calcThroughputScore(input.tpsPerChip, input.modelSizeB)
  const efficiencyScore = calcEfficiencyScore(input.mfu, input.mbu)
  const balanceScore = calcBalanceScore(input.mfu, input.mbu)
  const memoryScore = calcMemoryScore(input.memoryUsedGB, input.memoryCapacityGB)
  const communicationScore = calcCommunicationScore(
    input.prefillCommLatency,
    input.prefillComputeLatency,
    input.decodeCommLatency,
    input.decodeComputeLatency
  )

  const partialScores = {
    latencyScore,
    throughputScore,
    efficiencyScore,
    balanceScore,
    memoryScore,
    communicationScore,
  }

  return {
    ...partialScores,
    overallScore: calcOverallScore(partialScores),
  }
}

/**
 * 从 PlanAnalysisResult 风格的数据中提取评分输入
 * 用于兼容现有数据结构
 */
export function extractScoreInputFromPlan(plan: {
  ttft?: number
  tpot?: number
  tps?: number
  tps_per_chip?: number
  mfu?: number
  mbu?: number
  dram_occupy?: number
  memory_capacity_gb?: number
  stats?: Record<string, unknown>
}): ScoreInput {
  const stats = plan.stats as Record<string, unknown> || {}
  const prefillStats = stats.prefill as Record<string, number> || {}
  const decodeStats = stats.decode as Record<string, number> || {}

  // 显存：从 dram_occupy (bytes) 转换为 GB
  const memoryUsedGB = plan.dram_occupy ? plan.dram_occupy / (1024 * 1024 * 1024) : 0

  return {
    ttft: plan.ttft || 0,
    tpot: plan.tpot || 0,
    tps: plan.tps || 0,
    tpsPerChip: plan.tps_per_chip || 0,
    mfu: plan.mfu || 0,
    mbu: plan.mbu || 0,
    memoryUsedGB,
    memoryCapacityGB: plan.memory_capacity_gb || 80, // 默认 80GB
    prefillCommLatency: prefillStats.commTime ? prefillStats.commTime / 1000 : 0,
    prefillComputeLatency: prefillStats.computeTime ? prefillStats.computeTime / 1000 : 0,
    decodeCommLatency: decodeStats.commTime ? decodeStats.commTime / 1000 : 0,
    decodeComputeLatency: decodeStats.computeTime ? decodeStats.computeTime / 1000 : 0,
  }
}

/**
 * 将六维评分转换为雷达图数据格式
 */
export function scoresToRadarData(scores: ScoreResult): number[] {
  return [
    scores.latencyScore,
    scores.throughputScore,
    scores.efficiencyScore,
    scores.balanceScore,
    scores.memoryScore,
    scores.communicationScore,
  ]
}
