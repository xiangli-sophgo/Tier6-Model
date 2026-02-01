/**
 * Gantt 数据处理工具函数
 *
 * 提供任务聚合、分析等功能
 */

import type {
  GanttTask,
  GanttTaskExtended,
  LayerBreakdown,
  CommTypeBreakdown,
  CommTypeDetail,
} from './types'

// 导入统一的格式化工具
export {
  formatBytes,
  formatTime,
  formatTimeMs,
  formatFlops,
  formatPercent,
  formatGemmShape,
} from '../formatters'

// ============================================
// 任务类型分类
// ============================================

/** 计算类型任务 */
const COMPUTE_TASK_TYPES = new Set([
  'compute',
  'embedding',
  'layernorm',
  'attention_qkv',
  'attention_score',
  'attention_softmax',
  'attention_output',
  'ffn_gate',
  'ffn_up',
  'ffn_down',
  'lm_head',
  'rmsnorm_q_lora',
  'rmsnorm_kv_lora',
  'mm_q_lora_a',
  'mm_q_lora_b',
  'mm_kv_lora_a',
  'attn_fc',
  'bmm_qk',
  'bmm_sv',
  'moe_gate',
  'moe_expert',
  'moe_shared_expert',
])

/** 访存类型任务 */
const MEMORY_TASK_TYPES = new Set([
  'pcie_h2d',
  'pcie_d2h',
  'hbm_write',
  'hbm_read',
  'weight_load',
  'kv_cache_read',
  'kv_cache_write',
])

/** 通信类型任务 */
const COMM_TASK_TYPES = {
  tp: new Set(['tp_comm']),
  pp: new Set(['pp_comm']),
  ep: new Set(['ep_comm', 'ep_dispatch', 'ep_combine']),
  sp: new Set(['sp_allgather', 'sp_reduce_scatter']),
}

/** 判断任务类型分类 */
export function getTaskCategory(type: string): 'compute' | 'memory' | 'tp' | 'pp' | 'ep' | 'sp' | 'other' {
  if (COMPUTE_TASK_TYPES.has(type)) return 'compute'
  if (MEMORY_TASK_TYPES.has(type)) return 'memory'
  if (COMM_TASK_TYPES.tp.has(type)) return 'tp'
  if (COMM_TASK_TYPES.pp.has(type)) return 'pp'
  if (COMM_TASK_TYPES.ep.has(type)) return 'ep'
  if (COMM_TASK_TYPES.sp.has(type)) return 'sp'
  return 'other'
}

// ============================================
// 聚合函数
// ============================================

/**
 * 按层聚合任务数据
 * @param tasks GanttTask 数组
 * @param phase 可选，过滤特定阶段
 * @returns 按层索引聚合的 LayerBreakdown 数组
 */
export function aggregateTasksByLayer(
  tasks: GanttTask[],
  phase?: 'prefill' | 'decode'
): LayerBreakdown[] {
  const layerMap = new Map<string, LayerBreakdown>()

  for (const task of tasks) {
    // 过滤阶段
    if (phase && task.phase !== phase) continue

    // 获取层索引 (如果没有则跳过)
    const layerIndex = task.layer
    if (layerIndex === undefined || layerIndex === null) continue

    const phaseKey = task.phase as 'prefill' | 'decode'
    const key = `${layerIndex}-${phaseKey}`

    // 获取或创建聚合数据
    let breakdown = layerMap.get(key)
    if (!breakdown) {
      breakdown = {
        layerIndex,
        phase: phaseKey,
        computeTime: 0,
        memoryTime: 0,
        commTime: { tp: 0, pp: 0, ep: 0, sp: 0 },
        totalTime: 0,
        flops: 0,
        dramTraffic: 0,
        taskCount: 0,
      }
      layerMap.set(key, breakdown)
    }

    // 计算任务持续时间 (us)
    const duration = (task.end - task.start) * 1000 // ms to us

    // 根据任务类型分类聚合
    const category = getTaskCategory(task.type)
    const extTask = task as GanttTaskExtended

    switch (category) {
      case 'compute':
        breakdown.computeTime += extTask.compute_time_us ?? duration
        if (extTask.flops) breakdown.flops += extTask.flops
        break
      case 'memory':
        breakdown.memoryTime += extTask.memory_time_us ?? duration
        if (extTask.dram_traffic_bytes) breakdown.dramTraffic += extTask.dram_traffic_bytes
        break
      case 'tp':
        breakdown.commTime.tp += extTask.comm_time_us ?? duration
        break
      case 'pp':
        breakdown.commTime.pp += extTask.comm_time_us ?? duration
        break
      case 'ep':
        breakdown.commTime.ep += extTask.comm_time_us ?? duration
        break
      case 'sp':
        breakdown.commTime.sp += extTask.comm_time_us ?? duration
        break
    }

    breakdown.taskCount++
  }

  // 计算总时间并转换为数组
  const result: LayerBreakdown[] = []
  for (const breakdown of layerMap.values()) {
    breakdown.totalTime =
      breakdown.computeTime +
      breakdown.memoryTime +
      breakdown.commTime.tp +
      breakdown.commTime.pp +
      breakdown.commTime.ep +
      breakdown.commTime.sp
    result.push(breakdown)
  }

  // 按层索引排序
  result.sort((a, b) => a.layerIndex - b.layerIndex)

  return result
}

/**
 * 按通信类型聚合
 * @param tasks GanttTask 数组
 * @returns CommTypeBreakdown 通信分解数据
 */
export function aggregateCommByType(tasks: GanttTask[]): CommTypeBreakdown {
  const breakdown: CommTypeBreakdown = {
    totalTime: 0,
    breakdown: {
      tp: { time: 0, volume: 0, count: 0 },
      pp: { time: 0, volume: 0, count: 0 },
      ep: { time: 0, volume: 0, count: 0 },
      sp: { time: 0, volume: 0, count: 0 },
    },
    bottleneckLayers: [],
  }

  // 用于计算瓶颈层
  const layerCommTime = new Map<number, number>()

  for (const task of tasks) {
    const category = getTaskCategory(task.type)
    if (!['tp', 'pp', 'ep', 'sp'].includes(category)) continue

    const duration = (task.end - task.start) * 1000 // ms to us
    const extTask = task as GanttTaskExtended

    const detail = breakdown.breakdown[category as keyof typeof breakdown.breakdown]
    detail.time += extTask.comm_time_us ?? duration
    detail.volume += extTask.comm_size_bytes ?? 0
    detail.count++
    if (extTask.comm_algorithm && !detail.algorithm) {
      detail.algorithm = extTask.comm_algorithm
    }

    // 累计层通信时间
    if (task.layer !== undefined) {
      const current = layerCommTime.get(task.layer) || 0
      layerCommTime.set(task.layer, current + (extTask.comm_time_us ?? duration))
    }
  }

  // 计算总时间
  breakdown.totalTime =
    breakdown.breakdown.tp.time +
    breakdown.breakdown.pp.time +
    breakdown.breakdown.ep.time +
    breakdown.breakdown.sp.time

  // 找出通信瓶颈层 (时间最长的前3层)
  const sortedLayers = Array.from(layerCommTime.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([layer]) => layer)
  breakdown.bottleneckLayers = sortedLayers

  return breakdown
}

// ============================================
// 颜色常量 - 从设计令牌导入
// ============================================

export { TIME_BREAKDOWN_COLORS, TIME_BREAKDOWN_LABELS } from '../design-tokens'

// ============================================
// 统计计算函数
// ============================================

/**
 * 计算利用率统计
 * @param layers LayerBreakdown 数组
 * @returns 平均计算占比、访存占比、通信占比
 */
export function computeUtilizationStats(layers: LayerBreakdown[]): {
  avgComputeRatio: number
  avgMemoryRatio: number
  avgCommRatio: number
} {
  if (layers.length === 0) {
    return { avgComputeRatio: 0, avgMemoryRatio: 0, avgCommRatio: 0 }
  }

  let totalCompute = 0
  let totalMemory = 0
  let totalComm = 0
  let totalTime = 0

  for (const layer of layers) {
    totalCompute += layer.computeTime
    totalMemory += layer.memoryTime
    totalComm +=
      layer.commTime.tp +
      layer.commTime.pp +
      layer.commTime.ep +
      layer.commTime.sp
    totalTime += layer.totalTime
  }

  return {
    avgComputeRatio: totalTime > 0 ? totalCompute / totalTime : 0,
    avgMemoryRatio: totalTime > 0 ? totalMemory / totalTime : 0,
    avgCommRatio: totalTime > 0 ? totalComm / totalTime : 0,
  }
}

/**
 * 找出瓶颈层
 * @param layers LayerBreakdown 数组
 * @param topK 返回前 K 个
 * @returns 按总时间排序的前 K 层
 */
export function findBottleneckLayers(
  layers: LayerBreakdown[],
  topK: number = 3
): LayerBreakdown[] {
  return [...layers]
    .sort((a, b) => b.totalTime - a.totalTime)
    .slice(0, topK)
}
