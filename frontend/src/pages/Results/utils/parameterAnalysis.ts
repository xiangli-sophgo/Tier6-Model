/**
 * 参数分析数据聚合工具
 * 用于从实验结果中提取参数、聚合统计数据
 */

import type { EvaluationTask, EvaluationResult } from '@/api/results'

// ============================================
// 类型定义
// ============================================

/** 单参数敏感度数据点 */
export interface SensitivityDataPoint {
  /** 参数值 */
  value: number
  /** 平均性能 */
  mean_performance: number
  /** 最小性能 */
  min_performance: number
  /** 最大性能 */
  max_performance: number
  /** 样本数 */
  count: number
}

/** 双参数热力图数据 */
export interface HeatmapData {
  /** X轴参数名 */
  param_x: string
  /** Y轴参数名 */
  param_y: string
  /** X轴取值列表 */
  x_values: number[]
  /** Y轴取值列表 */
  y_values: number[]
  /** 网格数据点 */
  data: Array<{
    x_value: number
    y_value: number
    mean_performance: number
    count: number
  }>
}

/** 性能指标类型 */
export type MetricType =
  | 'tps' | 'tps_per_chip' | 'tps_per_batch'
  | 'tpot' | 'ttft' | 'end_to_end_latency'
  | 'mfu' | 'mbu'
  | 'score' | 'chips'
  | 'dram_occupy' | 'flops'
  | 'cost_total' | 'cost_server' | 'cost_interconnect' | 'cost_per_chip' | 'cost_dfop'

/** 参数信息 */
export interface ParameterInfo {
  /** 参数路径 (如 'parallelism.tp') */
  path: string
  /** 参数的所有取值 */
  values: Set<number>
}

// ============================================
// 工具函数
// ============================================

/**
 * 从嵌套对象中提取参数值
 * 支持路径解析，如 'parallelism.tp' -> config.parallelism.tp
 */
export function extractParamValue(config: Record<string, any>, path: string): number | null {
  const keys = path.split('.')
  let current: any = config

  for (const key of keys) {
    if (current && typeof current === 'object' && key in current) {
      current = current[key]
    } else {
      return null
    }
  }

  // 确保返回的是数值类型
  const value = Number(current)
  return isNaN(value) ? null : value
}

/**
 * 从任务中提取性能指标值
 */
export function extractMetricValue(task: EvaluationTask, metric: MetricType): number | null {
  if (!task.result) return null

  // 成本相关指标在 result.cost 下
  if (metric.startsWith('cost_')) {
    const costKey = metric.replace('cost_', '') as keyof NonNullable<typeof task.result.cost>
    if (task.result.cost && costKey in task.result.cost) {
      const value = task.result.cost[costKey]
      return typeof value === 'number' ? value : null
    }
    // 特殊处理 cost_total -> total_cost
    if (metric === 'cost_total' && task.result.cost?.total_cost !== undefined) {
      return task.result.cost.total_cost
    }
    if (metric === 'cost_server' && task.result.cost?.server_cost !== undefined) {
      return task.result.cost.server_cost
    }
    if (metric === 'cost_interconnect' && task.result.cost?.interconnect_cost !== undefined) {
      return task.result.cost.interconnect_cost
    }
    if (metric === 'cost_per_chip' && task.result.cost?.cost_per_chip !== undefined) {
      return task.result.cost.cost_per_chip
    }
    if (metric === 'cost_dfop' && task.result.cost?.dfop !== undefined) {
      return task.result.cost.dfop
    }
    return null
  }

  // 显存占用需要转换单位（字节 -> GB）
  if (metric === 'dram_occupy') {
    const value = task.result.dram_occupy
    if (typeof value === 'number') {
      return value / (1024 * 1024 * 1024) // 转换为 GB
    }
    return null
  }

  // 计算量需要转换单位（FLOPs -> TFLOPs）
  if (metric === 'flops') {
    const value = task.result.flops
    if (typeof value === 'number') {
      return value / 1e12 // 转换为 TFLOPs
    }
    return null
  }

  // 其他指标直接从 result 中提取
  if (metric in task.result) {
    const value = task.result[metric as keyof typeof task.result]
    return typeof value === 'number' ? value : null
  }

  // 如果 result 中没有，尝试从 results 数组的第一个结果中提取
  if (task.results && task.results.length > 0) {
    const firstResult = task.results[0]
    if (metric in firstResult) {
      const value = firstResult[metric as keyof EvaluationResult]
      return typeof value === 'number' ? value : null
    }
  }

  return null
}

/**
 * 从结果列表中提取所有参数及其取值
 */
export function extractParametersFromResults(tasks: EvaluationTask[]): Map<string, ParameterInfo> {
  const parametersMap = new Map<string, ParameterInfo>()

  // 定义要提取的参数路径
  const parameterPaths = [
    // 并行策略
    'parallelism.dp',
    'parallelism.tp',
    'parallelism.pp',
    'parallelism.ep',
    'parallelism.sp',
    'parallelism.moe_tp',
    // 推理配置
    'inference.batch_size',
    'inference.input_seq_length',
    'inference.output_seq_length',
    // 模型配置
    'model.hidden_size',
    'model.num_layers',
    'model.num_attention_heads',
    'model.intermediate_size',
    // 硬件参数 (注意：新版本在 topology_config.chips 中)
    'hardware.compute_tflops_fp8',
    'hardware.compute_tflops_bf16',
    'hardware.memory_capacity_gb',
    'hardware.memory_bandwidth_gbps',
  ]

  tasks.forEach(task => {
    if (!task.config_snapshot) return

    const config = task.config_snapshot

    // 处理新旧格式兼容
    const normalizedConfig: Record<string, any> = {
      parallelism: task.result?.parallelism || {},
      inference: config.benchmark_config?.inference || config.inference || {},
      model: config.benchmark_config?.model || config.model || {},
      hardware: config.topology_config?.chips || config.topology_config?.hardware_params || config.hardware || {},
    }

    parameterPaths.forEach(path => {
      const value = extractParamValue(normalizedConfig, path)
      if (value !== null) {
        if (!parametersMap.has(path)) {
          parametersMap.set(path, {
            path,
            values: new Set(),
          })
        }
        parametersMap.get(path)!.values.add(value)
      }
    })
  })

  // 过滤掉只有单一值的参数（这些参数不适合做敏感度分析）
  const result = new Map<string, ParameterInfo>()
  parametersMap.forEach((info, path) => {
    if (info.values.size > 1) {
      result.set(path, info)
    }
  })

  return result
}

/**
 * 单参数敏感度数据聚合
 */
export function aggregateSensitivityData(
  tasks: EvaluationTask[],
  parameter: string,
  metric: MetricType
): SensitivityDataPoint[] {
  // 按参数值分组
  const groups = new Map<number, number[]>()

  tasks.forEach(task => {
    if (!task.config_snapshot || task.status !== 'completed') return

    const config = task.config_snapshot
    const normalizedConfig: Record<string, any> = {
      parallelism: task.result?.parallelism || {},
      inference: config.benchmark_config?.inference || config.inference || {},
      model: config.benchmark_config?.model || config.model || {},
      hardware: config.topology_config?.chips || config.topology_config?.hardware_params || config.hardware || {},
    }

    const paramValue = extractParamValue(normalizedConfig, parameter)
    const metricValue = extractMetricValue(task, metric)

    if (paramValue !== null && metricValue !== null) {
      if (!groups.has(paramValue)) {
        groups.set(paramValue, [])
      }
      groups.get(paramValue)!.push(metricValue)
    }
  })

  // 计算每组的统计数据
  const dataPoints: SensitivityDataPoint[] = []

  groups.forEach((values, paramValue) => {
    if (values.length === 0) return

    const mean = values.reduce((sum, v) => sum + v, 0) / values.length
    const min = Math.min(...values)
    const max = Math.max(...values)

    dataPoints.push({
      value: paramValue,
      mean_performance: mean,
      min_performance: min,
      max_performance: max,
      count: values.length,
    })
  })

  // 按参数值排序
  return dataPoints.sort((a, b) => a.value - b.value)
}

/**
 * 双参数热力图数据聚合
 */
export function aggregateHeatmapData(
  tasks: EvaluationTask[],
  paramX: string,
  paramY: string,
  metric: MetricType
): HeatmapData {
  // 按参数组合分组
  const groups = new Map<string, { x: number; y: number; values: number[] }>()
  const xValuesSet = new Set<number>()
  const yValuesSet = new Set<number>()

  tasks.forEach(task => {
    if (!task.config_snapshot || task.status !== 'completed') return

    const config = task.config_snapshot
    const normalizedConfig: Record<string, any> = {
      parallelism: task.result?.parallelism || {},
      inference: config.benchmark_config?.inference || config.inference || {},
      model: config.benchmark_config?.model || config.model || {},
      hardware: config.topology_config?.chips || config.topology_config?.hardware_params || config.hardware || {},
    }

    const xValue = extractParamValue(normalizedConfig, paramX)
    const yValue = extractParamValue(normalizedConfig, paramY)
    const metricValue = extractMetricValue(task, metric)

    if (xValue !== null && yValue !== null && metricValue !== null) {
      const key = `${xValue},${yValue}`
      if (!groups.has(key)) {
        groups.set(key, { x: xValue, y: yValue, values: [] })
      }
      groups.get(key)!.values.push(metricValue)

      xValuesSet.add(xValue)
      yValuesSet.add(yValue)
    }
  })

  // 计算每个网格点的平均值
  const data = Array.from(groups.values()).map(group => ({
    x_value: group.x,
    y_value: group.y,
    mean_performance:
      group.values.reduce((sum, v) => sum + v, 0) / group.values.length,
    count: group.values.length,
  }))

  // 排序取值列表
  const x_values = Array.from(xValuesSet).sort((a, b) => a - b)
  const y_values = Array.from(yValuesSet).sort((a, b) => a - b)

  return {
    param_x: paramX,
    param_y: paramY,
    x_values,
    y_values,
    data,
  }
}

/**
 * 获取指标的显示名称和单位
 */
export function getMetricLabel(metric: MetricType): { name: string; unit: string } {
  const labels: Record<MetricType, { name: string; unit: string }> = {
    // 吞吐量指标
    tps: { name: '集群吞吐量', unit: 'tokens/s' },
    tps_per_chip: { name: '单芯片吞吐量', unit: 'tokens/s' },
    tps_per_batch: { name: '单请求吞吐量', unit: 'tokens/s' },
    // 延迟指标
    tpot: { name: '每Token延迟', unit: 'ms' },
    ttft: { name: '首Token延迟', unit: 'ms' },
    end_to_end_latency: { name: '端到端延迟', unit: 'ms' },
    // 利用率指标
    mfu: { name: '算力利用率', unit: '%' },
    mbu: { name: '带宽利用率', unit: '%' },
    // 资源指标
    score: { name: '综合得分', unit: '' },
    chips: { name: '芯片数量', unit: '' },
    dram_occupy: { name: '显存占用', unit: 'GB' },
    flops: { name: '计算量', unit: 'TFLOPs' },
    // 成本指标
    cost_total: { name: '总成本', unit: '$' },
    cost_server: { name: '服务器成本', unit: '$' },
    cost_interconnect: { name: '互联成本', unit: '$' },
    cost_per_chip: { name: '单芯片成本', unit: '$' },
    cost_dfop: { name: 'DFOP', unit: '$/TPS' },
  }
  return labels[metric] || { name: metric, unit: '' }
}
