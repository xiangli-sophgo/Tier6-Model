/**
 * 任务字段分类器
 * 将任务的所有可展示字段分类到不同的组
 */

import type { EvaluationTask } from '@/api/results'

export interface ClassifiedTaskFields {
  important: string[] // 重要统计
  config: string[] // 配置参数
  stats: string[] // 搜索统计
  performance: string[] // 性能指标
  time: string[] // 时间信息
}

/**
 * 从任务中提取所有可展示的字段键
 */
export function extractTaskFields(tasks: EvaluationTask[]): string[] {
  const fields = new Set<string>()

  // 配置字段
  fields.add('benchmark_name')
  fields.add('topology_config_name')

  // 搜索统计字段（从 search_stats 中提取）
  tasks.forEach(task => {
    if (task.search_stats) {
      Object.keys(task.search_stats).forEach(key => {
        fields.add(`search_stats_${key}`)
      })
    }
  })

  // 性能指标字段（从 result 中提取）
  fields.add('throughput')
  fields.add('tps_per_chip')
  fields.add('tpot')
  fields.add('ttft')
  fields.add('mfu')
  fields.add('score')
  fields.add('chips')

  // 并行策略字段
  fields.add('parallelism_dp')
  fields.add('parallelism_tp')
  fields.add('parallelism_pp')
  fields.add('parallelism_ep')
  fields.add('parallelism_sp')

  // 时间字段
  fields.add('created_at')

  return Array.from(fields)
}

/**
 * 分类任务字段
 */
export function classifyTaskFields(fieldKeys: string[]): ClassifiedTaskFields {
  const classified: ClassifiedTaskFields = {
    important: [],
    config: [],
    stats: [],
    performance: [],
    time: [],
  }

  const performanceFields = ['throughput', 'tps_per_chip', 'tpot', 'ttft', 'mfu', 'score', 'chips']
  const parallelismFields = ['parallelism_dp', 'parallelism_tp', 'parallelism_pp', 'parallelism_ep', 'parallelism_sp']

  fieldKeys.forEach(key => {
    if (key.startsWith('search_stats_')) {
      classified.stats.push(key)
    } else if (performanceFields.includes(key)) {
      classified.performance.push(key)
    } else if (parallelismFields.includes(key)) {
      classified.config.push(key)
    } else if (key.endsWith('_at')) {
      classified.time.push(key)
    } else if (['benchmark_name', 'topology_config_name'].includes(key)) {
      classified.config.push(key)
    }
  })

  return classified
}

/**
 * 带层级结构的分类（用于树形显示）
 */
export function classifyTaskFieldsWithHierarchy(fieldKeys: string[]): ClassifiedTaskFields {
  return classifyTaskFields(fieldKeys)
}
