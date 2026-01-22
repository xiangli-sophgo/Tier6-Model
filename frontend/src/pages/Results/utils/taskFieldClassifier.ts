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

  // 基础字段
  fields.add('task_id')
  fields.add('status')
  fields.add('progress')
  fields.add('message')
  fields.add('error')

  // 配置字段
  fields.add('benchmark_name')
  fields.add('topology_config_name')
  fields.add('search_mode')

  // 搜索统计字段（从 search_stats 中提取）
  tasks.forEach(task => {
    if (task.search_stats) {
      Object.keys(task.search_stats).forEach(key => {
        fields.add(`search_stats_${key}`)
      })
    }
  })

  // 性能指标字段（从最佳结果中提取）
  fields.add('best_chips')
  fields.add('best_tp')
  fields.add('best_ep')
  fields.add('best_pp')
  fields.add('best_dp')
  fields.add('best_sp')
  fields.add('best_moe_tp')
  fields.add('best_throughput')
  fields.add('best_tps_per_chip')
  fields.add('best_ttft')
  fields.add('best_tpot')
  fields.add('best_mfu')
  fields.add('best_mbu')
  fields.add('best_score')
  fields.add('best_is_feasible')

  // 时间字段
  fields.add('created_at')
  fields.add('started_at')
  fields.add('completed_at')

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

  fieldKeys.forEach(key => {
    if (key === 'task_id' || key === 'status' || key === 'progress') {
      classified.important.push(key)
    } else if (key.startsWith('search_stats_')) {
      classified.stats.push(key)
    } else if (key.startsWith('best_')) {
      classified.performance.push(key)
    } else if (key.endsWith('_at')) {
      classified.time.push(key)
    } else if (['benchmark_name', 'topology_config_name', 'search_mode'].includes(key)) {
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
