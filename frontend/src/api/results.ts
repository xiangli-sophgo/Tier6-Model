/**
 * Results Management API Client
 *
 * 结果管理相关的 HTTP 请求封装
 */

import axios from 'axios'

const API_BASE_URL = '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  withCredentials: true, // 支持跨域请求携带凭证
})

// ============================================
// 类型定义
// ============================================

export interface EvaluationResult {
  id: number
  task_id: number
  dp: number
  tp: number
  pp: number
  ep: number
  sp: number
  moe_tp?: number
  chips: number
  tps: number           // 集群总吞吐 (tokens/s)
  tps_per_batch: number // 单请求TPS (tokens/s per request)
  tps_per_chip: number  // 单芯片TPS (tokens/s per chip)
  ttft: number
  tpot: number
  mfu: number
  mbu: number
  score: number
  is_feasible: number
  infeasible_reason?: string
  full_result: Record<string, unknown>
  created_at: string
}

export interface EvaluationTask {
  id: number
  task_id: string
  result_id?: number  // 结果 ID（每个结果展开为单独的行）
  result_rank?: number  // 结果排名（按分数）
  experiment_id: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  message?: string
  error?: string
  created_at: string
  started_at?: string
  completed_at?: string
  search_mode: string
  manual_parallelism?: Record<string, unknown>
  search_constraints?: Record<string, unknown>
  search_stats?: Record<string, unknown>
  config_snapshot?: {
    // 旧格式
    model?: Record<string, unknown>
    hardware?: Record<string, unknown>
    inference?: Record<string, unknown>
    topology?: Record<string, unknown>
    // 新格式 (v2.1.3+)
    benchmark_config?: {
      model?: Record<string, unknown>
      inference?: Record<string, unknown>
    }
    topology_config?: Record<string, unknown>
    // 任务配置
    max_workers?: number
  }
  benchmark_name?: string
  topology_config_name?: string
  results?: EvaluationResult[]
  result?: {  // 当前结果的性能指标
    tps: number           // 集群总吞吐 (tokens/s)
    tps_per_batch: number // 单请求TPS (tokens/s per request)
    tps_per_chip: number  // 单芯片TPS (tokens/s per chip)
    tpot: number          // Time Per Output Token (ms)
    ttft: number          // Time To First Token (ms)
    mfu: number           // 模型算力利用率
    mbu: number           // 内存带宽利用率
    score: number         // 综合得分
    chips: number         // 芯片数
    dram_occupy: number   // 显存占用 (字节)
    flops: number         // 计算量 (FLOPs)
    cost?: {              // 成本分析结果
      server_cost: number
      rdma_cost: number
      per_chip_cost: number
      interconnect_cost: number
      total_cost: number
      bandwidth_gbps: number
      lanes: number
      lane_cost: number
      cost_per_chip: number
      cost_per_million_tokens: number
      dfop: number
      model_size_gb: number
    }
    parallelism?: {       // 并行策略
      dp: number
      tp: number
      pp: number
      ep: number
      sp: number
      moe_tp?: number
    }
  }
}

export interface Experiment {
  id: number
  name: string
  description?: string
  created_at: string
  updated_at: string
  total_tasks: number
  completed_tasks: number
  tasks?: EvaluationTask[]
}

export interface ResultsPageResponse {
  total: number
  page: number
  pageSize: number
  results: EvaluationResult[]
}

// ============================================
// 实验相关 API
// ============================================

/**
 * 获取所有实验列表
 */
export async function listExperiments(): Promise<Experiment[]> {
  const response = await api.get('/evaluation/experiments')
  return response.data.experiments || []
}

/**
 * 获取实验详情
 */
export async function getExperimentDetail(experimentId: number): Promise<Experiment> {
  const response = await api.get(`/evaluation/experiments/${experimentId}`)
  return response.data
}

/**
 * 更新实验信息
 */
export async function updateExperiment(
  experimentId: number,
  data: Partial<Experiment>
): Promise<Experiment> {
  const response = await api.patch(`/evaluation/experiments/${experimentId}`, data)
  return response.data
}

/**
 * 删除实验
 */
export async function deleteExperiment(experimentId: number): Promise<void> {
  await api.delete(`/evaluation/experiments/${experimentId}`)
}

/**
 * 批量删除实验
 */
export async function deleteExperimentsBatch(
  experimentIds: number[]
): Promise<{ success: boolean; message: string; deleted_count: number }> {
  const response = await api.post('/evaluation/experiments/batch-delete', {
    experiment_ids: experimentIds,
  })
  return response.data
}

// ============================================
// 任务相关 API
// ============================================

/**
 * 获取实验的所有任务
 */
export async function getExperimentTasks(experimentId: number): Promise<EvaluationTask[]> {
  const response = await api.get(`/evaluation/experiments/${experimentId}`)
  return response.data.tasks || []
}

/**
 * 获取任务详情
 */
export async function getTaskDetail(taskId: string): Promise<EvaluationTask> {
  const response = await api.get(`/evaluation/tasks/${taskId}`)
  return response.data
}

/**
 * 任务结果响应类型
 */
export interface TaskResultsResponse {
  task_id: string
  status: string
  experiment_name: string
  top_k_plans: Array<{
    parallelism: {
      dp: number
      tp: number
      pp: number
      ep: number
      sp: number
      moe_tp?: number
    }
    chips: number
    is_feasible: boolean
    tps: number           // 集群总吞吐 (tokens/s)
    tps_per_batch: number // 单请求TPS (tokens/s per request)
    tps_per_chip: number  // 单芯片TPS (tokens/s per chip)
    ttft: number
    tpot: number
    mfu: number
    mbu: number
    score: number
    dram_occupy?: number  // 显存占用 (字节)，后端返回
    flops?: number        // 计算量 (FLOPs)，后端返回
    cost?: {              // 成本评估结果
      server_cost: number           // 服务器总成本 ($)
      rdma_cost: number             // RDMA 网卡成本 ($)
      per_chip_cost: number         // 每芯片附加成本 ($)
      interconnect_cost: number     // 互联总成本 ($)
      total_cost: number            // 总成本 ($)
      bandwidth_gbps: number        // 互联带宽需求 (Gbps)
      lanes: number                 // 所需 lane 数量
      lane_cost: number             // 单 lane 成本 ($/lane)
      cost_per_chip: number         // 单芯片摊派成本 ($)
      cost_per_million_tokens: number  // 每百万 tokens 成本 ($/M tokens)
      dfop: number                  // DFOP: 每 TPS 成本 ($/TPS)
      model_size_gb: number         // 模型大小 (GB)
    }
    stats?: Record<string, unknown>      // 完整的统计数据
    gantt_chart?: Record<string, unknown>  // 甘特图数据（用于可视化）
  }>
  infeasible_plans: unknown[]
  search_stats: Record<string, unknown>
}

/**
 * 获取任务的所有结果
 */
export async function getTaskResults(taskId: string): Promise<TaskResultsResponse> {
  const response = await api.get(`/evaluation/tasks/${taskId}/results`)
  return response.data
}

// ============================================
// 结果相关 API
// ============================================

/**
 * 分页获取实验的结果
 */
export async function getResultsPage(
  experimentId: number,
  page: number = 1,
  pageSize: number = 50,
  sortField: string = 'id',
  sortOrder: 'asc' | 'desc' = 'asc'
): Promise<ResultsPageResponse> {
  const response = await api.get(`/evaluation/experiments/${experimentId}/results`, {
    params: {
      page,
      pageSize,
      sortField,
      sortOrder,
    },
  })
  return response.data
}

/**
 * 删除单个结果
 */
export async function deleteResult(experimentId: number, resultId: number): Promise<void> {
  await api.delete(`/evaluation/experiments/${experimentId}/results/${resultId}`)
}

/**
 * 批量删除结果
 */
export async function deleteResultsBatch(
  experimentId: number,
  resultIds: number[]
): Promise<{ message: string }> {
  const response = await api.post(
    `/evaluation/experiments/${experimentId}/results/batch-delete`,
    { result_ids: resultIds }
  )
  return response.data
}

/**
 * 导出结果为CSV
 */
export async function exportResultsCSV(
  experimentId: number,
  columns?: string[]
): Promise<Blob> {
  const response = await api.get(`/evaluation/experiments/${experimentId}/results/export`, {
    params: {
      format: 'csv',
      columns: columns?.join(','),
    },
    responseType: 'blob',
  })
  return response.data
}

// ============================================
// 导入导出相关 API
// ============================================

/**
 * 导出实验配置
 */
export async function exportExperiments(experimentIds?: number[]): Promise<Record<string, unknown>> {
  const params = new URLSearchParams()
  if (experimentIds && experimentIds.length > 0) {
    params.append('experiment_ids', experimentIds.join(','))
  }
  const response = await api.get('/evaluation/experiments/export', { params })
  return response.data
}

/**
 * 下载导出的实验为 JSON 文件
 */
export async function downloadExperimentJSON(experimentIds?: number[]): Promise<void> {
  try {
    const data = await exportExperiments(experimentIds)

    // 创建 Blob 和下载链接
    const element = document.createElement('a')
    element.setAttribute(
      'href',
      'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2))
    )
    element.setAttribute(
      'download',
      `experiments_${new Date().toISOString().slice(0, 10)}.json`
    )
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  } catch (error) {
    throw error
  }
}

/**
 * 检查导入文件的有效性
 */
export async function checkImportFile(
  file: File
): Promise<{
  valid: boolean
  error?: string
  experiments?: Array<{
    id?: number
    name: string
    description?: string
    total_tasks: number
    completed_tasks: number
    conflict: boolean
    existing_id?: number
  }>
  temp_file_id?: string
}> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/evaluation/experiments/check-import', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * 执行导入操作
 */
export async function executeImport(
  tempFileId: string,
  configs: Array<{
    original_name: string
    action: 'rename' | 'overwrite' | 'skip'
    new_name?: string
  }>
): Promise<{
  success: boolean
  imported_count: number
  skipped_count: number
  overwritten_count: number
  message: string
}> {
  const response = await api.post('/evaluation/experiments/execute-import', {
    temp_file_id: tempFileId,
    configs,
  })
  return response.data
}

// ============================================
// 列配置方案相关 API（使用 localStorage）
// ============================================

import {
  ColumnPreset,
  PresetsFile,
  getColumnPresets as _getColumnPresets,
  getColumnPresetsByExperiment as _getColumnPresetsByExperiment,
  saveColumnPresets as _saveColumnPresets,
  addColumnPreset as _addColumnPreset,
  deleteColumnPreset as _deleteColumnPreset,
} from '../utils/storage'

// 导出类型定义
export type { ColumnPreset, PresetsFile }

/**
 * 获取所有列配置方案
 */
export async function getColumnPresets(): Promise<PresetsFile> {
  return Promise.resolve(_getColumnPresets())
}

/**
 * 获取指定实验的列配置方案
 */
export async function getColumnPresetsByExperiment(experimentId: number): Promise<{ presets: ColumnPreset[] }> {
  return Promise.resolve(_getColumnPresetsByExperiment(experimentId))
}

/**
 * 保存所有列配置方案
 */
export async function saveColumnPresets(presetsFile: PresetsFile): Promise<{ message: string; count: number }> {
  return Promise.resolve(_saveColumnPresets(presetsFile))
}

/**
 * 添加或更新单个列配置方案
 */
export async function addColumnPreset(preset: ColumnPreset): Promise<{ message: string; preset: ColumnPreset }> {
  return Promise.resolve(_addColumnPreset(preset))
}

/**
 * 删除列配置方案
 */
export async function deleteColumnPreset(experimentId: number, name: string): Promise<{ message: string }> {
  return Promise.resolve(_deleteColumnPreset(experimentId, name))
}
