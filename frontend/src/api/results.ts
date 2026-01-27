/**
 * Results Management API Client
 *
 * 结果管理相关的 HTTP 请求封装
 */

import axios from 'axios'

const API_PORT = import.meta.env.VITE_API_PORT || '8001'
const API_BASE_URL = `http://localhost:${API_PORT}/api`

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
  throughput: number
  tps_per_chip: number
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
    model: Record<string, unknown>
    inference: Record<string, unknown>
    topology: Record<string, unknown>
  }
  benchmark_name?: string
  topology_config_name?: string
  results?: EvaluationResult[]
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
): Promise<void> {
  await api.patch(`/evaluation/experiments/${experimentId}`, data)
}

/**
 * 删除实验
 */
export async function deleteExperiment(experimentId: number): Promise<void> {
  await api.delete(`/evaluation/experiments/${experimentId}`)
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
    throughput: number
    tps_per_chip: number
    ttft: number
    tpot: number
    mfu: number
    mbu: number
    score: number
    dram_occupy?: number  // 显存占用 (字节)，后端返回
    flops?: number        // 计算量 (FLOPs)，后端返回
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
