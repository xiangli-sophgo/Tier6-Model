/**
 * Task Management API Client
 *
 * 任务管理相关的 HTTP 请求封装
 */

import axios from 'axios'

const API_PORT = import.meta.env.VITE_API_PORT || '8001'
const API_BASE_URL = `http://localhost:${API_PORT}/api`

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2分钟超时
})

// ============================================
// 类型定义
// ============================================

export interface EvaluationRequest {
  experiment_name: string
  description?: string
  topology: Record<string, unknown>
  model: Record<string, unknown>
  hardware: Record<string, unknown>
  inference: Record<string, unknown>
  search_mode: 'manual' | 'auto'
  manual_parallelism?: Record<string, unknown>
  search_constraints?: Record<string, unknown>
}

export interface TaskSubmitResponse {
  task_id: string
  message: string
}

export interface TaskStatus {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  message: string
  error?: string
  search_stats?: {
    total_plans: number
    feasible_plans: number
    infeasible_plans: number
  }
  created_at: string
  started_at?: string
  completed_at?: string
  experiment_name: string
}

export interface TaskResult {
  task_id: string
  status: string
  experiment_name: string
  top_k_plans: unknown[]
  infeasible_plans: unknown[]
  search_stats?: {
    total_plans: number
    feasible_plans: number
    infeasible_plans: number
  }
}

export interface TaskListItem {
  task_id: string
  experiment_name: string
  status: string
  progress: number
  message: string
  created_at: string
  started_at?: string
  completed_at?: string
}

// ============================================
// API 函数
// ============================================

/**
 * 提交评估任务
 */
export async function submitEvaluation(request: EvaluationRequest): Promise<TaskSubmitResponse> {
  const response = await api.post<TaskSubmitResponse>('/evaluation/submit', request)
  return response.data
}

/**
 * 获取任务状态
 */
export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const response = await api.get<TaskStatus>(`/evaluation/tasks/${taskId}`)
  return response.data
}

/**
 * 获取任务结果
 */
export async function getTaskResults(taskId: string): Promise<TaskResult> {
  const response = await api.get<TaskResult>(`/evaluation/tasks/${taskId}/results`)
  return response.data
}

/**
 * 取消任务
 */
export async function cancelTask(taskId: string): Promise<void> {
  await api.post(`/evaluation/tasks/${taskId}/cancel`)
}

/**
 * 删除任务
 */
export async function deleteTask(taskId: string): Promise<void> {
  await api.delete(`/evaluation/tasks/${taskId}`)
}

/**
 * 获取任务列表
 */
export async function getTasks(params?: {
  status?: string
  experiment_name?: string
  limit?: number
}): Promise<{ tasks: TaskListItem[] }> {
  const response = await api.get<{ tasks: TaskListItem[] }>('/evaluation/tasks', { params })
  return response.data
}

/**
 * 获取运行中的任务
 */
export async function getRunningTasks(): Promise<{ tasks: TaskListItem[] }> {
  const response = await api.get<{ tasks: TaskListItem[] }>('/evaluation/running')
  return response.data
}

/**
 * 获取实验列表
 */
export async function getExperiments(): Promise<{
  experiments: Array<{
    id: number
    name: string
    description: string
    total_tasks: number
    completed_tasks: number
    created_at: string
  }>
}> {
  const response = await api.get('/evaluation/experiments')
  return response.data
}

/**
 * 获取实验详情
 */
export async function getExperimentDetails(experimentId: number): Promise<{
  id: number
  name: string
  description: string
  model_config: Record<string, unknown>
  hardware_config: Record<string, unknown>
  inference_config: Record<string, unknown>
  total_tasks: number
  completed_tasks: number
  created_at: string
  tasks: TaskListItem[]
}> {
  const response = await api.get(`/evaluation/experiments/${experimentId}`)
  return response.data
}
