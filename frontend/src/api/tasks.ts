/**
 * Task Management API Client
 *
 * 任务管理相关的 HTTP 请求封装
 */

import { longApiClient as api } from './client'

// ============================================
// 类型定义
// ============================================

/**
 * 评估请求
 *
 * 前端传递完整配置内容，后端直接使用：
 * - benchmark_name: 配置来源标记（用于显示）
 * - benchmark_config: 完整 benchmark 配置（model + inference）
 * - topology_config_name: 拓扑配置来源标记（用于显示）
 * - topology_config: 完整拓扑配置
 */
export interface EvaluationRequest {
  experiment_name: string
  description?: string

  // 配置来源标记（必填，用于显示和追溯）
  benchmark_name: string           // Benchmark 名称（如 DeepSeek-V3-671B-S32K-O1K-W8A8-B1）
  topology_config_name: string     // Topology 名称（如 P1-R1-B1-C8）

  // 完整配置内容（必填）
  benchmark_config: {
    model: Record<string, unknown>
    inference: Record<string, unknown>
  }
  topology_config: Record<string, unknown>  // 完整拓扑配置

  // 搜索配置
  search_mode: 'manual' | 'auto' | 'sweep'
  manual_parallelism?: Record<string, unknown>
  search_constraints?: Record<string, unknown>

  // 任务配置
  max_workers?: number             // 本任务的最大并发数（默认 4）
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

// ============================================
// 执行器配置 API
// ============================================

export interface ExecutorConfig {
  max_workers: number
  running_tasks: number
  active_tasks: number
  note: string
}

export interface ExecutorConfigUpdateRequest {
  max_workers: number
}

export interface ExecutorConfigUpdateResponse {
  success: boolean
  message: string
  current_max_workers: number
  new_max_workers: number
}

/**
 * 获取执行器配置
 */
export async function getExecutorConfig(): Promise<ExecutorConfig> {
  const response = await api.get<ExecutorConfig>('/evaluation/config')
  return response.data
}

/**
 * 更新执行器配置
 */
export async function updateExecutorConfig(
  request: ExecutorConfigUpdateRequest
): Promise<ExecutorConfigUpdateResponse> {
  const response = await api.put<ExecutorConfigUpdateResponse>('/evaluation/config', request)
  return response.data
}
