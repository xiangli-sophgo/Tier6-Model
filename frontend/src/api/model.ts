/**
 * Model API Client
 *
 * 模型相关的 HTTP 请求封装
 */

import type { LLMModelConfig } from '../utils/llmDeployment/types'
import { apiClient as api } from './client'

// ============================================
// 类型定义
// ============================================

export interface ModelParamsResponse {
  total_params: number
  total_params_b: number
  active_params: number
  active_params_b: number
  weight_size_bytes: number
  weight_size_gb: number
  breakdown: Record<string, unknown>
}

// ============================================
// API 函数
// ============================================

/**
 * 计算模型参数量
 *
 * @param model 模型配置
 * @returns 参数量信息
 */
export async function calculateModelParams(
  model: LLMModelConfig | Record<string, unknown>
): Promise<ModelParamsResponse> {
  const response = await api.post<ModelParamsResponse>('/model/calculate-params', { model_config: model })
  return response.data
}
