/**
 * Model API Client
 *
 * 模型相关的 HTTP 请求封装
 */

import axios from 'axios'
import type { LLMModelConfig } from '../utils/llmDeployment/types'

const API_PORT = import.meta.env.VITE_API_PORT || '8001'
const API_BASE_URL = `http://localhost:${API_PORT}/api`

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10秒超时
})

// ============================================
// 类型定义
// ============================================

export interface ModelParamsResponse {
  params: number
  formatted: string
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
  const response = await api.post<ModelParamsResponse>('/model/calculate-params', model)
  return response.data
}

/**
 * 格式化参数量显示（本地实现，无需调用后端）
 *
 * @param params 参数量
 * @returns 格式化字符串
 */
export function formatParams(params: number): string {
  if (params >= 1e12) {
    return `${(params / 1e12).toFixed(1)}T`
  } else if (params >= 1e9) {
    return `${(params / 1e9).toFixed(1)}B`
  } else if (params >= 1e6) {
    return `${(params / 1e6).toFixed(1)}M`
  } else if (params >= 1e3) {
    return `${(params / 1e3).toFixed(1)}K`
  }
  return String(params)
}
