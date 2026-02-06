/**
 * Benchmark API Client
 *
 * Benchmark 配置相关的 HTTP 请求封装
 */

import axios from 'axios'
import type { LLMModelConfig, InferenceConfig } from '../utils/llmDeployment/types'

const API_BASE_URL = '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10秒超时
})

// ============================================
// 类型定义
// ============================================

export interface BenchmarkConfig {
  id: string
  name: string
  model: LLMModelConfig
  inference: InferenceConfig
}

export interface BenchmarkListItem {
  id: string
  name: string
  model_name: string
  model_type: string
  batch_size: number
  input_seq_length: number
  output_seq_length: number
}

// ============================================
// API 函数
// ============================================

/**
 * 获取 Benchmark 配置列表
 */
export async function listBenchmarks(): Promise<BenchmarkListItem[]> {
  const response = await api.get<BenchmarkListItem[]>('/benchmarks')
  return response.data
}

/**
 * 获取单个 Benchmark 配置
 */
export async function getBenchmark(benchmarkId: string): Promise<BenchmarkConfig> {
  const response = await api.get<BenchmarkConfig>(`/benchmarks/${benchmarkId}`)
  return response.data
}

/**
 * 创建新的 Benchmark 配置
 */
export async function createBenchmark(benchmark: BenchmarkConfig): Promise<BenchmarkConfig> {
  const response = await api.post<BenchmarkConfig>('/benchmarks', benchmark)
  return response.data
}

/**
 * 更新 Benchmark 配置
 */
export async function updateBenchmark(
  benchmarkId: string,
  benchmark: BenchmarkConfig
): Promise<BenchmarkConfig> {
  const response = await api.put<BenchmarkConfig>(`/benchmarks/${benchmarkId}`, benchmark)
  return response.data
}

/**
 * 删除 Benchmark 配置
 */
export async function deleteBenchmark(benchmarkId: string): Promise<void> {
  await api.delete(`/benchmarks/${benchmarkId}`)
}
