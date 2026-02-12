/**
 * 共享 axios 客户端
 * 统一 baseURL、超时、错误拦截器
 */

import axios from 'axios'

const API_BASE_URL = '/api'

/** 标准 API 客户端 (10s 超时) */
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
})

/** 长时间运行的 API 客户端 (120s 超时, withCredentials) */
export const longApiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  withCredentials: true,
})

// 统一错误拦截器
const errorInterceptor = (error: any) => {
  if (error.response) {
    console.error(`[API Error] ${error.response.status}: ${error.config?.url}`)
  } else if (error.request) {
    console.error(`[API Error] No response: ${error.config?.url}`)
  }
  return Promise.reject(error)
}

apiClient.interceptors.response.use((r) => r, errorInterceptor)
longApiClient.interceptors.response.use((r) => r, errorInterceptor)
