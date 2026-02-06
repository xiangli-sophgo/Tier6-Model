/**
 * WebSocket Hook for Task Updates
 *
 * 连接到后端 WebSocket 端点，实时接收任务状态更新
 */

import { useEffect, useRef, useCallback } from 'react'

export interface SubTaskProgress {
  candidate_index: number
  parallelism: {
    dp: number
    tp: number
    pp: number
    ep: number
    sp: number
    moe_tp?: number
  }
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  chips?: number
}

export interface TaskUpdate {
  type: 'task_update'
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  message: string
  error?: string
  search_stats?: {
    total_plans?: number
    feasible_plans?: number
    infeasible_plans?: number
    sub_tasks?: SubTaskProgress[]
  }
}

interface UseTaskWebSocketOptions {
  /** 任务更新回调 */
  onTaskUpdate?: (update: TaskUpdate) => void
  /** WebSocket 连接成功回调 */
  onConnect?: () => void
  /** WebSocket 断开连接回调 */
  onDisconnect?: () => void
  /** 是否自动重连（默认 true） */
  autoReconnect?: boolean
}

/**
 * 使用 WebSocket 订阅任务更新
 *
 * @example
 * useTaskWebSocket({
 *   onTaskUpdate: (update) => {
 *     console.log('Task updated:', update)
 *   }
 * })
 */
export function useTaskWebSocket(options: UseTaskWebSocketOptions = {}) {
  const {
    onTaskUpdate,
    onConnect,
    onDisconnect,
    autoReconnect = true,
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectCountRef = useRef(0)

  // 使用 ref 保存最新的回调，避免闭包问题
  const onTaskUpdateRef = useRef(onTaskUpdate)
  const onConnectRef = useRef(onConnect)
  const onDisconnectRef = useRef(onDisconnect)

  // 每次渲染时更新 ref
  useEffect(() => {
    onTaskUpdateRef.current = onTaskUpdate
    onConnectRef.current = onConnect
    onDisconnectRef.current = onDisconnect
  })

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return // 已连接
    }

    // 使用 Vite 代理路径，连接到后端的 WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/api/ws/tasks`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        reconnectCountRef.current = 0
        onConnectRef.current?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TaskUpdate
          if (data.type === 'task_update') {
            onTaskUpdateRef.current?.(data)
          }
          // heartbeat 消息静默处理
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
      }

      ws.onclose = () => {
        onDisconnectRef.current?.()

        // 自动重连
        if (autoReconnect) {
          reconnectCountRef.current++
          const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current), 30000) // 指数退避，最大 30 秒
          reconnectTimerRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error)
    }
  }, [autoReconnect]) // 只依赖 autoReconnect，回调通过 ref 获取

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
    reconnect: connect,
    disconnect,
  }
}
