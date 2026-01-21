/**
 * WebSocket Hook for Task Updates
 *
 * 连接到后端 WebSocket 端点，实时接收任务状态更新
 */

import { useEffect, useRef, useCallback } from 'react'

export interface TaskUpdate {
  type: 'task_update'
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

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return // 已连接
    }

    const apiPort = import.meta.env.VITE_API_PORT || '8001'
    const wsUrl = `ws://localhost:${apiPort}/ws/tasks`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('[WebSocket] Connected')
        reconnectCountRef.current = 0
        onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TaskUpdate
          if (data.type === 'task_update') {
            onTaskUpdate?.(data)
          }
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
      }

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected')
        onDisconnect?.()

        // 自动重连
        if (autoReconnect) {
          reconnectCountRef.current++
          const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current), 30000) // 指数退避，最大 30 秒
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectCountRef.current})`)
          reconnectTimerRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error)
    }
  }, [onTaskUpdate, onConnect, onDisconnect, autoReconnect])

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
