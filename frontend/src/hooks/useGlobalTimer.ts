/**
 * 全局定时器 Hook
 *
 * 使用单一定时器管理所有订阅者的时间更新，避免每个组件创建独立的 setInterval
 * 这可以显著减少高频状态更新带来的性能开销
 */

import { useEffect, useState, useCallback } from 'react'

// 全局定时器管理（单例模式）
class GlobalTimerManager {
  private static instance: GlobalTimerManager
  private timer: ReturnType<typeof setInterval> | null = null
  private subscribers = new Set<() => void>()

  static getInstance() {
    if (!GlobalTimerManager.instance) {
      GlobalTimerManager.instance = new GlobalTimerManager()
    }
    return GlobalTimerManager.instance
  }

  subscribe(callback: () => void) {
    this.subscribers.add(callback)
    if (this.subscribers.size === 1) {
      this.startTimer()
    }
    return () => {
      this.subscribers.delete(callback)
      if (this.subscribers.size === 0) {
        this.stopTimer()
      }
    }
  }

  private startTimer() {
    if (this.timer) return
    this.timer = setInterval(() => {
      this.subscribers.forEach(cb => cb())
    }, 1000)
  }

  private stopTimer() {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
  }
}

// 获取全局单例（稳定引用，不会变化）
const globalTimerManager = GlobalTimerManager.getInstance()

/**
 * 使用全局定时器来追踪经过的时间
 *
 * @param startTime 开始时间戳（毫秒）
 * @param endTime 结束时间戳（毫秒），如果提供则返回固定值
 * @param enabled 是否启用定时更新
 * @returns 格式化的经过时间字符串
 */
export function useElapsedTime(
  startTime: number | undefined,
  endTime?: number,
  enabled = true
): string {
  const [, forceUpdate] = useState(0)

  const getElapsedTime = useCallback(() => {
    if (!startTime) return '0s'
    const end = endTime || Date.now()
    const elapsed = Math.floor((end - startTime) / 1000)
    if (elapsed < 60) return `${elapsed}s`
    if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`
    const h = Math.floor(elapsed / 3600)
    const m = Math.floor((elapsed % 3600) / 60)
    const s = elapsed % 60
    return `${h}h ${m}m ${s}s`
  }, [startTime, endTime])

  useEffect(() => {
    // 如果有 endTime 或禁用，不需要订阅更新
    if (endTime || !enabled || !startTime) return

    const unsubscribe = globalTimerManager.subscribe(() => {
      forceUpdate(n => n + 1)
    })

    return unsubscribe
  }, [startTime, endTime, enabled])

  return getElapsedTime()
}

/**
 * 使用全局定时器获取当前时间戳（每秒更新）
 *
 * @param enabled 是否启用
 * @returns 当前时间戳（秒级精度）
 */
export function useCurrentTimestamp(enabled = true): number {
  const [timestamp, setTimestamp] = useState(() => Math.floor(Date.now() / 1000))

  useEffect(() => {
    if (!enabled) return

    const unsubscribe = globalTimerManager.subscribe(() => {
      setTimestamp(Math.floor(Date.now() / 1000))
    })

    return unsubscribe
  }, [enabled])

  return timestamp
}
