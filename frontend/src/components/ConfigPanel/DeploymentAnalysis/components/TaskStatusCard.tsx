/**
 * 任务状态卡片组件
 *
 * 显示运行中的评估任务进度和状态
 */

import React from 'react'
import { Loader2, CheckCircle, XCircle, StopCircle, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { InfoTooltip } from '@/components/ui/info-tooltip'
import { useCurrentTimestamp } from '@/hooks/useGlobalTimer'

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
  top_plan?: {
    parallelism: {
      dp: number
      tp: number
      pp: number
      ep: number
      sp: number
      moe_tp?: number
    }
    throughput: number
    tps_per_chip: number
    ttft: number
    tpot: number
    mfu: number
    mbu: number
    score: number
  }
  experiment_name: string
  created_at: string
}

interface TaskStatusCardProps {
  task: TaskStatus
  startTime?: number
  onCancel?: () => void
  onClose?: () => void
}

const getStatusColor = (status: string) => {
  const colorMap: Record<string, string> = {
    pending: '#faad14',
    running: '#1890ff',
    completed: '#52c41a',
    failed: '#ff4d4f',
    cancelled: '#8c8c8c',
  }
  return colorMap[status] || '#1890ff'
}

const getStatusIcon = (status: string) => {
  const iconMap: Record<string, React.ReactNode> = {
    pending: <Loader2 className="h-3 w-3 animate-spin" />,
    running: <Loader2 className="h-3 w-3 animate-spin" />,
    completed: <CheckCircle className="h-3 w-3" />,
    failed: <XCircle className="h-3 w-3" />,
    cancelled: <StopCircle className="h-3 w-3" />,
  }
  return iconMap[status] || <Loader2 className="h-3 w-3 animate-spin" />
}

const getStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return textMap[status] || status
}

export const TaskStatusCard: React.FC<TaskStatusCardProps> = ({
  task,
  startTime,
  onCancel,
  onClose,
}) => {
  // 使用全局定时器 hook 替代独立的 setInterval
  const isRunning = task.status === 'running' || task.status === 'pending'
  const currentTimestamp = useCurrentTimestamp(isRunning)

  // 计算已用时间（秒）
  const elapsedTime = startTime && isRunning
    ? Math.floor((currentTimestamp * 1000 - startTime) / 1000)
    : 0

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = seconds % 60
    if (h > 0) {
      return `${h}h${m}m${s}s`
    }
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  // 根据进度估计总耗时和剩余时间
  const estimateTotalTime = () => {
    if (task.progress <= 0 || elapsedTime <= 0) return null
    const totalTime = Math.ceil(elapsedTime / (task.progress / 100))
    const remainingTime = Math.max(0, totalTime - elapsedTime)
    return { totalTime, remainingTime }
  }

  const timeEstimate = estimateTotalTime()

  return (
    <div
      className={`mb-3 rounded-lg border p-3 ${task.status === 'failed' ? 'bg-red-50' : 'bg-white'}`}
      style={{ borderLeft: `4px solid ${getStatusColor(task.status)}` }}
    >
        {/* 头部：标题 + 操作按钮 */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className="flex items-center gap-1 text-xs"
              style={{ borderColor: getStatusColor(task.status), color: getStatusColor(task.status) }}
            >
              {getStatusIcon(task.status)}
              {getStatusText(task.status)}
            </Badge>
            <span className="font-semibold text-sm">
              {task.experiment_name || task.task_id.slice(0, 8)}
            </span>
          </div>
          <div className="flex items-center gap-1">
            {onCancel && task.status === 'running' && (
              <InfoTooltip content="取消任务">
                <Button variant="ghost" size="icon" className="h-6 w-6 text-red-500" onClick={onCancel}>
                  <StopCircle className="h-4 w-4" />
                </Button>
              </InfoTooltip>
            )}
            {onClose && task.status === 'failed' && (
              <InfoTooltip content="关闭">
                <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}>
                  <X className="h-4 w-4" />
                </Button>
              </InfoTooltip>
            )}
          </div>
        </div>

        {/* 时间信息 */}
        {startTime && (task.status === 'running' || task.status === 'pending') && (
          <div className="flex items-center gap-3 mb-2 text-[11px]">
            <InfoTooltip content="已用时间">
              <span className="text-gray-500 font-medium">⏱️ {formatTime(elapsedTime)}</span>
            </InfoTooltip>
            {timeEstimate && (
              <>
                <InfoTooltip content="预计总时间">
                  <span className="text-gray-400">/ {formatTime(timeEstimate.totalTime)}</span>
                </InfoTooltip>
                <InfoTooltip content="剩余时间">
                  <span className="text-yellow-600 font-medium">剩余: {formatTime(timeEstimate.remainingTime)}</span>
                </InfoTooltip>
              </>
            )}
          </div>
        )}

        {/* 进度条（仅运行中任务） */}
        {task.status === 'running' && (
          <div className="flex items-center gap-2 mb-2">
            <Progress value={Math.round(task.progress)} className="flex-1 h-2" />
            <span className="text-xs text-gray-500">{Math.round(task.progress)}%</span>
          </div>
        )}

        {/* 消息 */}
        {task.message && (
          <p className="text-xs text-gray-500 mb-2">{task.message}</p>
        )}

        {/* 搜索统计（运行中或完成时） */}
        {task.search_stats && (task.status === 'running' || task.status === 'completed') && (
          <div className="flex gap-2 text-[11px] mb-2">
            <span className="text-gray-500">
              总方案: <span className="font-semibold">{task.search_stats.total_plans}</span>
            </span>
            <span className="text-green-600">
              可行: <span className="font-semibold">{task.search_stats.feasible_plans}</span>
            </span>
            <span className="text-gray-500">
              不可行: {task.search_stats.infeasible_plans}
            </span>
          </div>
        )}

        {/* Benchmark 信息（完成时显示最优方案） */}
        {task.status === 'completed' && task.top_plan && (
          <div className="bg-green-50 border border-green-200 rounded p-2 mt-1">
            <div className="mb-1.5">
              <span className="font-semibold text-xs text-green-600">
                🏆 最优方案 (得分: {task.top_plan.score.toFixed(2)})
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 text-[11px]">
              <div>
                <span className="text-gray-500">并行策略:</span>
                <div>
                  DP={task.top_plan.parallelism.dp}, TP={task.top_plan.parallelism.tp},
                  PP={task.top_plan.parallelism.pp}, EP={task.top_plan.parallelism.ep}
                </div>
              </div>
              <div>
                <span className="text-gray-500">性能指标:</span>
                <div>吞吐量: {task.top_plan.throughput.toFixed(2)} tokens/s</div>
              </div>
              <div>
                <span className="text-gray-500">TTFT:</span>
                <div>{task.top_plan.ttft.toFixed(2)} ms</div>
              </div>
              <div>
                <span className="text-gray-500">TPOT:</span>
                <div>{task.top_plan.tpot.toFixed(3)} ms/token</div>
              </div>
              <div>
                <span className="text-gray-500">MFU:</span>
                <div>{(task.top_plan.mfu * 100).toFixed(1)}%</div>
              </div>
              <div>
                <span className="text-gray-500">MBU:</span>
                <div>{(task.top_plan.mbu * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        )}

        {/* 错误信息 */}
        {task.error && task.status === 'failed' && (
          <div className="bg-red-50 border border-red-200 rounded p-2 mt-1">
            <span className="text-[11px] text-red-500 font-mono whitespace-pre-wrap">
              {task.error.length > 200 ? `${task.error.slice(0, 200)}...` : task.error}
            </span>
          </div>
        )}
    </div>
  )
}
