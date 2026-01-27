/**
 * 分析任务列表组件
 *
 * - 运行中的任务显示为独立卡片（带实时进度）
 * - 已完成的任务显示在历史列表中
 */

import React, { useState, useEffect } from 'react'
import {
  Loader2,
  CheckCircle,
  XCircle,
  StopCircle,
  Trash2,
  Trash,
  Eye,
  RefreshCw,
  Zap,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { AnalysisTask } from '../shared'

interface AnalysisTaskListProps {
  tasks: AnalysisTask[]
  onViewTask: (task: AnalysisTask) => void
  onCancelTask: (taskId: string) => void
  onDeleteTask: (taskId: string) => void
  onClearCompleted: () => void
  onRefresh: () => void
}

// 格式化耗时
const formatDuration = (startTime: number, endTime?: number): string => {
  const end = endTime || Date.now()
  const duration = end - startTime
  if (duration < 1000) return `${duration}ms`
  if (duration < 60000) return `${Math.round(duration / 1000)}s`
  return `${Math.floor(duration / 60000)}m ${Math.floor((duration % 60000) / 1000)}s`
}

// 获取运行时间（动态更新）
const getElapsedTime = (startTime: number | null): string => {
  if (!startTime) return '0s'
  return formatDuration(startTime)
}

// 格式化并行策略
const formatParallelism = (p: AnalysisTask['parallelism']): string => {
  const parts: string[] = []
  if (p.tp > 1) parts.push(`TP${p.tp}`)
  if (p.pp > 1) parts.push(`PP${p.pp}`)
  if (p.dp > 1) parts.push(`DP${p.dp}`)
  if (p.ep > 1) parts.push(`EP${p.ep}`)
  if (p.moe_tp && p.moe_tp > 1) parts.push(`MoE_TP${p.moe_tp}`)
  return parts.length > 0 ? parts.join(' ') : 'TP1'
}

// 状态标签
const StatusBadge: React.FC<{ status: AnalysisTask['status'] }> = ({ status }) => {
  const configs: Record<string, { className: string; text: string; icon: React.ReactNode }> = {
    running: {
      className: 'bg-blue-100 text-blue-700 border-blue-200',
      text: '运行中',
      icon: <Loader2 className="h-3 w-3 animate-spin" />,
    },
    completed: {
      className: 'bg-green-100 text-green-700 border-green-200',
      text: '已完成',
      icon: <CheckCircle className="h-3 w-3" />,
    },
    failed: {
      className: 'bg-red-100 text-red-700 border-red-200',
      text: '失败',
      icon: <XCircle className="h-3 w-3" />,
    },
    cancelled: {
      className: 'bg-gray-100 text-gray-600 border-gray-200',
      text: '已取消',
      icon: <StopCircle className="h-3 w-3" />,
    },
  }

  const config = configs[status]
  if (!config) return null

  return (
    <Badge variant="outline" className={`gap-1 ${config.className}`}>
      {config.icon}
      {config.text}
    </Badge>
  )
}

// 统计项组件
const StatItem: React.FC<{ title: string; value: string | number; valueColor?: string }> = ({
  title,
  value,
  valueColor,
}) => (
  <div className="text-center">
    <div className="text-xs text-gray-500 mb-1">{title}</div>
    <div className="text-xl font-semibold" style={valueColor ? { color: valueColor } : undefined}>
      {value}
    </div>
  </div>
)

// 运行中任务卡片组件
interface RunningTaskCardProps {
  task: AnalysisTask
  onCancel: () => void
}

const RunningTaskCard: React.FC<RunningTaskCardProps> = ({ task, onCancel }) => {
  const [elapsedTime, setElapsedTime] = useState(getElapsedTime(task.startTime))

  // 每秒更新运行时间
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(getElapsedTime(task.startTime))
    }, 1000)
    return () => clearInterval(timer)
  }, [task.startTime])

  const progress = task.progress
    ? Math.round((task.progress.current / task.progress.total) * 100)
    : 0

  const taskName = task.experimentName || task.benchmarkName || task.modelName

  return (
    <div className="mb-3 p-3 border border-blue-200 bg-blue-50 rounded-lg">
      {/* 标题栏 */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-yellow-500" />
          <span className="font-medium text-sm">{taskName}</span>
          <StatusBadge status={task.status} />
        </div>
        <Button variant="destructive" size="sm" onClick={onCancel}>
          <StopCircle className="h-3.5 w-3.5 mr-1" />
          取消
        </Button>
      </div>

      {/* 统计数据 */}
      <div className="grid grid-cols-3 gap-4 mb-3">
        <StatItem title="进度" value={`${progress}%`} valueColor="#1890ff" />
        <StatItem title="运行时间" value={elapsedTime} valueColor="#1890ff" />
        <StatItem title="芯片数" value={task.chips || '-'} />
      </div>

      {/* 分隔线 */}
      <div className="border-t border-blue-200 my-3" />

      {/* 进度条 */}
      <Progress value={progress} className="h-2" />

      {/* 详情 */}
      <div className="mt-2 text-xs text-gray-600 flex items-center gap-2">
        <span>模式: {task.mode === 'auto' ? '自动搜索' : '手动'}</span>
        <span className="text-gray-300">|</span>
        <span>策略: {formatParallelism(task.parallelism)}</span>
      </div>
    </div>
  )
}

// 历史任务列表项
interface HistoryTaskItemProps {
  task: AnalysisTask
  onView: () => void
  onDelete: () => void
}

const HistoryTaskItem: React.FC<HistoryTaskItemProps> = ({ task, onView, onDelete }) => {
  return (
    <TooltipProvider>
      <div className="flex items-center justify-between p-2 bg-gray-50 mb-1 rounded border border-gray-100">
        <div className="flex-1 min-w-0">
          {/* 标题行 */}
          <div className="flex items-center gap-2 flex-wrap">
            <StatusBadge status={task.status} />
            <span className="text-[13px] font-medium">
              {task.experimentName || task.benchmarkName || task.modelName}
            </span>
            <span className="text-xs text-gray-400">|</span>
            <span className="text-xs text-gray-600">{formatParallelism(task.parallelism)}</span>
          </div>

          {/* 描述行 */}
          <div className="text-xs mt-1">
            {task.status === 'completed' ? (
              <span className="text-green-600">
                TTFT: {task.ttft?.toFixed(1)}ms · TPOT: {task.tpot?.toFixed(2)}ms · {formatDuration(task.startTime, task.endTime)}
              </span>
            ) : task.status === 'failed' ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-red-500 cursor-help">
                    {task.error?.slice(0, 50)}{task.error && task.error.length > 50 ? '...' : ''}
                  </span>
                </TooltipTrigger>
                <TooltipContent>{task.error}</TooltipContent>
              </Tooltip>
            ) : (
              <span className="text-gray-500">已取消 · {formatDuration(task.startTime, task.endTime)}</span>
            )}
          </div>
        </div>

        {/* 操作按钮 */}
        <div className="flex items-center gap-1 ml-2">
          {task.status === 'completed' && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={onView}>
                  <Eye className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>查看结果</TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0 text-red-500 hover:text-red-600"
                onClick={onDelete}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>删除</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  )
}

export const AnalysisTaskList: React.FC<AnalysisTaskListProps> = ({
  tasks,
  onViewTask,
  onCancelTask,
  onDeleteTask,
  onClearCompleted,
  onRefresh,
}) => {
  // 分离运行中的任务和历史任务
  const runningTasks = tasks.filter(t => t.status === 'running')
  const historyTasks = tasks.filter(t => t.status !== 'running')

  if (tasks.length === 0) {
    return (
      <div className="py-8 text-center">
        <div className="text-gray-400 text-sm">暂无分析任务</div>
      </div>
    )
  }

  return (
    <div>
      {/* 运行中的任务卡片 */}
      {runningTasks.map(task => (
        <RunningTaskCard
          key={task.id}
          task={task}
          onCancel={() => onCancelTask(task.id)}
        />
      ))}

      {/* 历史任务列表 */}
      {historyTasks.length > 0 && (
        <>
          <div
            className={`flex justify-between items-center mb-2 px-1 ${
              runningTasks.length > 0 ? 'mt-4' : ''
            }`}
          >
            <span className="text-xs text-gray-600">
              历史任务 ({historyTasks.length})
            </span>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={onRefresh}>
                <RefreshCw className="h-3 w-3 mr-1" />
                刷新
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 text-xs text-red-500 hover:text-red-600">
                    <Trash className="h-3 w-3 mr-1" />
                    清空
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>确认清空</AlertDialogTitle>
                    <AlertDialogDescription>
                      确定清空所有历史任务？此操作不可撤销。
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>取消</AlertDialogCancel>
                    <AlertDialogAction
                      className="bg-red-600 hover:bg-red-700"
                      onClick={onClearCompleted}
                    >
                      确定
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </div>

          <div className="max-h-[200px] overflow-y-auto">
            {historyTasks.map((task) => (
              <HistoryTaskItem
                key={task.id}
                task={task}
                onView={() => onViewTask(task)}
                onDelete={() => onDeleteTask(task.id)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export default AnalysisTaskList
