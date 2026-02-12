/**
 * 分析任务列表组件
 *
 * - 运行中的任务显示为独立卡片（带实时进度）
 * - 已完成的任务显示在历史列表中
 */

import React, { useState } from 'react'
import { formatNumber, getMetricDecimals } from '@/utils/formatters'
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
  Info,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { InfoTooltip } from '@/components/ui/info-tooltip'
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { AnalysisTask } from '../shared'
import { useElapsedTime } from '@/hooks/useGlobalTimer'

interface AnalysisTaskListProps {
  tasks: AnalysisTask[]
  onViewTask: (task: AnalysisTask) => void
  onCancelTask: (taskId: string) => void
  onDeleteTask: (taskId: string) => void
  onClearCompleted: () => void
  onRefresh: () => void
}

// 格式化耗时（用于历史记录列表）
const formatDuration = (startTime: number, endTime?: number): string => {
  const end = endTime || Date.now()
  const duration = end - startTime
  if (duration < 1000) return `${duration}ms`
  if (duration < 60000) return `${Math.round(duration / 1000)}s`
  return `${Math.floor(duration / 60000)}m ${Math.floor((duration % 60000) / 1000)}s`
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

// 运行中/失败任务卡片组件
interface RunningTaskCardProps {
  task: AnalysisTask
  onCancel: () => void
  onDelete: () => void
}

const RunningTaskCard: React.FC<RunningTaskCardProps> = ({ task, onCancel, onDelete }) => {
  const [showProgressModal, setShowProgressModal] = useState(false)

  const isTerminated = task.status === 'failed' || task.status === 'completed' || task.status === 'cancelled'

  // 使用全局定时器 hook 替代独立的 setInterval
  const elapsedTime = useElapsedTime(task.startTime, task.endTime, !isTerminated)

  const isFailed = task.status === 'failed'

  const progress = task.progress
    ? Math.round((task.progress.current / task.progress.total) * 100)
    : 0

  const taskName = task.experimentName || task.benchmarkName || task.modelName

  // 是否为自动搜索模式（多个候选方案并行评估）
  const isAutoMode = task.mode === 'auto' && task.subTasks && task.subTasks.length > 0

  // 计算子任务统计
  const getSubTaskStats = () => {
    if (!task.subTasks) return { running: 0, completed: 0, pending: 0, failed: 0 }
    return {
      running: task.subTasks.filter(t => t.status === 'running').length,
      completed: task.subTasks.filter(t => t.status === 'completed').length,
      pending: task.subTasks.filter(t => t.status === 'pending').length,
      failed: task.subTasks.filter(t => t.status === 'failed').length,
    }
  }

  // 处理进度条点击
  const handleProgressClick = () => {
    if (isAutoMode) {
      setShowProgressModal(true)
    }
  }

  const stats = getSubTaskStats()

  return (
    <div className={`mb-3 p-3 border rounded-lg ${
      isFailed
        ? 'border-red-200 bg-red-50'
        : 'border-blue-200 bg-blue-50'
    }`}>
      {/* 标题栏 */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Zap className={`h-4 w-4 ${isFailed ? 'text-red-500' : 'text-yellow-500'}`} />
          <span className="font-medium text-sm">{taskName}</span>
          <StatusBadge status={task.status} />
        </div>
        {isFailed ? (
          <Button variant="outline" size="sm" onClick={onDelete} className="text-red-500 hover:text-red-600 border-red-200">
            <Trash2 className="h-3.5 w-3.5 mr-1" />
            关闭
          </Button>
        ) : (
          <Button variant="destructive" size="sm" onClick={onCancel}>
            <StopCircle className="h-3.5 w-3.5 mr-1" />
            取消
          </Button>
        )}
      </div>

      {/* 统计数据 */}
      {isAutoMode ? (
        // 自动模式：显示候选方案统计
        <div className="grid grid-cols-3 gap-4 mb-3">
          <StatItem title="进度" value={`${progress}%`} valueColor="#1890ff" />
          <StatItem title="运行时间" value={elapsedTime} valueColor="#1890ff" />
          <StatItem title="候选方案" value={`${stats.completed}/${task.subTasks?.length || 0}`} valueColor="#52c41a" />
        </div>
      ) : (
        // 手动模式：显示芯片数
        <div className="grid grid-cols-3 gap-4 mb-3">
          <StatItem title="进度" value={`${progress}%`} valueColor="#1890ff" />
          <StatItem title="运行时间" value={elapsedTime} valueColor="#1890ff" />
          <StatItem title="芯片数" value={task.chips || '-'} />
        </div>
      )}

      {/* 分隔线 */}
      <div className={`border-t my-3 ${isFailed ? 'border-red-200' : 'border-blue-200'}`} />

      {isFailed ? (
        /* 失败时显示错误信息 */
        <div className="p-3 bg-red-100 rounded-md">
          <div className="flex items-start gap-2">
            <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-red-700 break-all">
              {task.error || '未知错误'}
            </div>
          </div>
        </div>
      ) : (
        /* 进度条（自动模式可点击查看详情） */
        <div
          style={{
            cursor: isAutoMode ? 'pointer' : 'default',
          }}
          onClick={handleProgressClick}
        >
          <Progress value={progress} className="h-2" />
        </div>
      )}

      {/* 详情 */}
      <div className="mt-2 text-xs text-gray-600 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span>模式: {task.mode === 'auto' ? '自动搜索' : '手动'}</span>
          <span className="text-gray-300">|</span>
          <span>策略: {formatParallelism(task.parallelism)}</span>
        </div>
        {isAutoMode && !isFailed && (
          <span className="text-xs text-gray-500 flex items-center gap-1">
            <Info className="h-3 w-3" />
            点击进度条查看详情
          </span>
        )}
      </div>

      {/* 子任务详情弹窗 */}
      <Dialog open={showProgressModal} onOpenChange={setShowProgressModal}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>候选方案评估进度</DialogTitle>
          </DialogHeader>

          {task.subTasks && (
            <div>
              {/* 统计信息 */}
              <div className="grid grid-cols-4 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
                <StatItem title="运行中" value={stats.running} valueColor="#1890ff" />
                <StatItem title="已完成" value={stats.completed} valueColor="#52c41a" />
                <StatItem title="等待中" value={stats.pending} valueColor="#faad14" />
                <StatItem title="失败" value={stats.failed} valueColor="#ff4d4f" />
              </div>

              {/* 运行中的子任务列表 */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700 mb-2">运行中的方案</h4>
                {task.subTasks.filter(t => t.status === 'running').length > 0 ? (
                  task.subTasks
                    .filter(t => t.status === 'running')
                    .map((subTask) => (
                      <div key={subTask.candidateIndex} className="p-3 border border-blue-200 bg-blue-50 rounded">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-blue-600">
                            方案 {subTask.candidateIndex + 1}
                          </span>
                          <span className="text-xs text-gray-600">
                            {formatParallelism(subTask.parallelism)}
                            {subTask.chips && ` · ${subTask.chips}芯片`}
                          </span>
                        </div>
                        <Progress value={subTask.progress} className="h-2" />
                        <div className="text-xs text-gray-500 mt-1">{subTask.progress}%</div>
                      </div>
                    ))
                ) : (
                  <div className="text-center py-6 text-gray-400 text-sm">暂无运行中的方案</div>
                )}
              </div>

              {/* 已完成的子任务列表（折叠显示） */}
              {stats.completed > 0 && (
                <details className="mt-4">
                  <summary className="text-sm font-medium text-gray-700 cursor-pointer mb-2">
                    已完成的方案 ({stats.completed})
                  </summary>
                  <div className="space-y-2 mt-2">
                    {task.subTasks
                      .filter(t => t.status === 'completed')
                      .map((subTask) => (
                        <div key={subTask.candidateIndex} className="p-2 border border-green-200 bg-green-50 rounded flex items-center justify-between">
                          <span className="text-sm text-green-700">
                            方案 {subTask.candidateIndex + 1}
                          </span>
                          <span className="text-xs text-gray-600">
                            {formatParallelism(subTask.parallelism)}
                            {subTask.chips && ` · ${subTask.chips}芯片`}
                          </span>
                        </div>
                      ))}
                  </div>
                </details>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
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
                TTFT: {formatNumber(task.ttft, getMetricDecimals('ttft'))}ms · TPOT: {formatNumber(task.tpot, getMetricDecimals('tpot'))}ms · {formatDuration(task.startTime, task.endTime)}
              </span>
            ) : task.status === 'failed' ? (
              <InfoTooltip content={task.error || ''}>
                <span className="text-red-500 cursor-help">
                  {task.error?.slice(0, 50)}{task.error && task.error.length > 50 ? '...' : ''}
                </span>
              </InfoTooltip>
            ) : (
              <span className="text-gray-500">已取消 · {formatDuration(task.startTime, task.endTime)}</span>
            )}
          </div>
        </div>

        {/* 操作按钮 */}
        <div className="flex items-center gap-1 ml-2">
          {task.status === 'completed' && (
            <InfoTooltip content="查看结果">
              <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={onView}>
                <Eye className="h-3.5 w-3.5" />
              </Button>
            </InfoTooltip>
          )}
          <InfoTooltip content="删除">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-red-500 hover:text-red-600"
              onClick={onDelete}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </InfoTooltip>
        </div>
      </div>
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
  // 分离运行中/失败的任务和历史任务
  // 失败的任务也显示为卡片，方便查看错误详情
  const runningTasks = tasks.filter(t => t.status === 'running' || t.status === 'failed')
  const historyTasks = tasks.filter(t => t.status !== 'running' && t.status !== 'failed')

  if (tasks.length === 0) {
    return (
      <div className="py-8 text-center">
        <div className="text-gray-400 text-sm">暂无分析任务</div>
      </div>
    )
  }

  return (
    <div>
      {/* 运行中/失败的任务卡片 */}
      {runningTasks.map(task => (
        <RunningTaskCard
          key={task.id}
          task={task}
          onCancel={() => onCancelTask(task.id)}
          onDelete={() => onDeleteTask(task.id)}
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
