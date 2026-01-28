/**
 * 最近任务列表组件
 */

import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  History,
  ArrowRight,
  CheckCircle,
  Clock,
  Loader2,
  XCircle,
  Zap,
} from 'lucide-react'

interface TaskSummary {
  task_id: string
  experiment_name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress?: number
  message?: string
  created_at: string
}

interface RecentTasksProps {
  tasks: TaskSummary[]
  loading: boolean
  onNavigate: () => void
  onTaskClick?: (task: TaskSummary) => void
}

const getStatusConfig = (status: string) => {
  const statusMap: Record<
    string,
    { variant: 'default' | 'processing' | 'success' | 'destructive' | 'warning'; text: string; icon: React.ReactNode }
  > = {
    pending: { variant: 'default', text: '等待中', icon: <Clock className="h-3 w-3" /> },
    running: { variant: 'processing', text: '运行中', icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    completed: { variant: 'success', text: '已完成', icon: <CheckCircle className="h-3 w-3" /> },
    failed: { variant: 'destructive', text: '失败', icon: <XCircle className="h-3 w-3" /> },
    cancelled: { variant: 'warning', text: '已取消', icon: <XCircle className="h-3 w-3" /> },
  }
  return statusMap[status] || { variant: 'default' as const, text: status, icon: null }
}

export const RecentTasks: React.FC<RecentTasksProps> = ({ tasks, loading, onNavigate, onTaskClick }) => {
  return (
    <Card className="border-0">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-6">
        <CardTitle className="flex items-center gap-3 text-lg">
          <div className="rounded-lg bg-blue-50 p-2">
            <History className="h-5 w-5 text-blue-600" />
          </div>
          <span>最近任务</span>
        </CardTitle>
        <Button
          variant="ghost"
          onClick={onNavigate}
          className="h-auto gap-1 p-0 text-sm text-blue-600 hover:text-blue-700"
        >
          查看全部 <ArrowRight className="h-3 w-3" />
        </Button>
      </CardHeader>

      <CardContent>
        {tasks.length === 0 && !loading ? (
          <div className="py-12 text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-blue-50">
              <Zap className="h-8 w-8 text-blue-300" />
            </div>
            <div>
              <p className="text-sm text-text-secondary">暂无任务记录</p>
            </div>
            <Button
              className="mt-6 bg-blue-600 hover:bg-blue-700"
              onClick={onNavigate}
            >
              开始第一次评估
            </Button>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
          </div>
        ) : (
          <div className="space-y-3">
            {tasks.map((task) => {
              const statusConfig = getStatusConfig(task.status)
              return (
                <div
                  key={task.task_id}
                  className="rounded-lg border border-blue-50 bg-blue-50/30 p-4 transition-all hover:bg-blue-50/50 hover:border-blue-200 cursor-pointer"
                  onClick={() => onTaskClick?.(task)}
                >
                  <div className="mb-2 flex items-center gap-3">
                    <span className="text-sm font-semibold text-text-primary">
                      {task.experiment_name || task.task_id.slice(0, 8)}
                    </span>
                    <Badge variant={statusConfig.variant} className="gap-1">
                      {statusConfig.icon}
                      {statusConfig.text}
                    </Badge>
                  </div>
                  {task.message && (
                    <p className="text-xs text-text-secondary">
                      {task.message}
                    </p>
                  )}
                  <div className="mt-2">
                    <p className="text-[11px] text-text-muted">
                      {new Date(task.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
