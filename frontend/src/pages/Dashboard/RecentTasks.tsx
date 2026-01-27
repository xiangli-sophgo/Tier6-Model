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

export const RecentTasks: React.FC<RecentTasksProps> = ({ tasks, loading, onNavigate }) => {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <History className="h-4 w-4 text-blue-500" />
          <span>最近任务</span>
        </CardTitle>
        <Button
          variant="link"
          onClick={onNavigate}
          className="h-auto p-0 text-sm"
        >
          查看全部 <ArrowRight className="ml-1 h-3 w-3" />
        </Button>
      </CardHeader>

      <CardContent>
        {tasks.length === 0 && !loading ? (
          <div className="py-10 text-center">
            <Zap className="mx-auto mb-4 h-12 w-12 text-border" />
            <div>
              <p className="text-sm text-text-secondary">暂无任务记录</p>
            </div>
            <Button
              className="mt-4"
              onClick={onNavigate}
            >
              开始第一次评估
            </Button>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-6 w-6 animate-spin text-text-secondary" />
          </div>
        ) : (
          <div className="space-y-4">
            {tasks.map((task) => {
              const statusConfig = getStatusConfig(task.status)
              return (
                <div key={task.task_id} className="border-b border-border pb-4 last:border-0 last:pb-0">
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-sm font-medium">
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
                  <div className="mt-1">
                    <p className="text-[11px] text-text-secondary">
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
