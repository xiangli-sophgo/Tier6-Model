/**
 * 任务历史表格组件
 *
 * 显示所有历史评估任务
 */

import React, { useState } from 'react'
import {
  RefreshCw,
  Trash2,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  StopCircle,
  Eye,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
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

export interface TaskHistoryItem {
  task_id: string
  experiment_name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  message: string
  created_at: string
  started_at?: string
  completed_at?: string
}

interface TaskHistoryTableProps {
  tasks: TaskHistoryItem[]
  loading: boolean
  onRefresh: () => void
  onViewResult: (taskId: string) => void
  onDelete: (taskId: string) => void
}

const getStatusBadge = (status: string) => {
  const statusMap: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline'; text: string; icon: React.ReactNode }> = {
    pending: { variant: 'secondary', text: '等待中', icon: <Clock className="h-3 w-3" /> },
    running: { variant: 'default', text: '运行中', icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    completed: { variant: 'default', text: '已完成', icon: <CheckCircle className="h-3 w-3" /> },
    failed: { variant: 'destructive', text: '失败', icon: <XCircle className="h-3 w-3" /> },
    cancelled: { variant: 'outline', text: '已取消', icon: <StopCircle className="h-3 w-3" /> },
  }
  const { variant, text, icon } = statusMap[status] || { variant: 'secondary' as const, text: status, icon: null }

  const colorClasses: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-700 border-gray-200',
    running: 'bg-blue-100 text-blue-700 border-blue-200',
    completed: 'bg-green-100 text-green-700 border-green-200',
    failed: 'bg-red-100 text-red-700 border-red-200',
    cancelled: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  }

  return (
    <Badge variant={variant} className={`gap-1 ${colorClasses[status] || ''}`}>
      {icon}
      {text}
    </Badge>
  )
}

export const TaskHistoryTable: React.FC<TaskHistoryTableProps> = ({
  tasks,
  loading,
  onRefresh,
  onViewResult,
  onDelete,
}) => {
  const [currentPage, setCurrentPage] = useState(1)
  const pageSize = 10
  const totalPages = Math.ceil(tasks.length / pageSize)
  const paginatedTasks = tasks.slice((currentPage - 1) * pageSize, currentPage * pageSize)

  return (
    <div className="mt-6">
      <div className="flex justify-between items-center mb-3">
        <span className="font-semibold text-sm">历史任务</span>
        <Button variant="outline" size="sm" onClick={onRefresh}>
          <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
          刷新
        </Button>
      </div>

      {tasks.length === 0 ? (
        <div className="py-10 text-center border rounded-md bg-gray-50">
          <div className="text-gray-400 text-sm mb-2">暂无历史任务</div>
          <div className="text-gray-400 text-xs">运行评估后会在这里显示历史记录</div>
        </div>
      ) : (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">实验名称</TableHead>
                <TableHead className="w-[100px]">状态</TableHead>
                <TableHead className="w-[80px]">进度</TableHead>
                <TableHead>消息</TableHead>
                <TableHead className="w-[160px]">创建时间</TableHead>
                <TableHead className="w-[120px]">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8">
                    <Loader2 className="h-5 w-5 animate-spin mx-auto text-gray-400" />
                  </TableCell>
                </TableRow>
              ) : (
                paginatedTasks.map((task) => (
                  <TableRow key={task.task_id}>
                    <TableCell className="font-medium text-[13px]">{task.experiment_name}</TableCell>
                    <TableCell>{getStatusBadge(task.status)}</TableCell>
                    <TableCell className="text-gray-500 text-sm">
                      {task.status === 'running' ? `${Math.round(task.progress)}%` : null}
                    </TableCell>
                    <TableCell className="text-gray-500 text-xs truncate max-w-[200px]">
                      {task.message}
                    </TableCell>
                    <TableCell className="text-gray-500 text-xs">
                      {new Date(task.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        {task.status === 'completed' && (
                          <Button
                            variant="link"
                            size="sm"
                            className="h-7 px-2 text-blue-600"
                            onClick={() => onViewResult(task.task_id)}
                          >
                            <Eye className="h-3.5 w-3.5 mr-1" />
                            查看
                          </Button>
                        )}
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button variant="link" size="sm" className="h-7 px-2 text-red-600">
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>确认删除</AlertDialogTitle>
                              <AlertDialogDescription>
                                确定要删除此任务记录吗？
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>取消</AlertDialogCancel>
                              <AlertDialogAction
                                className="bg-red-600 hover:bg-red-700"
                                onClick={() => onDelete(task.task_id)}
                              >
                                删除
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          {/* 分页 */}
          {totalPages > 1 && (
            <div className="flex justify-between items-center mt-3 text-sm">
              <span className="text-gray-500">共 {tasks.length} 个任务</span>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage((p) => p - 1)}
                >
                  上一页
                </Button>
                <span className="text-gray-600">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage((p) => p + 1)}
                >
                  下一页
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
