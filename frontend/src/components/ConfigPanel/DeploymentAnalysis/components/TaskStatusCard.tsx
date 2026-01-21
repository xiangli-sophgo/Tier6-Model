/**
 * 任务状态卡片组件
 *
 * 显示运行中的评估任务进度和状态
 */

import React, { useEffect, useState } from 'react'
import { Card, Progress, Button, Tag, Space, Typography, Tooltip } from 'antd'
import {
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
  CloseOutlined,
} from '@ant-design/icons'

const { Text } = Typography

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
    pending: <LoadingOutlined spin />,
    running: <LoadingOutlined spin />,
    completed: <CheckCircleOutlined />,
    failed: <CloseCircleOutlined />,
    cancelled: <StopOutlined />,
  }
  return iconMap[status] || <LoadingOutlined spin />
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
  const [elapsedTime, setElapsedTime] = useState(0)

  // 计算已用时间
  useEffect(() => {
    if (!startTime) return
    if (task.status !== 'running' && task.status !== 'pending') return

    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)

    return () => clearInterval(timer)
  }, [startTime, task.status])

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  return (
    <Card
      size="small"
      style={{
        marginBottom: 12,
        borderLeft: `4px solid ${getStatusColor(task.status)}`,
        background: task.status === 'failed' ? '#fff1f0' : '#ffffff',
      }}
      bodyStyle={{ padding: 12 }}
      extra={
        <Space size={4}>
          {onCancel && task.status === 'running' && (
            <Tooltip title="取消任务">
              <Button
                type="text"
                size="small"
                icon={<StopOutlined />}
                onClick={onCancel}
                danger
              />
            </Tooltip>
          )}
          {onClose && task.status === 'failed' && (
            <Tooltip title="关闭">
              <Button
                type="text"
                size="small"
                icon={<CloseOutlined />}
                onClick={onClose}
              />
            </Tooltip>
          )}
        </Space>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {/* 标题行 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag
              color={getStatusColor(task.status)}
              icon={getStatusIcon(task.status)}
              style={{ margin: 0, fontSize: 12 }}
            >
              {getStatusText(task.status)}
            </Tag>
            <Text strong style={{ fontSize: 14 }}>
              {task.experiment_name || task.task_id.slice(0, 8)}
            </Text>
          </div>
          {startTime && (task.status === 'running' || task.status === 'pending') && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              {formatTime(elapsedTime)}
            </Text>
          )}
        </div>

        {/* 进度条（仅运行中任务） */}
        {task.status === 'running' && (
          <Progress
            percent={Math.round(task.progress)}
            size="small"
            strokeColor={getStatusColor(task.status)}
            showInfo={true}
            format={(percent) => `${percent}%`}
          />
        )}

        {/* 消息 */}
        {task.message && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {task.message}
          </Text>
        )}

        {/* 搜索统计（运行中或完成时） */}
        {task.search_stats && (task.status === 'running' || task.status === 'completed') && (
          <div style={{ display: 'flex', gap: 8, fontSize: 11 }}>
            <Text type="secondary">
              总方案: <Text strong>{task.search_stats.total_plans}</Text>
            </Text>
            <Text type="success">
              可行: <Text strong>{task.search_stats.feasible_plans}</Text>
            </Text>
            <Text type="secondary">
              不可行: <Text>{task.search_stats.infeasible_plans}</Text>
            </Text>
          </div>
        )}

        {/* 错误信息 */}
        {task.error && task.status === 'failed' && (
          <div
            style={{
              background: '#fff2f0',
              border: '1px solid #ffccc7',
              borderRadius: 4,
              padding: 8,
              marginTop: 4,
            }}
          >
            <Text type="danger" style={{ fontSize: 11, fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {task.error.length > 200 ? `${task.error.slice(0, 200)}...` : task.error}
            </Text>
          </div>
        )}
      </div>
    </Card>
  )
}
