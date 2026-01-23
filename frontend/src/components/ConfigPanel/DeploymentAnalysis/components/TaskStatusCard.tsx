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
    <Card
      size="small"
      style={{
        marginBottom: 12,
        borderLeft: `4px solid ${getStatusColor(task.status)}`,
        background: task.status === 'failed' ? '#fff1f0' : '#ffffff',
      }}
      styles={{ body: { padding: 12 } }}
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
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <Tooltip title="已用时间">
                <Text type="secondary" style={{ fontSize: 11, fontWeight: 500 }}>
                  ⏱️ {formatTime(elapsedTime)}
                </Text>
              </Tooltip>
              {timeEstimate && (
                <>
                  <Tooltip title="预计总时间">
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      / {formatTime(timeEstimate.totalTime)}
                    </Text>
                  </Tooltip>
                  <Tooltip title="剩余时间">
                    <Text style={{ fontSize: 11, color: '#faad14', fontWeight: 500 }}>
                      剩余: {formatTime(timeEstimate.remainingTime)}
                    </Text>
                  </Tooltip>
                </>
              )}
            </div>
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

        {/* Benchmark 信息（完成时显示最优方案） */}
        {task.status === 'completed' && task.top_plan && (
          <div
            style={{
              background: '#f6ffed',
              border: '1px solid #b7eb8f',
              borderRadius: 4,
              padding: 8,
              marginTop: 4,
            }}
          >
            <div style={{ marginBottom: 6 }}>
              <Text strong style={{ fontSize: 12, color: '#52c41a' }}>
                🏆 最优方案 (得分: {task.top_plan.score.toFixed(2)})
              </Text>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
              <div>
                <Text type="secondary">并行策略:</Text>
                <div>
                  DP={task.top_plan.parallelism.dp}, TP={task.top_plan.parallelism.tp},
                  PP={task.top_plan.parallelism.pp}, EP={task.top_plan.parallelism.ep}
                </div>
              </div>
              <div>
                <Text type="secondary">性能指标:</Text>
                <div>
                  吞吐量: {task.top_plan.throughput.toFixed(2)} tokens/s
                </div>
              </div>
              <div>
                <Text type="secondary">TTFT:</Text>
                <div>{task.top_plan.ttft.toFixed(2)} ms</div>
              </div>
              <div>
                <Text type="secondary">TPOT:</Text>
                <div>{task.top_plan.tpot.toFixed(3)} ms/token</div>
              </div>
              <div>
                <Text type="secondary">MFU:</Text>
                <div>{(task.top_plan.mfu * 100).toFixed(1)}%</div>
              </div>
              <div>
                <Text type="secondary">MBU:</Text>
                <div>{(task.top_plan.mbu * 100).toFixed(1)}%</div>
              </div>
            </div>
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
