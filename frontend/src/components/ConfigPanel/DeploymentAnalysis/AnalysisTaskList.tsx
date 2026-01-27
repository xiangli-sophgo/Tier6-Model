/**
 * 分析任务列表组件
 *
 * - 运行中的任务显示为独立卡片（带实时进度）
 * - 已完成的任务显示在历史列表中
 */

import React, { useState, useEffect } from 'react'
import {
  List, Button, Tag, Space, Tooltip, Progress, Empty, Popconfirm,
  Card, Row, Col, Statistic, Divider,
} from 'antd'
import {
  SyncOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  ClearOutlined,
  EyeOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
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
const StatusTag: React.FC<{ status: AnalysisTask['status'] }> = ({ status }) => {
  switch (status) {
    case 'running':
      return <Tag icon={<SyncOutlined spin />} color="processing">运行中</Tag>
    case 'completed':
      return <Tag icon={<CheckCircleOutlined />} color="success">已完成</Tag>
    case 'failed':
      return <Tag icon={<CloseCircleOutlined />} color="error">失败</Tag>
    case 'cancelled':
      return <Tag icon={<StopOutlined />} color="default">已取消</Tag>
    default:
      return null
  }
}

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
    <Card
      size="small"
      title={
        <Space>
          <ThunderboltOutlined style={{ color: '#faad14' }} />
          <span>{taskName}</span>
          <StatusTag status={task.status} />
        </Space>
      }
      extra={
        <Button
          icon={<StopOutlined />}
          danger
          size="small"
          onClick={onCancel}
        >
          取消
        </Button>
      }
      style={{ marginBottom: 12, border: '1px solid #91d5ff', background: '#e6f7ff' }}
    >
      <Row gutter={[16, 8]}>
        <Col span={8}>
          <Statistic
            title="进度"
            value={progress}
            suffix="%"
            valueStyle={{ color: '#1890ff', fontSize: 20 }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="运行时间"
            value={elapsedTime}
            valueStyle={{ color: '#1890ff', fontSize: 20 }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="芯片数"
            value={task.chips || '-'}
            valueStyle={{ fontSize: 20 }}
          />
        </Col>
      </Row>

      <Divider style={{ margin: '12px 0' }} />

      <Progress
        percent={progress}
        status="active"
        strokeColor={{ from: '#1890ff', to: '#4096ff' }}
      />

      <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
        <Space split={<Divider type="vertical" />}>
          <span>模式: {task.mode === 'auto' ? '自动搜索' : '手动'}</span>
          <span>策略: {formatParallelism(task.parallelism)}</span>
        </Space>
      </div>
    </Card>
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
    <List.Item
      style={{
        padding: '8px 12px',
        background: '#fafafa',
        marginBottom: 4,
        borderRadius: 4,
        border: '1px solid #f0f0f0',
      }}
      actions={[
        <Space key="actions" size={0}>
          {task.status === 'completed' && (
            <Tooltip title="查看结果">
              <Button
                size="small"
                type="text"
                icon={<EyeOutlined />}
                onClick={onView}
              />
            </Tooltip>
          )}
          <Tooltip title="删除">
            <Button
              size="small"
              type="text"
              icon={<DeleteOutlined />}
              onClick={onDelete}
              danger
            />
          </Tooltip>
        </Space>,
      ]}
    >
      <List.Item.Meta
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <StatusTag status={task.status} />
            <span style={{ fontSize: 13, fontWeight: 500 }}>
              {task.experimentName || task.benchmarkName || task.modelName}
            </span>
            <span style={{ fontSize: 12, color: '#8c8c8c' }}>|</span>
            <span style={{ fontSize: 12, color: '#666' }}>{formatParallelism(task.parallelism)}</span>
          </div>
        }
        description={
          <div style={{ fontSize: 12, marginTop: 4 }}>
            {task.status === 'completed' ? (
              <span style={{ color: '#52c41a' }}>
                TTFT: {task.ttft?.toFixed(1)}ms · TPOT: {task.tpot?.toFixed(2)}ms · {formatDuration(task.startTime, task.endTime)}
              </span>
            ) : task.status === 'failed' ? (
              <Tooltip title={task.error}>
                <span style={{ color: '#ff4d4f' }}>
                  {task.error?.slice(0, 50)}{task.error && task.error.length > 50 ? '...' : ''}
                </span>
              </Tooltip>
            ) : (
              <span style={{ color: '#999' }}>已取消 · {formatDuration(task.startTime, task.endTime)}</span>
            )}
          </div>
        }
      />
    </List.Item>
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
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="暂无分析任务"
        style={{ padding: '20px 0' }}
      />
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
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 8,
            marginTop: runningTasks.length > 0 ? 16 : 0,
            padding: '0 4px',
          }}>
            <span style={{ fontSize: 12, color: '#666' }}>
              历史任务 ({historyTasks.length})
            </span>
            <Space size={4}>
              <Button size="small" type="text" icon={<ReloadOutlined />} onClick={onRefresh}>
                刷新
              </Button>
              <Popconfirm
                title="确定清空历史任务？"
                onConfirm={onClearCompleted}
                okText="确定"
                cancelText="取消"
                placement="topRight"
              >
                <Button size="small" type="text" icon={<ClearOutlined />} danger>
                  清空
                </Button>
              </Popconfirm>
            </Space>
          </div>

          <List
            size="small"
            dataSource={historyTasks}
            style={{ maxHeight: 200, overflowY: 'auto' }}
            renderItem={(task) => (
              <HistoryTaskItem
                key={task.id}
                task={task}
                onView={() => onViewTask(task)}
                onDelete={() => onDeleteTask(task.id)}
              />
            )}
          />
        </>
      )}
    </div>
  )
}

export default AnalysisTaskList
