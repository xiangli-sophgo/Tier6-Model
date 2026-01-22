/**
 * 分析任务列表组件
 *
 * 显示正在进行和已完成的分析任务，支持：
 * - 实时进度显示
 * - 查看结果
 * - 取消任务
 * - 删除任务
 * - 清空列表
 */

import React from 'react'
import { List, Button, Tag, Space, Tooltip, Progress, Empty, Popconfirm } from 'antd'
import {
  SyncOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  ClearOutlined,
  EyeOutlined,
  ReloadOutlined,
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
  if (duration < 60000) return `${(duration / 1000).toFixed(1)}s`
  return `${Math.floor(duration / 60000)}m ${Math.floor((duration % 60000) / 1000)}s`
}

// 估算剩余时间
const estimateRemainingTime = (
  startTime: number,
  current: number,
  total: number
): { elapsed: string; remaining: string; total: string } | null => {
  if (current <= 0 || total <= 0) return null
  const elapsed = Date.now() - startTime
  const avgTimePerItem = elapsed / current
  const remainingItems = total - current
  const remainingMs = avgTimePerItem * remainingItems
  const totalMs = elapsed + remainingMs

  const formatMs = (ms: number): string => {
    if (ms < 1000) return `${Math.round(ms)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(0)}s`
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
  }

  return {
    elapsed: formatMs(elapsed),
    remaining: formatMs(remainingMs),
    total: formatMs(totalMs),
  }
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

export const AnalysisTaskList: React.FC<AnalysisTaskListProps> = ({
  tasks,
  onViewTask,
  onCancelTask,
  onDeleteTask,
  onClearCompleted,
  onRefresh,
}) => {
  const runningCount = tasks.filter(t => t.status === 'running').length
  const completedCount = tasks.filter(t => t.status !== 'running').length

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
      {/* 工具栏 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
        padding: '0 4px',
      }}>
        <span style={{ fontSize: 12, color: '#666' }}>
          {runningCount > 0 && <span style={{ color: '#1890ff' }}>{runningCount} 个运行中 · </span>}
          共 {tasks.length} 个任务
        </span>
        <Space size={4}>
          <Tooltip title="刷新列表">
            <Button size="small" type="text" icon={<ReloadOutlined />} onClick={onRefresh} />
          </Tooltip>
          {completedCount > 0 && (
            <Popconfirm
              title="确定清空已完成的任务？"
              onConfirm={onClearCompleted}
              okText="确定"
              cancelText="取消"
            >
              <Tooltip title="清空已完成">
                <Button size="small" type="text" icon={<ClearOutlined />} danger />
              </Tooltip>
            </Popconfirm>
          )}
        </Space>
      </div>

      {/* 任务列表 */}
      <List
        size="small"
        dataSource={tasks}
        style={{ maxHeight: 300, overflowY: 'auto' }}
        renderItem={(task) => (
          <List.Item
            style={{
              padding: '8px 12px',
              background: task.status === 'running' ? '#e6f7ff' : '#fafafa',
              marginBottom: 4,
              borderRadius: 4,
              border: task.status === 'running' ? '1px solid #91d5ff' : '1px solid #f0f0f0',
            }}
            actions={[
              task.status === 'running' ? (
                <Tooltip title="取消" key="cancel">
                  <Button
                    size="small"
                    type="text"
                    icon={<StopOutlined />}
                    onClick={() => onCancelTask(task.id)}
                    danger
                  />
                </Tooltip>
              ) : (
                <Space key="actions" size={0}>
                  {task.status === 'completed' && (
                    <Tooltip title="查看结果">
                      <Button
                        size="small"
                        type="text"
                        icon={<EyeOutlined />}
                        onClick={() => onViewTask(task)}
                      />
                    </Tooltip>
                  )}
                  <Tooltip title="删除">
                    <Button
                      size="small"
                      type="text"
                      icon={<DeleteOutlined />}
                      onClick={() => onDeleteTask(task.id)}
                      danger
                    />
                  </Tooltip>
                </Space>
              ),
            ]}
          >
            <List.Item.Meta
              title={
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <StatusTag status={task.status} />
                  {task.experimentName && (
                    <>
                      <span style={{ fontSize: 13, fontWeight: 500 }}>{task.experimentName}</span>
                      <span style={{ fontSize: 12, color: '#8c8c8c' }}>|</span>
                    </>
                  )}
                  <span style={{ fontSize: 12, color: task.experimentName ? '#666' : '#1a1a1a', fontWeight: task.experimentName ? 400 : 500 }}>
                    {task.benchmarkName || task.modelName}
                  </span>
                  <span style={{ fontSize: 12, color: '#8c8c8c' }}>|</span>
                  <span style={{ fontSize: 12, color: '#666' }}>{formatParallelism(task.parallelism)}</span>
                </div>
              }
              description={
                <div style={{ fontSize: 12, marginTop: 4 }}>
                  {task.status === 'running' && task.progress ? (
                    (() => {
                      const timeEstimate = estimateRemainingTime(
                        task.startTime,
                        task.progress.current,
                        task.progress.total
                      )
                      return (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                          <Progress
                            percent={Math.round((task.progress.current / task.progress.total) * 100)}
                            size="small"
                            style={{ width: 100, margin: 0 }}
                            showInfo={false}
                          />
                          <span style={{ color: '#1890ff' }}>
                            {task.progress.current}/{task.progress.total}
                          </span>
                          {timeEstimate ? (
                            <>
                              <span style={{ color: '#999' }}>
                                已用: {timeEstimate.elapsed}
                              </span>
                              <span style={{ color: '#faad14', fontWeight: 500 }}>
                                剩余: {timeEstimate.remaining}
                              </span>
                              <span style={{ color: '#999', fontSize: 11 }}>
                                (预计总时长: {timeEstimate.total})
                              </span>
                            </>
                          ) : (
                            <span style={{ color: '#999' }}>{formatDuration(task.startTime)}</span>
                          )}
                        </div>
                      )
                    })()
                  ) : task.status === 'running' ? (
                    <span style={{ color: '#1890ff' }}>
                      {task.mode === 'auto' ? '生成候选方案中...' : '模拟中...'} · {formatDuration(task.startTime)}
                    </span>
                  ) : task.status === 'completed' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      <span style={{ color: '#52c41a' }}>
                        TTFT: {task.ttft?.toFixed(1)}ms · TPOT: {task.tpot?.toFixed(2)}ms · {formatDuration(task.startTime, task.endTime)}
                      </span>
                      {(task.throughput !== undefined || task.mfu !== undefined || task.mbu !== undefined) && (
                        <div style={{ fontSize: 11, color: '#666', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {task.throughput !== undefined && (
                            <span>Throughput: {task.throughput.toFixed(2)} tokens/s</span>
                          )}
                          {task.mfu !== undefined && (
                            <span>MFU: {(task.mfu * 100).toFixed(1)}%</span>
                          )}
                          {task.mbu !== undefined && (
                            <span>MBU: {(task.mbu * 100).toFixed(1)}%</span>
                          )}
                        </div>
                      )}
                    </div>
                  ) : task.status === 'failed' ? (
                    <Tooltip title={task.error}>
                      <span style={{ color: '#ff4d4f' }}>
                        {task.error?.slice(0, 50)}{task.error && task.error.length > 50 ? '...' : ''}
                      </span>
                    </Tooltip>
                  ) : (
                    <span style={{ color: '#999' }}>已取消</span>
                  )}
                </div>
              }
            />
          </List.Item>
        )}
      />
    </div>
  )
}

export default AnalysisTaskList
