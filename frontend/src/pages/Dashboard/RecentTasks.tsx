/**
 * 最近任务列表组件
 */

import React from 'react'
import { Card, List, Tag, Typography, Button, Empty, Space } from 'antd'
import {
  HistoryOutlined,
  ArrowRightOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

const { Text } = Typography

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
}

const getStatusConfig = (status: string) => {
  const statusMap: Record<
    string,
    { color: string; text: string; icon: React.ReactNode }
  > = {
    pending: { color: 'default', text: '等待中', icon: <ClockCircleOutlined /> },
    running: { color: 'processing', text: '运行中', icon: <SyncOutlined spin /> },
    completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
    failed: { color: 'error', text: '失败', icon: <CloseCircleOutlined /> },
    cancelled: { color: 'warning', text: '已取消', icon: <CloseCircleOutlined /> },
  }
  return statusMap[status] || { color: 'default', text: status, icon: null }
}

export const RecentTasks: React.FC<RecentTasksProps> = ({ tasks, loading }) => {
  const navigate = useNavigate()

  return (
    <Card
      title={
        <Space>
          <HistoryOutlined style={{ color: '#1890ff' }} />
          <span>最近任务</span>
        </Space>
      }
      extra={
        <Button
          type="link"
          onClick={() => navigate('/deployment')}
          style={{ padding: 0 }}
        >
          查看全部 <ArrowRightOutlined />
        </Button>
      }
    >
      {tasks.length === 0 && !loading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <ThunderboltOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
          <div>
            <Text type="secondary">暂无任务记录</Text>
          </div>
          <Button
            type="primary"
            style={{ marginTop: 16 }}
            onClick={() => navigate('/deployment')}
          >
            开始第一次评估
          </Button>
        </div>
      ) : (
        <List
          loading={loading}
          dataSource={tasks}
          renderItem={(task) => {
            const statusConfig = getStatusConfig(task.status)
            return (
              <List.Item style={{ padding: '16px 0' }}>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      marginBottom: 4,
                    }}
                  >
                    <Text strong style={{ fontSize: 14 }}>
                      {task.experiment_name || task.task_id.slice(0, 8)}
                    </Text>
                    <Tag icon={statusConfig.icon} color={statusConfig.color}>
                      {statusConfig.text}
                    </Tag>
                  </div>
                  {task.message && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {task.message}
                    </Text>
                  )}
                  <div style={{ marginTop: 4 }}>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {new Date(task.created_at).toLocaleString()}
                    </Text>
                  </div>
                </div>
              </List.Item>
            )
          }}
        />
      )}
    </Card>
  )
}
