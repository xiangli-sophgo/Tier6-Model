/**
 * 任务历史表格组件
 *
 * 显示所有历史评估任务
 */

import React from 'react'
import { Table, Button, Tag, Space, Empty, Popconfirm, Typography } from 'antd'
import {
  ReloadOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  StopOutlined,
  EyeOutlined,
} from '@ant-design/icons'

const { Text } = Typography

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

const getStatusTag = (status: string) => {
  const statusMap: Record<string, { color: string; text: string; icon: React.ReactNode }> = {
    pending: { color: 'default', text: '等待中', icon: <ClockCircleOutlined /> },
    running: { color: 'processing', text: '运行中', icon: <SyncOutlined spin /> },
    completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
    failed: { color: 'error', text: '失败', icon: <CloseCircleOutlined /> },
    cancelled: { color: 'warning', text: '已取消', icon: <StopOutlined /> },
  }
  const { color, text, icon } = statusMap[status] || { color: 'default', text: status, icon: null }
  return (
    <Tag color={color} icon={icon}>
      {text}
    </Tag>
  )
}

export const TaskHistoryTable: React.FC<TaskHistoryTableProps> = ({
  tasks,
  loading,
  onRefresh,
  onViewResult,
  onDelete,
}) => {
  const columns = [
    {
      title: '实验名称',
      dataIndex: 'experiment_name',
      key: 'experiment_name',
      width: 200,
      render: (name: string) => (
        <Text strong style={{ fontSize: 13 }}>
          {name}
        </Text>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: getStatusTag,
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 80,
      render: (progress: number, record: TaskHistoryItem) => {
        if (record.status === 'running') {
          return <Text type="secondary">{Math.round(progress)}%</Text>
        }
        return null
      },
    },
    {
      title: '消息',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
      render: (message: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {message}
        </Text>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (time: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {new Date(time).toLocaleString()}
        </Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_: unknown, record: TaskHistoryItem) => (
        <Space size="small">
          {record.status === 'completed' && (
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => onViewResult(record.task_id)}
            >
              查看
            </Button>
          )}
          <Popconfirm
            title="确认删除"
            description="确定要删除此任务记录吗？"
            onConfirm={() => onDelete(record.task_id)}
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <Text strong style={{ fontSize: 14 }}>
          历史任务
        </Text>
        <Button icon={<ReloadOutlined />} onClick={onRefresh} size="small">
          刷新
        </Button>
      </div>
      <Table
        dataSource={tasks}
        columns={columns}
        rowKey="task_id"
        loading={loading}
        size="small"
        pagination={{
          pageSize: 10,
          showSizeChanger: false,
          showTotal: (total) => <Text type="secondary">共 {total} 个任务</Text>,
        }}
        locale={{
          emptyText: (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="暂无历史任务"
              style={{ padding: '20px 0' }}
            >
              <Text type="secondary" style={{ fontSize: 12 }}>
                运行评估后会在这里显示历史记录
              </Text>
            </Empty>
          ),
        }}
      />
    </div>
  )
}
