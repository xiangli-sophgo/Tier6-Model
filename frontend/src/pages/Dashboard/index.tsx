/**
 * Dashboard 首页
 * 显示系统概览、快速操作和最近任务
 */

import React, { useEffect, useState } from 'react'
import { Row, Col, Card, Space, Typography, Skeleton } from 'antd'
import { useNavigate } from 'react-router-dom'
import {
  ThunderboltOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ApartmentOutlined,
  BarChartOutlined,
  PartitionOutlined,
  RocketOutlined,
} from '@ant-design/icons'
import { getTasks, getRunningTasks } from '@/api/tasks'
import { StatCard } from './StatCard'
import { QuickAction } from './QuickAction'
import { RecentTasks } from './RecentTasks'

const { Title } = Typography

export const Dashboard: React.FC = () => {
  const navigate = useNavigate()

  const [stats, setStats] = useState({
    totalTasks: 0,
    runningTasks: 0,
    completedToday: 0,
    totalExperiments: 0,
  })
  const [recentTasks, setRecentTasks] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    setLoading(true)
    try {
      // 加载最近任务
      const tasksData = await getTasks({ limit: 5 })
      const tasks = tasksData.tasks || []
      setRecentTasks(tasks)

      // 加载运行中任务
      const runningData = await getRunningTasks()
      const runningCount = runningData.tasks.length

      // 计算今日完成任务
      const today = new Date().toDateString()
      const completedToday = tasks.filter(
        (t: any) =>
          t.status === 'completed' &&
          t.completed_at &&
          new Date(t.completed_at).toDateString() === today
      ).length

      // 统计唯一实验数
      const uniqueExperiments = new Set(tasks.map((t: any) => t.experiment_name))

      setStats({
        totalTasks: tasks.length,
        runningTasks: runningCount,
        completedToday: completedToday,
        totalExperiments: uniqueExperiments.size,
      })
    } catch (error) {
      console.error('加载 Dashboard 数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24, height: '100%', overflow: 'auto' }}>
      {/* 欢迎区域 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={3} style={{ marginBottom: 0 }}>
          Tier6+ 互联建模平台
        </Title>
      </div>

      {/* 快速操作 */}
      <Card
        title={
          <Space>
            <RocketOutlined style={{ color: '#1890ff' }} />
            <span>快速操作</span>
          </Space>
        }
        style={{ marginBottom: 24 }}
        bodyStyle={{ padding: 16 }}
      >
        <div style={{ display: 'flex', gap: 16 }}>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<ApartmentOutlined />}
              title="互联拓扑"
              description="配置Tier6+互联拓扑"
              color="#1890ff"
              onClick={() => navigate('/topology')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<ThunderboltOutlined />}
              title="部署分析"
              description="评估 LLM 部署推理方案"
              color="#52c41a"
              onClick={() => navigate('/deployment')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<DatabaseOutlined />}
              title="结果汇总"
              description="查看历史评估结果"
              color="#722ed1"
              onClick={() => navigate('/results')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<BarChartOutlined />}
              title="结果分析"
              description="性能指标详细分析"
              color="#fa8c16"
              onClick={() => navigate('/analysis')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<PartitionOutlined />}
              title="知识网络"
              description="分布式计算知识图谱"
              color="#13c2c2"
              onClick={() => navigate('/knowledge')}
            />
          </div>
        </div>
      </Card>

      {/* 最近任务 */}
      <RecentTasks tasks={recentTasks} loading={loading} />
    </div>
  )
}
