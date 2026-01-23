/**
 * Dashboard 概览
 * 显示系统概览、快速操作和最近任务
 */

import React, { useEffect, useState } from 'react'
import { Card, Space, Typography } from 'antd'
import {
  ThunderboltOutlined,
  DatabaseOutlined,
  ApartmentOutlined,
  PartitionOutlined,
  RocketOutlined,
} from '@ant-design/icons'
import { getTasks, getRunningTasks } from '@/api/tasks'
import { QuickAction } from './QuickAction'
import { RecentTasks } from './RecentTasks'
import { useWorkbench, ViewMode } from '@/contexts/WorkbenchContext'

const { Title } = Typography

export const Dashboard: React.FC = () => {
  const { ui } = useWorkbench()

  // 导航到指定视图
  const navigateTo = (mode: ViewMode) => {
    ui.setViewMode(mode)
  }

  const [, setStats] = useState({
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
    <div style={{ padding: 24, height: '100%', overflow: 'auto', background: '#fff' }}>
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
        styles={{ body: { padding: 16 } }}
      >
        <div style={{ display: 'flex', gap: 16 }}>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<ApartmentOutlined />}
              title="互联拓扑"
              description="配置Tier6+互联拓扑"
              color="#1890ff"
              onClick={() => navigateTo('topology')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<ThunderboltOutlined />}
              title="部署分析"
              description="评估 LLM 部署推理方案"
              color="#52c41a"
              onClick={() => navigateTo('deployment')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<DatabaseOutlined />}
              title="结果管理"
              description="实验结果查看与详细分析"
              color="#722ed1"
              onClick={() => navigateTo('results')}
            />
          </div>
          <div style={{ flex: 1 }}>
            <QuickAction
              icon={<PartitionOutlined />}
              title="知识网络"
              description="分布式计算知识图谱"
              color="#13c2c2"
              onClick={() => navigateTo('knowledge')}
            />
          </div>
        </div>
      </Card>

      {/* 最近任务 */}
      <RecentTasks tasks={recentTasks} loading={loading} onNavigate={() => navigateTo('results')} />
    </div>
  )
}
