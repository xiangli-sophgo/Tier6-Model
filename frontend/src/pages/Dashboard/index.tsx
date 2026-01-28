/**
 * Dashboard 概览
 * 显示系统概览、快速操作和最近任务
 */

import React, { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import {
  Zap,
  Database,
  Network,
  GitFork,
  Rocket,
} from 'lucide-react'
import { getTasks, getRunningTasks } from '@/api/tasks'
import { QuickAction } from './QuickAction'
import { RecentTasks } from './RecentTasks'
import { useWorkbench, ViewMode } from '@/contexts/WorkbenchContext'

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
    <div className="h-full flex flex-col bg-gradient-to-b from-gray-50 to-white">
      {/* 标题栏 */}
      <div className="px-8 py-6 border-b border-blue-100 bg-gradient-to-r from-blue-50 to-white flex-shrink-0" style={{boxShadow: '0 2px 12px rgba(37, 99, 235, 0.08)'}}>
        <h3 className="m-0 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-2xl font-bold text-transparent">
          Tier6+ 互联建模平台
        </h3>
      </div>

      {/* 内容区 */}
      <div className="flex-1 overflow-auto p-8">
      {/* 欢迎区域 */}
      <div className="mb-8">

      {/* 快速操作 */}
      <Card className="mb-8 border-0">
        <CardHeader className="pb-6">
          <CardTitle className="flex items-center gap-3 text-lg">
            <div className="rounded-lg bg-blue-50 p-2">
              <Rocket className="h-5 w-5 text-blue-600" />
            </div>
            <span>快速操作</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-6 pb-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <QuickAction
              icon={<Network className="h-6 w-6" />}
              title="互联拓扑"
              description="配置Tier6+互联拓扑"
              color="#2563EB"
              onClick={() => navigateTo('topology')}
            />
            <QuickAction
              icon={<Zap className="h-6 w-6" />}
              title="部署分析"
              description="评估 LLM 部署推理方案"
              color="#06B6D4"
              onClick={() => navigateTo('deployment')}
            />
            <QuickAction
              icon={<Database className="h-6 w-6" />}
              title="结果管理"
              description="实验结果查看与详细分析"
              color="#7C3AED"
              onClick={() => navigateTo('results')}
            />
            <QuickAction
              icon={<GitFork className="h-6 w-6" />}
              title="知识网络"
              description="分布式计算知识图谱"
              color="#059669"
              onClick={() => navigateTo('knowledge')}
            />
          </div>
        </CardContent>
      </Card>

      {/* 最近任务 */}
      <RecentTasks
        tasks={recentTasks}
        loading={loading}
        onNavigate={() => navigateTo('results')}
        onTaskClick={() => navigateTo('results')}
      />
      </div>
      </div>
    </div>
  )
}
