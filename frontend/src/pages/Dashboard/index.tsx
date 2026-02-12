/**
 * Dashboard 概览
 * 显示系统概览、快速操作和最近任务
 */

import React, { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import {
  Zap,
  Database,
  Network,
  GitFork,
  Rocket,
} from 'lucide-react'
import { getTasks } from '@/api/tasks'
import { QuickAction } from './QuickAction'
import { RecentTasks } from './RecentTasks'
import { PageHeader } from '@/components/ui/page-header'
import { useWorkbench, ViewMode } from '@/contexts/WorkbenchContext'

export const Dashboard: React.FC = () => {
  const { ui } = useWorkbench()

  // 导航到指定视图
  const navigateTo = (mode: ViewMode) => {
    ui.setViewMode(mode)
  }

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
      // 过滤掉无效任务（缺少必要字段）
      const validTasks = tasks.filter((t: any) => t.task_id && t.status)
      setRecentTasks(validTasks)
    } catch (error) {
      console.error('加载 Dashboard 数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-gray-50 to-white">
      {/* 标题栏 */}
      <PageHeader title="Tier6+ 互联建模平台" />

      {/* 内容区 */}
      <div className="flex-1 overflow-auto p-8">
      {/* 欢迎区域 */}
      <div className="mb-8">

      {/* 快速操作 */}
      <Card className="mb-8 shadow-none hover:shadow-md transition-shadow duration-300">
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
