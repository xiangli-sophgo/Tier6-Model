/**
 * 主布局组件
 * 包含左侧导航和内容区域
 * 使用全局视角模式：所有页面都保持挂载，通过 viewMode 控制显示
 */

import React from 'react'
import { useWorkbench } from '@/contexts/WorkbenchContext'
import { Sidebar } from './Sidebar'
import { Dashboard } from '@/pages/Dashboard'
import { TopologySetup } from '@/pages/TopologySetup'
import { DeploymentAnalysis } from '@/pages/DeploymentAnalysis'
import { Results } from '@/pages/Results'
import { Knowledge } from '@/pages/Knowledge'

export const MainLayout: React.FC = () => {
  const { ui } = useWorkbench()

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 h-screen overflow-hidden">
        <main className="h-full overflow-hidden bg-gray-100">
          {/* 全局视角模式：所有页面保持挂载，通过显示/隐藏来切换 */}
          {/* 这样可以保留所有组件的状态和位置（如知识图谱的节点坐标） */}

          {/* Dashboard */}
          <div className={`${ui.viewMode === 'dashboard' ? 'block' : 'hidden'} h-full w-full`}>
            <Dashboard />
          </div>

          {/* Topology Setup */}
          <div className={`${ui.viewMode === 'topology' ? 'flex' : 'hidden'} h-full w-full`}>
            <TopologySetup />
          </div>

          {/* Deployment Analysis */}
          <div className={`${ui.viewMode === 'deployment' ? 'flex' : 'hidden'} h-full w-full`}>
            <DeploymentAnalysis />
          </div>

          {/* Results */}
          <div className={`${ui.viewMode === 'results' ? 'flex' : 'hidden'} h-full w-full`}>
            <Results />
          </div>

          {/* Knowledge Network */}
          <div className={`${ui.viewMode === 'knowledge' ? 'flex' : 'hidden'} h-full w-full`}>
            <Knowledge />
          </div>
        </main>
      </div>
    </div>
  )
}
