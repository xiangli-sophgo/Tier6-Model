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
    <div className="h-screen w-screen overflow-hidden">
      <Sidebar />
      <div className={`h-screen transition-all duration-300 ${ui.sidebarCollapsed ? 'ml-16' : 'ml-[180px]'}`}>
        <main className="h-full w-full bg-gray-100 relative">
          {/* 全局视角模式：所有页面保持挂载，通过显示/隐藏来切换 */}
          {/* 这样可以保留所有组件的状态和位置（如知识图谱的节点坐标） */}

          {/* Dashboard */}
          <div className={`${ui.viewMode === 'dashboard' ? 'block' : 'hidden'} absolute inset-0 overflow-auto`}>
            <Dashboard />
          </div>

          {/* Topology Setup */}
          <div className={`${ui.viewMode === 'topology' ? 'flex' : 'hidden'} absolute inset-0 overflow-auto`}>
            <TopologySetup />
          </div>

          {/* Deployment Analysis */}
          <div className={`${ui.viewMode === 'deployment' ? 'flex' : 'hidden'} absolute inset-0 overflow-auto`}>
            <DeploymentAnalysis />
          </div>

          {/* Results */}
          <div
            className={`${ui.viewMode === 'results' ? 'flex' : 'hidden'} absolute inset-0 overflow-y-auto overflow-x-hidden`}
            style={{ overscrollBehavior: 'none' }}
          >
            <Results />
          </div>

          {/* Knowledge Network */}
          <div className={`${ui.viewMode === 'knowledge' ? 'flex' : 'hidden'} absolute inset-0 overflow-auto`}>
            <Knowledge />
          </div>
        </main>
      </div>
    </div>
  )
}
