/**
 * 主布局组件
 * 包含左侧导航和内容区域
 * 使用全局视角模式：所有页面都保持挂载，通过 viewMode 控制显示
 */

import React from 'react'
import { Layout } from 'antd'
import { useWorkbench } from '@/contexts/WorkbenchContext'
import { Sidebar } from './Sidebar'
import { Dashboard } from '@/pages/Dashboard'
import { TopologySetup } from '@/pages/TopologySetup'
import { DeploymentAnalysis } from '@/pages/DeploymentAnalysis'
import { Results } from '@/pages/Results'
import { Knowledge } from '@/pages/Knowledge'

const { Content } = Layout

export const MainLayout: React.FC = () => {
  const { ui } = useWorkbench()

  return (
    <Layout style={{ height: '100vh' }}>
      <Sidebar />
      <Layout style={{ height: '100vh', overflow: 'hidden' }}>
        <Content style={{ height: '100%', overflow: 'hidden', background: '#f0f2f5' }}>
          {/* 全局视角模式：所有页面保持挂载，通过显示/隐藏来切换 */}
          {/* 这样可以保留所有组件的状态和位置（如知识图谱的节点坐标） */}

          {/* Dashboard */}
          <div style={{ display: ui.viewMode === 'dashboard' ? 'block' : 'none', height: '100%', width: '100%' }}>
            <Dashboard />
          </div>

          {/* Topology Setup */}
          <div style={{ display: ui.viewMode === 'topology' ? 'flex' : 'none', height: '100%' }}>
            <TopologySetup />
          </div>

          {/* Deployment Analysis */}
          <div style={{ display: ui.viewMode === 'deployment' ? 'flex' : 'none', height: '100%' }}>
            <DeploymentAnalysis />
          </div>

          {/* Results */}
          <div style={{ display: ui.viewMode === 'results' ? 'flex' : 'none', height: '100%' }}>
            <Results />
          </div>

          {/* Knowledge Network */}
          <div style={{ display: ui.viewMode === 'knowledge' ? 'flex' : 'none', height: '100%' }}>
            <Knowledge />
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}
