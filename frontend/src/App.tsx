import React from 'react'
import { MainLayout } from './layouts/MainLayout'
import { WorkbenchProvider } from './contexts/WorkbenchContext'

/**
 * App 根组件 - 使用全局视角模式
 * 所有页面都保持挂载，通过 WorkbenchContext 的 viewMode 状态来控制显示
 * 这样可以保留所有组件的状态（如知识图谱的节点位置），避免不必要的卸载和重新初始化
 */
const App: React.FC = () => {
  return (
    <WorkbenchProvider>
      <MainLayout />
    </WorkbenchProvider>
  )
}

export default App
