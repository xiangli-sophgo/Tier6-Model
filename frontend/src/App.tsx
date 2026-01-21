import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { MainLayout } from './layouts/MainLayout'
import { Dashboard } from './pages/Dashboard'
import { TopologySetup } from './pages/TopologySetup'
import { DeploymentAnalysis } from './pages/DeploymentAnalysis'
import { Results } from './pages/Results'
import { ResultAnalysis } from './pages/ResultAnalysis'
import { Knowledge } from './pages/Knowledge'
import { WorkbenchProvider } from './contexts/WorkbenchContext'

/**
 * App 根组件 - 配置路由
 */
const App: React.FC = () => {
  return (
    <WorkbenchProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="topology" element={<TopologySetup />} />
            <Route path="deployment" element={<DeploymentAnalysis />} />
            <Route path="results" element={<Results />} />
            <Route path="analysis" element={<ResultAnalysis />} />
            <Route path="knowledge" element={<Knowledge />} />
            {/* 重定向未知路由到首页 */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </WorkbenchProvider>
  )
}

export default App
