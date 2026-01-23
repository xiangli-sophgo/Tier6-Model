/**
 * 分析状态管理 Context
 * 负责管理部署分析结果、历史记录和流量分析
 */
import React, { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode } from 'react'
import { TopologyTrafficResult } from '../utils/llmDeployment/types'
import { DeploymentAnalysisData, AnalysisHistoryItem, AnalysisViewMode } from '../components/ConfigPanel/shared'

// ============================================
// 常量
// ============================================
const ANALYSIS_HISTORY_KEY = 'llm-deployment-analysis-history'
const MAX_HISTORY_ITEMS = 20

// ============================================
// 类型定义
// ============================================
export interface AnalysisContextType {
  deploymentAnalysisData: DeploymentAnalysisData | null
  analysisViewMode: AnalysisViewMode
  analysisHistory: AnalysisHistoryItem[]
  trafficResult: TopologyTrafficResult | null
  setDeploymentAnalysisData: (data: DeploymentAnalysisData | null) => void
  setAnalysisViewMode: (mode: AnalysisViewMode) => void
  setTrafficResult: (result: TopologyTrafficResult | null) => void
  handleAddToHistory: (item: Omit<AnalysisHistoryItem, 'id' | 'timestamp'>) => void
  handleLoadFromHistory: (item: AnalysisHistoryItem) => void
  handleDeleteHistory: (id: string) => void
  handleClearHistory: () => void
}

// ============================================
// Context 创建
// ============================================
const AnalysisContext = createContext<AnalysisContextType | null>(null)

export const useAnalysis = () => {
  const context = useContext(AnalysisContext)
  if (!context) {
    throw new Error('useAnalysis must be used within AnalysisProvider')
  }
  return context
}

// ============================================
// Provider 实现
// ============================================
interface AnalysisProviderProps {
  children: ReactNode
}

export const AnalysisProvider: React.FC<AnalysisProviderProps> = ({ children }) => {
  const [deploymentAnalysisData, setDeploymentAnalysisData] = useState<DeploymentAnalysisData | null>(null)
  const [analysisViewMode, setAnalysisViewMode] = useState<AnalysisViewMode>('history')
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>(() => {
    try {
      const stored = localStorage.getItem(ANALYSIS_HISTORY_KEY)
      return stored ? JSON.parse(stored) : []
    } catch { return [] }
  })
  const [trafficResult, setTrafficResult] = useState<TopologyTrafficResult | null>(null)

  // 分析完成后自动切换到详情视图
  const prevResultRef = useRef<typeof deploymentAnalysisData>(null)
  useEffect(() => {
    if (deploymentAnalysisData?.result && !prevResultRef.current?.result) {
      // 分析完成后切换到详情视图（保持在当前页面）
      setAnalysisViewMode('detail')
    }
    if (deploymentAnalysisData?.history) {
      setAnalysisHistory(deploymentAnalysisData.history)
    }
    prevResultRef.current = deploymentAnalysisData
  }, [deploymentAnalysisData])

  // 从历史记录加载
  const handleLoadFromHistory = useCallback((item: AnalysisHistoryItem) => {
    setDeploymentAnalysisData(prev => ({
      result: item.result,
      topKPlans: item.topKPlans || [item.result],
      hardware: item.hardwareConfig,
      model: item.modelConfig,
      inference: item.inferenceConfig,
      loading: false,
      errorMsg: null,
      searchStats: null,
      onSelectPlan: (plan) => {
        setDeploymentAnalysisData(d => d ? { ...d, result: plan } : null)
      },
      onMapToTopology: prev?.onMapToTopology,
      onClearTraffic: prev?.onClearTraffic,
      canMapToTopology: false,
      viewMode: 'detail' as const,
      onViewModeChange: prev?.onViewModeChange || (() => {}),
      // 历史由 AnalysisContext 统一管理，这里保持兼容
      history: prev?.history || [],
      onLoadFromHistory: prev?.onLoadFromHistory || (() => {}),
      onDeleteHistory: prev?.onDeleteHistory || (() => {}),
      onClearHistory: prev?.onClearHistory || (() => {}),
    }))
    setAnalysisViewMode('detail')
  }, [])

  // 删除历史记录
  const handleDeleteHistory = useCallback((id: string) => {
    setAnalysisHistory(prev => {
      const updated = prev.filter(h => h.id !== id)
      localStorage.setItem(ANALYSIS_HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
  }, [])

  // 清空历史记录
  const handleClearHistory = useCallback(() => {
    localStorage.setItem(ANALYSIS_HISTORY_KEY, '[]')
    setAnalysisHistory([])
  }, [])

  // 添加到历史记录
  const handleAddToHistory = useCallback((item: Omit<AnalysisHistoryItem, 'id' | 'timestamp'>) => {
    setAnalysisHistory(prev => {
      const newItem: AnalysisHistoryItem = {
        ...item,
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
      }
      const updated = [newItem, ...prev].slice(0, MAX_HISTORY_ITEMS)
      localStorage.setItem(ANALYSIS_HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
  }, [])

  const contextValue: AnalysisContextType = {
    deploymentAnalysisData,
    analysisViewMode,
    analysisHistory,
    trafficResult,
    setDeploymentAnalysisData,
    setAnalysisViewMode,
    setTrafficResult,
    handleLoadFromHistory,
    handleDeleteHistory,
    handleClearHistory,
    handleAddToHistory,
  }

  return (
    <AnalysisContext.Provider value={contextValue}>
      {children}
    </AnalysisContext.Provider>
  )
}

export default AnalysisContext
