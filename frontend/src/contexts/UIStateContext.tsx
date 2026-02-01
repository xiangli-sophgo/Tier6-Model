/**
 * UI 状态管理 Context
 * 负责管理全局 UI 状态，如页面模式、选中节点、聚焦层级等
 */
import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { NodeDetail, LinkDetail } from '../components/TopologyGraph'

// ============================================
// 类型定义
// ============================================
export type ViewMode = 'dashboard' | 'topology' | 'deployment' | 'results' | 'knowledge' | '3d' | 'playground'

export interface UIStateContextType {
  viewMode: ViewMode
  selectedNode: NodeDetail | null
  selectedLink: LinkDetail | null
  focusedLevel: 'datacenter' | 'pod' | 'rack' | 'board' | null
  // 拓扑页面视图模式（3D/2D切换）
  topologyPageViewMode: '3d' | 'topology'
  // 侧边栏折叠状态
  sidebarCollapsed: boolean
  setViewMode: (mode: ViewMode) => void
  setSelectedNode: (node: NodeDetail | null) => void
  setSelectedLink: (link: LinkDetail | null) => void
  setFocusedLevel: (level: 'datacenter' | 'pod' | 'rack' | 'board' | null) => void
  setTopologyPageViewMode: (mode: '3d' | 'topology') => void
  setSidebarCollapsed: (collapsed: boolean) => void
  toggleSidebar: () => void
}

// ============================================
// Context 创建
// ============================================
const UIStateContext = createContext<UIStateContextType | null>(null)

export const useUIState = () => {
  const context = useContext(UIStateContext)
  if (!context) {
    throw new Error('useUIState must be used within UIStateProvider')
  }
  return context
}

// ============================================
// Provider 实现
// ============================================
interface UIStateProviderProps {
  children: ReactNode
  onViewModeChange?: (mode: ViewMode) => void  // 可选的回调，用于在切换视图时清理知识图谱状态
}

export const UIStateProvider: React.FC<UIStateProviderProps> = ({ children, onViewModeChange }) => {
  const [viewMode, setViewModeInternal] = useState<ViewMode>('dashboard')
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null)
  const [selectedLink, setSelectedLink] = useState<LinkDetail | null>(null)
  const [focusedLevel, setFocusedLevel] = useState<'datacenter' | 'pod' | 'rack' | 'board' | null>(null)
  // 拓扑页面视图模式（默认2D）
  const [topologyPageViewMode, setTopologyPageViewMode] = useState<'3d' | 'topology'>('topology')
  // 侧边栏折叠状态（从 localStorage 恢复）
  const [sidebarCollapsed, setSidebarCollapsedInternal] = useState<boolean>(() => {
    const saved = localStorage.getItem('tier6_sidebar_collapsed')
    return saved === 'true'
  })

  // 设置侧边栏折叠状态并持久化
  const setSidebarCollapsed = useCallback((collapsed: boolean) => {
    setSidebarCollapsedInternal(collapsed)
    localStorage.setItem('tier6_sidebar_collapsed', String(collapsed))
  }, [])

  // 切换侧边栏折叠状态
  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed(!sidebarCollapsed)
  }, [sidebarCollapsed, setSidebarCollapsed])

  // 切换视图模式
  const setViewMode = useCallback((mode: ViewMode) => {
    setViewModeInternal(mode)
    // 通知外部（用于清理知识图谱选中状态等）
    if (onViewModeChange) {
      onViewModeChange(mode)
    }
  }, [onViewModeChange])

  const contextValue: UIStateContextType = {
    viewMode,
    selectedNode,
    selectedLink,
    focusedLevel,
    topologyPageViewMode,
    sidebarCollapsed,
    setViewMode,
    setSelectedNode,
    setSelectedLink,
    setFocusedLevel,
    setTopologyPageViewMode,
    setSidebarCollapsed,
    toggleSidebar,
  }

  return (
    <UIStateContext.Provider value={contextValue}>
      {children}
    </UIStateContext.Provider>
  )
}

export default UIStateContext
