/**
 * Tier6+ 工作台统一状态管理（重构版）
 * 组合多个独立的 Context，提供统一的访问接口
 * 保持向后兼容，现有代码无需修改
 */
import React, { createContext, useContext, useCallback, useMemo, ReactNode } from 'react'
import { HierarchicalTopology } from '../types'
import { useViewNavigation } from '../hooks/useViewNavigation'
import { loadBackendChipPresets } from '../utils/llmDeployment/presets'

// 导入拆分的 Context
import { TopologyProvider, useTopology, TopologyContextType } from './TopologyContext'
import { ConnectionProvider, useConnection, ConnectionContextType } from './ConnectionContext'
import { AnalysisProvider, useAnalysis, AnalysisContextType } from './AnalysisContext'
import { KnowledgeGraphProvider, useKnowledgeGraph, KnowledgeGraphContextType } from './KnowledgeGraphContext'
import { UIStateProvider, useUIState, UIStateContextType, ViewMode } from './UIStateContext'

// 导出类型供外部使用
export type { ViewMode }

// ============================================
// 类型定义
// ============================================

// 完整 Context 类型（向后兼容）
interface WorkbenchContextType {
  topology: TopologyContextType
  connection: ConnectionContextType
  analysis: AnalysisContextType
  knowledge: KnowledgeGraphContextType
  ui: UIStateContextType
  navigation: ReturnType<typeof useViewNavigation>
  // 当前视图连接（计算属性）
  currentViewConnections: HierarchicalTopology['connections']
  getCurrentLevel: () => 'datacenter' | 'pod' | 'rack' | 'board'
}

// ============================================
// Context 创建
// ============================================
const WorkbenchContext = createContext<WorkbenchContextType | null>(null)

export const useWorkbench = () => {
  const context = useContext(WorkbenchContext)
  if (!context) {
    throw new Error('useWorkbench must be used within WorkbenchProvider')
  }
  return context
}

// ============================================
// 内部组合组件
// ============================================
interface WorkbenchInnerProps {
  children: ReactNode
}

const WorkbenchInner: React.FC<WorkbenchInnerProps> = ({ children }) => {
  const topology = useTopology()
  const analysis = useAnalysis()
  const knowledge = useKnowledgeGraph()
  const ui = useUIState()

  // 视图导航
  const navigation = useViewNavigation(topology.topology)

  // 监听视图模式变化，离开 knowledge 页面时清空选中节点
  React.useEffect(() => {
    if (ui.viewMode !== 'knowledge') {
      knowledge.clearSelectedNodes()
    }
  }, [ui.viewMode, knowledge])

  // 获取当前层级
  const getCurrentLevel = useCallback(() => {
    if (navigation.currentBoard) return 'board'
    if (navigation.currentRack) return 'rack'
    if (navigation.currentPod) return 'pod'
    return 'datacenter'
  }, [navigation.currentBoard, navigation.currentRack, navigation.currentPod])

  // 计算当前视图的连接
  const currentViewConnections = useMemo(() => {
    if (!topology.topology) return []
    const currentLevel = getCurrentLevel()

    if (currentLevel === 'datacenter') {
      const podIds = new Set(topology.topology.pods.map(p => p.id))
      const dcSwitchIds = new Set(
        (topology.topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_pod')
          .map(s => s.id)
      )
      return topology.topology.connections.filter(c => {
        const sourceInDc = podIds.has(c.source) || dcSwitchIds.has(c.source)
        const targetInDc = podIds.has(c.target) || dcSwitchIds.has(c.target)
        return sourceInDc && targetInDc
      })
    } else if (currentLevel === 'pod' && navigation.currentPod) {
      const rackIds = new Set(navigation.currentPod.racks.map(r => r.id))
      const podSwitchIds = new Set(
        (topology.topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_rack' && s.parent_id === navigation.currentPod!.id)
          .map(s => s.id)
      )
      return topology.topology.connections.filter(c => {
        const sourceInPod = rackIds.has(c.source) || podSwitchIds.has(c.source)
        const targetInPod = rackIds.has(c.target) || podSwitchIds.has(c.target)
        return sourceInPod && targetInPod
      })
    } else if (currentLevel === 'rack' && navigation.currentRack) {
      const boardIds = new Set(navigation.currentRack.boards.map(b => b.id))
      const rackSwitchIds = new Set(
        (topology.topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_board' && s.parent_id === navigation.currentRack!.id)
          .map(s => s.id)
      )
      return topology.topology.connections.filter(c => {
        const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
        const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
        return sourceInRack && targetInRack
      })
    } else if (currentLevel === 'board' && navigation.currentBoard) {
      const chipIds = new Set(navigation.currentBoard.chips.map(c => c.id))
      return topology.topology.connections.filter(c =>
        chipIds.has(c.source) && chipIds.has(c.target)
      )
    }
    return []
  }, [topology.topology, navigation.currentPod, navigation.currentRack, navigation.currentBoard, getCurrentLevel])

  const contextValue: WorkbenchContextType = {
    topology,
    connection: useConnection(),
    analysis,
    knowledge,
    ui,
    navigation,
    currentViewConnections,
    getCurrentLevel,
  }

  return (
    <WorkbenchContext.Provider value={contextValue}>
      {children}
    </WorkbenchContext.Provider>
  )
}

// ============================================
// Provider 实现
// ============================================
interface WorkbenchProviderProps {
  children: ReactNode
}

export const WorkbenchProvider: React.FC<WorkbenchProviderProps> = ({ children }) => {
  // 初始化时加载后端芯片预设
  React.useEffect(() => {
    loadBackendChipPresets().then(() => {
      console.log('后端芯片预设加载完成')
    })
  }, [])

  // 处理视图模式变化（用于清理知识图谱状态）
  const handleViewModeChange = useCallback((_mode: ViewMode) => {
    // 这个回调会在 UIStateProvider 内部被调用
    // 如果需要在切换视图时做其他清理，可以在这里添加
  }, [])

  return (
    <TopologyProvider>
      <AnalysisProvider>
        <KnowledgeGraphProvider>
          <UIStateProvider onViewModeChange={handleViewModeChange}>
            <ConnectionProvider>
              <WorkbenchInner>
                {children}
              </WorkbenchInner>
            </ConnectionProvider>
          </UIStateProvider>
        </KnowledgeGraphProvider>
      </AnalysisProvider>
    </TopologyProvider>
  )
}

export default WorkbenchContext
