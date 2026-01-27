/**
 * 连接编辑状态管理 Context
 * 负责管理拓扑连接的手动编辑、批量连接等功能
 */
import React, { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode } from 'react'
import { toast } from 'sonner'
import {
  ManualConnectionConfig,
  ManualConnection,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
  MultiLevelViewOptions,
} from '../types'
import { getLevelConnectionDefaults } from '../api/topology'
import { useTopology } from './TopologyContext'

// ============================================
// 常量
// ============================================
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'

// ============================================
// 连接检查辅助函数
// ============================================
interface ConnectionLike {
  source: string
  target: string
}

const connectionExists = (source: string, target: string, connections: ConnectionLike[]): boolean => {
  return connections.some(c =>
    (c.source === source && c.target === target) ||
    (c.source === target && c.target === source)
  )
}

const findConnection = <T extends ConnectionLike>(source: string, target: string, connections: T[]): T | undefined => {
  return connections.find(c =>
    (c.source === source && c.target === target) ||
    (c.source === target && c.target === source)
  )
}

// ============================================
// 类型定义
// ============================================
export interface ConnectionContextType {
  manualConnectionConfig: ManualConnectionConfig
  connectionMode: ConnectionMode
  selectedNodes: Set<string>
  targetNodes: Set<string>
  sourceNode: string | null
  layoutType: LayoutType
  multiLevelOptions: MultiLevelViewOptions
  // 操作方法
  setManualConnectionConfig: (config: ManualConnectionConfig) => void
  setConnectionMode: (mode: ConnectionMode) => void
  setSelectedNodes: (nodes: Set<string>) => void
  setTargetNodes: (nodes: Set<string>) => void
  setSourceNode: (nodeId: string | null) => void
  setLayoutType: (type: LayoutType) => void
  setMultiLevelOptions: (options: MultiLevelViewOptions) => void
  handleManualConnect: (sourceId: string, targetId: string, level: HierarchyLevel) => void
  handleBatchConnect: (level: HierarchyLevel) => void
  handleDeleteManualConnection: (connectionId: string) => void
  handleDeleteConnection: (source: string, target: string) => void
  handleUpdateConnectionParams: (source: string, target: string, bandwidth?: number, latency?: number) => void
}

// ============================================
// Context 创建
// ============================================
const ConnectionContext = createContext<ConnectionContextType | null>(null)

export const useConnection = () => {
  const context = useContext(ConnectionContext)
  if (!context) {
    throw new Error('useConnection must be used within ConnectionProvider')
  }
  return context
}

// ============================================
// Provider 实现
// ============================================
interface ConnectionProviderProps {
  children: ReactNode
}

export const ConnectionProvider: React.FC<ConnectionProviderProps> = ({ children }) => {
  // 直接从 TopologyContext 获取 topology 和 setTopology
  const { topology, setTopology } = useTopology()
  const [manualConnectionConfig, setManualConnectionConfigRaw] = useState<ManualConnectionConfig>(() => {
    try {
      const cachedStr = localStorage.getItem(CONFIG_CACHE_KEY)
      if (cachedStr) {
        const cached = JSON.parse(cachedStr)
        if (cached.manualConnectionConfig) return cached.manualConnectionConfig
      }
    } catch { /* ignore */ }
    return { enabled: false, mode: 'append', connections: [] }
  })
  const [connectionMode, setConnectionModeRaw] = useState<ConnectionMode>('view')
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set())
  const [targetNodes, setTargetNodes] = useState<Set<string>>(new Set())
  const [sourceNode, setSourceNode] = useState<string | null>(null)
  const [layoutType, setLayoutType] = useState<LayoutType>('auto')
  const [multiLevelOptions, setMultiLevelOptions] = useState<MultiLevelViewOptions>({
    enabled: false,
    levelPair: 'pod_rack',
    expandedContainers: new Set(),
  })

  // 层级连接默认参数（仅用于初始化）
  const [, setLevelConnectionDefaults] = useState<{
    datacenter: { bandwidth: number; latency: number }
    pod: { bandwidth: number; latency: number }
    rack: { bandwidth: number; latency: number }
    board: { bandwidth: number; latency: number }
  } | null>(null)

  // 加载层级默认参数
  useEffect(() => {
    getLevelConnectionDefaults().then((defaults) => {
      setLevelConnectionDefaults(defaults)
      setManualConnectionConfigRaw(prev => ({
        ...prev,
        level_defaults: { ...defaults, ...prev.level_defaults },
      }))
    }).catch(console.error)
  }, [])

  // 层级默认参数变化时更新连接
  const prevLevelDefaults = useRef(manualConnectionConfig.level_defaults)
  useEffect(() => {
    const levelDefaults = manualConnectionConfig.level_defaults
    if (!levelDefaults || !topology) return
    // 只有当 level_defaults 变化时才更新
    if (prevLevelDefaults.current === levelDefaults) return
    prevLevelDefaults.current = levelDefaults

    setTopology(prev => {
      if (!prev) return prev
      return {
        ...prev,
        connections: prev.connections.map(conn => {
          let level: 'datacenter' | 'pod' | 'rack' | 'board' | undefined
          if (conn.type === 'intra') level = 'board'
          else if (conn.type === 'inter') level = 'rack'
          const defaults = level ? levelDefaults[level] : undefined
          if (defaults) {
            return {
              ...conn,
              bandwidth: defaults.bandwidth ?? conn.bandwidth,
              latency: defaults.latency ?? conn.latency,
            }
          }
          return conn
        }),
      }
    })
  }, [manualConnectionConfig.level_defaults, topology, setTopology])

  // 设置手动连接配置
  const setManualConnectionConfig = useCallback((config: ManualConnectionConfig) => {
    setManualConnectionConfigRaw(config)
    if (!config.enabled) {
      setConnectionModeRaw('view')
      setSelectedNodes(new Set())
      setTargetNodes(new Set())
      setSourceNode(null)
    }
  }, [])

  // 设置连接模式
  const setConnectionMode = useCallback((mode: ConnectionMode) => {
    setConnectionModeRaw(mode)
    if (mode === 'view') {
      setSelectedNodes(new Set())
      setTargetNodes(new Set())
      setSourceNode(null)
    } else if (mode === 'select_source') {
      setTargetNodes(new Set())
      setSourceNode(null)
    }
  }, [])

  // 手动连接
  const handleManualConnect = useCallback((sourceId: string, targetId: string, level: HierarchyLevel) => {
    if (connectionExists(sourceId, targetId, manualConnectionConfig.connections)) {
      toast.warning(`手动连接已存在: ${sourceId} ↔ ${targetId}`)
      return
    }
    if (topology?.connections && connectionExists(sourceId, targetId, topology.connections)) {
      toast.warning(`自动连接已存在: ${sourceId} ↔ ${targetId}`)
      return
    }
    const newConnection: ManualConnection = {
      id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      source: sourceId,
      target: targetId,
      hierarchy_level: level,
      created_at: new Date().toISOString(),
    }
    setManualConnectionConfigRaw(prev => ({
      ...prev,
      connections: [...prev.connections, newConnection],
    }))
    toast.success(`已添加连接: ${sourceId} ↔ ${targetId}`)
  }, [manualConnectionConfig.connections, topology?.connections])

  // 批量连接
  const handleBatchConnect = useCallback((level: HierarchyLevel) => {
    if (selectedNodes.size === 0 || targetNodes.size === 0) {
      toast.warning('请先选择源节点和目标节点')
      return
    }
    let addedCount = 0
    const newConnections: ManualConnection[] = []
    selectedNodes.forEach(sourceId => {
      targetNodes.forEach(targetId => {
        if (sourceId === targetId) return
        const existsManual = connectionExists(sourceId, targetId, manualConnectionConfig.connections)
        const existsAuto = topology?.connections && connectionExists(sourceId, targetId, topology.connections)
        const existsNew = connectionExists(sourceId, targetId, newConnections)
        if (!existsManual && !existsAuto && !existsNew) {
          newConnections.push({
            id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}_${addedCount}`,
            source: sourceId,
            target: targetId,
            hierarchy_level: level,
            created_at: new Date().toISOString(),
          })
          addedCount++
        }
      })
    })
    if (newConnections.length > 0) {
      setManualConnectionConfigRaw(prev => ({
        ...prev,
        connections: [...prev.connections, ...newConnections],
      }))
      toast.success(`已添加 ${newConnections.length} 条连接`)
    } else {
      toast.warning('所有连接已存在')
    }
    setSelectedNodes(new Set())
    setTargetNodes(new Set())
    setConnectionModeRaw('select_source')
  }, [selectedNodes, targetNodes, manualConnectionConfig.connections, topology?.connections])

  // 删除手动连接
  const handleDeleteManualConnection = useCallback((connectionId: string) => {
    setManualConnectionConfigRaw(prev => ({
      ...prev,
      connections: prev.connections.filter(c => c.id !== connectionId),
    }))
    toast.success('已删除连接')
  }, [])

  // 删除连接
  const handleDeleteConnection = useCallback((source: string, target: string) => {
    const manualConn = findConnection(source, target, manualConnectionConfig.connections)
    if (manualConn) {
      setManualConnectionConfigRaw(prev => ({
        ...prev,
        connections: prev.connections.filter(c => c.id !== manualConn.id),
      }))
      toast.success('已删除手动连接')
    } else {
      setTopology(prev => {
        if (!prev) return prev
        return {
          ...prev,
          connections: prev.connections.filter(c =>
            !((c.source === source && c.target === target) ||
              (c.source === target && c.target === source))
          ),
        }
      })
      toast.success('已删除连接')
    }
  }, [manualConnectionConfig.connections, setTopology])

  // 更新连接参数
  const handleUpdateConnectionParams = useCallback((source: string, target: string, bandwidth?: number, latency?: number) => {
    const manualConn = findConnection(source, target, manualConnectionConfig.connections)
    if (manualConn) {
      setManualConnectionConfigRaw(prev => ({
        ...prev,
        connections: prev.connections.map(c =>
          c.id === manualConn.id ? { ...c, bandwidth, latency } : c
        ),
      }))
    } else {
      setTopology(prev => {
        if (!prev) return prev
        return {
          ...prev,
          connections: prev.connections.map(c => {
            if ((c.source === source && c.target === target) ||
                (c.source === target && c.target === source)) {
              return { ...c, bandwidth, latency }
            }
            return c
          }),
        }
      })
    }
  }, [manualConnectionConfig.connections, setTopology])

  const contextValue: ConnectionContextType = {
    manualConnectionConfig,
    connectionMode,
    selectedNodes,
    targetNodes,
    sourceNode,
    layoutType,
    multiLevelOptions,
    setManualConnectionConfig,
    setConnectionMode,
    setSelectedNodes,
    setTargetNodes,
    setSourceNode,
    setLayoutType,
    setMultiLevelOptions,
    handleManualConnect,
    handleBatchConnect,
    handleDeleteManualConnection,
    handleDeleteConnection,
    handleUpdateConnectionParams,
  }

  return (
    <ConnectionContext.Provider value={contextValue}>
      {children}
    </ConnectionContext.Provider>
  )
}

export default ConnectionContext
