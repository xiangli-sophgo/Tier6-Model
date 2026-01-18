/**
 * Tier6+ 工作台统一状态管理
 * 整合拓扑、连接、分析等相关状态，减少 prop drilling
 */
import React, { createContext, useContext, useState, useCallback, useEffect, useRef, useMemo, ReactNode } from 'react'
import { message } from 'antd'
import * as d3Force from 'd3-force'
import {
  HierarchicalTopology,
  ManualConnectionConfig,
  ManualConnection,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
  MultiLevelViewOptions,
} from '../types'
import { TopologyTrafficResult } from '../utils/llmDeployment/types'
import { DeploymentAnalysisData, AnalysisHistoryItem, AnalysisViewMode } from '../components/ConfigPanel/shared'
import { getTopology, generateTopology, getLevelConnectionDefaults } from '../api/topology'
import { useViewNavigation } from '../hooks/useViewNavigation'
import { NodeDetail, LinkDetail } from '../components/TopologyGraph'
import { ForceKnowledgeNode, KnowledgeCategory, CORE_RELATION_TYPES, MAX_EDGES_PER_NODE } from '../components/KnowledgeGraph'
import knowledgeData from '../data/knowledge-graph'

// ============================================
// 常量
// ============================================
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'
const ANALYSIS_HISTORY_KEY = 'llm-deployment-analysis-history'

// 知识图谱力导向布局参数 - 简化稳定版
const KNOWLEDGE_FORCE_CONFIG = {
  // 斥力配置
  chargeStrength: -400,
  chargeDistanceMax: 500,
  // 连接力配置
  linkDistance: 80,
  linkStrength: 0.3,
  // 径向力配置（主导布局）
  radialStrength: 0.1,
  radialMinRadius: 50,
  radialMaxRadius: 500,
  // 中心引力
  centerStrength: 0.05,
  // 碰撞配置
  collisionRadius: 35,
  collisionStrength: 1,
  collisionIterations: 4,
  // 预热 tick 数量
  warmupTicks: 300,
}

const KNOWLEDGE_CATEGORY_ORDER: KnowledgeCategory[] = [
  'hardware', 'interconnect', 'parallel', 'inference',
  'model', 'communication', 'protocol', 'system'
]

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
// Context 类型定义
// ============================================

// 拓扑状态
interface TopologyState {
  topology: HierarchicalTopology | null
  loading: boolean
  loadTopology: () => Promise<void>
  handleGenerate: (config: GenerateConfig) => Promise<void>
}

// 生成配置类型
interface GenerateConfig {
  pod_count: number
  racks_per_pod: number
  board_configs: {
    u1: { count: number; chips: { npu: number; cpu: number } }
    u2: { count: number; chips: { npu: number; cpu: number } }
    u4: { count: number; chips: { npu: number; cpu: number } }
  }
  rack_config?: {
    total_u: number
    boards: Array<{
      id: string
      name: string
      u_height: number
      count: number
      chips: Array<{ name: string; count: number }>
    }>
  }
  switch_config?: any
  manual_connections?: ManualConnectionConfig
}

// 连接编辑状态
interface ConnectionState {
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

// 分析状态
interface AnalysisState {
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

// 知识图谱视口类型
interface KnowledgeViewBox {
  x: number
  y: number
  width: number
  height: number
}

// UI 状态
interface UIState {
  viewMode: '3d' | 'topology' | 'analysis' | 'knowledge'
  selectedNode: NodeDetail | null
  selectedLink: LinkDetail | null
  focusedLevel: 'datacenter' | 'pod' | 'rack' | 'board' | null
  // 知识图谱状态
  knowledgeSelectedNodes: ForceKnowledgeNode[]  // 支持多个选中节点（用于详情卡片）
  knowledgeHighlightedNodeId: string | null  // 当前高亮的节点ID（用于图中高亮效果）
  knowledgeVisibleCategories: Set<KnowledgeCategory>
  knowledgeNodes: ForceKnowledgeNode[]
  knowledgeInitialized: boolean
  knowledgeViewBox: KnowledgeViewBox | null
  setViewMode: (mode: '3d' | 'topology' | 'analysis' | 'knowledge') => void
  setSelectedNode: (node: NodeDetail | null) => void
  setSelectedLink: (link: LinkDetail | null) => void
  setFocusedLevel: (level: 'datacenter' | 'pod' | 'rack' | 'board' | null) => void
  addKnowledgeSelectedNode: (node: ForceKnowledgeNode) => void  // 添加节点到列表
  removeKnowledgeSelectedNode: (nodeId: string) => void  // 从列表移除节点
  clearKnowledgeHighlight: () => void  // 清除高亮（不影响详情卡片）
  setKnowledgeVisibleCategories: (categories: Set<KnowledgeCategory>) => void
  setKnowledgeNodes: (nodes: ForceKnowledgeNode[]) => void
  setKnowledgeInitialized: (initialized: boolean) => void
  setKnowledgeViewBox: (viewBox: KnowledgeViewBox) => void
  resetKnowledgeCategories: () => void
}

// 完整 Context 类型
interface WorkbenchContextType {
  topology: TopologyState
  connection: ConnectionState
  analysis: AnalysisState
  ui: UIState
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
// Provider 实现
// ============================================
interface WorkbenchProviderProps {
  children: ReactNode
}

export const WorkbenchProvider: React.FC<WorkbenchProviderProps> = ({ children }) => {
  // ==================== 拓扑状态 ====================
  const [topology, setTopology] = useState<HierarchicalTopology | null>(null)
  const [loading, setLoading] = useState(true)

  // 视图导航
  const navigation = useViewNavigation(topology)

  // 层级连接默认参数（仅用于初始化）
  const [, setLevelConnectionDefaults] = useState<{
    datacenter: { bandwidth: number; latency: number }
    pod: { bandwidth: number; latency: number }
    rack: { bandwidth: number; latency: number }
    board: { bandwidth: number; latency: number }
  } | null>(null)

  // ==================== 连接编辑状态 ====================
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

  // ==================== 分析状态 ====================
  const [deploymentAnalysisData, setDeploymentAnalysisData] = useState<DeploymentAnalysisData | null>(null)
  const [analysisViewMode, setAnalysisViewMode] = useState<AnalysisViewMode>('history')
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>(() => {
    try {
      const stored = localStorage.getItem(ANALYSIS_HISTORY_KEY)
      return stored ? JSON.parse(stored) : []
    } catch { return [] }
  })
  const [trafficResult, setTrafficResult] = useState<TopologyTrafficResult | null>(null)

  // ==================== UI 状态 ====================
  const [viewMode, setViewModeInternal] = useState<'3d' | 'topology' | 'analysis' | 'knowledge'>('topology')
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null)
  const [selectedLink, setSelectedLink] = useState<LinkDetail | null>(null)
  const [focusedLevel, setFocusedLevel] = useState<'datacenter' | 'pod' | 'rack' | 'board' | null>(null)
  // 知识图谱状态
  const [knowledgeSelectedNodes, setKnowledgeSelectedNodes] = useState<ForceKnowledgeNode[]>([])
  const [knowledgeHighlightedNodeId, setKnowledgeHighlightedNodeId] = useState<string | null>(null)
  const [knowledgeVisibleCategories, setKnowledgeVisibleCategories] = useState<Set<KnowledgeCategory>>(
    new Set(['hardware', 'interconnect', 'parallel', 'communication', 'model', 'inference', 'protocol', 'system'])
  )
  const [knowledgeNodes, setKnowledgeNodes] = useState<ForceKnowledgeNode[]>([])
  const [knowledgeInitialized, setKnowledgeInitialized] = useState(false)
  const [knowledgeViewBox, setKnowledgeViewBox] = useState<KnowledgeViewBox | null>(null)
  const resetKnowledgeCategories = useCallback(() => {
    setKnowledgeVisibleCategories(new Set(['hardware', 'interconnect', 'parallel', 'communication', 'model', 'inference', 'protocol', 'system']))
  }, [])
  // 添加知识节点到选中列表（新节点放在最前面，如果已存在则移到最前面），同时设置高亮
  const addKnowledgeSelectedNode = useCallback((node: ForceKnowledgeNode) => {
    setKnowledgeSelectedNodes(prev => {
      const filtered = prev.filter(n => n.id !== node.id)
      return [node, ...filtered]
    })
    setKnowledgeHighlightedNodeId(node.id)
  }, [])
  // 从选中列表移除节点
  const removeKnowledgeSelectedNode = useCallback((nodeId: string) => {
    setKnowledgeSelectedNodes(prev => prev.filter(n => n.id !== nodeId))
    // 如果移除的是高亮节点，清除高亮
    setKnowledgeHighlightedNodeId(prev => prev === nodeId ? null : prev)
  }, [])
  // 清除高亮（不影响详情卡片）
  const clearKnowledgeHighlight = useCallback(() => {
    setKnowledgeHighlightedNodeId(null)
  }, [])
  // 切换视图模式，离开knowledge时清空选中节点和高亮
  const setViewMode = useCallback((mode: '3d' | 'topology' | 'analysis' | 'knowledge') => {
    if (viewMode === 'knowledge' && mode !== 'knowledge') {
      setKnowledgeSelectedNodes([])
      setKnowledgeHighlightedNodeId(null)
    }
    setViewModeInternal(mode)
  }, [viewMode])

  // ==================== 拓扑操作 ====================
  const loadTopology = useCallback(async () => {
    setLoading(true)
    try {
      const cachedStr = localStorage.getItem(CONFIG_CACHE_KEY)
      if (cachedStr) {
        const cached = JSON.parse(cachedStr)
        const data = await generateTopology({
          pod_count: cached.podCount,
          racks_per_pod: cached.racksPerPod,
          board_configs: cached.boardConfigs,
          switch_config: cached.switchConfig,
          manual_connections: cached.manualConnectionConfig,
        })
        setTopology(data)
      } else {
        const data = await getTopology()
        setTopology(data)
      }
    } catch (error) {
      console.error('加载拓扑失败:', error)
      message.error('加载拓扑数据失败')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleGenerate = useCallback(async (config: GenerateConfig) => {
    try {
      const data = await generateTopology(config)
      setTopology(data)
    } catch (error) {
      console.error('生成拓扑失败:', error)
      message.error('生成拓扑失败')
    }
  }, [])

  // 初始加载
  useEffect(() => {
    loadTopology()
  }, [loadTopology])

  // ==================== 知识图谱预初始化 ====================
  // 在应用启动时就初始化知识图谱布局，避免切换页面时节点乱飞
  useEffect(() => {
    // 如果已经初始化过，跳过
    if (knowledgeInitialized) return

    // 使用 requestIdleCallback 或 setTimeout 避免阻塞主线程
    const initKnowledgeGraph = () => {
      const centerX = 600
      const centerY = 400
      const data = knowledgeData

      // 计算节点度数
      const initDegreeMap = new Map<string, number>()
      data.nodes.forEach(n => initDegreeMap.set(n.id, 0))
      data.relations.forEach(r => {
        initDegreeMap.set(r.source, (initDegreeMap.get(r.source) || 0) + 1)
        initDegreeMap.set(r.target, (initDegreeMap.get(r.target) || 0) + 1)
      })
      const maxDegree = Math.max(...initDegreeMap.values(), 1)

      // 初始化节点位置 - 爆炸式发散：度数决定半径，类别决定角度
      const totalCategories = KNOWLEDGE_CATEGORY_ORDER.length
      const initialNodes: ForceKnowledgeNode[] = data.nodes.map((node) => {
        const category = node.category as KnowledgeCategory
        const categoryIndex = KNOWLEDGE_CATEGORY_ORDER.indexOf(category)
        const degree = initDegreeMap.get(node.id) || 0

        // 爆炸式初始位置：高度数靠中心，低度数在外围
        const degreeRatio = degree / maxDegree
        const distanceRatio = Math.pow(1 - degreeRatio, 1.5)
        const minRadius = KNOWLEDGE_FORCE_CONFIG.radialMinRadius
        const maxRadius = KNOWLEDGE_FORCE_CONFIG.radialMaxRadius
        const radius = minRadius + distanceRatio * (maxRadius - minRadius)

        // 同类别节点基础角度相近，形成"颜色射线束"
        const categoryAngle = (categoryIndex / totalCategories) * 2 * Math.PI - Math.PI / 2
        const angleSpread = Math.PI / totalCategories * 0.8
        const randomOffset = (Math.random() - 0.5) * angleSpread
        const angle = categoryAngle + randomOffset
        const jitter = Math.random() * 30

        return {
          ...node,
          category,
          x: centerX + Math.cos(angle) * (radius + jitter),
          y: centerY + Math.sin(angle) * (radius + jitter),
          vx: 0,
          vy: 0,
        }
      })

      // 筛选可见关系用于力导向布局
      const visibleNodeIds = new Set(initialNodes.map(n => n.id))
      const coreRelations = data.relations.filter(
        r => visibleNodeIds.has(r.source) && visibleNodeIds.has(r.target) && CORE_RELATION_TYPES.has(r.type)
      )
      const nodeEdgeCount = new Map<string, number>()
      const visibleRelations = coreRelations.filter(r => {
        const sourceCount = nodeEdgeCount.get(r.source) || 0
        const targetCount = nodeEdgeCount.get(r.target) || 0
        if (sourceCount >= MAX_EDGES_PER_NODE || targetCount >= MAX_EDGES_PER_NODE) {
          return false
        }
        nodeEdgeCount.set(r.source, sourceCount + 1)
        nodeEdgeCount.set(r.target, targetCount + 1)
        return true
      })

      // 创建力导向模拟 - 简化稳定版
      const simulation = d3Force.forceSimulation<ForceKnowledgeNode>(initialNodes)
        .force('charge', d3Force.forceManyBody<ForceKnowledgeNode>()
          .strength(KNOWLEDGE_FORCE_CONFIG.chargeStrength)
          .distanceMax(KNOWLEDGE_FORCE_CONFIG.chargeDistanceMax)
        )
        .force('link', d3Force.forceLink<ForceKnowledgeNode, d3Force.SimulationLinkDatum<ForceKnowledgeNode>>(
          visibleRelations.map(r => ({ source: r.source, target: r.target }))
        )
          .id(d => d.id)
          .distance(KNOWLEDGE_FORCE_CONFIG.linkDistance)
          .strength(KNOWLEDGE_FORCE_CONFIG.linkStrength)
        )
        .force('radial', d3Force.forceRadial<ForceKnowledgeNode>(
          (d) => {
            const degree = initDegreeMap.get(d.id) || 0
            const radiusFactor = Math.pow(1 - degree / maxDegree, 1.2)
            return KNOWLEDGE_FORCE_CONFIG.radialMinRadius +
                   radiusFactor * (KNOWLEDGE_FORCE_CONFIG.radialMaxRadius - KNOWLEDGE_FORCE_CONFIG.radialMinRadius)
          },
          centerX, centerY
        ).strength(KNOWLEDGE_FORCE_CONFIG.radialStrength))
        .force('centerX', d3Force.forceX(centerX).strength(KNOWLEDGE_FORCE_CONFIG.centerStrength))
        .force('centerY', d3Force.forceY(centerY).strength(KNOWLEDGE_FORCE_CONFIG.centerStrength))
        .force('collision', d3Force.forceCollide<ForceKnowledgeNode>()
          .radius(KNOWLEDGE_FORCE_CONFIG.collisionRadius)
          .strength(KNOWLEDGE_FORCE_CONFIG.collisionStrength)
          .iterations(KNOWLEDGE_FORCE_CONFIG.collisionIterations)
        )
        .stop()

      // Warmup：静默运行直到稳定
      for (let i = 0; i < KNOWLEDGE_FORCE_CONFIG.warmupTicks; i++) {
        simulation.tick()
      }

      // 清零速度确保静止
      initialNodes.forEach(node => {
        node.vx = 0
        node.vy = 0
      })

      // 计算适合所有节点的视口
      const padding = 100
      const minX = Math.min(...initialNodes.map(n => n.x ?? 0)) - padding
      const maxX = Math.max(...initialNodes.map(n => n.x ?? 0)) + padding
      const minY = Math.min(...initialNodes.map(n => n.y ?? 0)) - padding
      const maxY = Math.max(...initialNodes.map(n => n.y ?? 0)) + padding
      const width = Math.max(maxX - minX, 400)
      const height = Math.max(maxY - minY, 300)

      // 保存预计算的结果
      setKnowledgeNodes(initialNodes)
      setKnowledgeViewBox({ x: minX, y: minY, width, height })
      setKnowledgeInitialized(true)
    }

    // 使用 requestIdleCallback 在空闲时初始化，如果不支持则使用 setTimeout
    if ('requestIdleCallback' in window) {
      (window as typeof window & { requestIdleCallback: (cb: () => void) => number }).requestIdleCallback(initKnowledgeGraph)
    } else {
      setTimeout(initKnowledgeGraph, 100)
    }
  }, [knowledgeInitialized])

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

  // 层级默认参数变化时更新连接 - 使用 useEffect 依赖触发（修复 setTimeout 竞态问题）
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
  }, [manualConnectionConfig.level_defaults, topology])

  // ==================== 连接编辑操作 ====================
  const setManualConnectionConfig = useCallback((config: ManualConnectionConfig) => {
    setManualConnectionConfigRaw(config)
    if (!config.enabled) {
      setConnectionModeRaw('view')
      setSelectedNodes(new Set())
      setTargetNodes(new Set())
      setSourceNode(null)
    }
  }, [])

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

  const handleManualConnect = useCallback((sourceId: string, targetId: string, level: HierarchyLevel) => {
    if (connectionExists(sourceId, targetId, manualConnectionConfig.connections)) {
      message.warning(`手动连接已存在: ${sourceId} ↔ ${targetId}`)
      return
    }
    if (topology?.connections && connectionExists(sourceId, targetId, topology.connections)) {
      message.warning(`自动连接已存在: ${sourceId} ↔ ${targetId}`)
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
    message.success(`已添加连接: ${sourceId} ↔ ${targetId}`)
  }, [manualConnectionConfig.connections, topology?.connections])

  const handleBatchConnect = useCallback((level: HierarchyLevel) => {
    if (selectedNodes.size === 0 || targetNodes.size === 0) {
      message.warning('请先选择源节点和目标节点')
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
      message.success(`已添加 ${newConnections.length} 条连接`)
    } else {
      message.warning('所有连接已存在')
    }
    setSelectedNodes(new Set())
    setTargetNodes(new Set())
    setConnectionModeRaw('select_source')
  }, [selectedNodes, targetNodes, manualConnectionConfig.connections, topology?.connections])

  const handleDeleteManualConnection = useCallback((connectionId: string) => {
    setManualConnectionConfigRaw(prev => ({
      ...prev,
      connections: prev.connections.filter(c => c.id !== connectionId),
    }))
    message.success('已删除连接')
  }, [])

  const handleDeleteConnection = useCallback((source: string, target: string) => {
    const manualConn = findConnection(source, target, manualConnectionConfig.connections)
    if (manualConn) {
      setManualConnectionConfigRaw(prev => ({
        ...prev,
        connections: prev.connections.filter(c => c.id !== manualConn.id),
      }))
      message.success('已删除手动连接')
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
      message.success('已删除连接')
    }
  }, [manualConnectionConfig.connections])

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
  }, [manualConnectionConfig.connections])

  // ==================== 分析操作 ====================
  const prevResultRef = useRef<typeof deploymentAnalysisData>(null)
  useEffect(() => {
    if (deploymentAnalysisData?.result && !prevResultRef.current?.result) {
      setViewMode('analysis')
      setAnalysisViewMode('detail')
    }
    if (deploymentAnalysisData?.history) {
      setAnalysisHistory(deploymentAnalysisData.history)
    }
    prevResultRef.current = deploymentAnalysisData
  }, [deploymentAnalysisData])

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
      // 历史由 WorkbenchContext 统一管理，这里保持兼容
      history: prev?.history || [],
      onLoadFromHistory: prev?.onLoadFromHistory || (() => {}),
      onDeleteHistory: prev?.onDeleteHistory || (() => {}),
      onClearHistory: prev?.onClearHistory || (() => {}),
    }))
    setAnalysisViewMode('detail')
  }, [])

  const handleDeleteHistory = useCallback((id: string) => {
    setAnalysisHistory(prev => {
      const updated = prev.filter(h => h.id !== id)
      localStorage.setItem(ANALYSIS_HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
  }, [])

  const handleClearHistory = useCallback(() => {
    localStorage.setItem(ANALYSIS_HISTORY_KEY, '[]')
    setAnalysisHistory([])
  }, [])

  const MAX_HISTORY_ITEMS = 20
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

  // ==================== 计算属性 ====================
  const getCurrentLevel = useCallback(() => {
    if (navigation.currentBoard) return 'board'
    if (navigation.currentRack) return 'rack'
    if (navigation.currentPod) return 'pod'
    return 'datacenter'
  }, [navigation.currentBoard, navigation.currentRack, navigation.currentPod])

  const currentViewConnections = useMemo(() => {
    if (!topology) return []
    const currentLevel = getCurrentLevel()
    if (currentLevel === 'datacenter') {
      const podIds = new Set(topology.pods.map(p => p.id))
      const dcSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_pod')
          .map(s => s.id)
      )
      return topology.connections.filter(c => {
        const sourceInDc = podIds.has(c.source) || dcSwitchIds.has(c.source)
        const targetInDc = podIds.has(c.target) || dcSwitchIds.has(c.target)
        return sourceInDc && targetInDc
      })
    } else if (currentLevel === 'pod' && navigation.currentPod) {
      const rackIds = new Set(navigation.currentPod.racks.map(r => r.id))
      const podSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_rack' && s.parent_id === navigation.currentPod!.id)
          .map(s => s.id)
      )
      return topology.connections.filter(c => {
        const sourceInPod = rackIds.has(c.source) || podSwitchIds.has(c.source)
        const targetInPod = rackIds.has(c.target) || podSwitchIds.has(c.target)
        return sourceInPod && targetInPod
      })
    } else if (currentLevel === 'rack' && navigation.currentRack) {
      const boardIds = new Set(navigation.currentRack.boards.map(b => b.id))
      const rackSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_board' && s.parent_id === navigation.currentRack!.id)
          .map(s => s.id)
      )
      return topology.connections.filter(c => {
        const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
        const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
        return sourceInRack && targetInRack
      })
    } else if (currentLevel === 'board' && navigation.currentBoard) {
      const chipIds = new Set(navigation.currentBoard.chips.map(c => c.id))
      return topology.connections.filter(c =>
        chipIds.has(c.source) && chipIds.has(c.target)
      )
    }
    return []
  }, [topology, navigation.currentPod, navigation.currentRack, navigation.currentBoard, getCurrentLevel])

  // ==================== 组装 Context 值 ====================
  const contextValue: WorkbenchContextType = {
    topology: {
      topology,
      loading,
      loadTopology,
      handleGenerate,
    },
    connection: {
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
    },
    analysis: {
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
    },
    ui: {
      viewMode,
      selectedNode,
      selectedLink,
      focusedLevel,
      knowledgeSelectedNodes,
      knowledgeHighlightedNodeId,
      knowledgeVisibleCategories,
      knowledgeNodes,
      knowledgeInitialized,
      knowledgeViewBox,
      setViewMode,
      setSelectedNode,
      setSelectedLink,
      setFocusedLevel,
      addKnowledgeSelectedNode,
      removeKnowledgeSelectedNode,
      clearKnowledgeHighlight,
      setKnowledgeVisibleCategories,
      setKnowledgeNodes,
      setKnowledgeInitialized,
      setKnowledgeViewBox,
      resetKnowledgeCategories,
    },
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

export default WorkbenchContext
