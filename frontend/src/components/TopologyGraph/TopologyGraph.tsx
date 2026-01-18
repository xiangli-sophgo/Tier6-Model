import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react'
import { Modal, Button, Space, Typography, Breadcrumb } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined } from '@ant-design/icons'
import {
  HierarchyLevel,
} from '../../types'
import {
  TopologyGraphProps,
  Node,
  LayoutType,
} from './shared'
// renderNodeShape 已被统一的 renderNode 函数替代
import { ControlPanel, TorusArcs } from './components'
import { ForceLayoutManager, ForceNode } from './layouts'
import { computeTopologyData } from './computeTopologyData'
import { MultiLevelView } from './MultiLevelView'
import { SingleLevelView } from './SingleLevelView'

const { Text } = Typography

// ManualConnectionLine 已提取到独立文件，renderNode 统一渲染所有节点类型

// ============================================
// 模块级常量（避免组件重渲染时重新创建）
// ============================================

// 节点尺寸配置（统一管理）- 单层级视图
const NODE_SIZE_CONFIG: Record<string, { w: number; h: number; labelY: number; fontSize: number }> = {
  switch: { w: 61, h: 24, labelY: 4, fontSize: 14 },
  pod: { w: 56, h: 32, labelY: 9, fontSize: 15 },
  rack: { w: 36, h: 56, labelY: 6, fontSize: 14 },
  board: { w: 64, h: 36, labelY: 5, fontSize: 15 },
  chip: { w: 40, h: 40, labelY: 5, fontSize: 14 },
  default: { w: 50, h: 36, labelY: 0, fontSize: 11 },
}

// 多层级视图的节点尺寸配置（可单独调整）
const MULTI_LEVEL_NODE_SIZE_CONFIG: Record<string, { w: number; h: number; labelY: number; fontSize: number }> = {
  switch: { w: 61, h: 24, labelY: 5, fontSize: 14 },
  pod: { w: 56, h: 32, labelY: 9, fontSize: 16 },
  rack: { w: 36, h: 56, labelY: 6, fontSize: 16 },
  board: { w: 64, h: 36, labelY: 5, fontSize: 16 },
  chip: { w: 40, h: 40, labelY: 5, fontSize: 16 },
  default: { w: 50, h: 36, labelY: 0, fontSize: 11 },
}

// 格式化标签（名称第一个字符大写 + 编号）
const formatNodeLabel = (label: string): string => {
  const match = label.match(/\d+/)
  const num = match ? match[0] : ''
  const firstChar = label.charAt(0).toUpperCase()
  return `${firstChar}${num}`
}

export const TopologyGraph: React.FC<TopologyGraphProps> = ({
  visible,
  onClose,
  topology,
  currentLevel,
  currentPod,
  currentRack,
  currentBoard,
  onNodeDoubleClick,
  onNodeClick,
  onLinkClick,
  selectedNodeId = null,
  selectedLinkId = null,
  onNavigateBack: _onNavigateBack,
  onBreadcrumbClick,
  breadcrumbs = [],
  canGoBack: _canGoBack = false,
  embedded = false,
  // 手动连线相关
  connectionMode = 'view',
  selectedNodes = new Set<string>(),
  onSelectedNodesChange,
  targetNodes = new Set<string>(),
  onTargetNodesChange,
  sourceNode: _sourceNode = null,
  onSourceNodeChange: _onSourceNodeChange,
  onManualConnect: _onManualConnect,
  manualConnections = [],
  onDeleteManualConnection: _onDeleteManualConnection,
  onDeleteConnection: _onDeleteConnection,
  layoutType = 'auto',
  onLayoutTypeChange,
  // 多层级视图相关
  multiLevelOptions,
  onMultiLevelOptionsChange,
  // 流量热力图
  trafficResult,
}) => {
  void _onNavigateBack
  void _canGoBack
  void _sourceNode
  void _onSourceNodeChange
  void _onManualConnect
  void _onDeleteManualConnection
  void _onDeleteConnection
  void onMultiLevelOptionsChange
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null)
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)

  // 力导向布局动态模拟
  const forceManagerRef = useRef<ForceLayoutManager | null>(null)
  const [forceNodes, setForceNodes] = useState<ForceNode[]>([])
  const [isForceSimulating, setIsForceSimulating] = useState(false)
  const isForceMode = layoutType === 'force' && !multiLevelOptions?.enabled

  // 多层级模式：悬停的层级索引（用于抬起上方层级）
  const [hoveredLayerIndex, setHoveredLayerIndex] = useState<number | null>(null)

  // 容器展开动画状态
  const [expandingContainer, setExpandingContainer] = useState<{
    id: string
    type: string
  } | null>(null)

  // 包装双击事件，先触发展开动画，再执行实际导航
  // TODO: 动画功能暂时禁用，直接导航
  const handleNodeDoubleClickWithAnimation = useCallback((nodeId: string, nodeType: string) => {
    // if (multiLevelOptions?.enabled && (nodeType === 'pod' || nodeType === 'rack' || nodeType === 'board')) {
    //   // 触发展开动画（动画完成后由 onTransitionEnd 回调处理导航）
    //   setExpandingContainer({ id: nodeId, type: nodeType })
    // } else {
    //   onNodeDoubleClick?.(nodeId, nodeType)
    // }
    onNodeDoubleClick?.(nodeId, nodeType)
  }, [onNodeDoubleClick])

  // 展开动画完成回调
  const handleExpandAnimationEnd = useCallback((nodeId: string, nodeType: string) => {
    setExpandingContainer(null)
    onNodeDoubleClick?.(nodeId, nodeType)
  }, [onNodeDoubleClick])

  // 容器收缩动画状态（从单层级切换到多层级时使用）
  const [collapsingContainer, setCollapsingContainer] = useState<{
    id: string
    type: string
  } | null>(null)
  // 收缩动画是否已开始（用于两阶段动画：先渲染展开状态，再过渡到正常状态）
  const [collapseAnimationStarted, setCollapseAnimationStarted] = useState(false)

  // 视图切换淡入效果
  const [viewFadeIn, setViewFadeIn] = useState(false)
  const prevMultiLevelEnabled = useRef(multiLevelOptions?.enabled)

  // 检测视图切换，触发动画
  useEffect(() => {
    if (prevMultiLevelEnabled.current && !multiLevelOptions?.enabled) {
      // 从多层级切换到单层级，触发淡入
      setViewFadeIn(true)
      const timer = setTimeout(() => setViewFadeIn(false), 50)
      return () => clearTimeout(timer)
    } else if (!prevMultiLevelEnabled.current && multiLevelOptions?.enabled) {
      // 从单层级切换到多层级，触发收缩动画
      // 根据当前层级确定要收缩到哪个容器
      let containerId = ''
      let containerType = ''
      if (currentRack) {
        containerId = currentRack.id
        containerType = 'rack'
      } else if (currentPod) {
        containerId = currentPod.id
        containerType = 'pod'
      }
      if (containerId) {
        setCollapsingContainer({ id: containerId, type: containerType })
        setCollapseAnimationStarted(false)  // 重置动画开始标志
        // 动画完成后清除状态
        const timer = setTimeout(() => {
          setCollapsingContainer(null)
          setCollapseAnimationStarted(false)
        }, 600)
        return () => clearTimeout(timer)
      }
    }
    prevMultiLevelEnabled.current = multiLevelOptions?.enabled
  }, [multiLevelOptions?.enabled, currentPod, currentRack])

  // 收缩动画：在下一帧开始动画（从展开状态过渡到正常状态）
  useEffect(() => {
    if (collapsingContainer && !collapseAnimationStarted) {
      const frameId = requestAnimationFrame(() => {
        setCollapseAnimationStarted(true)
      })
      return () => cancelAnimationFrame(frameId)
    }
  }, [collapsingContainer, collapseAnimationStarted])

  // 构建链路流量查找表
  const linkTrafficMap = useMemo(() => {
    const map = new Map<string, { trafficMb: number; utilizationPercent: number; groups: string[] }>()
    if (trafficResult?.linkTraffic) {
      for (const lt of trafficResult.linkTraffic) {
        // 使用双向key
        map.set(`${lt.source}->${lt.target}`, {
          trafficMb: lt.trafficMb,
          utilizationPercent: lt.utilizationPercent,
          groups: lt.contributingGroups,
        })
        map.set(`${lt.target}->${lt.source}`, {
          trafficMb: lt.trafficMb,
          utilizationPercent: lt.utilizationPercent,
          groups: lt.contributingGroups,
        })
      }
    }
    return map
  }, [trafficResult])

  // 计算最大流量（用于宽度归一化）
  const maxTrafficMb = useMemo(() => {
    if (!trafficResult?.linkTraffic || trafficResult.linkTraffic.length === 0) return 0
    return Math.max(...trafficResult.linkTraffic.map(lt => lt.trafficMb))
  }, [trafficResult])

  // 获取边的热力图样式
  const getTrafficHeatmapStyle = useCallback((source: string, target: string) => {
    const lt = linkTrafficMap.get(`${source}->${target}`)
    if (!lt) return null

    // 根据利用率计算颜色
    const u = Math.min(Math.max(lt.utilizationPercent, 0), 100)
    let color: string
    if (u < 30) {
      const t = u / 30
      color = `rgb(${Math.round(100 * t)}, 200, ${Math.round(100 * (1 - t))})`
    } else if (u < 60) {
      const t = (u - 30) / 30
      color = `rgb(${Math.round(100 + 155 * t)}, 200, 0)`
    } else if (u < 80) {
      const t = (u - 60) / 20
      color = `rgb(255, ${Math.round(200 - 100 * t)}, 0)`
    } else {
      const t = (u - 80) / 20
      color = `rgb(255, ${Math.round(100 - 100 * t)}, 0)`
    }

    // 根据流量计算宽度 (2-6px)
    const width = maxTrafficMb > 0 ? 2 + (lt.trafficMb / maxTrafficMb) * 4 : 2

    return {
      stroke: color,
      strokeWidth: width,
      trafficMb: lt.trafficMb,
      utilization: lt.utilizationPercent,
    }
  }, [linkTrafficMap, maxTrafficMb])

  // 节点形状渲染函数（仅渲染形状和标签，不包含外层g）
  const renderNodeShape = useCallback((node: Node, useMultiLevelConfig: boolean = false) => {
    const nodeType = node.isSwitch ? 'switch' : node.type.toLowerCase()
    const config = useMultiLevelConfig ? MULTI_LEVEL_NODE_SIZE_CONFIG : NODE_SIZE_CONFIG
    const size = config[nodeType] || config.default
    const halfW = size.w / 2
    const halfH = size.h / 2

    return (
      <>
        {/* 根据节点类型渲染不同形状 */}
        {nodeType === 'switch' && (
          <>
            <rect x={-halfW} y={-halfH} width={size.w} height={size.h} rx={3} fill={node.color} stroke="#fff" strokeWidth={1.5} />
            {/* 端口 */}
            <rect x={-24} y={-6} width={5} height={6} rx={1} fill="rgba(255,255,255,0.5)" />
            <rect x={-17} y={-6} width={5} height={6} rx={1} fill="rgba(255,255,255,0.5)" />
            <rect x={-10} y={-6} width={5} height={6} rx={1} fill="rgba(255,255,255,0.5)" />
            <rect x={5} y={-6} width={5} height={6} rx={1} fill="rgba(255,255,255,0.5)" />
            <rect x={12} y={-6} width={5} height={6} rx={1} fill="rgba(255,255,255,0.5)" />
            <rect x={19} y={-6} width={5} height={6} rx={1} fill="rgba(255,255,255,0.5)" />
            {/* 指示灯 */}
            <circle cx={-22} cy={6} r={2} fill="#4ade80" />
            <circle cx={-16} cy={6} r={2} fill="#4ade80" />
          </>
        )}

        {nodeType === 'pod' && (
          <>
            <rect x={-halfW} y={-halfH + 6} width={size.w} height={size.h - 6} rx={3} fill={node.color} stroke="#fff" strokeWidth={1.5} />
            <polygon points={`${-halfW},-${halfH - 6} 0,-${halfH} ${halfW},-${halfH - 6}`} fill={node.color} stroke="#fff" strokeWidth={1.5} />
            {/* 装饰 */}
            <rect x={-20} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
            <rect x={-6} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
            <rect x={8} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
          </>
        )}

        {nodeType === 'rack' && (
          <>
            <rect x={-halfW} y={-halfH} width={size.w} height={size.h} rx={3} fill={node.color} stroke="#fff" strokeWidth={1.5} />
            {/* 层级线 */}
            <line x1={-14} y1={-16} x2={14} y2={-16} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
            <line x1={-14} y1={-4} x2={14} y2={-4} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
            <line x1={-14} y1={8} x2={14} y2={8} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
            <line x1={-14} y1={20} x2={14} y2={20} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
            {/* 指示灯 */}
            <circle cx={10} cy={-22} r={2} fill="#4ade80" />
            <circle cx={10} cy={-10} r={2} fill="#4ade80" />
          </>
        )}

        {nodeType === 'board' && (
          <>
            <rect x={-halfW} y={-halfH} width={size.w} height={size.h} rx={2} fill={node.color} stroke="#fff" strokeWidth={1.5} />
            {/* 电路线装饰 */}
            <path d="M-24,-10 L-24,-2 L-16,-2 L-16,6 L-8,6" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} fill="none" />
            <path d="M8,-10 L8,0 L16,0 L16,8 L24,8" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} fill="none" />
            {/* 芯片 */}
            <rect x={-8} y={-8} width={16} height={16} rx={1} fill="rgba(0,0,0,0.2)" stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
          </>
        )}

        {nodeType === 'chip' && (
          <>
            <rect x={-halfW} y={-halfH} width={size.w} height={size.h} rx={2} fill={node.color} stroke="#fff" strokeWidth={1.5} />
            {/* 引脚 */}
            <rect x={-12} y={-halfH - 4} width={4} height={4} fill={node.color} />
            <rect x={-2} y={-halfH - 4} width={4} height={4} fill={node.color} />
            <rect x={8} y={-halfH - 4} width={4} height={4} fill={node.color} />
            <rect x={-12} y={halfH} width={4} height={4} fill={node.color} />
            <rect x={-2} y={halfH} width={4} height={4} fill={node.color} />
            <rect x={8} y={halfH} width={4} height={4} fill={node.color} />
            {/* 内核 */}
            <rect x={-10} y={-10} width={20} height={20} rx={1} fill="rgba(255,255,255,0.15)" />
          </>
        )}

        {/* 默认形状（未匹配的类型） */}
        {!['switch', 'pod', 'rack', 'board', 'chip'].includes(nodeType) && (
          <rect x={-halfW} y={-halfH} width={size.w} height={size.h} rx={6} fill={node.color} stroke="#fff" strokeWidth={1.5} />
        )}

        {/* 统一标签渲染 */}
        <text y={size.labelY} textAnchor="middle" fill="#fff" fontSize={size.fontSize} fontWeight={600} style={{ pointerEvents: 'none' }}>
          {formatNodeLabel(node.label)}
        </text>
      </>
    )
  }, [])

  // 统一节点渲染函数（带完整g包装） - 适用于Switch面板、容器内节点等场景
  const renderNode = useCallback((
    node: Node,
    options: {
      keyPrefix: string
      scale?: number
      isSelected?: boolean
      onClick?: () => void
      useMultiLevelConfig?: boolean
    }
  ) => {
    const { keyPrefix, scale = 1, isSelected = false, onClick, useMultiLevelConfig = false } = options

    return (
      <g
        key={`${keyPrefix}-${node.id}`}
        transform={scale === 1 ? `translate(${node.x}, ${node.y})` : `translate(${node.x}, ${node.y}) scale(${scale})`}
        style={{
          cursor: 'pointer',
          filter: isSelected
            ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.6)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.3))'
            : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
          transition: 'filter 0.15s ease',
        }}
        onClick={(e) => {
          e.stopPropagation()
          onClick?.()
        }}
      >
        {renderNodeShape(node, useMultiLevelConfig)}
      </g>
    )
  }, [renderNodeShape])

  // 手动调整模式开关（内部状态）
  const [isManualMode, setIsManualMode] = useState(false)

  // 手动布局缓存key（按层级、路径和布局类型区分）
  const getManualPositionsCacheKey = (layout: LayoutType) => {
    const pathKey = currentLevel === 'datacenter' ? 'dc' :
      currentLevel === 'pod' ? `pod_${currentPod?.id}` :
      currentLevel === 'rack' ? `rack_${currentRack?.id}` :
      `board_${currentBoard?.id}`
    return `tier6_manual_positions_${pathKey}_${layout}`
  }

  // 手动布局：按布局类型分开存储位置（从localStorage加载）
  const [manualPositionsByLayout, setManualPositionsByLayout] = useState<Record<LayoutType, Record<string, { x: number; y: number }>>>(() => {
    const result: Record<LayoutType, Record<string, { x: number; y: number }>> = {
      auto: {},
      circle: {},
      grid: {},
      force: {},
    }
    try {
      const pathKey = currentLevel === 'datacenter' ? 'dc' : currentLevel
      for (const layout of ['auto', 'circle', 'grid', 'force'] as LayoutType[]) {
        const cached = localStorage.getItem(`tier6_manual_positions_${pathKey}_${layout}`)
        if (cached) result[layout] = JSON.parse(cached)
      }
    } catch (e) { /* ignore */ }
    return result
  })

  // 当前布局的手动位置（便捷访问）
  const manualPositions = manualPositionsByLayout[layoutType] || {}

  // 拖动状态
  const [draggingNode, setDraggingNode] = useState<string | null>(null)
  const [dragStart, setDragStart] = useState<{ x: number; y: number; nodeX: number; nodeY: number } | null>(null)

  // 撤销/重做历史
  const [history, setHistory] = useState<Record<string, { x: number; y: number }>[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const maxHistoryLength = 50

  // 辅助线状态（支持水平、垂直和圆形）
  const [alignmentLines, setAlignmentLines] = useState<{ type: 'h' | 'v' | 'circle'; pos: number; center?: { x: number; y: number } }[]>([])

  // 层级/路径变化时，加载对应的手动位置
  useEffect(() => {
    const result: Record<LayoutType, Record<string, { x: number; y: number }>> = {
      auto: {},
      circle: {},
      grid: {},
      force: {},
    }
    try {
      for (const layout of ['auto', 'circle', 'grid', 'force'] as LayoutType[]) {
        const key = getManualPositionsCacheKey(layout)
        const cached = localStorage.getItem(key)
        if (cached) result[layout] = JSON.parse(cached)
      }
    } catch (e) { /* ignore */ }
    setManualPositionsByLayout(result)
    // 重置历史和手动模式
    setHistory([])
    setHistoryIndex(-1)
    setIsManualMode(false)
  }, [currentLevel, currentPod?.id, currentRack?.id, currentBoard?.id])

  // 手动位置变化时自动保存（只保存当前布局）
  useEffect(() => {
    const positions = manualPositionsByLayout[layoutType]
    if (positions && Object.keys(positions).length > 0) {
      try {
        const key = getManualPositionsCacheKey(layoutType)
        localStorage.setItem(key, JSON.stringify(positions))
      } catch (e) { /* ignore */ }
    }
  }, [manualPositionsByLayout, layoutType])


  // 获取当前层级对应的 HierarchyLevel
  const getCurrentHierarchyLevel = (): HierarchyLevel => {
    switch (currentLevel) {
      case 'datacenter': return 'datacenter'
      case 'pod': return 'pod'
      case 'rack': return 'rack'
      case 'board': return 'board'
      default: return 'datacenter'
    }
  }

  // 根据当前层级生成节点和边
  const { nodes, edges, title, directTopology, switchPanelWidth } = useMemo(() => {
    return computeTopologyData({
      topology,
      currentLevel,
      currentPod: currentPod ?? null,
      currentRack: currentRack ?? null,
      currentBoard: currentBoard ?? null,
      layoutType,
      multiLevelOptions,
      manualConnections,
    })
  }, [topology, currentLevel, currentPod, currentRack, currentBoard, layoutType, multiLevelOptions, manualConnections])


  // 力导向动态模拟管理
  useEffect(() => {
    if (!isForceMode) {
      // 非力导向模式，停止并清理模拟
      if (forceManagerRef.current) {
        forceManagerRef.current.destroy()
        forceManagerRef.current = null
      }
      setForceNodes([])
      setIsForceSimulating(false)
      return
    }

    // 初始化力导向模拟
    if (!forceManagerRef.current) {
      forceManagerRef.current = new ForceLayoutManager({
        width: 800,
        height: 600,
        chargeStrength: -300,
        linkDistance: 100,
        collisionRadius: 35,
      })
    }

    const manager = forceManagerRef.current

    // 初始化节点和边
    const initialNodes = manager.initialize(nodes, edges, {
      width: 800,
      height: 600,
    })

    // 设置 tick 回调
    manager.setOnTick((updatedNodes) => {
      setForceNodes([...updatedNodes])
    })

    manager.setOnEnd(() => {
      setIsForceSimulating(false)
    })

    setForceNodes(initialNodes)
    setIsForceSimulating(true)

    // 开始模拟
    manager.start(1.0)

    return () => {
      manager.stop()
    }
  }, [isForceMode, nodes.length, edges.length])

  // 切换到手动模式时，如果没有保存的位置，使用当前布局位置作为初始值
  useEffect(() => {
    if (isManualMode && Object.keys(manualPositions).length === 0 && nodes.length > 0) {
      // 没有保存的位置，使用当前布局的位置
      const currentPositions: Record<string, { x: number; y: number }> = {}
      nodes.forEach(node => {
        currentPositions[node.id] = { x: node.x, y: node.y }
      })
      setManualPositionsByLayout(prev => ({
        ...prev,
        [layoutType]: currentPositions
      }))
    }
  }, [isManualMode, nodes.length, layoutType])

  // 更新当前布局的手动位置
  const setManualPositions = useCallback((updater: Record<string, { x: number; y: number }> | ((prev: Record<string, { x: number; y: number }>) => Record<string, { x: number; y: number }>)) => {
    setManualPositionsByLayout(prev => ({
      ...prev,
      [layoutType]: typeof updater === 'function' ? updater(prev[layoutType] || {}) : updater
    }))
  }, [layoutType])

  // 当节点列表变化时（数量或ID变化），重置手动位置
  const prevNodeIdsRef = useRef<string>('')
  useEffect(() => {
    const currentNodeIds = nodes.map(n => n.id).sort().join(',')
    if (prevNodeIdsRef.current && prevNodeIdsRef.current !== currentNodeIds) {
      // 节点列表发生变化，重置当前布局的手动位置
      setManualPositionsByLayout(prev => ({
        ...prev,
        [layoutType]: {}
      }))
      setHistory([])
      setHistoryIndex(-1)
    }
    prevNodeIdsRef.current = currentNodeIds
  }, [nodes, layoutType])

  // 应用手动位置调整后的节点列表
  const displayNodes = useMemo(() => {
    // 力导向模式：使用动态模拟的位置
    if (isForceMode && forceNodes.length > 0) {
      return forceNodes as Node[]
    }
    // 手动模式：应用手动调整的位置
    if (isManualMode) {
      return nodes.map(node => {
        const manualPos = manualPositions[node.id]
        if (manualPos) {
          return { ...node, x: manualPos.x, y: manualPos.y }
        }
        return node
      })
    }
    // 其他模式：使用静态计算的位置
    return nodes
  }, [nodes, manualPositions, isManualMode, isForceMode, forceNodes])

  // 根据节点数量计算缩放系数
  const nodeScale = useMemo(() => {
    const deviceNodes = displayNodes.filter(n => !n.isSwitch)
    const count = deviceNodes.length
    if (count <= 4) return 1
    if (count <= 8) return 0.85
    if (count <= 16) return 0.7
    if (count <= 32) return 0.55
    if (count <= 64) return 0.45
    return 0.35
  }, [displayNodes])

  // 创建节点位置映射
  const nodePositions = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    displayNodes.forEach(node => {
      map.set(node.id, { x: node.x, y: node.y })
    })
    return map
  }, [displayNodes])

  const handleZoomIn = () => setZoom(z => Math.min(z + 0.2, 2))
  const handleZoomOut = () => setZoom(z => Math.max(z - 0.2, 0.5))

  // 对齐吸附阈值
  const SNAP_THRESHOLD = 10

  // 圆形布局的参数（与 circleLayout 函数一致）
  const CIRCLE_CENTER = { x: 400, y: 300 }
  const CIRCLE_RADIUS = Math.min(800, 600) * 0.35  // 210

  // 检测对齐并返回吸附后的位置
  const checkAlignment = (x: number, y: number, excludeNodeId: string) => {
    const lines: { type: 'h' | 'v' | 'circle'; pos: number; center?: { x: number; y: number } }[] = []
    let snappedX = x
    let snappedY = y

    // 环形布局：优先检测圆形轨迹吸附
    if (layoutType === 'circle') {
      const dx = x - CIRCLE_CENTER.x
      const dy = y - CIRCLE_CENTER.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      // 检测是否接近圆形轨迹
      if (Math.abs(distance - CIRCLE_RADIUS) < SNAP_THRESHOLD * 2) {
        // 吸附到圆上：保持角度，调整距离到半径
        const angle = Math.atan2(dy, dx)
        snappedX = CIRCLE_CENTER.x + CIRCLE_RADIUS * Math.cos(angle)
        snappedY = CIRCLE_CENTER.y + CIRCLE_RADIUS * Math.sin(angle)
        lines.push({ type: 'circle', pos: CIRCLE_RADIUS, center: CIRCLE_CENTER })
      }
    }

    // 获取其他节点的位置
    const otherNodes = displayNodes.filter(n => n.id !== excludeNodeId)

    for (const node of otherNodes) {
      // 水平对齐检测
      if (Math.abs(node.y - y) < SNAP_THRESHOLD) {
        snappedY = node.y
        lines.push({ type: 'h', pos: node.y })
      }
      // 垂直对齐检测
      if (Math.abs(node.x - x) < SNAP_THRESHOLD) {
        snappedX = node.x
        lines.push({ type: 'v', pos: node.x })
      }
    }

    return { snappedX, snappedY, lines }
  }

  // 保存历史记录
  const saveToHistory = useCallback((positions: Record<string, { x: number; y: number }>) => {
    setHistory(prev => {
      // 删除当前位置之后的历史（重做时）
      const newHistory = prev.slice(0, historyIndex + 1)
      newHistory.push({ ...positions })
      // 限制历史长度
      if (newHistory.length > maxHistoryLength) {
        newHistory.shift()
        return newHistory
      }
      return newHistory
    })
    setHistoryIndex(prev => Math.min(prev + 1, maxHistoryLength - 1))
  }, [historyIndex])

  // 撤销
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1
      setHistoryIndex(newIndex)
      setManualPositions(history[newIndex] || {})
    } else if (historyIndex === 0) {
      setHistoryIndex(-1)
      setManualPositions({})
    }
  }, [history, historyIndex])

  // 重做
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1
      setHistoryIndex(newIndex)
      setManualPositions(history[newIndex])
    }
  }, [history, historyIndex])

  // 键盘快捷键：Ctrl+Z 撤销，Ctrl+Y 重做
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isManualMode) return
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault()
          handleUndo()
        } else if (e.key === 'y' || (e.key === 'z' && e.shiftKey)) {
          e.preventDefault()
          handleRedo()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isManualMode, handleUndo, handleRedo])

  // 计算屏幕坐标到SVG坐标的转换比例
  const getScreenToSvgScale = useCallback(() => {
    if (!svgRef.current) return { scaleX: 1, scaleY: 1 }
    const rect = svgRef.current.getBoundingClientRect()
    // viewBox 尺寸（考虑 zoom）
    const viewBoxWidth = 800 / zoom
    const viewBoxHeight = 600 / zoom
    // 屏幕像素到 SVG 坐标的比例
    const scaleX = viewBoxWidth / rect.width
    const scaleY = viewBoxHeight / rect.height
    return { scaleX, scaleY }
  }, [zoom])

  // 拖动处理（支持手动模式和力导向模式）
  const handleDragStart = (nodeId: string, e: React.MouseEvent) => {
    // 查找节点：先在 displayNodes 中找，再在多层级视图的 singleLevelData 中找
    const findNode = () => {
      const node = displayNodes.find(n => n.id === nodeId)
      if (node) return node
      // 在多层级视图的容器中查找
      for (const container of displayNodes) {
        if (container.isContainer && container.singleLevelData) {
          const slNode = container.singleLevelData.nodes.find((n: any) => n.id === nodeId)
          if (slNode) return slNode
        }
      }
      return null
    }

    // 力导向模式：直接拖拽（无需 Shift 键）
    if (isForceMode) {
      e.preventDefault()
      e.stopPropagation()
      const node = findNode()
      if (!node) return
      setDraggingNode(nodeId)
      setDragStart({ x: e.clientX, y: e.clientY, nodeX: node.x, nodeY: node.y })
      // 固定节点位置
      forceManagerRef.current?.fixNode(nodeId, node.x, node.y)
      return
    }
    // 手动模式：需要 Shift 键
    if (!isManualMode || !e.shiftKey) return
    e.preventDefault()
    e.stopPropagation()
    const node = findNode()
    if (!node) return
    setDraggingNode(nodeId)
    setDragStart({ x: e.clientX, y: e.clientY, nodeX: node.x, nodeY: node.y })
  }

  const handleDragMove = (e: React.MouseEvent) => {
    if (!draggingNode || !dragStart) return
    e.preventDefault()

    // 使用正确的坐标转换
    const { scaleX, scaleY } = getScreenToSvgScale()
    const dx = (e.clientX - dragStart.x) * scaleX
    const dy = (e.clientY - dragStart.y) * scaleY
    const rawX = dragStart.nodeX + dx
    const rawY = dragStart.nodeY + dy

    // 检查是否是多层级视图中的节点
    const isMultiLevelNode = displayNodes.some(n =>
      n.isContainer && n.singleLevelData?.nodes.some((sn: any) => sn.id === draggingNode)
    )

    // 力导向模式：通过物理模拟更新位置
    if (isForceMode) {
      if (isMultiLevelNode) {
        // 多层级视图中的节点：直接更新位置
        setManualPositions(prev => ({
          ...prev,
          [draggingNode]: { x: rawX, y: rawY }
        }))
      } else {
        // 单层级视图中的节点：通过力导向管理器更新
        forceManagerRef.current?.dragNode(draggingNode, rawX, rawY)
      }
      return
    }

    // 手动模式：检测对齐并更新位置
    const { snappedX, snappedY, lines } = checkAlignment(rawX, rawY, draggingNode)
    setAlignmentLines(lines)

    setManualPositions(prev => ({
      ...prev,
      [draggingNode]: {
        x: snappedX,
        y: snappedY,
      }
    }))
  }

  const handleDragEnd = () => {
    if (draggingNode) {
      // 检查是否是多层级视图中的节点
      const isMultiLevelNode = displayNodes.some(n =>
        n.isContainer && n.singleLevelData?.nodes.some((sn: any) => sn.id === draggingNode)
      )

      // 力导向模式：释放节点，让物理模拟继续
      if (isForceMode) {
        if (isMultiLevelNode) {
          // 多层级视图中的节点：保存到历史记录
          saveToHistory(manualPositions)
        } else {
          // 单层级视图中的节点：释放力导向模拟
          forceManagerRef.current?.releaseNode(draggingNode)
        }
      } else {
        // 手动模式：保存到历史记录
        saveToHistory(manualPositions)
      }
    }
    setDraggingNode(null)
    setDragStart(null)
    setAlignmentLines([])
  }

  // 重置当前布局的手动位置
  const handleResetManualPositions = useCallback(() => {
    setManualPositionsByLayout(prev => ({
      ...prev,
      [layoutType]: {}
    }))
    setHistory([])
    setHistoryIndex(-1)
    try {
      const key = getManualPositionsCacheKey(layoutType)
      localStorage.removeItem(key)
    } catch (e) { /* ignore */ }
  }, [layoutType, currentLevel, currentPod?.id, currentRack?.id, currentBoard?.id])

  // 工具栏组件
  const toolbar = (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: embedded ? '12px 16px' : 0,
      background: embedded ? '#fff' : 'transparent',
      borderBottom: embedded ? '1px solid #f0f0f0' : 'none',
    }}>
      {embedded && breadcrumbs.length > 0 ? (
        <Breadcrumb
          items={breadcrumbs.map((item, index) => ({
            key: item.id,
            title: (
              <a
                onClick={(e) => {
                  e.preventDefault()
                  onBreadcrumbClick?.(index)
                }}
                style={{
                  cursor: index < breadcrumbs.length - 1 ? 'pointer' : 'default',
                  color: index < breadcrumbs.length - 1 ? '#1890ff' : 'rgba(0, 0, 0, 0.88)',
                  fontWeight: index === breadcrumbs.length - 1 ? 500 : 400,
                }}
              >
                {item.label}
              </a>
            ),
          }))}
        />
      ) : (
        <span style={{ fontWeight: 500 }}>{title || '抽象拓扑图'}</span>
      )}
      <Space>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {directTopology !== 'none' ? `拓扑: ${
            directTopology === 'full_mesh' ? '全连接' :
            directTopology === 'ring' ? '环形' :
            directTopology === 'torus_2d' ? '2D Torus' :
            directTopology === 'torus_3d' ? '3D Torus' :
            directTopology === 'hypercube' ? '超立方体' :
            directTopology === 'star' ? '星形' : directTopology
          }` : ''}
        </Text>
        <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
        <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn} />
      </Space>
    </div>
  )

  // 图形内容
  const graphContent = (
    <div style={{
      width: '100%',
      height: embedded ? '100%' : 650,
      overflow: 'hidden',
      background: '#fafafa',
      position: 'relative',
    }}>
      {/* 悬浮面包屑导航 */}
      {embedded && breadcrumbs.length > 0 && (
        <div style={{
          position: 'absolute',
          top: 16,
          left: 16,
          zIndex: 100,
          background: '#fff',
          padding: '10px 16px',
          borderRadius: 10,
          border: '1px solid rgba(0, 0, 0, 0.08)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.06)',
        }}>
          <Breadcrumb
            items={breadcrumbs.map((item, index) => ({
              key: item.id,
              title: (
                <a
                  onClick={(e) => {
                    e.preventDefault()
                    onBreadcrumbClick?.(index)
                  }}
                  style={{
                    cursor: index < breadcrumbs.length - 1 ? 'pointer' : 'default',
                    color: index < breadcrumbs.length - 1 ? '#2563eb' : '#171717',
                    fontWeight: index === breadcrumbs.length - 1 ? 500 : 400,
                  }}
                >
                  {item.label}
                </a>
              ),
            }))}
          />
        </div>
      )}

      {/* 右上角控制面板悬浮框 */}
      {embedded && (
        <ControlPanel
          multiLevelOptions={multiLevelOptions}
          onMultiLevelOptionsChange={onMultiLevelOptionsChange}
          currentLevel={currentLevel}
          layoutType={layoutType}
          onLayoutTypeChange={onLayoutTypeChange}
          isForceMode={isForceMode}
          isForceSimulating={isForceSimulating}
          isManualMode={isManualMode}
          setIsManualMode={setIsManualMode}
          manualPositions={manualPositions}
          historyIndex={historyIndex}
          historyLength={history.length}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onReset={handleResetManualPositions}
          onLayoutChange={() => {
            setHistory([])
            setHistoryIndex(-1)
          }}
        />
      )}

        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`${400 - 400/zoom} ${300 - 300/zoom} ${800 / zoom} ${600 / zoom}`}
          style={{
            display: 'block',
            opacity: viewFadeIn ? 0 : 1,
            transition: 'opacity 0.4s ease-out',
          }}
          onMouseMove={handleDragMove}
          onMouseUp={handleDragEnd}
          onMouseLeave={handleDragEnd}
        >
          {/* 背景层 - 用于点击空白区域清除选中状态 */}
          <rect
            x={400 - 400/zoom}
            y={300 - 300/zoom}
            width={800 / zoom}
            height={600 / zoom}
            fill="transparent"
            onClick={() => {
              // 在连接模式下，点击空白处重置悬停状态
              if (connectionMode !== 'view') {
                setHoveredLayerIndex(null)
              } else if (!isManualMode) {
                onNodeClick?.(null)
                onLinkClick?.(null)
              }
            }}
          />
          {/* 定义箭头标记 */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#999" />
            </marker>
            {/* 手动连接箭头 */}
            <marker
              id="arrowhead-manual"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#52c41a" />
            </marker>
            {/* 跨层级连线3D效果 - 渐变定义 */}
            <linearGradient id="interLevelGradient-horizontal" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#faad14" stopOpacity="1"/>
              <stop offset="30%" stopColor="#faad14" stopOpacity="0.4"/>
              <stop offset="70%" stopColor="#faad14" stopOpacity="0.4"/>
              <stop offset="100%" stopColor="#faad14" stopOpacity="1"/>
            </linearGradient>
            <linearGradient id="interLevelGradient-vertical" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#faad14" stopOpacity="1"/>
              <stop offset="30%" stopColor="#faad14" stopOpacity="0.4"/>
              <stop offset="70%" stopColor="#faad14" stopOpacity="0.4"/>
              <stop offset="100%" stopColor="#faad14" stopOpacity="1"/>
            </linearGradient>
            {/* 跨层级连线3D效果 - 发光滤镜 */}
            <filter id="interLevelGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur"/>
              <feComposite in="SourceGraphic" in2="blur" operator="over"/>
            </filter>
            {/* 跨层级连线3D效果 - 阴影滤镜 */}
            <filter id="interLevel3DShadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="2" dy="4" stdDeviation="2" floodColor="#000" floodOpacity="0.3"/>
            </filter>
          </defs>

          {/* 手动布局时的辅助对齐线 */}
          {isManualMode && alignmentLines.map((line, idx) => {
            if (line.type === 'h') {
              return (
                <line
                  key={`align-h-${idx}`}
                  x1={0}
                  y1={line.pos}
                  x2={800}
                  y2={line.pos}
                  stroke="#1890ff"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.8}
                />
              )
            } else if (line.type === 'v') {
              return (
                <line
                  key={`align-v-${idx}`}
                  x1={line.pos}
                  y1={0}
                  x2={line.pos}
                  y2={600}
                  stroke="#1890ff"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.8}
                />
              )
            } else if (line.type === 'circle' && line.center) {
              return (
                <circle
                  key={`align-circle-${idx}`}
                  cx={line.center.x}
                  cy={line.center.y}
                  r={line.pos}
                  fill="none"
                  stroke="#52c41a"
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  opacity={0.6}
                />
              )
            }
            return null
          })}

          {/* 多层级模式：使用独立组件渲染 */}
          {multiLevelOptions?.enabled && (
              <MultiLevelView
                displayNodes={displayNodes}
                edges={edges}
                manualConnections={manualConnections}
                zoom={zoom}
                selectedNodeId={selectedNodeId}
                selectedLinkId={selectedLinkId}
                hoveredLayerIndex={hoveredLayerIndex}
                setHoveredLayerIndex={setHoveredLayerIndex}
                expandingContainer={expandingContainer}
                collapsingContainer={collapsingContainer}
                collapseAnimationStarted={collapseAnimationStarted}
                onExpandAnimationEnd={handleExpandAnimationEnd}
                connectionMode={connectionMode}
                onNodeClick={onNodeClick}
                onLinkClick={onLinkClick}
                onNodeDoubleClick={handleNodeDoubleClickWithAnimation}
                layoutType={layoutType}
                renderNode={renderNode}
                getCurrentHierarchyLevel={getCurrentHierarchyLevel}
                handleDragMove={handleDragMove}
                handleDragEnd={handleDragEnd}
                selectedNodes={selectedNodes}
                targetNodes={targetNodes}
                onSelectedNodesChange={onSelectedNodesChange}
                onTargetNodesChange={onTargetNodesChange}
              />
          )}

          {/* Torus/FullMesh2D 弧线连接 */}
          {(directTopology === 'torus_2d' || directTopology === 'torus_3d' || directTopology === 'full_mesh_2d') && (
            <TorusArcs
              nodes={displayNodes.filter(n => !n.isSwitch)}
              directTopology={directTopology}
              opacity={0.6}
              getNodePosition={isManualMode ? (node) => manualPositions[node.id] || { x: node.x, y: node.y } : undefined}
              selectedLinkId={selectedLinkId}
              onLinkClick={onLinkClick}
              connectionMode={connectionMode}
              isManualMode={isManualMode}
            />
          )}

          {/* 单层级模式：使用独立组件渲染 */}
          {!multiLevelOptions?.enabled && (
            <SingleLevelView
              displayNodes={displayNodes}
              nodes={nodes}
              edges={edges}
              manualConnections={manualConnections}
              nodePositions={nodePositions}
              zoom={zoom}
              selectedNodeId={selectedNodeId}
              selectedLinkId={selectedLinkId}
              hoveredNodeId={hoveredNodeId}
              setHoveredNodeId={setHoveredNodeId}
              connectionMode={connectionMode}
              isManualMode={isManualMode}
              isForceMode={isForceMode}
              directTopology={directTopology}
              switchPanelWidth={switchPanelWidth}
              onNodeClick={onNodeClick}
              onNodeDoubleClick={handleNodeDoubleClickWithAnimation}
              onLinkClick={onLinkClick}
              setTooltip={setTooltip}
              svgRef={svgRef}
              draggingNode={draggingNode}
              handleDragStart={handleDragStart}
              layoutType={layoutType}
              nodeScale={nodeScale}
              renderNode={renderNode}
              renderNodeShape={renderNodeShape}
              getCurrentHierarchyLevel={getCurrentHierarchyLevel}
              getTrafficHeatmapStyle={getTrafficHeatmapStyle}
              selectedNodes={selectedNodes}
              targetNodes={targetNodes}
              onSelectedNodesChange={onSelectedNodesChange}
              onTargetNodesChange={onTargetNodesChange}
            />
          )}



        </svg>

        {/* 图例 */}
        <div style={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          background: '#fff',
          padding: '8px 14px',
          borderRadius: 8,
          border: '1px solid rgba(0, 0, 0, 0.08)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.04)',
          fontSize: 12,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          <Text type="secondary">
            节点: {nodes.length} | 连接: {edges.length}
          </Text>
        </div>

        {/* 悬停提示 */}
        {tooltip && (
          <div style={{
            position: 'absolute',
            left: tooltip.x,
            top: tooltip.y,
            transform: 'translateX(-50%)',
            background: '#171717',
            color: '#fff',
            padding: '6px 10px',
            borderRadius: 6,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 1000,
          }}>
            {tooltip.content}
          </div>
        )}
      </div>
  )

  // 嵌入模式：直接渲染内容
  if (embedded) {
    return (
      <div style={{ width: '100%', height: '100%' }}>
        {graphContent}
      </div>
    )
  }

  // 弹窗模式
  return (
    <Modal
      title={toolbar}
      open={visible}
      onCancel={onClose}
      footer={null}
      width={900}
      styles={{ body: { padding: 0 } }}
    >
      {graphContent}
    </Modal>
  )
}
