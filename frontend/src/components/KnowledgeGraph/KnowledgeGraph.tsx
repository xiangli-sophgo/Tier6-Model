/**
 * 知识网络可视化组件
 * 使用D3力导向布局展示名词及其关系
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Input, Tag, Button, Spin, Typography } from 'antd'
import { SearchOutlined, ReloadOutlined, ApartmentOutlined } from '@ant-design/icons'
import * as d3Force from 'd3-force'
import {
  KnowledgeRelation,
  KnowledgeGraphData,
  ForceKnowledgeNode,
  KnowledgeCategory,
  CATEGORY_COLORS,
  CATEGORY_NAMES,
  RELATION_STYLES,
  CORE_RELATION_TYPES,
  MAX_EDGES_PER_NODE,
} from './types'
import { useWorkbench } from '../../contexts/WorkbenchContext'
import knowledgeData from '../../data/knowledge-graph'

const { Text } = Typography

// 力导向布局参数 - 简化稳定版
const FORCE_CONFIG = {
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

// 8个类别的顺序（用于初始角度分布）
const CATEGORY_ORDER: KnowledgeCategory[] = [
  'hardware', 'interconnect', 'parallel', 'inference',
  'model', 'communication', 'protocol', 'system'
]

// 节点半径范围
const NODE_RADIUS_MIN = 20
const NODE_RADIUS_MAX = 40

export const KnowledgeGraph: React.FC = () => {
  const { ui } = useWorkbench()
  const {
    knowledgeHighlightedNodeId: highlightedNodeId,
    knowledgeVisibleCategories: visibleCategories,
    knowledgeNodes,
    knowledgeInitialized,
    knowledgeViewBox: savedViewBox,
    addKnowledgeSelectedNode,
    clearKnowledgeHighlight,
    setKnowledgeVisibleCategories: setVisibleCategories,
    setKnowledgeNodes,
    setKnowledgeViewBox: saveViewBox,
    resetKnowledgeCategories,
  } = ui

  // 获取高亮节点用于高亮相邻节点和边
  const highlightedNode = useMemo(() =>
    knowledgeNodes.find(n => n.id === highlightedNodeId) || null
  , [knowledgeNodes, highlightedNodeId])

  // 本地状态
  const [relations, setRelations] = useState<KnowledgeRelation[]>([])
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  // Refs
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // 视口状态 - 优先使用保存的viewBox，避免切换页面时视口跳变
  const [viewBox, setViewBoxLocal] = useState(() =>
    savedViewBox || { x: 0, y: 0, width: 1200, height: 800 }
  )
  // 更新viewBox时同时保存到context
  const setViewBox = useCallback((newViewBox: typeof viewBox | ((prev: typeof viewBox) => typeof viewBox)) => {
    setViewBoxLocal(prev => {
      const next = typeof newViewBox === 'function' ? newViewBox(prev) : newViewBox
      saveViewBox(next)
      return next
    })
  }, [saveViewBox])
  const [isPanning, setIsPanning] = useState(false)
  const panStartRef = useRef({ x: 0, y: 0, viewX: 0, viewY: 0 })

  // 加载关系数据（节点数据已在 WorkbenchContext 中预初始化）
  useEffect(() => {
    const data = knowledgeData as KnowledgeGraphData
    // relations 是静态数据，每次组件挂载都需要加载
    setRelations(data.relations)
    // 如果已预初始化，直接完成加载；否则等待 context 初始化完成
    if (knowledgeInitialized) {
      setLoading(false)
    }
  }, [knowledgeInitialized])

  // 记录上一次的分类，用于检测分类切换
  const prevCategoriesRef = useRef<Set<KnowledgeCategory> | null>(null)

  // 静态渲染模式：已有稳定位置时不创建 simulation
  useEffect(() => {
    if (knowledgeNodes.length === 0 || relations.length === 0) return

    // 检查节点是否已有稳定位置（从 WorkbenchContext 预计算）
    const hasStablePositions = knowledgeNodes.every(n =>
      n.x !== undefined && n.y !== undefined
    )

    // 已有稳定位置，直接使用静态渲染，不创建 simulation
    if (hasStablePositions) {
      setLoading(false)
      prevCategoriesRef.current = new Set(visibleCategories)
      return
    }

    // 没有稳定位置的情况（理论上不应发生，因为 WorkbenchContext 会预计算）
    setLoading(false)
    prevCategoriesRef.current = new Set(visibleCategories)
  }, [knowledgeNodes, relations, visibleCategories])

  // 计算每个节点的连接数（度数）
  const nodeDegrees = useMemo(() => {
    const degrees = new Map<string, number>()
    knowledgeNodes.forEach(n => degrees.set(n.id, 0))
    relations.forEach(r => {
      degrees.set(r.source, (degrees.get(r.source) || 0) + 1)
      degrees.set(r.target, (degrees.get(r.target) || 0) + 1)
    })
    return degrees
  }, [knowledgeNodes, relations])

  // 根据度数计算节点半径
  const getNodeRadius = useCallback((nodeId: string): number => {
    const degree = nodeDegrees.get(nodeId) || 0
    const maxDegree = Math.max(...nodeDegrees.values(), 1)
    // 使用平方根缩放，让差异不会太大
    const ratio = Math.sqrt(degree / maxDegree)
    return NODE_RADIUS_MIN + ratio * (NODE_RADIUS_MAX - NODE_RADIUS_MIN)
  }, [nodeDegrees])

  // 获取相邻节点
  const getAdjacentNodeIds = useCallback((nodeId: string): Set<string> => {
    const adjacent = new Set<string>()
    relations.forEach(r => {
      if (r.source === nodeId) adjacent.add(r.target)
      if (r.target === nodeId) adjacent.add(r.source)
    })
    return adjacent
  }, [relations])

  // 搜索匹配
  const matchedNodeIds = useMemo(() => {
    if (!searchQuery.trim()) return null
    const query = searchQuery.toLowerCase()
    const matched = new Set<string>()
    knowledgeNodes.forEach(node => {
      if (
        node.name.toLowerCase().includes(query) ||
        node.fullName?.toLowerCase().includes(query) ||
        node.definition.toLowerCase().includes(query) ||
        node.aliases?.some(a => a.toLowerCase().includes(query))
      ) {
        matched.add(node.id)
      }
    })
    return matched
  }, [searchQuery, knowledgeNodes])

  // 节点点击 - 添加到选中列表
  const handleNodeClick = useCallback((node: ForceKnowledgeNode) => {
    addKnowledgeSelectedNode(node)
  }, [addKnowledgeSelectedNode])

  // 画布拖拽
  const handlePanStart = useCallback((e: React.MouseEvent) => {
    setIsPanning(true)
    panStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      viewX: viewBox.x,
      viewY: viewBox.y,
    }
  }, [viewBox])

  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return
    const dx = (e.clientX - panStartRef.current.x) * (viewBox.width / (containerRef.current?.clientWidth || 1))
    const dy = (e.clientY - panStartRef.current.y) * (viewBox.height / (containerRef.current?.clientHeight || 1))
    setViewBox(prev => ({
      ...prev,
      x: panStartRef.current.viewX - dx,
      y: panStartRef.current.viewY - dy,
    }))
  }, [isPanning, viewBox.width, viewBox.height])

  const handlePanEnd = useCallback(() => {
    setIsPanning(false)
  }, [])

  // 滚轮缩放
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const scaleFactor = e.deltaY > 0 ? 1.1 : 0.9
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return

    const mouseX = (e.clientX - rect.left) / rect.width
    const mouseY = (e.clientY - rect.top) / rect.height

    setViewBox(prev => {
      const newWidth = prev.width * scaleFactor
      const newHeight = prev.height * scaleFactor
      const newX = prev.x + (prev.width - newWidth) * mouseX
      const newY = prev.y + (prev.height - newHeight) * mouseY
      return { x: newX, y: newY, width: newWidth, height: newHeight }
    })
  }, [])

  // 重新布局 - 使用 warmup 模式：静默运行直到稳定后再渲染
  const handleRelayout = useCallback(() => {
    setLoading(true)

    const centerX = 600
    const centerY = 400
    const maxDegree = Math.max(...nodeDegrees.values(), 1)
    const totalCategories = CATEGORY_ORDER.length

    // 计算初始位置：高度数靠中心，低度数在外围
    const relayoutNodes: ForceKnowledgeNode[] = knowledgeNodes.map(node => {
      const categoryIndex = CATEGORY_ORDER.indexOf(node.category)
      const degree = nodeDegrees.get(node.id) || 0

      const degreeRatio = degree / maxDegree
      const distanceRatio = Math.pow(1 - degreeRatio, 1.5)
      const radius = FORCE_CONFIG.radialMinRadius +
                     distanceRatio * (FORCE_CONFIG.radialMaxRadius - FORCE_CONFIG.radialMinRadius)

      const categoryAngle = (categoryIndex / totalCategories) * 2 * Math.PI - Math.PI / 2
      const angleSpread = Math.PI / totalCategories * 0.8
      const randomOffset = (Math.random() - 0.5) * angleSpread
      const angle = categoryAngle + randomOffset
      const jitter = Math.random() * 30

      return {
        ...node,
        x: centerX + Math.cos(angle) * (radius + jitter),
        y: centerY + Math.sin(angle) * (radius + jitter),
        vx: 0,
        vy: 0,
      }
    })

    // 筛选可见关系用于力导向
    const visibleNodeIds = new Set(relayoutNodes.map(n => n.id))
    const coreRelations = relations.filter(
      r => visibleNodeIds.has(r.source) && visibleNodeIds.has(r.target) && CORE_RELATION_TYPES.has(r.type)
    )
    const nodeEdgeCount = new Map<string, number>()
    const visibleRelations = coreRelations.filter(r => {
      const sourceCount = nodeEdgeCount.get(r.source) || 0
      const targetCount = nodeEdgeCount.get(r.target) || 0
      if (sourceCount >= MAX_EDGES_PER_NODE || targetCount >= MAX_EDGES_PER_NODE) return false
      nodeEdgeCount.set(r.source, sourceCount + 1)
      nodeEdgeCount.set(r.target, targetCount + 1)
      return true
    })

    // 创建 simulation 并配置力
    const simulation = d3Force.forceSimulation<ForceKnowledgeNode>(relayoutNodes)
      .force('charge', d3Force.forceManyBody<ForceKnowledgeNode>()
        .strength(FORCE_CONFIG.chargeStrength)
        .distanceMax(FORCE_CONFIG.chargeDistanceMax)
      )
      .force('link', d3Force.forceLink<ForceKnowledgeNode, d3Force.SimulationLinkDatum<ForceKnowledgeNode>>(
        visibleRelations.map(r => ({ source: r.source, target: r.target }))
      )
        .id(d => d.id)
        .distance(FORCE_CONFIG.linkDistance)
        .strength(FORCE_CONFIG.linkStrength)
      )
      .force('radial', d3Force.forceRadial<ForceKnowledgeNode>(
        (d) => {
          const degree = nodeDegrees.get(d.id) || 0
          const radiusFactor = Math.pow(1 - degree / maxDegree, 1.2)
          return FORCE_CONFIG.radialMinRadius + radiusFactor * (FORCE_CONFIG.radialMaxRadius - FORCE_CONFIG.radialMinRadius)
        },
        centerX, centerY
      ).strength(FORCE_CONFIG.radialStrength))
      .force('centerX', d3Force.forceX(centerX).strength(FORCE_CONFIG.centerStrength))
      .force('centerY', d3Force.forceY(centerY).strength(FORCE_CONFIG.centerStrength))
      .force('collision', d3Force.forceCollide<ForceKnowledgeNode>()
        .radius(FORCE_CONFIG.collisionRadius)
        .strength(FORCE_CONFIG.collisionStrength)
        .iterations(FORCE_CONFIG.collisionIterations)
      )
      .stop()

    // Warmup：静默运行直到稳定
    for (let i = 0; i < FORCE_CONFIG.warmupTicks; i++) {
      simulation.tick()
    }

    // 清零速度确保静止
    relayoutNodes.forEach(node => {
      node.vx = 0
      node.vy = 0
    })

    // 更新节点位置
    setKnowledgeNodes([...relayoutNodes])
    setLoading(false)
  }, [knowledgeNodes, nodeDegrees, relations, setKnowledgeNodes])

  // 分类过滤切换 (普通点击切换，Ctrl+点击只显示该分类)
  const handleCategoryClick = useCallback((category: KnowledgeCategory, ctrlKey: boolean) => {
    if (ctrlKey) {
      // Ctrl+点击: 只显示该分类
      setVisibleCategories(new Set([category]))
    } else {
      // 普通点击: 切换该分类
      const next = new Set(visibleCategories)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      setVisibleCategories(next)
    }
  }, [visibleCategories, setVisibleCategories])

  // 计算可见的边（应用核心关系筛选和连接数限制，确保每个节点至少有一条连接）
  const visibleEdges = useMemo(() => {
    const visibleNodeIds = new Set(
      knowledgeNodes.filter(n => visibleCategories.has(n.category)).map(n => n.id)
    )
    // 所有可见关系（用于确保每个节点至少有一条连接）
    const allVisibleRelations = relations.filter(
      r => visibleNodeIds.has(r.source) && visibleNodeIds.has(r.target)
    )
    // 核心关系优先
    const coreRelations = allVisibleRelations.filter(r => CORE_RELATION_TYPES.has(r.type))

    const nodeEdgeCount = new Map<string, number>()
    const selectedEdges: typeof relations = []

    // 第一轮：添加核心关系（受连接数限制）
    for (const r of coreRelations) {
      const sourceCount = nodeEdgeCount.get(r.source) || 0
      const targetCount = nodeEdgeCount.get(r.target) || 0
      if (sourceCount < MAX_EDGES_PER_NODE && targetCount < MAX_EDGES_PER_NODE) {
        selectedEdges.push(r)
        nodeEdgeCount.set(r.source, sourceCount + 1)
        nodeEdgeCount.set(r.target, targetCount + 1)
      }
    }

    // 第二轮：确保每个节点至少有一条连接
    const connectedNodes = new Set<string>()
    selectedEdges.forEach(r => {
      connectedNodes.add(r.source)
      connectedNodes.add(r.target)
    })
    for (const nodeId of visibleNodeIds) {
      if (!connectedNodes.has(nodeId)) {
        // 找一条连接这个节点的关系
        const edge = allVisibleRelations.find(r => r.source === nodeId || r.target === nodeId)
        if (edge && !selectedEdges.includes(edge)) {
          selectedEdges.push(edge)
          connectedNodes.add(edge.source)
          connectedNodes.add(edge.target)
        }
      }
    }

    return selectedEdges
  }, [knowledgeNodes, relations, visibleCategories])

  // 渲染边
  const renderEdges = () => {
    const nodeMap = new Map(knowledgeNodes.map(n => [n.id, n]))
    return visibleEdges.map((rel, i) => {
      const source = nodeMap.get(rel.source)
      const target = nodeMap.get(rel.target)
      if (!source || !target) return null

      const style = RELATION_STYLES[rel.type] || RELATION_STYLES.related_to
      // 使用高亮节点判断高亮
      const isHighlighted = highlightedNode && (rel.source === highlightedNode.id || rel.target === highlightedNode.id)
      const isFiltered = matchedNodeIds && (!matchedNodeIds.has(rel.source) || !matchedNodeIds.has(rel.target))
      // 当有高亮节点时，非高亮的边变灰
      const isDimmed = highlightedNode && !isHighlighted

      return (
        <line
          key={`edge-${i}`}
          x1={source.x}
          y1={source.y}
          x2={target.x}
          y2={target.y}
          stroke={isDimmed ? '#ddd' : style.stroke}
          strokeWidth={isHighlighted ? 2.5 : 1.5}
          opacity={isFiltered ? 0.15 : isDimmed ? 0.2 : isHighlighted ? 1 : 0.6}
        />
      )
    })
  }

  // 渲染节点
  const renderNodes = () => {
    return knowledgeNodes.map(node => {
      if (!visibleCategories.has(node.category)) return null

      const radius = getNodeRadius(node.id)
      const color = CATEGORY_COLORS[node.category]
      // 是否是当前高亮节点
      const isHighlighted = node.id === highlightedNodeId
      const isHovered = hoveredNode === node.id
      // 使用高亮节点判断相邻
      const isAdjacent = highlightedNode ? getAdjacentNodeIds(highlightedNode.id).has(node.id) : false
      const isMatched = matchedNodeIds ? matchedNodeIds.has(node.id) : true
      const isFiltered = matchedNodeIds && !isMatched
      // 当有高亮节点时，非高亮且非相邻的节点变灰
      const isDimmed = highlightedNode && !isHighlighted && !isAdjacent
      // 根据节点大小调整字体
      const fontSize = Math.max(10, Math.min(14, radius * 0.5))

      return (
        <g
          key={node.id}
          transform={`translate(${node.x}, ${node.y})`}
          style={{ cursor: 'pointer' }}
          onClick={() => handleNodeClick(node)}
          onMouseEnter={() => setHoveredNode(node.id)}
          onMouseLeave={() => setHoveredNode(null)}
        >
          {/* 高亮外圈 */}
          {(isHighlighted || isAdjacent) && (
            <circle
              r={radius + 6}
              fill="none"
              stroke={color}
              strokeWidth={3}
              strokeOpacity={isHighlighted ? 0.6 : 0.3}
            />
          )}

          {/* 主圆形 */}
          <circle
            r={radius}
            fill={isDimmed ? '#ccc' : isFiltered ? `${color}40` : color}
            stroke={isDimmed ? '#999' : '#fff'}
            strokeWidth={2}
            style={{
              filter: isHovered && !isDimmed ? `drop-shadow(0 0 8px ${color})` : 'none',
              opacity: isDimmed ? 0.3 : isFiltered ? 0.4 : 1,
              transition: 'filter 0.2s, opacity 0.2s',
            }}
          />

          {/* 名称标签 */}
          <text
            y={4}
            textAnchor="middle"
            fill={isDimmed ? '#999' : '#fff'}
            fontSize={fontSize}
            fontWeight={600}
            style={{ pointerEvents: 'none', userSelect: 'none', opacity: isDimmed ? 0.5 : 1 }}
          >
            {node.name.length > 5 ? node.name.slice(0, 4) + '..' : node.name}
          </text>

          {/* Hover Tooltip */}
          {isHovered && (
            <g transform={`translate(0, ${-radius - 20})`}>
              <rect
                x={-60}
                y={-12}
                width={120}
                height={24}
                rx={4}
                fill="rgba(0,0,0,0.85)"
              />
              <text
                textAnchor="middle"
                fill="#fff"
                fontSize={12}
                y={4}
              >
                {node.fullName || node.name}
              </text>
            </g>
          )}
        </g>
      )
    })
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fafafa' }}>
      {/* 工具栏 */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #e5e5e5', background: '#fff', display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        {/* 搜索框 */}
        <Input
          placeholder="搜索名词..."
          prefix={<SearchOutlined style={{ color: '#666' }} />}
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          allowClear
          style={{ width: 200 }}
        />

        {/* 分类过滤 (Ctrl+点击只显示该分类) */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 }}>
          {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
            const category = key as KnowledgeCategory
            const isActive = visibleCategories.has(category)
            const count = knowledgeNodes.filter(n => n.category === category).length
            if (count === 0) return null
            return (
              <Tag
                key={category}
                color={isActive ? CATEGORY_COLORS[category] : undefined}
                style={{
                  cursor: 'pointer',
                  opacity: isActive ? 1 : 0.5,
                  borderColor: CATEGORY_COLORS[category],
                }}
                onClick={(e) => handleCategoryClick(category, e.ctrlKey || e.metaKey)}
              >
                {name} ({count})
              </Tag>
            )
          })}
        </div>

        {/* 重新布局按钮 */}
        <Button
          size="small"
          icon={<ApartmentOutlined />}
          onClick={handleRelayout}
        >
          重新布局
        </Button>

        {/* 全部显示按钮 */}
        {visibleCategories.size < 8 && (
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={resetKnowledgeCategories}
          >
            全部显示
          </Button>
        )}

        {/* 统计信息 */}
        <Text type="secondary" style={{ fontSize: 12 }}>
          {knowledgeNodes.filter(n => visibleCategories.has(n.category)).length} 个节点 · {visibleEdges.length} 条连接
        </Text>
      </div>

      {/* 画布 */}
      <div
        ref={containerRef}
        style={{ flex: 1, overflow: 'hidden', position: 'relative' }}
        onMouseDown={handlePanStart}
        onMouseMove={handlePanMove}
        onMouseUp={handlePanEnd}
        onMouseLeave={handlePanEnd}
        onWheel={handleWheel}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
          style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
        >
          {/* 背景 - 点击清除高亮 */}
          <rect
            x={viewBox.x - 1000}
            y={viewBox.y - 1000}
            width={viewBox.width + 2000}
            height={viewBox.height + 2000}
            fill="#fafafa"
            onClick={() => clearKnowledgeHighlight()}
          />

          {/* 边 */}
          <g>{renderEdges()}</g>

          {/* 节点 */}
          <g>{renderNodes()}</g>
        </svg>
      </div>
    </div>
  )
}
