/**
 * 知识网络可视化组件
 * 使用 react-force-graph 实现高性能力导向布局
 */
import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import { Input, Tag, Button, Typography, Switch } from 'antd'
import { SearchOutlined, ReloadOutlined, ApartmentOutlined, DragOutlined } from '@ant-design/icons'
import ForceGraph2D, { ForceGraphMethods, NodeObject, LinkObject } from 'react-force-graph-2d'
import * as d3Force from 'd3-force'
import {
  KnowledgeGraphData,
  KnowledgeCategory,
  CATEGORY_COLORS,
  CATEGORY_NAMES,
} from './types'
import { useWorkbench } from '../../contexts/WorkbenchContext'
import knowledgeData from '../../data/knowledge-graph'

const { Text } = Typography

// 节点半径范围
const NODE_RADIUS_MIN = 4
const NODE_RADIUS_MAX = 12

// react-force-graph 数据格式
interface GraphNode extends NodeObject {
  id: string
  name: string
  fullName?: string
  definition: string
  category: KnowledgeCategory
  source?: string
  aliases?: string[]
  degree?: number
  // react-force-graph 自动添加的属性
  x?: number
  y?: number
  vx?: number
  vy?: number
}

interface GraphLink extends LinkObject {
  source: string | GraphNode
  target: string | GraphNode
  type: string
  description?: string
}

interface KnowledgeGraphProps {
  renderMode?: 'toolbar-only' | 'canvas-only'  // 渲染模式：只渲染工具栏或只渲染画布
}

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ renderMode }) => {
  const { knowledge, ui } = useWorkbench()
  const {
    knowledgeHighlightedNodeId: highlightedNodeId,
    knowledgeVisibleCategories: visibleCategories,
    knowledgeNodes: cachedNodes,
    knowledgeViewBox: _cachedViewBox,
    knowledgeEnableDrag: enableDrag,
    knowledgeGraphActions,
    addKnowledgeSelectedNode,
    clearKnowledgeHighlight,
    setKnowledgeVisibleCategories: setVisibleCategories,
    setKnowledgeEnableDrag: setEnableDrag,
    setKnowledgeGraphActions,
    resetKnowledgeCategories,
  } = knowledge

  // 从原始数据获取关系
  const allRelations = useMemo(() => {
    const data = knowledgeData as KnowledgeGraphData
    return data.relations
  }, [])

  // 本地状态
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [searchResultCount, setSearchResultCount] = useState(0)

  // Refs
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink>>()
  const containerRef = useRef<HTMLDivElement>(null)

  // 获取节点列表 - 优先使用预初始化的缓存，否则从原始数据加载
  const allNodes = useMemo(() => {
    if (cachedNodes.length > 0) {
      return cachedNodes
    }
    const data = knowledgeData as KnowledgeGraphData
    return data.nodes
  }, [cachedNodes])

  // 计算每个节点的连接数（度数）
  const nodeDegrees = useMemo(() => {
    const degrees = new Map<string, number>()
    allNodes.forEach(n => degrees.set(n.id, 0))
    allRelations.forEach(r => {
      degrees.set(r.source, (degrees.get(r.source) || 0) + 1)
      degrees.set(r.target, (degrees.get(r.target) || 0) + 1)
    })
    return degrees
  }, [allNodes, allRelations])

  // 根据度数计算节点半径
  const getNodeRadius = useCallback((nodeId: string): number => {
    const degree = nodeDegrees.get(nodeId) || 0
    const maxDegree = Math.max(...nodeDegrees.values(), 1)
    // 使用平方根缩放，让差异不会太大
    const ratio = Math.sqrt(degree / maxDegree)
    return NODE_RADIUS_MIN + ratio * (NODE_RADIUS_MAX - NODE_RADIUS_MIN)
  }, [nodeDegrees])

  // 搜索匹配
  const matchedNodeIds = useMemo(() => {
    if (!searchQuery.trim()) {
      setSearchResultCount(0)
      return null
    }

    // 数据完整性检查
    if (!allNodes || allNodes.length === 0) {
      console.warn('⚠️ 搜索时 allNodes 为空', {
        allNodes: allNodes?.length || 0,
        cachedNodes: cachedNodes.length,
      })
      setSearchResultCount(0)
      return null
    }

    const query = searchQuery.toLowerCase()
    const matched = new Set<string>()

    allNodes.forEach(node => {
      if (
        node.name.toLowerCase().includes(query) ||
        node.fullName?.toLowerCase().includes(query) ||
        node.definition.toLowerCase().includes(query) ||
        node.aliases?.some(a => a.toLowerCase().includes(query))
      ) {
        matched.add(node.id)
      }
    })

    setSearchResultCount(matched.size)

    // 调试日志
    if (matched.size === 0) {
      console.log(`🔍 搜索 "${searchQuery}" - 未找到匹配节点（共${allNodes.length}个节点）`)
    } else {
      // 检查匹配节点是否可见
      const matchedNodes = Array.from(matched)
        .map(id => allNodes.find(n => n.id === id))
        .filter(Boolean) as GraphNode[]

      const visibleMatches = matchedNodes.filter(n => visibleCategories.has(n.category))
      const hiddenMatches = matchedNodes.filter(n => !visibleCategories.has(n.category))

      console.log(`🔍 搜索 "${searchQuery}" - 找到${matched.size}个匹配`)
      console.log(`  ├─ 🟢 可见: ${visibleMatches.length}个`)
      console.log(`  │   ${visibleMatches.map(n => `${n.name}(${n.category})`).join(', ') || '(无)'}`)
      if (hiddenMatches.length > 0) {
        console.log(`  └─ 🔴 被隐藏: ${hiddenMatches.length}个（需要启用分类）`)
        console.log(`    ${hiddenMatches.map(n => `${n.name}(${n.category})`).join(', ')}`)
      }
    }

    return matched
  }, [searchQuery, allNodes, cachedNodes.length, visibleCategories])

  // 过滤可见节点 - 保持原始节点引用，避免不必要的对象创建
  const visibleNodes = useMemo(() => {
    return allNodes.filter(n => visibleCategories.has(n.category))
  }, [allNodes, visibleCategories])

  // 过滤可见边（显示所有可见节点之间的关系）
  const visibleLinks = useMemo(() => {
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id))

    // 显示所有可见节点之间的关系
    return allRelations
      .filter(r => visibleNodeIds.has(r.source) && visibleNodeIds.has(r.target))
      .map(r => ({
        source: r.source,
        target: r.target,
        type: r.type,
        description: r.description,
      }))
  }, [visibleNodes, allRelations])

  // 预计算相邻节点映射 - 避免在 paintNode 中重复计算
  const adjacencyMap = useMemo(() => {
    const map = new Map<string, Set<string>>()
    visibleLinks.forEach(link => {
      const sourceId = typeof link.source === 'object' ? (link.source as GraphNode).id : String(link.source)
      const targetId = typeof link.target === 'object' ? (link.target as GraphNode).id : String(link.target)

      if (!map.has(sourceId)) map.set(sourceId, new Set())
      if (!map.has(targetId)) map.set(targetId, new Set())

      map.get(sourceId)!.add(targetId)
      map.get(targetId)!.add(sourceId)
    })
    return map
  }, [visibleLinks])

  // 构建图数据
  const graphData = useMemo(() => {
    return {
      nodes: visibleNodes,
      links: visibleLinks,
    }
  }, [visibleNodes, visibleLinks])

  // 节点点击 - 添加到选中列表
  const handleNodeClick = useCallback((node: GraphNode) => {
    // GraphNode 已经包含了 x, y 坐标（由 react-force-graph 添加）
    // 直接强制转换为 ForceKnowledgeNode（它们的结构兼容）
    addKnowledgeSelectedNode(node as any)
  }, [addKnowledgeSelectedNode])

  // 背景点击 - 清除高亮
  const handleBackgroundClick = useCallback(() => {
    clearKnowledgeHighlight()
  }, [clearKnowledgeHighlight])

  // 分类过滤切换
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

  // 重新布局辅助函数 - 重置节点位置并重新模拟
  const performRelayout = useCallback(() => {
    if (!graphRef.current) return

    // 1. 重置所有节点到随机圆形分布
    const radius = 300
    graphData.nodes.forEach(node => {
      // react-force-graph 会在运行时动态添加 x, y, vx, vy 属性
      const forceNode = node as GraphNode
      const angle = Math.random() * 2 * Math.PI
      const r = Math.sqrt(Math.random()) * radius
      forceNode.x = Math.cos(angle) * r
      forceNode.y = Math.sin(angle) * r
      forceNode.vx = 0
      forceNode.vy = 0
    })

    // 2. 重新加热模拟
    graphRef.current.d3ReheatSimulation()

    // 3. 布局稳定后自动适配视角
    setTimeout(() => {
      if (graphRef.current) {
        graphRef.current.zoomToFit(400, 20)
      }
    }, 1200)
  }, [graphData])

  // 重新布局 - 重置节点位置并重新模拟
  const handleRelayout = useCallback(() => {
    // 优先使用 Context 中的 actions（适用于 toolbar-only 模式）
    if (knowledgeGraphActions) {
      knowledgeGraphActions.relayout()
    }
    // 回退到本地 graphRef（适用于默认模式）
    else {
      performRelayout()
    }
  }, [knowledgeGraphActions, performRelayout])

  // 优化力导向布局参数 - 让布局更紧凑
  useEffect(() => {
    if (!graphRef.current) return

    const fg = graphRef.current

    // 配置力参数让布局更紧凑
    fg.d3Force('charge')?.strength(-50)   // 减小斥力，避免节点越来越分散
    fg.d3Force('link')?.distance(20)      // 连接距离
    fg.d3Force('center', d3Force.forceCenter(0, 0).strength(0.8))  // 增强中心引力，防止分散
    fg.d3Force('collision', d3Force.forceCollide(8))  // 适当的碰撞半径

    // 重新加热模拟以应用新参数
    fg.d3ReheatSimulation()
  }, [graphData])

  // 数据加载调试
  useEffect(() => {
    if (allNodes.length === 0) {
      console.warn('⚠️ KnowledgeGraph: allNodes 为空', {
        cachedNodesLength: cachedNodes.length,
        allNodesLength: allNodes.length,
        hasKnowledgeData: !!knowledgeData,
      })
    } 
  }, [allNodes.length, cachedNodes.length])

  // 搜索时自动启用匹配分类
  useEffect(() => {
    if (!searchQuery.trim() || !matchedNodeIds || matchedNodeIds.size === 0) {
      return
    }

    // 获取匹配节点的分类
    const matchedCategories = new Set<KnowledgeCategory>()
    matchedNodeIds.forEach(id => {
      const node = allNodes.find(n => n.id === id)
      if (node) {
        matchedCategories.add(node.category)
      }
    })

    // 如果有分类被隐藏，自动启用它们
    const categoriesToEnable = Array.from(matchedCategories).filter(
      cat => !visibleCategories.has(cat)
    )

    if (categoriesToEnable.length > 0) {
      const newVisible = new Set(visibleCategories)
      categoriesToEnable.forEach(cat => newVisible.add(cat))
      setVisibleCategories(newVisible)
      console.log(`🎯 搜索自动启用分类: ${categoriesToEnable.join(', ')}`)
    }
  }, [searchQuery, matchedNodeIds, allNodes, visibleCategories, setVisibleCategories])

  // 监听容器尺寸变化
  useEffect(() => {
    if (!containerRef.current) return

    const updateDimensions = () => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current
        setDimensions({ width: clientWidth, height: clientHeight })
      }
    }

    // 初始化尺寸
    updateDimensions()

    // 使用 ResizeObserver 监听容器尺寸变化
    const resizeObserver = new ResizeObserver(updateDimensions)
    resizeObserver.observe(containerRef.current)

    return () => resizeObserver.disconnect()
  }, [])

  // 自动适配视角 - 切换到知识图谱页面时
  useEffect(() => {
    if (ui.viewMode === 'knowledge' && graphRef.current && visibleNodes.length > 0) {
      // 延迟执行，等待布局稳定（200 ticks 需要约 1 秒）
      const timer = setTimeout(() => {
        if (graphRef.current) {
          graphRef.current.zoomToFit(400, 20)  // 400ms 动画，20px padding
        }
      }, 1200)  // 1.2 秒延迟，让布局基本稳定
      return () => clearTimeout(timer)
    }
  }, [ui.viewMode, visibleNodes.length])

  // 在 canvas-only 模式下，将 graphRef 的方法注册到 Context
  useEffect(() => {
    if (renderMode === 'canvas-only' && graphRef.current) {
      setKnowledgeGraphActions({
        reheatSimulation: () => {
          if (graphRef.current) {
            graphRef.current.d3ReheatSimulation()
          }
        },
        zoomToFit: (duration = 400, padding = 20) => {
          if (graphRef.current) {
            graphRef.current.zoomToFit(duration, padding)
          }
        },
        relayout: performRelayout
      })
    }

    // 组件卸载时清除
    return () => {
      if (renderMode === 'canvas-only') {
        setKnowledgeGraphActions(null)
      }
    }
  }, [renderMode, setKnowledgeGraphActions, performRelayout])

  // 节点渲染
  const paintNode = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    const radius = getNodeRadius(node.id!)
    const color = CATEGORY_COLORS[node.category]

    // 判断节点状态 - 使用预计算的 adjacencyMap
    const isHighlighted = node.id === highlightedNodeId
    const isHovered = hoveredNode?.id === node.id
    const isAdjacent = highlightedNodeId ? (adjacencyMap.get(highlightedNodeId)?.has(node.id!) || false) : false
    const isMatched = matchedNodeIds ? matchedNodeIds.has(node.id!) : true
    const isFiltered = matchedNodeIds && !isMatched
    const isDimmed = highlightedNodeId && !isHighlighted && !isAdjacent

    // 搜索时的外圈高亮（搜索匹配的节点）- 加强效果
    if (matchedNodeIds && isMatched && !isFiltered) {
      // 外圈 - 高对比度金色
      ctx.beginPath()
      ctx.arc(node.x!, node.y!, radius + 6, 0, 2 * Math.PI)
      ctx.strokeStyle = '#FFD700'  // 金色，更显眼
      ctx.lineWidth = 3.5
      ctx.globalAlpha = 0.8
      ctx.stroke()

      // 内圈 - 节点颜色的浅色
      ctx.beginPath()
      ctx.arc(node.x!, node.y!, radius + 2, 0, 2 * Math.PI)
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.globalAlpha = 0.6
      ctx.stroke()
      ctx.globalAlpha = 1
    }

    // 绘制高亮外圈（节点被选中时）
    if ((isHighlighted || isAdjacent) && !isDimmed) {
      ctx.beginPath()
      ctx.arc(node.x!, node.y!, radius + 3, 0, 2 * Math.PI)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.globalAlpha = isHighlighted ? 0.6 : 0.3
      ctx.stroke()
      ctx.globalAlpha = 1
    }

    // 绘制主圆形
    ctx.beginPath()
    ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI)

    if (isDimmed) {
      ctx.fillStyle = '#ccc'
      ctx.globalAlpha = 0.3
    } else if (isFiltered) {
      ctx.fillStyle = color + '4D' // 30% opacity
      ctx.globalAlpha = 0.3
    } else {
      ctx.fillStyle = color
      ctx.globalAlpha = 1
    }

    ctx.fill()

    // 边框
    ctx.strokeStyle = isDimmed ? '#999' : isFiltered ? '#ccc' : '#fff'
    ctx.lineWidth = 1.5
    ctx.stroke()

    ctx.globalAlpha = 1

    // 悬停效果 - 发光
    if (isHovered && !isDimmed) {
      ctx.shadowBlur = 10
      ctx.shadowColor = color
      ctx.beginPath()
      ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.shadowBlur = 0
    }

    // 绘制文字 - 根据节点大小动态调整字体和截断
    // 字体大小随节点变小，最小6px
    const fontSize = Math.max(6, radius * 0.6)
    // 估算可容纳的字符数（中文字符约占 fontSize 宽度，英文约占 fontSize * 0.6）
    const maxWidth = radius * 1.8  // 节点直径的 90%
    const avgCharWidth = fontSize * 0.8  // 平均字符宽度
    const maxChars = Math.floor(maxWidth / avgCharWidth)

    // 至少显示3个字符，如果名字太长就截断
    let label = node.name
    if (label.length > maxChars && maxChars >= 3) {
      label = label.slice(0, Math.max(3, maxChars - 2)) + '..'
    } else if (maxChars < 3 && label.length > 3) {
      // 如果节点太小连3个字符都放不下，强制显示3个字符但用更小的字体
      label = label.slice(0, 3)
    }

    ctx.font = `600 ${fontSize}px Sans-Serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = isDimmed ? '#999' : '#fff'
    ctx.globalAlpha = isDimmed ? 0.5 : 1
    ctx.fillText(label, node.x!, node.y!)
    ctx.globalAlpha = 1
  }, [highlightedNodeId, hoveredNode, matchedNodeIds, getNodeRadius, adjacencyMap])

  // 边渲染
  const paintLink = useCallback((link: GraphLink, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    const sourceNode = typeof link.source === 'object' ? link.source : null
    const targetNode = typeof link.target === 'object' ? link.target : null

    if (!sourceNode || !targetNode) return

    const sourceId = sourceNode.id!
    const targetId = targetNode.id!

    // 判断边状态
    const isHighlighted = highlightedNodeId && (sourceId === highlightedNodeId || targetId === highlightedNodeId)
    const isFiltered = matchedNodeIds && (!matchedNodeIds.has(sourceId) || !matchedNodeIds.has(targetId))
    const isDimmed = highlightedNodeId && !isHighlighted

    // 绘制边
    ctx.beginPath()
    ctx.moveTo(sourceNode.x!, sourceNode.y!)
    ctx.lineTo(targetNode.x!, targetNode.y!)

    if (isDimmed) {
      ctx.strokeStyle = '#ddd'
      ctx.globalAlpha = 0.2
    } else if (isFiltered) {
      ctx.strokeStyle = '#ccc'
      ctx.globalAlpha = 0.15
    } else if (isHighlighted) {
      ctx.strokeStyle = '#666'
      ctx.globalAlpha = 1
      ctx.lineWidth = 2
    } else {
      ctx.strokeStyle = '#94A3B8'
      ctx.globalAlpha = 0.6
      ctx.lineWidth = 1
    }

    ctx.stroke()
    ctx.globalAlpha = 1
  }, [highlightedNodeId, matchedNodeIds])

  // 只渲染工具栏
  if (renderMode === 'toolbar-only') {
    return (
      <div style={{
        width: '100%',
        padding: '12px 16px',
        borderBottom: '1px solid #e5e5e5',
        background: '#fff',
        display: 'flex',
        gap: 16,
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        {/* 左侧：搜索框 + 分类过滤 + 全部显示 */}
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', justifyContent: 'center', flex: 1, minWidth: 0 }}>
          {/* 搜索框 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
            <Input
              placeholder="搜索名词..."
              prefix={<SearchOutlined style={{ color: '#666' }} />}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              allowClear
              style={{ width: 200 }}
            />
            {/* 搜索结果反馈 */}
            {searchQuery.trim() && (
              <span style={{
                fontSize: 12,
                color: searchResultCount > 0 ? '#52c41a' : '#ff4d4f',
                fontWeight: 500,
                whiteSpace: 'nowrap',
              }}>
                {searchResultCount > 0
                  ? `找到 ${searchResultCount}`
                  : '未找到'}
              </span>
            )}
          </div>

          {/* 分类过滤 */}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', minWidth: 0 }}>
            {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
              const category = key as KnowledgeCategory
              const isActive = visibleCategories.has(category)
              const count = allNodes.filter(n => n.category === category).length
              if (count === 0) return null
              return (
                <Tag
                  key={category}
                  color={isActive ? CATEGORY_COLORS[category] : undefined}
                  style={{
                    cursor: 'pointer',
                    opacity: isActive ? 1 : 0.5,
                    borderColor: CATEGORY_COLORS[category],
                    fontSize: 12,
                    padding: '0 8px',
                    margin: 0
                  }}
                  onClick={(e) => handleCategoryClick(category, e.ctrlKey || e.metaKey)}
                >
                  {name}
                </Tag>
              )
            })}
          </div>

          {/* 全部显示按钮 */}
          {visibleCategories.size < 8 && (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={resetKnowledgeCategories}
              style={{ flexShrink: 0 }}
            >
              全部显示
            </Button>
          )}
        </div>

        {/* 右侧：重新布局 + 拖动开关 */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexShrink: 0 }}>
          {/* 重新布局按钮 */}
          <Button
            size="small"
            icon={<ApartmentOutlined />}
            onClick={handleRelayout}
          >
            重新布局
          </Button>

          {/* 拖动开关 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <DragOutlined style={{ color: enableDrag ? '#1890ff' : '#999' }} />
            <Switch
              size="small"
              checked={enableDrag}
              onChange={setEnableDrag}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {enableDrag ? '可拖动' : '不可拖动'}
            </Text>
          </div>
        </div>
      </div>
    )
  }

  // 只渲染画布
  if (renderMode === 'canvas-only') {
    return (
      <div ref={containerRef} style={{ width: '100%', height: '100%', overflow: 'hidden', position: 'relative', background: '#fafafa' }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeId="id"
          nodeLabel={(node: GraphNode) => node.fullName || node.name}
          nodeCanvasObject={paintNode}
          nodeCanvasObjectMode={() => 'replace'}
          linkCanvasObject={paintLink}
          linkCanvasObjectMode={() => 'replace'}
          onNodeClick={handleNodeClick}
          onNodeHover={setHoveredNode}
          onBackgroundClick={handleBackgroundClick}
          backgroundColor="#fafafa"
          cooldownTicks={200}
          warmupTicks={0}
          d3AlphaDecay={0.03}
          d3VelocityDecay={0.5}
          enableNodeDrag={enableDrag}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />
      </div>
    )
  }

  // 默认：渲染完整视图（工具栏 + 画布）
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', background: '#fafafa' }}>
      {/* 工具栏 */}
      <div style={{
        width: '100%',
        padding: '12px 16px',
        borderBottom: '1px solid #e5e5e5',
        background: '#fff',
        display: 'flex',
        gap: 16,
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        {/* 左侧：搜索框 + 分类过滤 + 全部显示 */}
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', justifyContent: 'center', flex: 1, minWidth: 0 }}>
          {/* 搜索框 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
            <Input
              placeholder="搜索名词..."
              prefix={<SearchOutlined style={{ color: '#666' }} />}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              allowClear
              style={{ width: 200 }}
            />
            {/* 搜索结果反馈 */}
            {searchQuery.trim() && (
              <span style={{
                fontSize: 12,
                color: searchResultCount > 0 ? '#52c41a' : '#ff4d4f',
                fontWeight: 500,
                whiteSpace: 'nowrap',
              }}>
                {searchResultCount > 0
                  ? `找到 ${searchResultCount}`
                  : '未找到'}
              </span>
            )}
          </div>

          {/* 分类过滤 */}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', minWidth: 0 }}>
            {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
              const category = key as KnowledgeCategory
              const isActive = visibleCategories.has(category)
              const count = allNodes.filter(n => n.category === category).length
              if (count === 0) return null
              return (
                <Tag
                  key={category}
                  color={isActive ? CATEGORY_COLORS[category] : undefined}
                  style={{
                    cursor: 'pointer',
                    opacity: isActive ? 1 : 0.5,
                    borderColor: CATEGORY_COLORS[category],
                    fontSize: 12,
                    padding: '0 8px',
                    margin: 0
                  }}
                  onClick={(e) => handleCategoryClick(category, e.ctrlKey || e.metaKey)}
                >
                  {name}
                </Tag>
              )
            })}
          </div>

          {/* 全部显示按钮 */}
          {visibleCategories.size < 8 && (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={resetKnowledgeCategories}
              style={{ flexShrink: 0 }}
            >
              全部显示
            </Button>
          )}
        </div>

        {/* 右侧：重新布局 + 拖动开关 */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexShrink: 0 }}>
          {/* 重新布局按钮 */}
          <Button
            size="small"
            icon={<ApartmentOutlined />}
            onClick={handleRelayout}
          >
            重新布局
          </Button>

          {/* 拖动开关 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <DragOutlined style={{ color: enableDrag ? '#1890ff' : '#999' }} />
            <Switch
              size="small"
              checked={enableDrag}
              onChange={setEnableDrag}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {enableDrag ? '可拖动' : '不可拖动'}
            </Text>
          </div>
        </div>
      </div>

      {/* 画布 */}
      <div ref={containerRef} style={{ flex: 1, width: '100%', overflow: 'hidden', position: 'relative' }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeId="id"
          nodeLabel={(node: GraphNode) => node.fullName || node.name}
          nodeCanvasObject={paintNode}
          nodeCanvasObjectMode={() => 'replace'}
          linkCanvasObject={paintLink}
          linkCanvasObjectMode={() => 'replace'}
          onNodeClick={handleNodeClick}
          onNodeHover={setHoveredNode}
          onBackgroundClick={handleBackgroundClick}
          backgroundColor="#fafafa"
          // ⚡ 让 ForceGraph2D 进行充分的力导向布局
          // 因为组件始终挂载（display: none），布局会在后台自动进行
          // 增加 cooldownTicks 让布局有更多时间稳定
          cooldownTicks={200}
          warmupTicks={0}
          d3AlphaDecay={0.03}
          d3VelocityDecay={0.5}
          enableNodeDrag={enableDrag}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />
      </div>
    </div>
  )
}
