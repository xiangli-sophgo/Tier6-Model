/**
 * 知识网络可视化组件
 * 使用 react-force-graph 实现高性能力导向布局
 */
import React, { useState, useRef, useCallback, useMemo } from 'react'
import { Input, Tag, Button, Typography, Switch } from 'antd'
import { SearchOutlined, ReloadOutlined, ApartmentOutlined, DragOutlined } from '@ant-design/icons'
import ForceGraph2D, { ForceGraphMethods, NodeObject, LinkObject } from 'react-force-graph-2d'
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
}

interface GraphLink extends LinkObject {
  source: string | GraphNode
  target: string | GraphNode
  type: string
  description?: string
}

export const KnowledgeGraph: React.FC = () => {
  const { ui } = useWorkbench()
  const {
    knowledgeHighlightedNodeId: highlightedNodeId,
    knowledgeVisibleCategories: visibleCategories,
    knowledgeNodes: cachedNodes,
    knowledgeViewBox: _cachedViewBox,
    addKnowledgeSelectedNode,
    clearKnowledgeHighlight,
    setKnowledgeVisibleCategories: setVisibleCategories,
    resetKnowledgeCategories,
  } = ui

  // 从原始数据获取关系
  const allRelations = useMemo(() => {
    const data = knowledgeData as KnowledgeGraphData
    return data.relations
  }, [])

  // 本地状态
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [enableDrag, setEnableDrag] = useState(false)

  // Refs
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink>>()

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
    if (!searchQuery.trim()) return null
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
    return matched
  }, [searchQuery, allNodes])

  // 过滤可见节点
  const visibleNodes = useMemo(() => {
    return allNodes
      .filter(n => visibleCategories.has(n.category))
      .map(n => ({
        ...n,
        degree: nodeDegrees.get(n.id) || 0,
      }))
  }, [allNodes, visibleCategories, nodeDegrees])

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

  // 构建图数据
  const graphData = useMemo(() => ({
    nodes: visibleNodes,
    links: visibleLinks,
  }), [visibleNodes, visibleLinks])

  // 获取相邻节点
  const getAdjacentNodeIds = useCallback((nodeId: string): Set<string> => {
    const adjacent = new Set<string>()
    visibleLinks.forEach(link => {
      const sourceId = typeof link.source === 'object' ? (link.source as GraphNode).id : String(link.source)
      const targetId = typeof link.target === 'object' ? (link.target as GraphNode).id : String(link.target)
      if (sourceId === nodeId) adjacent.add(targetId)
      if (targetId === nodeId) adjacent.add(sourceId)
    })
    return adjacent
  }, [visibleLinks])

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

  // 重新布局
  const handleRelayout = useCallback(() => {
    if (graphRef.current) {
      // 重新加热模拟
      graphRef.current.d3ReheatSimulation()
    }
  }, [])

  // 节点渲染
  const paintNode = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    const radius = getNodeRadius(node.id!)
    const color = CATEGORY_COLORS[node.category]

    // 判断节点状态
    const isHighlighted = node.id === highlightedNodeId
    const isHovered = hoveredNode?.id === node.id
    const isAdjacent = highlightedNodeId ? getAdjacentNodeIds(highlightedNodeId).has(node.id!) : false
    const isMatched = matchedNodeIds ? matchedNodeIds.has(node.id!) : true
    const isFiltered = matchedNodeIds && !isMatched
    const isDimmed = highlightedNodeId && !isHighlighted && !isAdjacent

    // 绘制高亮外圈
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
      ctx.fillStyle = color + '66' // 40% opacity
      ctx.globalAlpha = 0.4
    } else {
      ctx.fillStyle = color
      ctx.globalAlpha = 1
    }

    ctx.fill()

    // 边框
    ctx.strokeStyle = isDimmed ? '#999' : '#fff'
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
  }, [highlightedNodeId, hoveredNode, matchedNodeIds, getNodeRadius, getAdjacentNodeIds])

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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fafafa' }}>
      {/* 工具栏 */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #e5e5e5',
        background: '#fff',
        display: 'flex',
        gap: 16,
        alignItems: 'center',
        flexWrap: 'wrap'
      }}>
        {/* 搜索框 */}
        <Input
          placeholder="搜索名词..."
          prefix={<SearchOutlined style={{ color: '#666' }} />}
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          allowClear
          style={{ width: 200 }}
        />

        {/* 分类过滤 */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 }}>
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

        {/* 统计信息 */}
        <Text type="secondary" style={{ fontSize: 12 }}>
          {visibleNodes.length} 个节点 · {visibleLinks.length} 条连接
        </Text>
      </div>

      {/* 画布 */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
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
          cooldownTicks={100}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          enableNodeDrag={enableDrag}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />
      </div>
    </div>
  )
}
