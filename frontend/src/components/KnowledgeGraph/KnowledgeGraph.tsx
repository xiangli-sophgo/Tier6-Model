/**
 * 知识网络可视化组件
 * 使用 react-force-graph 实现高性能力导向布局
 */
import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Search, RotateCw, Network, Move } from 'lucide-react'
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
    knowledgeHoveredSearchResultId: hoveredSearchResultIdFromContext,
    knowledgeVisibleCategories: visibleCategories,
    knowledgeNodes: cachedNodes,
    knowledgeViewBox: _cachedViewBox,
    knowledgeEnableDrag: enableDrag,
    knowledgeGraphActions,
    addKnowledgeSelectedNode,
    clearKnowledgeHighlight,
    setKnowledgeHoveredSearchResultId,
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
  const [showSearchResults, setShowSearchResults] = useState(false)
  const [selectedSearchIndex, setSelectedSearchIndex] = useState(0)

  // Refs
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink>>()
  const containerRef = useRef<HTMLDivElement>(null)
  const searchContainerRef = useRef<HTMLDivElement>(null)

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
      setShowSearchResults(false)
      setKnowledgeHoveredSearchResultId(null)  // 清除悬停高亮
      return null
    }

    // 数据完整性检查
    if (!allNodes || allNodes.length === 0) {
      console.warn('[WARN] 搜索时 allNodes 为空', {
        allNodes: allNodes?.length || 0,
        cachedNodes: cachedNodes.length,
      })
      setSearchResultCount(0)
      setShowSearchResults(false)
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
    setShowSearchResults(matched.size > 0)
    setSelectedSearchIndex(0)

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

  // 获取匹配的节点列表（用于搜索结果显示）
  const matchedNodes = useMemo(() => {
    if (!matchedNodeIds || matchedNodeIds.size === 0) return []
    return Array.from(matchedNodeIds)
      .map(id => allNodes.find(n => n.id === id))
      .filter((n): n is GraphNode => n !== undefined)
      .filter(n => visibleCategories.has(n.category))
  }, [matchedNodeIds, allNodes, visibleCategories])

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

  // 聚焦到指定节点
  const focusOnNode = useCallback((node: GraphNode) => {
    // 优先使用 Context 中的 actions（适用于 toolbar-only 模式调用）
    const actions = knowledgeGraphActions || graphRef.current

    if (!actions) {
      console.warn('[WARN] 无法聚焦节点：graphRef 和 knowledgeGraphActions 均不可用')
      return
    }

    // 添加到选中列表
    addKnowledgeSelectedNode(node as any)

    // 缩放并居中到该节点
    if (node.x !== undefined && node.y !== undefined) {
      if ('centerAt' in actions) {
        actions.centerAt(node.x, node.y, 1000)
        actions.zoom(3, 1000)
        console.log(`🎯 聚焦到节点: ${node.name} (${node.x}, ${node.y})`)
      }
    } else {
      console.warn(`[WARN] 节点 ${node.name} 没有坐标信息`)
    }
  }, [addKnowledgeSelectedNode, knowledgeGraphActions])

  // 搜索结果点击处理
  const handleSearchResultClick = useCallback((node: GraphNode, index: number) => {
    console.log(`🖱️ 点击搜索结果: ${node.name}`, node)
    setSelectedSearchIndex(index)

    // 清除悬停高亮
    setKnowledgeHoveredSearchResultId(null)

    // 先关闭下拉列表
    setShowSearchResults(false)

    // 延迟聚焦，确保画布已渲染
    setTimeout(() => {
      focusOnNode(node)
    }, 100)
  }, [focusOnNode, setKnowledgeHoveredSearchResultId])

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
      console.warn('[WARN] KnowledgeGraph: allNodes 为空', {
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
  }, [searchQuery, matchedNodeIds, allNodes, visibleCategories])

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

  // 点击搜索框外部关闭下拉列表
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node

      // 如果点击的是搜索框容器内的元素，不关闭
      if (searchContainerRef.current && searchContainerRef.current.contains(target)) {
        return
      }

      // 否则关闭下拉列表并清除高亮
      setShowSearchResults(false)
      setKnowledgeHoveredSearchResultId(null)
    }

    if (showSearchResults) {
      // 延迟绑定，避免立即触发
      setTimeout(() => {
        document.addEventListener('mousedown', handleClickOutside, true)  // 使用捕获阶段
      }, 0)

      return () => document.removeEventListener('mousedown', handleClickOutside, true)
    }
  }, [showSearchResults, setKnowledgeHoveredSearchResultId])

  // 单个搜索结果自动聚焦
  useEffect(() => {
    if (matchedNodes.length === 1 && searchQuery.trim()) {
      const timer = setTimeout(() => {
        focusOnNode(matchedNodes[0])
      }, 500) // 500ms 延迟，避免频繁触发
      return () => clearTimeout(timer)
    }
  }, [matchedNodes, searchQuery, focusOnNode])

  // 节点渲染
  const paintNode = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    const radius = getNodeRadius(node.id!)
    const color = CATEGORY_COLORS[node.category]

    // 判断节点状态 - 使用预计算的 adjacencyMap
    const isHighlighted = node.id === highlightedNodeId
    const isHovered = hoveredNode?.id === node.id
    const isSearchResultHovered = node.id === hoveredSearchResultIdFromContext  // 搜索结果列表悬停（从 Context 读取）
    const isAdjacent = highlightedNodeId ? (adjacencyMap.get(highlightedNodeId)?.has(node.id!) || false) : false
    const isMatched = matchedNodeIds ? matchedNodeIds.has(node.id!) : true
    const isFiltered = matchedNodeIds && !isMatched
    const isDimmed = highlightedNodeId && !isHighlighted && !isAdjacent

    // 搜索时：未匹配的节点变暗，匹配的节点保持原样并加强高亮
    const isSearchActive = matchedNodeIds && matchedNodeIds.size > 0
    const shouldDimNode = isSearchActive && !isMatched

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
    const nodeRadius = isSearchResultHovered ? radius * 1.3 : radius  // 悬停时放大
    ctx.beginPath()
    ctx.arc(node.x!, node.y!, nodeRadius, 0, 2 * Math.PI)

    // 搜索结果悬停 - 强烈多重外发光
    if (isSearchResultHovered) {
      ctx.shadowBlur = 30
      ctx.shadowColor = 'rgba(168, 85, 247, 0.8)'  // 紫色
      ctx.fillStyle = color
      ctx.globalAlpha = 1
      ctx.fill()

      // 多重阴影效果
      ctx.shadowBlur = 20
      ctx.shadowColor = 'rgba(168, 85, 247, 0.6)'
      ctx.fill()
      ctx.shadowBlur = 10
      ctx.shadowColor = 'rgba(168, 85, 247, 0.4)'
      ctx.fill()
      ctx.shadowBlur = 0
    }
    // 普通搜索匹配 - 柔和外发光
    else if (isSearchActive && isMatched && !isFiltered) {
      ctx.shadowBlur = 20
      ctx.shadowColor = 'rgba(0, 217, 255, 0.7)'  // 青色
      ctx.fillStyle = color
      ctx.globalAlpha = 1
      ctx.fill()
      ctx.shadowBlur = 0
    }
    // 未匹配节点 - 强烈变暗
    else if (shouldDimNode) {
      ctx.fillStyle = '#ddd'
      ctx.globalAlpha = 0.08
      ctx.fill()
    }
    // 其他状态
    else if (isDimmed) {
      ctx.fillStyle = '#ccc'
      ctx.globalAlpha = 0.3
      ctx.fill()
    } else if (isFiltered) {
      ctx.fillStyle = color + '4D'
      ctx.globalAlpha = 0.3
      ctx.fill()
    } else {
      ctx.fillStyle = color
      ctx.globalAlpha = 1
      ctx.fill()
    }

    // 边框
    if (shouldDimNode) {
      ctx.strokeStyle = '#ccc'
      ctx.globalAlpha = 0.1
    } else if (isSearchResultHovered) {
      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 2.5
      ctx.globalAlpha = 1
    } else {
      ctx.strokeStyle = isDimmed ? '#999' : isFiltered ? '#ccc' : '#fff'
      ctx.lineWidth = 1.5
      ctx.globalAlpha = isDimmed || isFiltered ? 0.5 : 1
    }
    ctx.stroke()

    ctx.globalAlpha = 1
    ctx.shadowBlur = 0

    // Canvas悬停效果 - 发光（仅当不是搜索结果悬停时）
    if (isHovered && !isDimmed && !isSearchResultHovered) {
      ctx.shadowBlur = 15
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

    if (shouldDimNode) {
      ctx.fillStyle = '#aaa'
      ctx.globalAlpha = 0.3
    } else {
      ctx.fillStyle = isDimmed ? '#999' : '#fff'
      ctx.globalAlpha = isDimmed ? 0.5 : 1
    }

    ctx.fillText(label, node.x!, node.y!)
    ctx.globalAlpha = 1
  }, [highlightedNodeId, hoveredNode, hoveredSearchResultIdFromContext, matchedNodeIds, getNodeRadius, adjacencyMap])

  // 边渲染
  const paintLink = useCallback((link: GraphLink, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    const sourceNode = typeof link.source === 'object' ? link.source : null
    const targetNode = typeof link.target === 'object' ? link.target : null

    if (!sourceNode || !targetNode) return

    const sourceId = sourceNode.id!
    const targetId = targetNode.id!

    // 判断边状态
    const isHighlighted = highlightedNodeId && (sourceId === highlightedNodeId || targetId === highlightedNodeId)
    const isSearchActive = matchedNodeIds && matchedNodeIds.size > 0
    const isSearchFiltered = isSearchActive && (!matchedNodeIds.has(sourceId) || !matchedNodeIds.has(targetId))
    const isDimmed = highlightedNodeId && !isHighlighted

    // 绘制边
    ctx.beginPath()
    ctx.moveTo(sourceNode.x!, sourceNode.y!)
    ctx.lineTo(targetNode.x!, targetNode.y!)

    if (isDimmed) {
      ctx.strokeStyle = '#ddd'
      ctx.globalAlpha = 0.2
    } else if (isSearchFiltered) {
      // 搜索时未匹配的边 - 强烈变暗
      ctx.strokeStyle = '#e5e5e5'
      ctx.globalAlpha = 0.1
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
      <div className="flex w-full items-center justify-between gap-6 border-b border-blue-100 bg-gradient-to-r from-blue-50 to-white px-6 py-2.5" style={{boxShadow: '0 2px 8px rgba(37, 99, 235, 0.06)'}}>
        {/* 左侧：搜索框 */}
        <div ref={searchContainerRef} className="relative w-[280px]">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-secondary" />
          <Input
            placeholder="搜索名词..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onFocus={() => matchedNodes.length > 0 && setShowSearchResults(true)}
            className="pl-9"
          />
          {/* 搜索结果下拉列表 */}
          {showSearchResults && matchedNodes.length > 0 && (
            <div className="absolute top-full left-0 mt-2 w-[450px] max-h-[500px] overflow-y-auto bg-white rounded-lg shadow-xl border border-gray-200" style={{ zIndex: 9999 }}>
                  <div className="p-2">
                    <div className="flex items-center justify-between text-xs px-2 py-1 mb-1">
                      <span className="text-text-muted">
                        找到 {matchedNodes.length} 个匹配结果
                      </span>
                      {matchedNodes.length === 1 && (
                        <span className="text-blue-500 font-medium">
                          自动定位中...
                        </span>
                      )}
                    </div>
                    {matchedNodes.map((node, index) => (
                      <div
                        key={node.id}
                        className="flex items-start gap-3 p-2 rounded cursor-pointer hover:bg-gray-50"
                        onClick={() => handleSearchResultClick(node, index)}
                        onMouseEnter={() => setKnowledgeHoveredSearchResultId(node.id)}
                        onMouseLeave={() => setKnowledgeHoveredSearchResultId(null)}
                      >
                        {/* 颜色指示器 */}
                        <div
                          className="flex-shrink-0 w-3 h-3 rounded-full mt-1"
                          style={{ backgroundColor: CATEGORY_COLORS[node.category] }}
                        />
                        {/* 节点信息 */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-sm text-text-primary">
                              {node.name}
                            </span>
                            <Badge
                              variant="outline"
                              className="text-xs px-1 py-0"
                              style={{
                                borderColor: CATEGORY_COLORS[node.category],
                                color: CATEGORY_COLORS[node.category]
                              }}
                            >
                              {CATEGORY_NAMES[node.category]}
                            </Badge>
                          </div>
                          <p className="text-xs text-text-secondary line-clamp-2">
                            {node.definition}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
        </div>
        <div className="flex flex-1 items-center justify-center gap-3 min-w-0">

          {/* 分类过滤 */}
          <div className="flex flex-wrap gap-1 min-w-0">
            {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
              const category = key as KnowledgeCategory
              const isActive = visibleCategories.has(category)
              const count = allNodes.filter(n => n.category === category).length
              if (count === 0) return null
              return (
                <Badge
                  key={category}
                  variant={isActive ? 'default' : 'outline'}
                  className="cursor-pointer text-xs px-2 py-0 m-0"
                  style={{
                    opacity: isActive ? 1 : 0.5,
                    borderColor: CATEGORY_COLORS[category],
                    backgroundColor: isActive ? CATEGORY_COLORS[category] : undefined,
                    color: isActive ? '#fff' : undefined,
                  }}
                  onClick={(e) => handleCategoryClick(category, e.ctrlKey || e.metaKey)}
                >
                  {name}
                </Badge>
              )
            })}
          </div>

          {/* 全部显示按钮 */}
          {visibleCategories.size < 8 && (
            <Button
              size="sm"
              variant="outline"
              onClick={resetKnowledgeCategories}
            >
              <RotateCw className="mr-1 h-3 w-3" />
              全部
            </Button>
          )}
        </div>

        {/* 右侧：重新布局 + 拖动开关 */}
        <div className="flex items-center gap-3">
          {/* 重新布局按钮 */}
          <Button
            size="sm"
            variant="outline"
            onClick={handleRelayout}
          >
            <Network className="mr-1 h-3 w-3" />
            重新布局
          </Button>

          {/* 拖动开关 */}
          <div className="flex items-center gap-2">
            <Move className={`h-4 w-4 ${enableDrag ? 'text-blue-500' : 'text-text-muted'}`} />
            <Switch
              checked={enableDrag}
              onCheckedChange={setEnableDrag}
            />
            <span className="text-xs text-text-secondary whitespace-nowrap">
              {enableDrag ? '可拖动' : '不可拖动'}
            </span>
          </div>
        </div>
      </div>
    )
  }

  // 只渲染画布
  if (renderMode === 'canvas-only') {
    return (
      <div ref={containerRef} className="relative h-full w-full overflow-hidden bg-gradient-to-b from-gray-50 to-white">
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
          backgroundColor="#F8FAFB"
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
    <div className="flex h-full w-full flex-col bg-gradient-to-b from-gray-50 to-white">
      {/* 工具栏 */}
      <div className="flex w-full items-center justify-between gap-4 border-b border-blue-100 bg-gradient-to-r from-blue-50 to-white px-4 py-4" style={{boxShadow: '0 2px 8px rgba(37, 99, 235, 0.06)'}}>
        {/* 左侧：搜索框 + 分类过滤 + 全部显示 */}
        <div className="flex flex-1 items-center justify-center gap-4 min-w-0">
          {/* 搜索框 */}
          <div className="flex flex-shrink-0 items-center gap-2">
            <div ref={searchContainerRef} className="relative w-[200px]">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-secondary" />
              <Input
                placeholder="搜索名词..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onFocus={() => matchedNodes.length > 0 && setShowSearchResults(true)}
                className="pl-9"
              />
              {/* 搜索结果下拉列表 */}
              {showSearchResults && matchedNodes.length > 0 && (
                <div className="absolute top-full left-0 mt-2 w-[400px] max-h-[400px] overflow-y-auto bg-white rounded-lg shadow-xl border border-gray-200" style={{ zIndex: 9999 }}>
                  <div className="p-2">
                    <div className="flex items-center justify-between text-xs px-2 py-1 mb-1">
                      <span className="text-text-muted">
                        找到 {matchedNodes.length} 个匹配结果
                      </span>
                      {matchedNodes.length === 1 && (
                        <span className="text-blue-500 font-medium">
                          自动定位中...
                        </span>
                      )}
                    </div>
                    {matchedNodes.map((node, index) => (
                      <div
                        key={node.id}
                        className="flex items-start gap-3 p-2 rounded cursor-pointer hover:bg-gray-50"
                        onClick={() => handleSearchResultClick(node, index)}
                        onMouseEnter={() => setKnowledgeHoveredSearchResultId(node.id)}
                        onMouseLeave={() => setKnowledgeHoveredSearchResultId(null)}
                      >
                        {/* 颜色指示器 */}
                        <div
                          className="flex-shrink-0 w-3 h-3 rounded-full mt-1"
                          style={{ backgroundColor: CATEGORY_COLORS[node.category] }}
                        />
                        {/* 节点信息 */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-sm text-text-primary">
                              {node.name}
                            </span>
                            <Badge
                              variant="outline"
                              className="text-xs px-1 py-0"
                              style={{
                                borderColor: CATEGORY_COLORS[node.category],
                                color: CATEGORY_COLORS[node.category]
                              }}
                            >
                              {CATEGORY_NAMES[node.category]}
                            </Badge>
                          </div>
                          <p className="text-xs text-text-secondary line-clamp-2">
                            {node.definition}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {/* 搜索结果反馈 */}
            {searchQuery.trim() && (
              <span className={`text-xs font-medium whitespace-nowrap ${
                searchResultCount > 0 ? 'text-success' : 'text-error'
              }`}>
                {searchResultCount > 0
                  ? `找到 ${searchResultCount}`
                  : '未找到'}
              </span>
            )}
          </div>

          {/* 分类过滤 */}
          <div className="flex flex-wrap gap-1 min-w-0">
            {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
              const category = key as KnowledgeCategory
              const isActive = visibleCategories.has(category)
              const count = allNodes.filter(n => n.category === category).length
              if (count === 0) return null
              return (
                <Badge
                  key={category}
                  variant={isActive ? 'default' : 'outline'}
                  className="cursor-pointer text-xs px-2 py-0 m-0"
                  style={{
                    opacity: isActive ? 1 : 0.5,
                    borderColor: CATEGORY_COLORS[category],
                    backgroundColor: isActive ? CATEGORY_COLORS[category] : undefined,
                    color: isActive ? '#fff' : undefined,
                  }}
                  onClick={(e) => handleCategoryClick(category, e.ctrlKey || e.metaKey)}
                >
                  {name}
                </Badge>
              )
            })}
          </div>

          {/* 全部显示按钮 */}
          {visibleCategories.size < 8 && (
            <Button
              size="sm"
              variant="outline"
              onClick={resetKnowledgeCategories}
              className="flex-shrink-0"
            >
              <RotateCw className="mr-1 h-3 w-3" />
              全部显示
            </Button>
          )}
        </div>

        {/* 右侧：重新布局 + 拖动开关 */}
        <div className="flex flex-shrink-0 items-center gap-3">
          {/* 重新布局按钮 */}
          <Button
            size="sm"
            variant="outline"
            onClick={handleRelayout}
          >
            <Network className="mr-1 h-3 w-3" />
            重新布局
          </Button>

          {/* 拖动开关 */}
          <div className="flex items-center gap-2">
            <Move className={`h-4 w-4 ${enableDrag ? 'text-blue-500' : 'text-text-muted'}`} />
            <Switch
              checked={enableDrag}
              onCheckedChange={setEnableDrag}
            />
            <span className="text-xs text-text-secondary">
              {enableDrag ? '可拖动' : '不可拖动'}
            </span>
          </div>
        </div>
      </div>

      {/* 画布 */}
      <div ref={containerRef} className="relative flex-1 w-full overflow-hidden bg-gradient-to-b from-gray-50 to-white">
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
          backgroundColor="#F8FAFB"
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
