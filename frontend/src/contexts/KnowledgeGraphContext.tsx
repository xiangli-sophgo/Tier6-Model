/**
 * 知识图谱状态管理 Context
 * 负责管理知识图谱的节点、布局、选中状态等
 * ForceGraph2D 会自动进行力导向布局（在后台，即使 display: none）
 */
import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react'
import { ForceKnowledgeNode, KnowledgeCategory } from '../components/KnowledgeGraph'
import { KnowledgeGraphData } from '../components/KnowledgeGraph/types'
import knowledgeData from '../data/knowledge-graph'

// ============================================
// 类型定义
// ============================================

// 知识图谱视口类型
export interface KnowledgeViewBox {
  x: number
  y: number
  width: number
  height: number
}

// 知识图谱操作接口
export interface KnowledgeGraphActions {
  reheatSimulation: () => void
  zoomToFit: (duration?: number, padding?: number) => void
  relayout: () => void  // 完整的重新布局（重置位置 + 重新模拟）
}

export interface KnowledgeGraphContextType {
  knowledgeSelectedNodes: ForceKnowledgeNode[]  // 支持多个选中节点（用于详情卡片）
  knowledgeHighlightedNodeId: string | null  // 当前高亮的节点ID（用于图中高亮效果）
  knowledgeVisibleCategories: Set<KnowledgeCategory>
  knowledgeNodes: ForceKnowledgeNode[]
  knowledgeInitialized: boolean
  knowledgeViewBox: KnowledgeViewBox | null
  knowledgeEnableDrag: boolean  // 节点拖动开关
  knowledgeGraphActions: KnowledgeGraphActions | null  // 图谱操作方法
  addKnowledgeSelectedNode: (node: ForceKnowledgeNode) => void  // 添加节点到列表
  removeKnowledgeSelectedNode: (nodeId: string) => void  // 从列表移除节点
  clearKnowledgeHighlight: () => void  // 清除高亮（不影响详情卡片）
  setKnowledgeVisibleCategories: (categories: Set<KnowledgeCategory>) => void
  setKnowledgeNodes: (nodes: ForceKnowledgeNode[]) => void
  setKnowledgeInitialized: (initialized: boolean) => void
  setKnowledgeViewBox: (viewBox: KnowledgeViewBox) => void
  setKnowledgeEnableDrag: (enable: boolean) => void  // 设置拖动开关
  setKnowledgeGraphActions: (actions: KnowledgeGraphActions | null) => void  // 设置图谱操作方法
  resetKnowledgeCategories: () => void
  clearSelectedNodes: () => void  // 清空选中节点列表
}

// ============================================
// Context 创建
// ============================================
const KnowledgeGraphContext = createContext<KnowledgeGraphContextType | null>(null)

export const useKnowledgeGraph = () => {
  const context = useContext(KnowledgeGraphContext)
  if (!context) {
    throw new Error('useKnowledgeGraph must be used within KnowledgeGraphProvider')
  }
  return context
}

// ============================================
// Provider 实现
// ============================================
interface KnowledgeGraphProviderProps {
  children: ReactNode
}

export const KnowledgeGraphProvider: React.FC<KnowledgeGraphProviderProps> = ({ children }) => {
  const [knowledgeSelectedNodes, setKnowledgeSelectedNodes] = useState<ForceKnowledgeNode[]>([])
  const [knowledgeHighlightedNodeId, setKnowledgeHighlightedNodeId] = useState<string | null>(null)
  const [knowledgeVisibleCategories, setKnowledgeVisibleCategories] = useState<Set<KnowledgeCategory>>(
    new Set(['hardware', 'interconnect', 'parallel', 'communication', 'model', 'inference', 'protocol', 'system'])
  )
  const [knowledgeNodes, setKnowledgeNodes] = useState<ForceKnowledgeNode[]>([])
  const [knowledgeInitialized, setKnowledgeInitialized] = useState(false)
  const [knowledgeViewBox, setKnowledgeViewBox] = useState<KnowledgeViewBox | null>(null)
  const [knowledgeEnableDrag, setKnowledgeEnableDrag] = useState(false)
  const [knowledgeGraphActions, setKnowledgeGraphActions] = useState<KnowledgeGraphActions | null>(null)

  // 重置分类
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

  // 清空选中节点列表
  const clearSelectedNodes = useCallback(() => {
    setKnowledgeSelectedNodes([])
    setKnowledgeHighlightedNodeId(null)
  }, [])

  // ==================== 知识图谱预初始化（Web Worker 版）====================
  // 使用 Web Worker 在后台线程计算力导向布局，完全不阻塞主线程
  useEffect(() => {
    // 如果已经初始化过，跳过
    if (knowledgeInitialized) return

    const initKnowledgeGraphWithWorker = () => {
      const data = knowledgeData as KnowledgeGraphData

      // ⚡ 立即加载原始节点数据，设置随机初始位置
      // 这样节点不会从同一点开始，布局范围更合理

      // 设置随机初始位置（在一个圆形区域内）
      const radius = 300  // 初始分布半径
      setKnowledgeNodes(data.nodes.map((n): ForceKnowledgeNode => {
        const angle = Math.random() * 2 * Math.PI
        const r = Math.sqrt(Math.random()) * radius  // 平方根让分布更均匀
        return {
          ...n,
          x: Math.cos(angle) * r,
          y: Math.sin(angle) * r
        }
      }))
      setKnowledgeInitialized(true)

      // 不再使用 Worker，让 ForceGraph2D 自己进行力导向布局
      return

      /* 以下 Worker 代码已废弃，保留以备将来需要
      const centerX = 600
      const centerY = 400

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

      // 转换度数 Map 为普通对象（可序列化）
      const degreeMapObject: Record<string, number> = {}
      initDegreeMap.forEach((value, key) => {
        degreeMapObject[key] = value
      })

      // 🎯 不再立即设置初始位置，等待 Worker 完成后再设置
      // 这样可以避免"弹跳"效果（初始位置和优化位置差异太大）

      try {
        // 创建 Web Worker（在后台优化节点位置）
        const worker = new Worker(
          new URL('../workers/knowledge-graph-worker.ts', import.meta.url),
          { type: 'module' }
        )

        // 设置超时保护（10 秒）
        const timeout = setTimeout(() => {
          worker.terminate()
          // Worker 超时，使用初始位置
          setKnowledgeNodes(initialNodes)
          const padding = 100
          const minX = Math.min(...initialNodes.map(n => n.x ?? 0)) - padding
          const maxX = Math.max(...initialNodes.map(n => n.x ?? 0)) + padding
          const minY = Math.min(...initialNodes.map(n => n.y ?? 0)) - padding
          const maxY = Math.max(...initialNodes.map(n => n.y ?? 0)) + padding
          setKnowledgeViewBox({
            x: minX,
            y: minY,
            width: Math.max(maxX - minX, 400),
            height: Math.max(maxY - minY, 300),
          })
          setKnowledgeInitialized(true)
        }, 10000)

        // 监听 Worker 消息
        worker.onmessage = (e: MessageEvent) => {
          clearTimeout(timeout)
          worker.terminate()

          if (e.data.type === 'error') {
            // Worker 失败，使用初始位置
            setKnowledgeNodes(initialNodes)
            const padding = 100
            const minX = Math.min(...initialNodes.map(n => n.x ?? 0)) - padding
            const maxX = Math.max(...initialNodes.map(n => n.x ?? 0)) + padding
            const minY = Math.min(...initialNodes.map(n => n.y ?? 0)) - padding
            const maxY = Math.max(...initialNodes.map(n => n.y ?? 0)) + padding
            setKnowledgeViewBox({
              x: minX,
              y: minY,
              width: Math.max(maxX - minX, 400),
              height: Math.max(maxY - minY, 300),
            })
            setKnowledgeInitialized(true)
            return
          }

          if (e.data.type === 'success' && e.data.data) {
            // 合并 Worker 计算的位置到完整节点数据
            const computedPositions = new Map<string, { x: number; y: number; vx: number; vy: number }>(
              e.data.data.nodes.map((n: { id: string; x: number; y: number; vx: number; vy: number }) =>
                [n.id, { x: n.x, y: n.y, vx: n.vx, vy: n.vy }]
              )
            )
            const finalNodes = initialNodes.map(node => ({
              ...node,
              ...(computedPositions.get(node.id) || { x: node.x, y: node.y }),
            }))


            setKnowledgeNodes(finalNodes)
            setKnowledgeViewBox(e.data.data.viewBox)
            setKnowledgeInitialized(true)
          }
        }

        // 监听 Worker 错误
        worker.onerror = (error) => {
          clearTimeout(timeout)
          worker.terminate()
          // Worker 错误，使用初始位置
          setKnowledgeNodes(initialNodes)
          const padding = 100
          const minX = Math.min(...initialNodes.map(n => n.x ?? 0)) - padding
          const maxX = Math.max(...initialNodes.map(n => n.x ?? 0)) + padding
          const minY = Math.min(...initialNodes.map(n => n.y ?? 0)) - padding
          const maxY = Math.max(...initialNodes.map(n => n.y ?? 0)) + padding
          setKnowledgeViewBox({
            x: minX,
            y: minY,
            width: Math.max(maxX - minX, 400),
            height: Math.max(maxY - minY, 300),
          })
          setKnowledgeInitialized(true)
        }

        // 发送数据到 Worker
        worker.postMessage({
          type: 'compute',
          data: {
            nodes: initialNodes.map(n => ({
              id: n.id,
              name: n.name,
              definition: n.definition,
              category: n.category,
              x: n.x,
              y: n.y,
              vx: n.vx,
              vy: n.vy,
            })),
            relations: visibleRelations.map(r => ({
              source: r.source,
              target: r.target,
            })),
            degreeMap: degreeMapObject,
            maxDegree,
            centerX,
            centerY,
          },
        })

      } catch (error) {
        // Worker 创建失败，使用初始位置
        setKnowledgeNodes(initialNodes)
        const padding = 100
        const minX = Math.min(...initialNodes.map(n => n.x ?? 0)) - padding
        const maxX = Math.max(...initialNodes.map(n => n.x ?? 0)) + padding
        const minY = Math.min(...initialNodes.map(n => n.y ?? 0)) - padding
        const maxY = Math.max(...initialNodes.map(n => n.y ?? 0)) + padding
        setKnowledgeViewBox({
          x: minX,
          y: minY,
          width: Math.max(maxX - minX, 400),
          height: Math.max(maxY - minY, 300),
        })
        setKnowledgeInitialized(true)
      }
      */
    }

    // 立即执行初始化（不再使用 requestIdleCallback）
    initKnowledgeGraphWithWorker()
  }, [knowledgeInitialized])

  const contextValue: KnowledgeGraphContextType = {
    knowledgeSelectedNodes,
    knowledgeHighlightedNodeId,
    knowledgeVisibleCategories,
    knowledgeNodes,
    knowledgeInitialized,
    knowledgeViewBox,
    knowledgeEnableDrag,
    knowledgeGraphActions,
    addKnowledgeSelectedNode,
    removeKnowledgeSelectedNode,
    clearKnowledgeHighlight,
    setKnowledgeVisibleCategories,
    setKnowledgeNodes,
    setKnowledgeInitialized,
    setKnowledgeViewBox,
    setKnowledgeEnableDrag,
    setKnowledgeGraphActions,
    resetKnowledgeCategories,
    clearSelectedNodes,
  }

  return (
    <KnowledgeGraphContext.Provider value={contextValue}>
      {children}
    </KnowledgeGraphContext.Provider>
  )
}

export default KnowledgeGraphContext
