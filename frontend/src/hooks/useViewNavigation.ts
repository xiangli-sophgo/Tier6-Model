import { useState, useCallback, useMemo, useRef } from 'react'
import {
  ViewState,
  ViewLevel,
  BreadcrumbItem,
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
} from '../types'

// 根据路径深度获取层级
// depth 0 = 数据中心顶层(显示所有Pod)
// depth 1 = Pod内部(显示该Pod的所有Rack)
// depth 2 = Rack内部(显示该Rack的所有Board)
// depth 3+ = Board内部(显示该Board的所有Chip)
function getLevelFromDepth(depth: number): ViewLevel {
  if (depth === 0) return 'pod'      // 顶层
  if (depth === 1) return 'pod'      // Pod内部，仍是pod级别视图
  if (depth === 2) return 'rack'     // Rack内部
  return 'board'                      // Board内部
}

// 从拓扑数据中查找节点
function findNode(
  topology: HierarchicalTopology,
  path: string[]
): PodConfig | RackConfig | BoardConfig | null {
  if (path.length === 0) return null

  const podId = path[0]
  const pod = topology.pods.find(p => p.id === podId)
  if (!pod || path.length === 1) return pod || null

  const rackId = path[1]
  const rack = pod.racks.find(r => r.id === rackId)
  if (!rack || path.length === 2) return rack || null

  const boardId = path[2]
  const board = rack.boards.find(b => b.id === boardId)
  return board || null
}

// 获取节点标签
function _getNodeLabel(
  topology: HierarchicalTopology,
  path: string[]
): string {
  const node = findNode(topology, path)
  if (!node) return path[path.length - 1]
  return node.label
}
void _getNodeLabel // 标记为已使用，保留以备后用

export interface ViewNavigationReturn {
  viewState: ViewState
  breadcrumbs: BreadcrumbItem[]
  navigateTo: (nodeId: string) => void
  navigateToPod: (podId: string) => void
  navigateToRack: (podId: string, rackId: string) => void
  navigateToBoard: (podId: string, rackId: string, boardId: string) => void
  navigateBack: () => void
  navigateToBreadcrumb: (index: number) => void
  navigateToTop: () => void
  canGoBack: boolean
  // 历史导航
  navigateHistoryBack: () => void
  navigateHistoryForward: () => void
  canGoHistoryBack: boolean
  canGoHistoryForward: boolean
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
}

export function useViewNavigation(
  topology: HierarchicalTopology | null
): ViewNavigationReturn {
  const [viewState, setViewState] = useState<ViewState>({
    level: 'pod',
    path: [],
    selectedNode: undefined,
  })

  // 历史记录: 存储路径数组的历史
  const historyRef = useRef<string[][]>([[]])  // 初始为空路径
  const historyIndexRef = useRef(0)
  const isNavigatingHistoryRef = useRef(false)  // 是否正在进行历史导航

  // 添加到历史记录（仅在非历史导航时）
  const addToHistory = useCallback((path: string[]) => {
    if (isNavigatingHistoryRef.current) {
      isNavigatingHistoryRef.current = false
      return
    }
    // 如果不在历史末尾，清除后面的历史
    if (historyIndexRef.current < historyRef.current.length - 1) {
      historyRef.current = historyRef.current.slice(0, historyIndexRef.current + 1)
    }
    // 避免重复添加相同路径
    const lastPath = historyRef.current[historyRef.current.length - 1]
    if (JSON.stringify(lastPath) !== JSON.stringify(path)) {
      historyRef.current.push([...path])
      historyIndexRef.current = historyRef.current.length - 1
    }
  }, [])

  // 生成面包屑
  const breadcrumbs = useMemo(() => {
    const items: BreadcrumbItem[] = [
      { level: 'pod', id: 'root', label: '数据中心' }
    ]

    if (!topology || viewState.path.length === 0) return items

    // path[0] = podId, path[1] = rackId, path[2] = boardId
    const podId = viewState.path[0]
    const pod = topology.pods.find(p => p.id === podId)
    if (pod) {
      items.push({ level: 'pod', id: podId, label: pod.label })
    }

    if (viewState.path.length >= 2 && pod) {
      const rackId = viewState.path[1]
      const rack = pod.racks.find(r => r.id === rackId)
      if (rack) {
        items.push({ level: 'rack', id: rackId, label: rack.label })
      }
    }

    if (viewState.path.length >= 3 && pod) {
      const rackId = viewState.path[1]
      const rack = pod.racks.find(r => r.id === rackId)
      if (rack) {
        const boardId = viewState.path[2]
        const board = rack.boards.find(b => b.id === boardId)
        if (board) {
          items.push({ level: 'board', id: boardId, label: board.label })
        }
      }
    }

    return items
  }, [viewState.path, topology])

  // 导航到Pod内部
  const navigateToPod = useCallback((podId: string) => {
    const newPath = [podId]
    addToHistory(newPath)
    setViewState({
      level: 'pod',
      path: newPath,
      selectedNode: undefined,
    })
  }, [addToHistory])

  // 导航到子层级（通用）
  const navigateTo = useCallback((nodeId: string) => {
    setViewState(prev => {
      const newPath = [...prev.path, nodeId]
      addToHistory(newPath)
      const newLevel = getLevelFromDepth(newPath.length)
      return {
        level: newLevel,
        path: newPath,
        selectedNode: undefined,
      }
    })
  }, [addToHistory])

  // 直接导航到Rack (从Pod视图)
  const navigateToRack = useCallback((podId: string, rackId: string) => {
    const newPath = [podId, rackId]
    addToHistory(newPath)
    setViewState({
      level: 'rack',
      path: newPath,
      selectedNode: undefined,
    })
  }, [addToHistory])

  // 直接导航到Board (从任意视图)
  const navigateToBoard = useCallback((podId: string, rackId: string, boardId: string) => {
    const newPath = [podId, rackId, boardId]
    addToHistory(newPath)
    setViewState({
      level: 'board',
      path: newPath,
      selectedNode: undefined,
    })
  }, [addToHistory])

  // 返回上一级
  const navigateBack = useCallback(() => {
    setViewState(prev => {
      if (prev.path.length === 0) return prev
      const newPath = prev.path.slice(0, -1)
      addToHistory(newPath)
      const newLevel = getLevelFromDepth(newPath.length)
      return {
        level: newLevel,
        path: newPath,
        selectedNode: undefined,
      }
    })
  }, [addToHistory])

  // 通过面包屑导航
  const navigateToBreadcrumb = useCallback((index: number) => {
    setViewState(prev => {
      if (index === 0) {
        addToHistory([])
        return { level: 'pod', path: [], selectedNode: undefined }
      }
      const newPath = prev.path.slice(0, index)
      addToHistory(newPath)
      const newLevel = getLevelFromDepth(newPath.length)
      return {
        level: newLevel,
        path: newPath,
        selectedNode: undefined,
      }
    })
  }, [addToHistory])

  // 导航到顶层（数据中心视图）
  const navigateToTop = useCallback(() => {
    addToHistory([])
    setViewState({ level: 'pod', path: [], selectedNode: undefined })
  }, [addToHistory])

  // 历史后退（左方向键）
  const navigateHistoryBack = useCallback(() => {
    if (historyIndexRef.current > 0) {
      historyIndexRef.current--
      const targetPath = historyRef.current[historyIndexRef.current]
      isNavigatingHistoryRef.current = true
      const newLevel = getLevelFromDepth(targetPath.length)
      setViewState({
        level: newLevel,
        path: [...targetPath],
        selectedNode: undefined,
      })
    }
  }, [])

  // 历史前进（右方向键）
  const navigateHistoryForward = useCallback(() => {
    if (historyIndexRef.current < historyRef.current.length - 1) {
      historyIndexRef.current++
      const targetPath = historyRef.current[historyIndexRef.current]
      isNavigatingHistoryRef.current = true
      const newLevel = getLevelFromDepth(targetPath.length)
      setViewState({
        level: newLevel,
        path: [...targetPath],
        selectedNode: undefined,
      })
    }
  }, [])

  // 获取当前Pod
  const currentPod = useMemo(() => {
    if (!topology || viewState.path.length === 0) return null
    return topology.pods.find(p => p.id === viewState.path[0]) || null
  }, [topology, viewState.path])

  // 获取当前Rack
  const currentRack = useMemo(() => {
    if (!currentPod || viewState.path.length < 2) return null
    const rackIdInPath = viewState.path[1]
    return currentPod.racks.find(r => r.id === rackIdInPath) || null
  }, [currentPod, viewState.path])

  // 获取当前Board
  const currentBoard = useMemo(() => {
    if (!currentRack || viewState.path.length < 3) return null
    return currentRack.boards.find(b => b.id === viewState.path[2]) || null
  }, [currentRack, viewState.path])

  const canGoBack = viewState.path.length > 0

  // 历史导航状态 - 每次渲染时基于当前ref值计算
  const canGoHistoryBack = historyIndexRef.current > 0
  const canGoHistoryForward = historyIndexRef.current < historyRef.current.length - 1

  return {
    viewState,
    breadcrumbs,
    navigateTo,
    navigateToPod,
    navigateToRack,
    navigateToBoard,
    navigateBack,
    navigateToBreadcrumb,
    navigateToTop,
    canGoBack,
    navigateHistoryBack,
    navigateHistoryForward,
    canGoHistoryBack,
    canGoHistoryForward,
    currentPod,
    currentRack,
    currentBoard,
  }
}
