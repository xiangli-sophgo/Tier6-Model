/**
 * 拓扑流量图表组件
 *
 * 在拓扑图上可视化链路流量和带宽利用率
 * 复用 TopologyGraph 组件实现 2D 拓扑可视化
 * 支持面包屑导航、布局切换、多层级视图等交互
 */

import React, { useMemo, useState, useCallback } from 'react'
import type { HierarchicalTopology, LayoutType, MultiLevelViewOptions } from '@/types'
import type { LinkTrafficStats, TopologyTrafficResult } from '@/utils/llmDeployment/types'
import { TopologyGraph } from '@/components/TopologyGraph'
import type { BreadcrumbItem } from '@/components/TopologyGraph/shared'

type ViewLevel = 'datacenter' | 'pod' | 'rack' | 'board'

interface TopologyTrafficChartProps {
  topology: HierarchicalTopology | null
  linkTrafficStats: LinkTrafficStats[] | null
  height?: number
}

// 热力图颜色函数（纯函数，无组件依赖）
function getHeatmapColor(utilizationPercent: number): string {
  const u = Math.min(Math.max(utilizationPercent, 0), 100)

  if (u < 30) {
    return '#52c41a' // 绿色：低负载
  } else if (u < 60) {
    return '#faad14' // 黄色：中等负载
  } else if (u < 80) {
    return '#fa8c16' // 橙色：高负载
  } else {
    return '#f5222d' // 红色：瓶颈
  }
}

export const TopologyTrafficChart: React.FC<TopologyTrafficChartProps> = ({
  topology,
  linkTrafficStats,
  height,
}) => {
  // ============================================
  // 导航状态管理
  // ============================================
  const [viewPath, setViewPath] = useState<string[]>([])
  const [layoutType, setLayoutType] = useState<LayoutType>('auto')
  const [multiLevelOptions, setMultiLevelOptions] = useState<MultiLevelViewOptions>({
    enabled: false,
    levelPair: 'datacenter_pod',
    expandedContainers: new Set<string>(),
  })

  // 构建芯片 ID 到 label 的映射（用于显示友好名称）
  // 同时为芯片设置正确的 label（芯片型号，如 SG2262）
  const { topologyWithConnections, chipIdToLabel } = useMemo(() => {
    if (!topology) return { topologyWithConnections: null, chipIdToLabel: {} as Record<string, string> }

    // 深拷贝 topology 以修改芯片 label
    const updatedTopology = JSON.parse(JSON.stringify(topology)) as HierarchicalTopology

    // 全局芯片计数器（按芯片型号分组）
    const chipCounters: Record<string, number> = {}
    // 芯片 ID 到 label 的映射
    const idToLabel: Record<string, string> = {}

    // 遍历所有芯片，设置正确的 label
    updatedTopology.pods?.forEach((pod) => {
      pod.racks?.forEach((rack) => {
        rack.boards?.forEach((board) => {
          board.chips?.forEach((chip) => {
            // 获取芯片型号名称 (label 来自 topologyGenerator, 格式: "SG2262-0")
            // 提取型号部分 (去掉 -N 后缀)
            const chipAny = chip as any
            const rawLabel = chip.label || chipAny.name
            if (!rawLabel) {
              throw new Error(`[FAIL] 芯片 '${chip.id}' 缺少 label 和 name 字段`)
            }
            const chipName = rawLabel.replace(/-\d+$/, '')
            // 获取该型号的当前计数
            const count = chipCounters[chipName] || 0
            // 节点显示只用芯片型号（不带编号）
            chip.label = chipName
            // 详细列表和 tooltip 用 型号-编号 格式区分
            idToLabel[chip.id] = `${chipName}-${count}`
            // 递增计数器
            chipCounters[chipName] = count + 1
          })
        })
      })
    })

    // 如果没有 connections，从 linkTrafficStats 生成
    if (!updatedTopology.connections || updatedTopology.connections.length === 0) {
      if (linkTrafficStats && linkTrafficStats.length > 0) {
        updatedTopology.connections = linkTrafficStats.map(stat => ({
          source: stat.source,
          target: stat.target,
          type: (stat.linkType || 'c2c') as 'c2c' | 'b2b' | 'r2r' | 'p2p' | 'custom' | 'switch',
          bandwidth: stat.bandwidthGbps,
          latency: stat.latencyUs,
        }))
      }
    }

    return { topologyWithConnections: updatedTopology, chipIdToLabel: idToLabel }
  }, [topology, linkTrafficStats])

  // 根据导航路径计算当前层级和节点
  const { currentLevel, currentPod, currentRack, currentBoard } = useMemo(() => {
    if (!topologyWithConnections?.pods?.[0]) {
      return { currentLevel: 'datacenter' as ViewLevel, currentPod: null, currentRack: null, currentBoard: null }
    }

    // 根据 viewPath 确定层级
    if (viewPath.length === 0) {
      // 顶层：如果只有一个 Pod，自动进入 Pod 内部
      const pods = topologyWithConnections.pods
      if (pods.length === 1) {
        const pod = pods[0]
        const racks = pod.racks
        if (racks.length === 1) {
          const rack = racks[0]
          const boards = rack.boards
          if (boards.length === 1) {
            // 只有一个 Board，直接显示 Board 内部（芯片视图）
            return {
              currentLevel: 'board' as ViewLevel,
              currentPod: pod,
              currentRack: rack,
              currentBoard: boards[0],
            }
          }
          // 多个 Board，显示 Rack 内部
          return {
            currentLevel: 'rack' as ViewLevel,
            currentPod: pod,
            currentRack: rack,
            currentBoard: null,
          }
        }
        // 多个 Rack，显示 Pod 内部
        return {
          currentLevel: 'pod' as ViewLevel,
          currentPod: pod,
          currentRack: null,
          currentBoard: null,
        }
      }
      // 多个 Pod，显示数据中心级别
      return {
        currentLevel: 'datacenter' as ViewLevel,
        currentPod: null,
        currentRack: null,
        currentBoard: null,
      }
    }

    // 有导航路径，逐级查找
    const podId = viewPath[0]
    const pod = topologyWithConnections.pods.find(p => p.id === podId) ?? null
    if (!pod || viewPath.length === 1) {
      return {
        currentLevel: 'pod' as ViewLevel,
        currentPod: pod,
        currentRack: null,
        currentBoard: null,
      }
    }

    const rackId = viewPath[1]
    const rack = pod.racks.find(r => r.id === rackId) ?? null
    if (!rack || viewPath.length === 2) {
      return {
        currentLevel: 'rack' as ViewLevel,
        currentPod: pod,
        currentRack: rack,
        currentBoard: null,
      }
    }

    const boardId = viewPath[2]
    const board = rack.boards.find(b => b.id === boardId) ?? null
    return {
      currentLevel: 'board' as ViewLevel,
      currentPod: pod,
      currentRack: rack,
      currentBoard: board,
    }
  }, [topologyWithConnections, viewPath])

  // 生成面包屑
  const breadcrumbs = useMemo((): BreadcrumbItem[] => {
    const items: BreadcrumbItem[] = [
      { level: 'datacenter', id: 'root', label: '数据中心' },
    ]

    if (!topologyWithConnections || viewPath.length === 0) {
      // 即使没有显式 viewPath，如果自动深入了也要显示面包屑
      if (currentPod) {
        items.push({ level: 'pod', id: currentPod.id, label: currentPod.label })
      }
      if (currentRack) {
        items.push({ level: 'rack', id: currentRack.id, label: currentRack.label })
      }
      if (currentBoard) {
        items.push({ level: 'board', id: currentBoard.id, label: currentBoard.label })
      }
      return items
    }

    // 根据 viewPath 构建面包屑
    const podId = viewPath[0]
    const pod = topologyWithConnections.pods.find(p => p.id === podId)
    if (pod) {
      items.push({ level: 'pod', id: podId, label: pod.label })
    }

    if (viewPath.length >= 2 && pod) {
      const rackId = viewPath[1]
      const rack = pod.racks.find(r => r.id === rackId)
      if (rack) {
        items.push({ level: 'rack', id: rackId, label: rack.label })
      }
    }

    if (viewPath.length >= 3 && pod) {
      const rackId = viewPath[1]
      const rack = pod.racks.find(r => r.id === rackId)
      if (rack) {
        const boardId = viewPath[2]
        const board = rack.boards.find(b => b.id === boardId)
        if (board) {
          items.push({ level: 'board', id: boardId, label: board.label })
        }
      }
    }

    return items
  }, [topologyWithConnections, viewPath, currentPod, currentRack, currentBoard])

  // 面包屑点击导航
  const handleBreadcrumbClick = useCallback((index: number) => {
    if (index === 0) {
      // 回到顶层
      setViewPath([])
    } else {
      // 截断到对应层级
      // 需要根据面包屑的实际 path 计算
      // index=1 对应 pod, index=2 对应 rack, index=3 对应 board
      // 但如果是自动深入的（viewPath 为空），需要手动构建 path
      if (viewPath.length > 0) {
        setViewPath(viewPath.slice(0, index))
      } else {
        // 自动深入场景：根据 index 和当前节点重建路径
        const newPath: string[] = []
        if (index >= 1 && currentPod) newPath.push(currentPod.id)
        if (index >= 2 && currentRack) newPath.push(currentRack.id)
        if (index >= 3 && currentBoard) newPath.push(currentBoard.id)
        // 截断到点击的层级（不包含当前层级，因为面包屑点击意味着回到该层级）
        setViewPath(newPath.slice(0, index))
      }
    }
  }, [viewPath, currentPod, currentRack, currentBoard])

  // 双击节点导航到下一层
  const handleNodeDoubleClick = useCallback((nodeId: string, _nodeType: string) => {
    if (!topologyWithConnections) return

    // 根据当前层级确定要导航的路径
    if (viewPath.length === 0) {
      // 自动深入场景：根据当前实际层级判断
      if (currentLevel === 'datacenter') {
        setViewPath([nodeId])
      } else if (currentLevel === 'pod' && currentPod) {
        setViewPath([currentPod.id, nodeId])
      } else if (currentLevel === 'rack' && currentPod && currentRack) {
        setViewPath([currentPod.id, currentRack.id, nodeId])
      }
    } else {
      setViewPath([...viewPath, nodeId])
    }
  }, [topologyWithConnections, viewPath, currentLevel, currentPod, currentRack])

  // 检查数据有效性
  if (!topology || !linkTrafficStats || linkTrafficStats.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        <div className="text-center">
          <div className="mb-2">暂无流量数据</div>
          <div className="text-sm">运行仿真后将显示链路流量统计</div>
        </div>
      </div>
    )
  }

  // 将 LinkTrafficStats 转换为 TopologyTrafficResult 格式
  // 注意：TopologyGraph 组件只使用 linkTraffic 字段
  const trafficResult = useMemo((): Partial<TopologyTrafficResult> & { linkTraffic: TopologyTrafficResult['linkTraffic'] } => {
    // 转换链路流量数据
    const linkTraffic: TopologyTrafficResult['linkTraffic'] = linkTrafficStats.map(stat => ({
      source: stat.source,
      target: stat.target,
      trafficMb: stat.trafficMb,
      bandwidthGbps: stat.bandwidthGbps,
      utilizationPercent: stat.utilizationPercent, // 已经是 0-100 范围
      contributingGroups: stat.contributingTasks || [],
    }))

    // 计算瓶颈链路（利用率 > 80%）
    const bottleneckLinks = linkTrafficStats
      .filter(stat => stat.utilizationPercent > 80)
      .map(stat => `${stat.source}-${stat.target}`)

    // 计算最大和平均利用率
    const maxUtilization = Math.max(...linkTrafficStats.map(s => s.utilizationPercent))
    const avgUtilization = linkTrafficStats.reduce((sum, s) => sum + s.utilizationPercent, 0) / linkTrafficStats.length

    return {
      chipMapping: [], // TopologyGraph 不使用此字段
      communicationGroups: [],
      linkTraffic,
      bottleneckLinks,
      maxUtilization,
      avgUtilization,
    }
  }, [linkTrafficStats])

  // 统计信息
  const statsInfo = useMemo(() => {
    if (!linkTrafficStats || linkTrafficStats.length === 0) return null

    const totalTraffic = linkTrafficStats.reduce((sum, s) => sum + s.trafficMb, 0)
    const avgUtilization = linkTrafficStats.reduce((sum, s) => sum + s.utilizationPercent, 0) / linkTrafficStats.length
    const maxUtilization = Math.max(...linkTrafficStats.map(s => s.utilizationPercent))
    const bottleneckLinks = linkTrafficStats.filter(s => s.utilizationPercent > 80).length

    return {
      totalTraffic,
      avgUtilization,
      maxUtilization,
      bottleneckLinks,
      totalLinks: linkTrafficStats.length,
    }
  }, [linkTrafficStats])

  return (
    <div style={{ height: height || '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 顶部控制栏 + 统计摘要（合并为一行） */}
      <div className="flex items-center justify-between mb-2 px-1 flex-shrink-0">
        {/* 图例 */}
        <div className="flex items-center gap-2 text-[11px]">
          <span className="text-gray-500">利用率:</span>
          <div className="flex items-center gap-0.5">
            <div className="w-4 h-1.5 rounded-sm" style={{ backgroundColor: '#52c41a' }} />
            <span>0-30%</span>
          </div>
          <div className="flex items-center gap-0.5">
            <div className="w-4 h-1.5 rounded-sm" style={{ backgroundColor: '#faad14' }} />
            <span>30-60%</span>
          </div>
          <div className="flex items-center gap-0.5">
            <div className="w-4 h-1.5 rounded-sm" style={{ backgroundColor: '#fa8c16' }} />
            <span>60-80%</span>
          </div>
          <div className="flex items-center gap-0.5">
            <div className="w-4 h-1.5 rounded-sm" style={{ backgroundColor: '#f5222d' }} />
            <span>80%+</span>
          </div>
        </div>
        {/* 统计摘要 */}
        {statsInfo && (
          <div className="flex items-center gap-4 text-[11px] text-gray-600">
            <span>
              流量: <span className="text-blue-600 font-medium">{statsInfo.totalTraffic.toFixed(0)} MB</span>
            </span>
            <span>
              平均: <span className="text-blue-600 font-medium">{statsInfo.avgUtilization.toFixed(0)}%</span>
            </span>
            <span>
              峰值: <span className="font-medium" style={{ color: getHeatmapColor(statsInfo.maxUtilization) }}>{statsInfo.maxUtilization.toFixed(0)}%</span>
            </span>
            <span>
              瓶颈: <span className="text-red-600 font-medium">{statsInfo.bottleneckLinks}/{statsInfo.totalLinks}</span>
            </span>
          </div>
        )}
      </div>

      {/* 2D 拓扑流量图 - 使用 TopologyGraph 组件，flex-grow 填满剩余空间 */}
      <div className="border border-gray-200 rounded-lg overflow-hidden flex-grow">
        <TopologyGraph
          visible={true}
          onClose={() => {}}
          topology={topologyWithConnections}
          currentLevel={currentLevel}
          currentPod={currentPod}
          currentRack={currentRack}
          currentBoard={currentBoard}
          embedded={true}
          trafficResult={trafficResult as TopologyTrafficResult}
          layoutType={layoutType}
          onLayoutTypeChange={setLayoutType}
          multiLevelOptions={multiLevelOptions}
          onMultiLevelOptionsChange={setMultiLevelOptions}
          breadcrumbs={breadcrumbs}
          onBreadcrumbClick={handleBreadcrumbClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          canGoBack={viewPath.length > 0}
          onNavigateBack={() => setViewPath(prev => prev.slice(0, -1))}
        />
      </div>
    </div>
  )
}
