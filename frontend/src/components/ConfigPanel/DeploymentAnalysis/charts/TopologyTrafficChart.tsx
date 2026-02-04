/**
 * 拓扑流量图表组件
 *
 * 在拓扑图上可视化链路流量和带宽利用率
 * 复用 TopologyGraph 组件实现 2D 拓扑可视化
 */

import React, { useMemo } from 'react'
import type { HierarchicalTopology, PodConfig, RackConfig, BoardConfig } from '@/types'
import type { LinkTrafficStats, TopologyTrafficResult } from '@/utils/llmDeployment/types'
import { TopologyGraph } from '@/components/TopologyGraph'

interface TopologyTrafficChartProps {
  topology: HierarchicalTopology | null
  linkTrafficStats: LinkTrafficStats[] | null
  height?: number
}

export const TopologyTrafficChart: React.FC<TopologyTrafficChartProps> = ({
  topology,
  linkTrafficStats,
  height,
}) => {
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
            // 获取芯片型号名称（后端存储在 name 字段，如 SG2262）
            const chipAny = chip as any
            const chipName = chipAny.name || chip.type || 'Chip'
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

  // 从修改后的拓扑中提取层级信息（确保使用修改后的 chip.label）
  const { currentPod, currentRack, currentBoard } = useMemo(() => {
    if (!topologyWithConnections?.pods?.[0]) {
      return { currentPod: null, currentRack: null, currentBoard: null }
    }
    const pod = topologyWithConnections.pods[0] as PodConfig
    const rack = pod.racks?.[0] as RackConfig | undefined
    const board = rack?.boards?.[0] as BoardConfig | undefined
    return {
      currentPod: pod,
      currentRack: rack ?? null,
      currentBoard: board ?? null,
    }
  }, [topologyWithConnections])

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

  // 热力图颜色函数
  const getHeatmapColor = (utilizationPercent: number): string => {
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
          currentLevel="board"
          currentPod={currentPod}
          currentRack={currentRack}
          currentBoard={currentBoard}
          embedded={true}
          trafficResult={trafficResult as TopologyTrafficResult}
          layoutType="auto"
        />
      </div>
    </div>
  )
}
