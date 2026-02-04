/**
 * 拓扑流量图表组件
 *
 * 在拓扑图上可视化链路流量和带宽利用率
 */

import React, { useState, useMemo } from 'react'
import type { HierarchicalTopology } from '@/types'
import type { LinkTrafficStats } from '@/utils/llmDeployment/types'

interface TopologyTrafficChartProps {
  topology: HierarchicalTopology | null
  linkTrafficStats: LinkTrafficStats[] | null
  height?: number
}

export const TopologyTrafficChart: React.FC<TopologyTrafficChartProps> = ({
  topology,
  linkTrafficStats,
  height = 600,
}) => {
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')

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

  // 构建流量映射 (source-target -> LinkTrafficStats)
  const trafficMap = useMemo(() => {
    const map = new Map<string, LinkTrafficStats>()
    linkTrafficStats.forEach(stat => {
      // 双向映射（无向图）
      map.set(`${stat.source}-${stat.target}`, stat)
      map.set(`${stat.target}-${stat.source}`, stat)
    })
    return map
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

  // 计算连线宽度（2-6px）
  const getLineWidth = (utilizationPercent: number): number => {
    const u = Math.min(Math.max(utilizationPercent, 0), 100)
    return 2 + (u / 100) * 4
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
    <div style={{ height }}>
      {/* 顶部控制栏 */}
      <div className="flex items-center justify-between mb-3 px-1">
        {/* 视图切换 */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('2d')}
            className={`px-3 py-1 text-sm rounded transition-colors ${
              viewMode === '2d'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            2D 视图
          </button>
          <button
            onClick={() => setViewMode('3d')}
            className={`px-3 py-1 text-sm rounded transition-colors ${
              viewMode === '3d'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            3D 视图
          </button>
        </div>

        {/* 图例 */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-6 h-2" style={{ backgroundColor: '#52c41a' }} />
            <span>0-30%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-2" style={{ backgroundColor: '#faad14' }} />
            <span>30-60%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-2" style={{ backgroundColor: '#fa8c16' }} />
            <span>60-80%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-2" style={{ backgroundColor: '#f5222d' }} />
            <span>80-100%</span>
          </div>
        </div>
      </div>

      {/* 统计摘要 */}
      {statsInfo && (
        <div className="mb-3 px-1">
          <div className="flex items-center gap-6 text-sm text-gray-600">
            <div>
              <span className="font-medium">总流量:</span>{' '}
              <span className="text-blue-600 font-semibold">
                {statsInfo.totalTraffic.toFixed(1)} MB
              </span>
            </div>
            <div>
              <span className="font-medium">平均利用率:</span>{' '}
              <span className="text-blue-600 font-semibold">
                {statsInfo.avgUtilization.toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="font-medium">峰值利用率:</span>{' '}
              <span
                className="font-semibold"
                style={{ color: getHeatmapColor(statsInfo.maxUtilization) }}
              >
                {statsInfo.maxUtilization.toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="font-medium">瓶颈链路:</span>{' '}
              <span className="text-red-600 font-semibold">
                {statsInfo.bottleneckLinks} / {statsInfo.totalLinks}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 拓扑图占位符 - 待与 TopologyGraph 集成 */}
      <div className="border border-gray-200 rounded-lg bg-gray-50 flex items-center justify-center" style={{ height: height - 120 }}>
        <div className="text-center text-gray-500">
          <div className="mb-2 text-lg font-medium">拓扑流量可视化</div>
          <div className="text-sm">
            {linkTrafficStats.length} 条链路 | {viewMode === '2d' ? '2D' : '3D'} 模式
          </div>
          <div className="mt-4 text-xs text-gray-400">
            （需要集成 TopologyGraph 组件并实现自定义边渲染器）
          </div>
        </div>
      </div>

      {/* 详细列表（折叠） */}
      <details className="mt-3">
        <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
          详细链路列表 ({linkTrafficStats.length} 条)
        </summary>
        <div className="mt-2 max-h-60 overflow-y-auto">
          <table className="w-full text-xs border-collapse">
            <thead className="bg-gray-100 sticky top-0">
              <tr>
                <th className="px-2 py-1 text-left border">源芯片</th>
                <th className="px-2 py-1 text-left border">目标芯片</th>
                <th className="px-2 py-1 text-right border">流量 (MB)</th>
                <th className="px-2 py-1 text-right border">带宽 (Gbps)</th>
                <th className="px-2 py-1 text-right border">利用率</th>
                <th className="px-2 py-1 text-center border">类型</th>
              </tr>
            </thead>
            <tbody>
              {linkTrafficStats
                .sort((a, b) => b.utilizationPercent - a.utilizationPercent)
                .map((stat, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-2 py-1 border font-mono text-[10px]">{stat.source}</td>
                    <td className="px-2 py-1 border font-mono text-[10px]">{stat.target}</td>
                    <td className="px-2 py-1 border text-right">{stat.trafficMb.toFixed(1)}</td>
                    <td className="px-2 py-1 border text-right">{stat.bandwidthGbps.toFixed(0)}</td>
                    <td
                      className="px-2 py-1 border text-right font-semibold"
                      style={{ color: getHeatmapColor(stat.utilizationPercent) }}
                    >
                      {stat.utilizationPercent.toFixed(1)}%
                    </td>
                    <td className="px-2 py-1 border text-center">
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-200">
                        {stat.linkType.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
