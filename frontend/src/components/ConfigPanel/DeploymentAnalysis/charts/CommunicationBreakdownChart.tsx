/**
 * 通信分析视图
 *
 * 展示 TP/PP/EP/SP 通信时间占比和详细分析
 */

import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import type { GanttChartData, CommTypeBreakdown } from '../../../../utils/llmDeployment/types'
import {
  aggregateCommByType,
  formatBytes,
  formatTime,
  formatPercent,
  TIME_BREAKDOWN_COLORS,
} from '../../../../utils/llmDeployment/ganttDataUtils'

interface CommunicationBreakdownChartProps {
  data: GanttChartData | null
  height?: number
}

/** 通信类型配置 */
const COMM_TYPE_CONFIG = {
  tp: {
    label: 'TP 通信',
    color: TIME_BREAKDOWN_COLORS.tp,
    description: '张量并行 (AllReduce)',
  },
  pp: {
    label: 'PP 通信',
    color: TIME_BREAKDOWN_COLORS.pp,
    description: '流水线并行 (P2P Send/Recv)',
  },
  ep: {
    label: 'EP 通信',
    color: TIME_BREAKDOWN_COLORS.ep,
    description: '专家并行 (AllToAll)',
  },
  sp: {
    label: 'SP 通信',
    color: TIME_BREAKDOWN_COLORS.sp,
    description: '序列并行 (AllGather/ReduceScatter)',
  },
}

/** 环形图组件 */
const DonutChart: React.FC<{
  data: CommTypeBreakdown
  size?: number
}> = ({ data, size = 180 }) => {
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null)

  const center = size / 2
  const outerRadius = size / 2 - 10
  const innerRadius = outerRadius * 0.6
  const strokeWidth = outerRadius - innerRadius

  // 计算各段的角度
  const segments = useMemo(() => {
    const total = data.totalTime
    if (total === 0) return []

    const result: Array<{
      type: 'tp' | 'pp' | 'ep' | 'sp'
      startAngle: number
      endAngle: number
      value: number
      percent: number
    }> = []

    let currentAngle = -Math.PI / 2 // 从顶部开始

    const types: Array<'tp' | 'pp' | 'ep' | 'sp'> = ['tp', 'pp', 'ep', 'sp']
    for (const type of types) {
      const value = data.breakdown[type].time
      if (value <= 0) continue

      const percent = value / total
      const angle = percent * 2 * Math.PI

      result.push({
        type,
        startAngle: currentAngle,
        endAngle: currentAngle + angle,
        value,
        percent,
      })

      currentAngle += angle
    }

    return result
  }, [data])

  // 绘制圆弧路径
  const getArcPath = (startAngle: number, endAngle: number, isHovered: boolean) => {
    const r = isHovered ? outerRadius + 4 : outerRadius - strokeWidth / 2
    const startX = center + r * Math.cos(startAngle)
    const startY = center + r * Math.sin(startAngle)
    const endX = center + r * Math.cos(endAngle)
    const endY = center + r * Math.sin(endAngle)
    const largeArcFlag = endAngle - startAngle > Math.PI ? 1 : 0

    return `M ${center} ${center} L ${startX} ${startY} A ${r} ${r} 0 ${largeArcFlag} 1 ${endX} ${endY} Z`
  }

  if (data.totalTime === 0) {
    return (
      <div className="flex items-center justify-center" style={{ width: size, height: size }}>
        <span className="text-sm text-text-muted">无通信数据</span>
      </div>
    )
  }

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* 背景圆环 */}
      <circle
        cx={center}
        cy={center}
        r={outerRadius - strokeWidth / 2}
        fill="none"
        stroke="#f0f0f0"
        strokeWidth={strokeWidth}
      />

      {/* 各段 */}
      {segments.map((seg) => {
        const isHovered = hoveredSegment === seg.type
        return (
          <path
            key={seg.type}
            d={getArcPath(seg.startAngle, seg.endAngle, isHovered)}
            fill={COMM_TYPE_CONFIG[seg.type].color}
            opacity={hoveredSegment && !isHovered ? 0.5 : 1}
            style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
            onMouseEnter={() => setHoveredSegment(seg.type)}
            onMouseLeave={() => setHoveredSegment(null)}
          />
        )
      })}

      {/* 内圆 (遮挡形成环形) */}
      <circle
        cx={center}
        cy={center}
        r={innerRadius}
        fill="white"
      />

      {/* 中心文字 */}
      <text
        x={center}
        y={center - 8}
        textAnchor="middle"
        fontSize={12}
        fill="#666"
      >
        总通信
      </text>
      <text
        x={center}
        y={center + 10}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill="#333"
      >
        {formatTime(data.totalTime)}
      </text>

      {/* 悬浮时显示类型信息 */}
      {hoveredSegment && (
        <text
          x={center}
          y={center + 28}
          textAnchor="middle"
          fontSize={11}
          fill={COMM_TYPE_CONFIG[hoveredSegment as 'tp' | 'pp' | 'ep' | 'sp'].color}
        >
          {COMM_TYPE_CONFIG[hoveredSegment as 'tp' | 'pp' | 'ep' | 'sp'].label}
        </text>
      )}
    </svg>
  )
}

/** 通信详情卡片 */
const CommDetailCard: React.FC<{
  type: 'tp' | 'pp' | 'ep' | 'sp'
  detail: CommTypeBreakdown['breakdown']['tp']
  totalTime: number
}> = ({ type, detail, totalTime }) => {
  const config = COMM_TYPE_CONFIG[type]
  const percent = totalTime > 0 ? (detail.time / totalTime) * 100 : 0

  if (detail.time === 0 && detail.count === 0) {
    return null
  }

  return (
    <div className="rounded-lg border border-gray-100 bg-bg-surface p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className="h-3 w-3 rounded-sm"
            style={{ backgroundColor: config.color }}
          />
          <span className="font-medium text-sm text-text-primary">{config.label}</span>
        </div>
        <Badge variant="outline" className="text-xs">
          {formatPercent(percent / 100)}
        </Badge>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-text-muted">时间</span>
          <span className="text-text-primary font-medium">{formatTime(detail.time)}</span>
        </div>
        <Progress value={percent} className="h-1.5" />

        <div className="flex justify-between text-xs">
          <span className="text-text-muted">数据量</span>
          <span className="text-text-primary">{formatBytes(detail.volume)}</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-text-muted">通信次数</span>
          <span className="text-text-primary">{detail.count}</span>
        </div>

        {detail.algorithm && (
          <div className="flex justify-between text-xs">
            <span className="text-text-muted">算法</span>
            <span className="text-text-primary">{detail.algorithm}</span>
          </div>
        )}

        <div className="text-[10px] text-text-muted mt-1">
          {config.description}
        </div>
      </div>
    </div>
  )
}

export const CommunicationBreakdownChart: React.FC<CommunicationBreakdownChartProps> = ({
  data,
  height = 400,
}) => {
  // 聚合通信数据
  const commData = useMemo(() => {
    if (!data) return null
    return aggregateCommByType(data.tasks)
  }, [data])

  if (!data || data.tasks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-gray-400">
        <div className="text-sm">运行模拟以生成通信分析</div>
      </div>
    )
  }

  if (!commData || commData.totalTime === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-gray-400">
        <div className="text-sm">无通信数据 (可能是单芯片配置)</div>
      </div>
    )
  }

  // 计算带宽利用率 (基于数据量和时间的粗略估算)
  const totalVolume =
    commData.breakdown.tp.volume +
    commData.breakdown.pp.volume +
    commData.breakdown.ep.volume +
    commData.breakdown.sp.volume

  // 找出主要通信类型
  const dominantType = (['tp', 'pp', 'ep', 'sp'] as const).reduce((prev, curr) =>
    commData.breakdown[curr].time > commData.breakdown[prev].time ? curr : prev
  )

  return (
    <div style={{ minHeight: height }}>
      {/* 标题和摘要 */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-sm font-medium text-text-primary">通信时间分布</div>
          <div className="text-xs text-text-muted mt-0.5">
            主要瓶颈: <span style={{ color: COMM_TYPE_CONFIG[dominantType].color }}>{COMM_TYPE_CONFIG[dominantType].label}</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm font-medium text-text-primary">{formatTime(commData.totalTime)}</div>
          <div className="text-xs text-text-muted">总通信时间</div>
        </div>
      </div>

      {/* 主要内容区域 */}
      <div className="flex gap-6">
        {/* 左侧：环形图 */}
        <div className="flex-shrink-0 flex flex-col items-center">
          <DonutChart data={commData} size={180} />

          {/* 图例 */}
          <div className="mt-4 grid grid-cols-2 gap-2">
            {(['tp', 'pp', 'ep', 'sp'] as const).map((type) => {
              const detail = commData.breakdown[type]
              if (detail.time === 0) return null
              return (
                <div key={type} className="flex items-center gap-1.5 text-xs">
                  <div
                    className="h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: COMM_TYPE_CONFIG[type].color }}
                  />
                  <span className="text-text-muted">{COMM_TYPE_CONFIG[type].label}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* 右侧：详情卡片 */}
        <div className="flex-1 grid grid-cols-2 gap-3">
          {(['tp', 'pp', 'ep', 'sp'] as const).map((type) => (
            <CommDetailCard
              key={type}
              type={type}
              detail={commData.breakdown[type]}
              totalTime={commData.totalTime}
            />
          ))}
        </div>
      </div>

      {/* 瓶颈层提示 */}
      {commData.bottleneckLayers.length > 0 && (
        <div className="mt-4 p-3 rounded-lg bg-yellow-50 border border-yellow-200">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-yellow-700 font-medium">通信瓶颈层</span>
            <div className="flex gap-1.5">
              {commData.bottleneckLayers.map((layer) => (
                <Badge key={layer} variant="warning" className="text-xs">
                  Layer {layer}
                </Badge>
              ))}
            </div>
          </div>
          <div className="text-xs text-yellow-600 mt-1">
            这些层的通信开销最大，可考虑调整并行策略或优化通信算法
          </div>
        </div>
      )}

      {/* 统计摘要 */}
      <div className="mt-4 grid grid-cols-4 gap-3">
        <div className="text-center p-2 rounded-lg bg-bg-surface">
          <div className="text-lg font-semibold text-text-primary">
            {formatBytes(totalVolume)}
          </div>
          <div className="text-xs text-text-muted">总数据量</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-bg-surface">
          <div className="text-lg font-semibold text-text-primary">
            {commData.breakdown.tp.count +
              commData.breakdown.pp.count +
              commData.breakdown.ep.count +
              commData.breakdown.sp.count}
          </div>
          <div className="text-xs text-text-muted">通信次数</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-bg-surface">
          <div className="text-lg font-semibold" style={{ color: COMM_TYPE_CONFIG[dominantType].color }}>
            {COMM_TYPE_CONFIG[dominantType].label}
          </div>
          <div className="text-xs text-text-muted">主要类型</div>
        </div>
        <div className="text-center p-2 rounded-lg bg-bg-surface">
          <div className="text-lg font-semibold text-text-primary">
            {commData.bottleneckLayers.length}
          </div>
          <div className="text-xs text-text-muted">瓶颈层数</div>
        </div>
      </div>
    </div>
  )
}

export default CommunicationBreakdownChart
