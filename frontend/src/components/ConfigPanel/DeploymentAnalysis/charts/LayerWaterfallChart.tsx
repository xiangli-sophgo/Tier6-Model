/**
 * 层级瀑布图
 *
 * 水平堆叠条形图展示每层的时间分解（计算/访存/通信）
 */

import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { GanttChartData, LayerBreakdown } from '../../../../utils/llmDeployment/types'
import {
  formatTime,
  formatPercent,
} from '../../../../utils/formatters'
import {
  aggregateTasksByLayer,
  TIME_BREAKDOWN_COLORS,
  TIME_BREAKDOWN_LABELS,
} from '../../../../utils/llmDeployment/ganttDataUtils'

interface LayerWaterfallChartProps {
  data: GanttChartData | null
  height?: number
  onLayerClick?: (layerIndex: number) => void
}

/** 图表边距 */
const MARGIN = { top: 30, right: 60, bottom: 30, left: 60 }

/** 行高度 */
const ROW_HEIGHT = 24

/** 条形高度 */
const BAR_HEIGHT = 18

/** 悬浮提示框样式 - 统一浅色风格 */
const tooltipStyle: React.CSSProperties = {
  position: 'fixed',
  background: 'rgba(255, 255, 255, 0.98)',
  color: '#333',
  padding: '10px 14px',
  borderRadius: 8,
  border: '1px solid #e5e5e5',
  fontSize: 12,
  lineHeight: 1.6,
  pointerEvents: 'none',
  zIndex: 1000,
  minWidth: 220,
  maxWidth: 300,
  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
}

export const LayerWaterfallChart: React.FC<LayerWaterfallChartProps> = ({
  data,
  height: propHeight,
  onLayerClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)
  const [selectedPhase, setSelectedPhase] = useState<'prefill' | 'decode' | 'all'>('all')
  const [tooltip, setTooltip] = useState<{ layer: LayerBreakdown; x: number; y: number } | null>(null)
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null)

  // 监听容器宽度变化
  useEffect(() => {
    if (!containerRef.current) return
    setContainerWidth(containerRef.current.clientWidth)
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width
      if (width && width > 0) setContainerWidth(width)
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // 聚合数据
  const layerData = useMemo(() => {
    if (!data) return []
    const phase = selectedPhase === 'all' ? undefined : selectedPhase
    return aggregateTasksByLayer(data.tasks, phase)
  }, [data, selectedPhase])

  // 计算实际使用的类型（用于图例显示）
  const usedTypes = useMemo(() => {
    const types = new Set<string>()
    layerData.forEach(layer => {
      if (layer.computeTime > 0) types.add('compute')
      if (layer.memoryTime > 0) types.add('memory')
      if (layer.commTime.tp > 0) types.add('tp')
      if (layer.commTime.pp > 0) types.add('pp')
      if (layer.commTime.ep > 0) types.add('ep')
      if (layer.commTime.sp > 0) types.add('sp')
    })
    return types
  }, [layerData])

  // 计算最大时间用于比例尺
  const maxTime = useMemo(() => {
    if (layerData.length === 0) return 1
    return Math.max(...layerData.map(l => l.totalTime))
  }, [layerData])

  // 计算图表尺寸
  const chartWidth = containerWidth
  const chartHeight = propHeight ?? Math.max(200, MARGIN.top + MARGIN.bottom + layerData.length * ROW_HEIGHT)
  const innerWidth = chartWidth - MARGIN.left - MARGIN.right

  // X轴比例尺
  const xScale = useCallback((time: number) => {
    return MARGIN.left + (time / maxTime) * innerWidth
  }, [maxTime, innerWidth])

  // Y轴比例尺
  const yScale = useCallback((index: number) => {
    return MARGIN.top + index * ROW_HEIGHT
  }, [])

  // 悬浮处理（智能定位：右侧放不下时放到左侧）
  const handleLayerHover = useCallback((layer: LayerBreakdown, e: React.MouseEvent) => {
    const tooltipWidth = 240 // 预估 tooltip 宽度
    const offset = 10
    const windowWidth = window.innerWidth

    // 检测是否会超出右侧边界
    const wouldOverflowRight = e.clientX + tooltipWidth + offset > windowWidth

    // 如果超出右侧，放到鼠标左侧；否则放到右侧
    const x = wouldOverflowRight
      ? e.clientX - tooltipWidth - offset
      : e.clientX + offset

    setTooltip({ layer, x, y: e.clientY + offset })
    setHoveredLayer(layer.layerIndex)
  }, [])

  const handleLayerLeave = useCallback(() => {
    setTooltip(null)
    setHoveredLayer(null)
  }, [])

  // 点击处理
  const handleLayerClick = useCallback((layer: LayerBreakdown) => {
    if (onLayerClick) {
      onLayerClick(layer.layerIndex)
    }
  }, [onLayerClick])

  // 格式化时间刻度
  const formatTimeTick = (us: number): string => {
    if (us < 1000) return `${us.toFixed(0)}µs`
    if (us < 1000000) return `${(us / 1000).toFixed(1)}ms`
    return `${(us / 1000000).toFixed(1)}s`
  }

  // 生成时间刻度
  const timeTicks = useMemo(() => {
    const numTicks = Math.min(6, Math.floor(innerWidth / 80))
    const ticks: number[] = []
    for (let i = 0; i <= numTicks; i++) {
      ticks.push((maxTime / numTicks) * i)
    }
    return ticks
  }, [maxTime, innerWidth])

  if (!data || data.tasks.length === 0) {
    return (
      <div ref={containerRef} style={{ width: '100%' }}>
        <div className="flex flex-col items-center justify-center py-10 text-gray-400">
          <div className="text-sm">运行模拟以生成层级瀑布图</div>
        </div>
      </div>
    )
  }

  if (containerWidth === 0) {
    return <div ref={containerRef} style={{ width: '100%', height: 100 }} />
  }

  if (layerData.length === 0) {
    return (
      <div ref={containerRef} style={{ width: '100%' }}>
        <div className="flex flex-col items-center justify-center py-10 text-gray-400">
          <div className="text-sm">暂无层级数据</div>
        </div>
      </div>
    )
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {/* 工具栏 */}
      <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="text-[11px] text-gray-500">
          共 {layerData.length} 层 | 点击查看详情
        </span>
        <Select
          value={selectedPhase}
          onValueChange={(value) => setSelectedPhase(value as 'prefill' | 'decode' | 'all')}
        >
          <SelectTrigger className="w-[100px] h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">全部阶段</SelectItem>
            <SelectItem value="prefill">Prefill</SelectItem>
            <SelectItem value="decode">Decode</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* SVG 图表 */}
      <svg width={chartWidth} height={chartHeight}>
        {/* 背景网格 */}
        <g className="grid">
          {timeTicks.map((tick, i) => (
            <line
              key={i}
              x1={xScale(tick)}
              y1={MARGIN.top}
              x2={xScale(tick)}
              y2={chartHeight - MARGIN.bottom}
              stroke="#f0f0f0"
              strokeDasharray="3,3"
            />
          ))}
        </g>

        {/* 层标签和堆叠条形 */}
        {layerData.map((layer, index) => {
          const y = yScale(index)
          const barY = y + (ROW_HEIGHT - BAR_HEIGHT) / 2
          const isHovered = hoveredLayer === layer.layerIndex

          // 计算各部分的堆叠位置
          let currentX = MARGIN.left
          const segments: Array<{
            type: keyof typeof TIME_BREAKDOWN_COLORS
            x: number
            width: number
            time: number
          }> = []

          // 计算
          if (layer.computeTime > 0) {
            const width = (layer.computeTime / maxTime) * innerWidth
            segments.push({ type: 'compute', x: currentX, width, time: layer.computeTime })
            currentX += width
          }

          // 访存
          if (layer.memoryTime > 0) {
            const width = (layer.memoryTime / maxTime) * innerWidth
            segments.push({ type: 'memory', x: currentX, width, time: layer.memoryTime })
            currentX += width
          }

          // TP 通信
          if (layer.commTime.tp > 0) {
            const width = (layer.commTime.tp / maxTime) * innerWidth
            segments.push({ type: 'tp', x: currentX, width, time: layer.commTime.tp })
            currentX += width
          }

          // PP 通信
          if (layer.commTime.pp > 0) {
            const width = (layer.commTime.pp / maxTime) * innerWidth
            segments.push({ type: 'pp', x: currentX, width, time: layer.commTime.pp })
            currentX += width
          }

          // EP 通信
          if (layer.commTime.ep > 0) {
            const width = (layer.commTime.ep / maxTime) * innerWidth
            segments.push({ type: 'ep', x: currentX, width, time: layer.commTime.ep })
            currentX += width
          }

          // SP 通信
          if (layer.commTime.sp > 0) {
            const width = (layer.commTime.sp / maxTime) * innerWidth
            segments.push({ type: 'sp', x: currentX, width, time: layer.commTime.sp })
            currentX += width
          }

          return (
            <g
              key={`layer-${layer.layerIndex}-${layer.phase}`}
              style={{ cursor: onLayerClick ? 'pointer' : 'default' }}
              onMouseEnter={(e) => handleLayerHover(layer, e)}
              onMouseMove={(e) => handleLayerHover(layer, e)}
              onMouseLeave={handleLayerLeave}
              onClick={() => handleLayerClick(layer)}
            >
              {/* 背景高亮 */}
              {isHovered && (
                <rect
                  x={MARGIN.left}
                  y={barY - 2}
                  width={innerWidth}
                  height={BAR_HEIGHT + 4}
                  fill="rgba(24, 144, 255, 0.08)"
                  rx={4}
                />
              )}

              {/* 层标签 */}
              <text
                x={MARGIN.left - 8}
                y={y + ROW_HEIGHT / 2 + 4}
                textAnchor="end"
                fontSize={11}
                fill={isHovered ? '#1890ff' : '#333'}
                fontWeight={isHovered ? 600 : 500}
              >
                L{layer.layerIndex}
              </text>

              {/* 堆叠条形 */}
              {segments.map((seg, segIndex) => (
                <rect
                  key={segIndex}
                  x={seg.x}
                  y={barY}
                  width={Math.max(1, seg.width)}
                  height={BAR_HEIGHT}
                  fill={TIME_BREAKDOWN_COLORS[seg.type]}
                  opacity={isHovered ? 1 : 0.85}
                  rx={segIndex === 0 ? 3 : 0}
                  style={{
                    borderTopRightRadius: segIndex === segments.length - 1 ? 3 : 0,
                    borderBottomRightRadius: segIndex === segments.length - 1 ? 3 : 0,
                  }}
                />
              ))}

              {/* 总时间标签 */}
              <text
                x={xScale(layer.totalTime) + 6}
                y={y + ROW_HEIGHT / 2 + 4}
                fontSize={10}
                fill="#666"
              >
                {formatTimeTick(layer.totalTime)}
              </text>
            </g>
          )
        })}

        {/* X 轴 */}
        <g className="x-axis" transform={`translate(0, ${chartHeight - MARGIN.bottom})`}>
          <line
            x1={MARGIN.left}
            y1={0}
            x2={chartWidth - MARGIN.right}
            y2={0}
            stroke="#d9d9d9"
          />
          {timeTicks.map((tick, i) => (
            <g key={i} transform={`translate(${xScale(tick)}, 0)`}>
              <line y1={0} y2={5} stroke="#999" />
              <text
                y={18}
                textAnchor="middle"
                fontSize={10}
                fill="#666"
              >
                {formatTimeTick(tick)}
              </text>
            </g>
          ))}
        </g>
      </svg>

      {/* 悬浮提示框 - 统一浅色风格 */}
      {tooltip && (
        <div style={{ ...tooltipStyle, left: tooltip.x, top: tooltip.y }}>
          <div style={{ fontWeight: 600, marginBottom: 6, borderBottom: '1px solid #e5e5e5', paddingBottom: 4 }}>
            Layer {tooltip.layer.layerIndex}
            {tooltip.layer.phase && (
              <span style={{ marginLeft: 8, fontWeight: 400, color: '#666' }}>
                ({tooltip.layer.phase})
              </span>
            )}
          </div>
          <div style={{ marginBottom: 4 }}>总时间: {formatTime(tooltip.layer.totalTime)}</div>
          <div style={{ marginBottom: 4 }}>任务数: {tooltip.layer.taskCount}</div>
          <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 4, marginTop: 4 }}>
            {tooltip.layer.computeTime > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TIME_BREAKDOWN_COLORS.compute, flexShrink: 0 }} />
                <span style={{ minWidth: 60, whiteSpace: 'nowrap' }}>计算</span>
                <span style={{ color: '#666', textAlign: 'right', flex: 1, whiteSpace: 'nowrap' }}>
                  {formatTime(tooltip.layer.computeTime)} ({formatPercent(tooltip.layer.computeTime / tooltip.layer.totalTime)})
                </span>
              </div>
            )}
            {tooltip.layer.memoryTime > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TIME_BREAKDOWN_COLORS.memory, flexShrink: 0 }} />
                <span style={{ minWidth: 60, whiteSpace: 'nowrap' }}>访存</span>
                <span style={{ color: '#666', textAlign: 'right', flex: 1, whiteSpace: 'nowrap' }}>
                  {formatTime(tooltip.layer.memoryTime)} ({formatPercent(tooltip.layer.memoryTime / tooltip.layer.totalTime)})
                </span>
              </div>
            )}
            {tooltip.layer.commTime.tp > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TIME_BREAKDOWN_COLORS.tp, flexShrink: 0 }} />
                <span style={{ minWidth: 60, whiteSpace: 'nowrap' }}>TP通信</span>
                <span style={{ color: '#666', textAlign: 'right', flex: 1, whiteSpace: 'nowrap' }}>
                  {formatTime(tooltip.layer.commTime.tp)} ({formatPercent(tooltip.layer.commTime.tp / tooltip.layer.totalTime)})
                </span>
              </div>
            )}
            {tooltip.layer.commTime.pp > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TIME_BREAKDOWN_COLORS.pp, flexShrink: 0 }} />
                <span style={{ minWidth: 60, whiteSpace: 'nowrap' }}>PP通信</span>
                <span style={{ color: '#666', textAlign: 'right', flex: 1, whiteSpace: 'nowrap' }}>
                  {formatTime(tooltip.layer.commTime.pp)} ({formatPercent(tooltip.layer.commTime.pp / tooltip.layer.totalTime)})
                </span>
              </div>
            )}
            {tooltip.layer.commTime.ep > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TIME_BREAKDOWN_COLORS.ep, flexShrink: 0 }} />
                <span style={{ minWidth: 60, whiteSpace: 'nowrap' }}>EP通信</span>
                <span style={{ color: '#666', textAlign: 'right', flex: 1, whiteSpace: 'nowrap' }}>
                  {formatTime(tooltip.layer.commTime.ep)} ({formatPercent(tooltip.layer.commTime.ep / tooltip.layer.totalTime)})
                </span>
              </div>
            )}
            {tooltip.layer.commTime.sp > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TIME_BREAKDOWN_COLORS.sp, flexShrink: 0 }} />
                <span style={{ minWidth: 60, whiteSpace: 'nowrap' }}>SP通信</span>
                <span style={{ color: '#666', textAlign: 'right', flex: 1, whiteSpace: 'nowrap' }}>
                  {formatTime(tooltip.layer.commTime.sp)} ({formatPercent(tooltip.layer.commTime.sp / tooltip.layer.totalTime)})
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 图例 - 只显示实际使用的类型 */}
      {usedTypes.size > 0 && (
        <div style={{
          marginTop: 8,
          padding: '8px 0',
          borderTop: '1px solid #f0f0f0',
          display: 'flex',
          gap: 16,
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}>
          {Object.entries(TIME_BREAKDOWN_LABELS)
            .filter(([key]) => usedTypes.has(key))
            .map(([key, label]) => (
              <span
                key={key}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}
              >
                <span
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: 2,
                    backgroundColor: TIME_BREAKDOWN_COLORS[key as keyof typeof TIME_BREAKDOWN_COLORS],
                  }}
                />
                <span style={{ color: '#666' }}>{label}</span>
              </span>
            ))}
        </div>
      )}
    </div>
  )
}

export default LayerWaterfallChart
