/**
 * 甘特图组件 - 展示 LLM 推理时序
 *
 * 优化版：
 * - 任务聚合：当任务过多时按时间窗口聚合显示
 * - 缩放功能：支持滚轮缩放和拖拽平移
 * - 自适应宽度：响应容器宽度变化
 */

import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react'
import { Empty, Typography, Button, Space, Tooltip } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, ReloadOutlined } from '@ant-design/icons'
import type { GanttChartData, GanttTask } from '../../../../utils/llmDeployment/simulation/types'

const { Text } = Typography

interface GanttChartProps {
  data: GanttChartData | null
  showLegend?: boolean
}

/** 任务类型颜色映射 */
const TASK_COLORS: Record<string, string> = {
  // 计算任务 - 绿色系
  compute: '#52c41a',
  embedding: '#73d13d',
  layernorm: '#95de64',
  attention_qkv: '#389e0d',
  attention_score: '#52c41a',
  attention_softmax: '#73d13d',
  attention_output: '#95de64',
  ffn_gate: '#237804',
  ffn_up: '#389e0d',
  ffn_down: '#52c41a',
  lm_head: '#135200',
  // 数据搬运 - 橙色系
  pcie_h2d: '#fa8c16',
  pcie_d2h: '#ffa940',
  hbm_write: '#ffc53d',
  hbm_read: '#ffd666',
  weight_load: '#d48806',
  kv_cache_read: '#faad14',
  kv_cache_write: '#ffc53d',
  // 通信 - 蓝/紫色系
  tp_comm: '#1890ff',
  pp_comm: '#722ed1',
  ep_comm: '#eb2f96',
  // SP 通信 - 蓝色系 (序列并行)
  sp_allgather: '#2f54eb',
  sp_reduce_scatter: '#1d39c4',
  // DP 通信 - 深紫色 (数据并行梯度同步)
  dp_gradient_sync: '#531dab',
  // MLA - 青色系
  rmsnorm_q_lora: '#13c2c2',
  rmsnorm_kv_lora: '#36cfc9',
  mm_q_lora_a: '#5cdbd3',
  mm_q_lora_b: '#87e8de',
  mm_kv_lora_a: '#b5f5ec',
  attn_fc: '#08979c',
  bmm_qk: '#006d75',
  bmm_sv: '#00474f',
  // MoE - 品红色系
  moe_gate: '#f759ab',
  moe_expert: '#eb2f96',
  moe_shared_expert: '#c41d7f',
  ep_dispatch: '#9254de',
  ep_combine: '#722ed1',
  // 其他
  bubble: '#ff4d4f',
  idle: '#d9d9d9',
}

/** 任务类型标签 */
const TASK_LABELS: Record<string, string> = {
  compute: '计算',
  embedding: 'Embed',
  layernorm: 'LN',
  attention_qkv: 'QKV',
  attention_score: 'Score',
  attention_softmax: 'Softmax',
  attention_output: 'AttnOut',
  ffn_gate: 'Gate',
  ffn_up: 'Up',
  ffn_down: 'Down',
  lm_head: 'LMHead',
  pcie_h2d: 'H2D',
  pcie_d2h: 'D2H',
  hbm_write: 'HBM写',
  hbm_read: 'HBM读',
  weight_load: '权重',
  kv_cache_read: 'KV读',
  kv_cache_write: 'KV写',
  tp_comm: 'TP',
  pp_comm: 'PP',
  ep_comm: 'EP',
  sp_allgather: 'SP AG',
  sp_reduce_scatter: 'SP RS',
  dp_gradient_sync: 'DP Sync',
  moe_gate: 'MoE Gate',
  moe_expert: 'Expert',
  moe_shared_expert: 'Shared',
  ep_dispatch: 'Dispatch',
  ep_combine: 'Combine',
  bubble: '气泡',
  idle: '空闲',
}

/** 图例分组配置 */
const LEGEND_GROUPS = [
  { name: '计算', types: ['compute', 'attention_qkv', 'ffn_gate', 'lm_head'] },
  { name: '数据搬运', types: ['pcie_h2d', 'weight_load', 'kv_cache_read'] },
  { name: '通信', types: ['tp_comm', 'pp_comm', 'ep_comm', 'sp_allgather', 'dp_gradient_sync'] },
  { name: 'MoE', types: ['moe_gate', 'moe_expert', 'ep_dispatch'] },
  { name: '其他', types: ['bubble', 'idle'] },
]

/** 图表边距 */
const MARGIN = { top: 25, right: 20, bottom: 25, left: 80 }

/** 滚动条高度 */
const SCROLLBAR_HEIGHT = 10

/** 资源行高度 */
const ROW_HEIGHT = 28

/** 任务条高度 */
const BAR_HEIGHT = 22

/** 聚合后的任务段 */
interface AggregatedSegment {
  binIndex: number
  resourceId: string
  startTime: number
  endTime: number
  typeBreakdown: Map<string, number> // 各类型占比
  taskCount: number
  tasks: GanttTask[] // 原始任务引用
}

/** 悬浮提示框样式 */
const tooltipStyle: React.CSSProperties = {
  position: 'fixed',
  background: 'rgba(0, 0, 0, 0.9)',
  color: '#fff',
  padding: '10px 14px',
  borderRadius: 8,
  fontSize: 12,
  lineHeight: 1.6,
  pointerEvents: 'none',
  zIndex: 1000,
  maxWidth: 300,
  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
}

export const GanttChart: React.FC<GanttChartProps> = ({
  data,
  showLegend = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)
  const [tooltip, setTooltip] = useState<{ segment: AggregatedSegment; x: number; y: number } | null>(null)

  // 缩放和平移状态
  const [zoom, setZoom] = useState(1)
  const [panOffset, setPanOffset] = useState(0) // 时间轴偏移(ms)
  const [isDragging, setIsDragging] = useState(false)
  const [isScrollbarDragging, setIsScrollbarDragging] = useState(false)
  const dragStartRef = useRef({ x: 0, offset: 0 })

  const svgRef = useRef<SVGSVGElement>(null)

  // 监听容器宽度变化
  useEffect(() => {
    if (!containerRef.current) return
    // 初始化宽度
    setContainerWidth(containerRef.current.clientWidth)
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width
      if (width && width > 0) setContainerWidth(width)
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // 原生wheel事件监听（passive: false 才能阻止浏览器默认缩放）
  useEffect(() => {
    const svg = svgRef.current
    if (!svg || !data) return

    const handleNativeWheel = (e: WheelEvent) => {
      const totalDuration = data.timeRange.end - data.timeRange.start

      // Ctrl + 滚轮：缩放
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        if (e.deltaY < 0) {
          setZoom((z) => Math.min(z * 1.2, 50))
        } else {
          setZoom((z) => {
            const newZoom = Math.max(z / 1.2, 1)
            const newVisibleDuration = totalDuration / newZoom
            setPanOffset((offset) => Math.max(0, Math.min(offset, totalDuration - newVisibleDuration)))
            return newZoom
          })
        }
        return
      }

      // Shift + 滚轮：水平平移
      if (e.shiftKey) {
        setZoom((currentZoom) => {
          if (currentZoom > 1) {
            e.preventDefault()
            const visibleDuration = totalDuration / currentZoom
            const panStep = visibleDuration * 0.1
            const delta = e.deltaY > 0 ? panStep : -panStep
            setPanOffset((offset) => Math.max(0, Math.min(offset + delta, totalDuration - visibleDuration)))
          }
          return currentZoom
        })
      }
    }

    svg.addEventListener('wheel', handleNativeWheel, { passive: false })
    return () => svg.removeEventListener('wheel', handleNativeWheel)
  }, [data])

  // 计算图表尺寸 - 根据实际资源数量计算高度
  const chartWidth = containerWidth
  const chartHeight = data
    ? MARGIN.top + MARGIN.bottom + data.resources.length * ROW_HEIGHT
    : 100
  const innerWidth = chartWidth - MARGIN.left - MARGIN.right

  // 计算可见时间范围
  const visibleTimeRange = useMemo(() => {
    if (!data) return { start: 0, end: 1 }
    const totalDuration = data.timeRange.end - data.timeRange.start
    const visibleDuration = totalDuration / zoom
    const start = data.timeRange.start + panOffset
    // 限制边界
    const clampedStart = Math.max(data.timeRange.start, Math.min(start, data.timeRange.end - visibleDuration))
    const clampedEnd = clampedStart + visibleDuration
    return { start: clampedStart, end: clampedEnd }
  }, [data, zoom, panOffset])

  // X轴比例尺
  const xScale = useCallback((time: number) => {
    const { start, end } = visibleTimeRange
    return MARGIN.left + ((time - start) / (end - start)) * innerWidth
  }, [visibleTimeRange, innerWidth])

  // Y轴比例尺
  const yScale = useCallback((resourceIndex: number) => {
    return MARGIN.top + resourceIndex * ROW_HEIGHT
  }, [])

  // 资源索引映射
  const resourceIndexMap = useMemo(() => {
    if (!data) return new Map<string, number>()
    const map = new Map<string, number>()
    data.resources.forEach((r, i) => map.set(r.id, i))
    return map
  }, [data])

  // 聚合任务数据
  const aggregatedData = useMemo(() => {
    if (!data) return []

    const { start, end } = visibleTimeRange
    const duration = end - start

    // 根据缩放级别决定bin数量
    const numBins = Math.min(Math.max(50, Math.floor(innerWidth / 3)), 300)
    const binWidth = duration / numBins

    // 初始化bins: resourceId -> binIndex -> tasks
    const bins = new Map<string, Map<number, GanttTask[]>>()
    for (const resource of data.resources) {
      bins.set(resource.id, new Map())
    }

    // 将任务分配到bins
    for (const task of data.tasks) {
      // 跳过不在可见范围内的任务
      if (task.end < start || task.start > end) continue

      // 确定资源行
      const isNetworkTask = ['tp_comm', 'pp_comm', 'ep_comm', 'ep_dispatch', 'ep_combine', 'sp_allgather', 'sp_reduce_scatter', 'dp_gradient_sync'].includes(task.type)
      const resourceId = `stage${task.ppStage}_${isNetworkTask ? 'network' : 'compute'}`

      const resourceBins = bins.get(resourceId)
      if (!resourceBins) continue

      // 计算任务覆盖的bin范围
      const taskStart = Math.max(task.start, start)
      const taskEnd = Math.min(task.end, end)
      const startBin = Math.floor((taskStart - start) / binWidth)
      const endBin = Math.floor((taskEnd - start) / binWidth)

      for (let bin = startBin; bin <= endBin && bin < numBins; bin++) {
        if (bin < 0) continue
        if (!resourceBins.has(bin)) resourceBins.set(bin, [])
        resourceBins.get(bin)!.push(task)
      }
    }

    // 生成聚合段
    const segments: AggregatedSegment[] = []

    for (const [resourceId, resourceBins] of bins) {
      // 合并连续的非空bins
      let currentSegment: AggregatedSegment | null = null

      for (let bin = 0; bin < numBins; bin++) {
        const binTasks = resourceBins.get(bin)

        if (binTasks && binTasks.length > 0) {
          if (!currentSegment) {
            currentSegment = {
              binIndex: bin,
              resourceId,
              startTime: start + bin * binWidth,
              endTime: start + (bin + 1) * binWidth,
              typeBreakdown: new Map(),
              taskCount: 0,
              tasks: [],
            }
          } else {
            currentSegment.endTime = start + (bin + 1) * binWidth
          }

          // 统计类型分布
          for (const task of binTasks) {
            const type = task.type
            const existing = currentSegment.typeBreakdown.get(type) || 0
            currentSegment.typeBreakdown.set(type, existing + (task.end - task.start))
            if (!currentSegment.tasks.includes(task)) {
              currentSegment.tasks.push(task)
              currentSegment.taskCount++
            }
          }
        } else {
          // bin为空，结束当前段
          if (currentSegment) {
            segments.push(currentSegment)
            currentSegment = null
          }
        }
      }

      // 添加最后一个段
      if (currentSegment) {
        segments.push(currentSegment)
      }
    }

    return segments
  }, [data, visibleTimeRange, innerWidth])

  // 格式化时间
  const formatTime = (ms: number): string => {
    if (ms < 0.001) return `${(ms * 1000000).toFixed(0)}ns`
    if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`
    if (ms < 1000) return `${ms.toFixed(2)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  // 生成时间刻度
  const timeTicksData = useMemo(() => {
    const { start, end } = visibleTimeRange
    const duration = end - start
    const numTicks = Math.min(8, Math.floor(innerWidth / 80))

    const ticks: number[] = []
    for (let i = 0; i <= numTicks; i++) {
      ticks.push(start + (duration / numTicks) * i)
    }
    return ticks
  }, [visibleTimeRange, innerWidth])

  // 缩放处理
  const handleZoomIn = useCallback(() => {
    setZoom((z) => Math.min(z * 1.5, 50))
  }, [])

  const handleZoomOut = useCallback(() => {
    setZoom((z) => Math.max(z / 1.5, 1))
    setPanOffset((offset) => {
      if (!data) return offset
      const totalDuration = data.timeRange.end - data.timeRange.start
      const newVisibleDuration = totalDuration / Math.max(zoom / 1.5, 1)
      return Math.min(offset, totalDuration - newVisibleDuration)
    })
  }, [data, zoom])

  const handleReset = useCallback(() => {
    setZoom(1)
    setPanOffset(0)
  }, [])

  // 拖拽平移
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom <= 1) return
    setIsDragging(true)
    dragStartRef.current = { x: e.clientX, offset: panOffset }
  }, [zoom, panOffset])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !data) return
    const dx = e.clientX - dragStartRef.current.x
    const totalDuration = data.timeRange.end - data.timeRange.start
    const visibleDuration = totalDuration / zoom
    const pxPerMs = innerWidth / visibleDuration
    const newOffset = dragStartRef.current.offset - dx / pxPerMs
    setPanOffset(Math.max(0, Math.min(newOffset, totalDuration - visibleDuration)))
  }, [isDragging, data, zoom, innerWidth])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    setIsScrollbarDragging(false)
  }, [])

  // 滚动条拖拽
  const handleScrollbarMouseDown = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    setIsScrollbarDragging(true)
    dragStartRef.current = { x: e.clientX, offset: panOffset }
  }, [panOffset])

  const handleScrollbarMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isScrollbarDragging || !data) return
    const dx = e.clientX - dragStartRef.current.x
    const totalDuration = data.timeRange.end - data.timeRange.start
    const visibleDuration = totalDuration / zoom
    const scrollbarWidth = innerWidth
    const thumbWidth = scrollbarWidth / zoom
    const maxThumbOffset = scrollbarWidth - thumbWidth
    const pxPerMs = maxThumbOffset / (totalDuration - visibleDuration)
    const newOffset = dragStartRef.current.offset + dx / pxPerMs
    setPanOffset(Math.max(0, Math.min(newOffset, totalDuration - visibleDuration)))
  }, [isScrollbarDragging, data, zoom, innerWidth])

  // 点击滚动条轨道跳转
  const handleScrollbarTrackClick = useCallback((e: React.MouseEvent<SVGRectElement>) => {
    if (!data || zoom <= 1) return
    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const totalDuration = data.timeRange.end - data.timeRange.start
    const visibleDuration = totalDuration / zoom
    const ratio = clickX / innerWidth
    const targetOffset = ratio * totalDuration - visibleDuration / 2
    setPanOffset(Math.max(0, Math.min(targetOffset, totalDuration - visibleDuration)))
  }, [data, zoom, innerWidth])

  // Tooltip处理
  const handleSegmentHover = useCallback((segment: AggregatedSegment, e: React.MouseEvent) => {
    setTooltip({ segment, x: e.clientX + 10, y: e.clientY + 10 })
  }, [])

  const handleSegmentLeave = useCallback(() => {
    setTooltip(null)
  }, [])

  // 渲染聚合段
  const renderSegment = useCallback((segment: AggregatedSegment) => {
    const resourceIndex = resourceIndexMap.get(segment.resourceId)
    if (resourceIndex === undefined) return null

    const x = xScale(segment.startTime)
    const endX = xScale(segment.endTime)
    const width = Math.max(2, endX - x)
    const y = yScale(resourceIndex) + (ROW_HEIGHT - BAR_HEIGHT) / 2

    // 计算各类型占比并生成渐变
    const totalDuration = Array.from(segment.typeBreakdown.values()).reduce((a, b) => a + b, 0)
    const sortedTypes = Array.from(segment.typeBreakdown.entries())
      .sort((a, b) => b[1] - a[1])

    // 如果只有一种类型，直接用纯色
    if (sortedTypes.length === 1) {
      const [type] = sortedTypes[0]
      return (
        <rect
          key={`${segment.resourceId}-${segment.binIndex}`}
          x={x}
          y={y}
          width={width}
          height={BAR_HEIGHT}
          fill={TASK_COLORS[type] || '#999'}
          rx={2}
          style={{ cursor: 'pointer' }}
          onMouseEnter={(e) => handleSegmentHover(segment, e)}
          onMouseMove={(e) => handleSegmentHover(segment, e)}
          onMouseLeave={handleSegmentLeave}
        />
      )
    }

    // 多种类型：使用主导颜色，透明度表示混合程度
    const dominantType = sortedTypes[0][0]
    const dominantRatio = sortedTypes[0][1] / totalDuration

    return (
      <rect
        key={`${segment.resourceId}-${segment.binIndex}`}
        x={x}
        y={y}
        width={width}
        height={BAR_HEIGHT}
        fill={TASK_COLORS[dominantType] || '#999'}
        opacity={0.6 + dominantRatio * 0.4}
        rx={2}
        style={{ cursor: 'pointer' }}
        onMouseEnter={(e) => handleSegmentHover(segment, e)}
        onMouseMove={(e) => handleSegmentHover(segment, e)}
        onMouseLeave={handleSegmentLeave}
      />
    )
  }, [resourceIndexMap, xScale, yScale, handleSegmentHover, handleSegmentLeave])

  if (!data || data.tasks.length === 0) {
    return (
      <div ref={containerRef} style={{ width: '100%' }}>
        <Empty
          description="运行模拟以生成甘特图"
          style={{ marginTop: 20 }}
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      </div>
    )
  }

  // 等待容器宽度初始化
  if (containerWidth === 0) {
    return <div ref={containerRef} style={{ width: '100%', height: 100 }} />
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      {/* 工具栏 */}
      <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text type="secondary" style={{ fontSize: 10 }}>
          Ctrl+滚轮: 缩放 | Shift+滚轮: 平移 | 拖拽: 平移
        </Text>
        <Space size="small">
          <Text type="secondary" style={{ fontSize: 11 }}>
            {zoom.toFixed(1)}x
          </Text>
          <Tooltip title="放大 (Ctrl+滚轮↑)">
            <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn} />
          </Tooltip>
          <Tooltip title="缩小 (Ctrl+滚轮↓)">
            <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut} disabled={zoom <= 1} />
          </Tooltip>
          <Tooltip title="重置视图">
            <Button size="small" icon={<ReloadOutlined />} onClick={handleReset} disabled={zoom === 1} />
          </Tooltip>
        </Space>
      </div>

      {/* SVG 图表 */}
      <svg
        ref={svgRef}
        width={chartWidth}
        height={chartHeight + (zoom > 1 ? SCROLLBAR_HEIGHT + 6 : 0)}
        onMouseDown={handleMouseDown}
        onMouseMove={(e) => { handleMouseMove(e); handleScrollbarMouseMove(e) }}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging || isScrollbarDragging ? 'grabbing' : zoom > 1 ? 'grab' : 'default' }}
      >
        {/* 背景网格 */}
        <g className="grid">
          {timeTicksData.map((tick, i) => (
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

        {/* Prefill/Decode 分界线 */}
        {data.phaseTransition && data.phaseTransition >= visibleTimeRange.start && data.phaseTransition <= visibleTimeRange.end && (
          <g>
            <line
              x1={xScale(data.phaseTransition)}
              y1={MARGIN.top - 5}
              x2={xScale(data.phaseTransition)}
              y2={chartHeight - MARGIN.bottom}
              stroke="#ff4d4f"
              strokeWidth={2}
              strokeDasharray="6,4"
            />
            <text
              x={xScale(data.phaseTransition) - 35}
              y={MARGIN.top - 10}
              fontSize={11}
              fontWeight={500}
              fill="#ff4d4f"
            >
              Prefill
            </text>
            <text
              x={xScale(data.phaseTransition) + 8}
              y={MARGIN.top - 10}
              fontSize={11}
              fontWeight={500}
              fill="#1890ff"
            >
              Decode
            </text>
          </g>
        )}

        {/* 资源行标签和背景 */}
        <g className="y-axis">
          {data.resources.map((resource, i) => (
            <g key={resource.id} transform={`translate(0, ${yScale(i)})`}>
              <text
                x={MARGIN.left - 8}
                y={ROW_HEIGHT / 2 + 4}
                textAnchor="end"
                fontSize={11}
                fill="#333"
                fontWeight={500}
              >
                {resource.name}
              </text>
              <rect
                x={MARGIN.left}
                y={(ROW_HEIGHT - BAR_HEIGHT) / 2}
                width={innerWidth}
                height={BAR_HEIGHT}
                fill={i % 2 === 0 ? '#fafafa' : '#fff'}
                stroke="#f0f0f0"
                rx={3}
              />
            </g>
          ))}
        </g>

        {/* 聚合任务段 */}
        <g className="tasks">
          {aggregatedData.map(renderSegment)}
        </g>

        {/* 时间轴 */}
        <g className="x-axis" transform={`translate(0, ${chartHeight - MARGIN.bottom})`}>
          <line
            x1={MARGIN.left}
            y1={0}
            x2={chartWidth - MARGIN.right}
            y2={0}
            stroke="#d9d9d9"
          />
          {timeTicksData.map((tick, i) => (
            <g key={i} transform={`translate(${xScale(tick)}, 0)`}>
              <line y1={0} y2={5} stroke="#999" />
              <text
                y={18}
                textAnchor="middle"
                fontSize={10}
                fill="#666"
              >
                {formatTime(tick)}
              </text>
            </g>
          ))}
        </g>

        {/* 滚动条 - 仅在缩放时显示 */}
        {zoom > 1 && data && (
          <g className="scrollbar" transform={`translate(0, ${chartHeight + 2})`}>
            {/* 轨道 */}
            <rect
              x={MARGIN.left}
              y={0}
              width={innerWidth}
              height={SCROLLBAR_HEIGHT}
              fill="#f0f0f0"
              rx={5}
              style={{ cursor: 'pointer' }}
              onClick={handleScrollbarTrackClick}
            />
            {/* 滑块 */}
            <rect
              x={MARGIN.left + (panOffset / (data.timeRange.end - data.timeRange.start)) * innerWidth}
              y={1}
              width={Math.max(30, innerWidth / zoom)}
              height={SCROLLBAR_HEIGHT - 2}
              fill={isScrollbarDragging ? '#1890ff' : '#999'}
              rx={4}
              style={{ cursor: 'grab' }}
              onMouseDown={handleScrollbarMouseDown}
            />
          </g>
        )}
      </svg>

      {/* 悬浮提示框 */}
      {tooltip && (
        <div style={{ ...tooltipStyle, left: tooltip.x, top: tooltip.y }}>
          <div style={{ fontWeight: 600, marginBottom: 6, borderBottom: '1px solid rgba(255,255,255,0.2)', paddingBottom: 4 }}>
            时间段: {formatTime(tooltip.segment.startTime)} - {formatTime(tooltip.segment.endTime)}
          </div>
          <div style={{ marginBottom: 4 }}>任务数: {tooltip.segment.taskCount}</div>
          <div style={{ marginBottom: 4 }}>类型分布:</div>
          <div style={{ maxHeight: 150, overflowY: 'auto' }}>
            {Array.from(tooltip.segment.typeBreakdown.entries())
              .sort((a, b) => b[1] - a[1])
              .slice(0, 8)
              .map(([type, duration]) => {
                const total = Array.from(tooltip.segment.typeBreakdown.values()).reduce((a, b) => a + b, 0)
                const pct = ((duration / total) * 100).toFixed(1)
                return (
                  <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TASK_COLORS[type] || '#999', flexShrink: 0 }} />
                    <span style={{ flex: 1 }}>{TASK_LABELS[type] || type}</span>
                    <span style={{ color: 'rgba(255,255,255,0.7)' }}>{pct}%</span>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* 图例和统计信息 */}
      <div style={{
        marginTop: 8,
        padding: '8px 0',
        borderTop: '1px solid #f0f0f0',
        overflow: 'hidden',
      }}>
        {/* 图例 */}
        {showLegend && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
            {LEGEND_GROUPS.map((group) => (
              <div key={group.name} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontSize: 10, color: '#999' }}>{group.name}:</span>
                {group.types.map((type) => (
                  <span
                    key={type}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 2, fontSize: 10 }}
                  >
                    <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: TASK_COLORS[type], flexShrink: 0 }} />
                    <span style={{ color: '#666' }}>{TASK_LABELS[type]}</span>
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* 统计信息 */}
        <div style={{ display: 'flex', gap: 12, fontSize: 10, color: '#666', flexWrap: 'wrap' }}>
          <span>总时长: {formatTime(data.timeRange.end - data.timeRange.start)}</span>
          {data.phaseTransition && (
            <>
              <span>Prefill: {formatTime(data.phaseTransition)}</span>
              <span>Decode: {formatTime(data.timeRange.end - data.phaseTransition)}</span>
            </>
          )}
          <span>任务数: {data.tasks.length.toLocaleString()}</span>
        </div>
      </div>
    </div>
  )
}

export default GanttChart
