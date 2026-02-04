/**
 * Canvas 甘特图组件 - 高性能 LLM 推理时序可视化
 *
 * 相比 SVG 版本的优化：
 * - Canvas 2D 渲染，支持 10000+ 任务流畅显示
 * - 智能任务聚合，根据缩放级别自动调整
 * - 框选放大、类型过滤等高级交互
 * - 简化的颜色系统（5大类）
 */

import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react'
import { ZoomIn, ZoomOut, RotateCcw, Filter } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { InfoTooltip } from '@/components/ui/info-tooltip'
import type { GanttChartData, GanttTask } from '../../../../utils/llmDeployment/types'
import {
  GANTT_CATEGORY_COLORS,
  getTaskCategory,
} from './chartTheme'

interface GanttChartCanvasProps {
  data: GanttChartData | null
  showLegend?: boolean
  onTaskClick?: (task: GanttTask) => void
}

/** 图表边距 */
const MARGIN = { top: 30, right: 20, bottom: 30, left: 100 }

/** 滚动条高度 */
const SCROLLBAR_HEIGHT = 10

/** 资源行高度 */
const ROW_HEIGHT = 32

/** 任务条高度 */
const BAR_HEIGHT = 24

/** 任务类型大类 */
const TASK_CATEGORIES = [
  { key: 'compute', name: '计算', color: GANTT_CATEGORY_COLORS.compute.primary },
  { key: 'memory', name: '访存', color: GANTT_CATEGORY_COLORS.memory.primary },
  { key: 'tp', name: 'TP通信', color: GANTT_CATEGORY_COLORS.tp.primary },
  { key: 'pp', name: 'PP通信', color: GANTT_CATEGORY_COLORS.pp.primary },
  { key: 'ep', name: 'EP/MoE', color: GANTT_CATEGORY_COLORS.ep.primary },
  { key: 'other', name: '其他', color: GANTT_CATEGORY_COLORS.other.primary },
] as const

/** 聚合后的任务段 */
interface AggregatedSegment {
  resourceIndex: number
  startTime: number
  endTime: number
  category: keyof typeof GANTT_CATEGORY_COLORS
  taskCount: number
  tasks: GanttTask[]
  dominantRatio: number // 主导类型占比
}

/** 悬浮提示框数据 */
interface TooltipData {
  segment: AggregatedSegment
  x: number
  y: number
}

export const GanttChartCanvas: React.FC<GanttChartCanvasProps> = ({
  data,
  showLegend = true,
  onTaskClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)
  const [tooltip, setTooltip] = useState<TooltipData | null>(null)

  // 缩放和平移状态
  const [zoom, setZoom] = useState(1)
  const [panOffset, setPanOffset] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const dragStartRef = useRef({ x: 0, offset: 0 })

  // 框选状态
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionStart, setSelectionStart] = useState<{ x: number; y: number } | null>(null)
  const [selectionEnd, setSelectionEnd] = useState<{ x: number; y: number } | null>(null)

  // 过滤状态
  const [visibleCategories, setVisibleCategories] = useState<Set<string>>(
    new Set(TASK_CATEGORIES.map(c => c.key))
  )

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

  // 计算图表尺寸
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
    const clampedStart = Math.max(data.timeRange.start, Math.min(start, data.timeRange.end - visibleDuration))
    return { start: clampedStart, end: clampedStart + visibleDuration }
  }, [data, zoom, panOffset])

  // 资源索引映射
  const resourceIndexMap = useMemo(() => {
    if (!data) return new Map<string, number>()
    const map = new Map<string, number>()
    data.resources.forEach((r, i) => map.set(r.id, i))
    return map
  }, [data])

  // 聚合任务数据
  const aggregatedData = useMemo((): AggregatedSegment[] => {
    if (!data) return []

    const { start, end } = visibleTimeRange
    const duration = end - start
    const numBins = Math.min(Math.max(50, Math.floor(innerWidth / 4)), 400)
    const binWidth = duration / numBins

    // 初始化bins
    const bins = new Map<number, Map<number, GanttTask[]>>()
    for (let i = 0; i < data.resources.length; i++) {
      bins.set(i, new Map())
    }

    // 将任务分配到bins
    for (const task of data.tasks) {
      if (task.end < start || task.start > end) continue

      // 检查类型过滤
      const category = getTaskCategory(task.type)
      if (!visibleCategories.has(category)) continue

      // 确定资源行
      const isNetworkTask = ['tp_comm', 'pp_comm', 'ep_comm', 'ep_dispatch', 'ep_combine', 'sp_allgather', 'sp_reduce_scatter', 'dp_gradient_sync'].includes(task.type)
      const resourceId = `stage${task.ppStage}_${isNetworkTask ? 'network' : 'compute'}`
      const resourceIndex = resourceIndexMap.get(resourceId)
      if (resourceIndex === undefined) continue

      const resourceBins = bins.get(resourceIndex)
      if (!resourceBins) continue

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

    for (const [resourceIndex, resourceBins] of bins) {
      let currentSegment: AggregatedSegment | null = null
      let currentCategory: keyof typeof GANTT_CATEGORY_COLORS | null = null

      for (let bin = 0; bin < numBins; bin++) {
        const binTasks = resourceBins.get(bin)

        if (binTasks && binTasks.length > 0) {
          // 计算主导类型
          const categoryDurations = new Map<string, number>()
          for (const task of binTasks) {
            const cat = getTaskCategory(task.type)
            categoryDurations.set(cat, (categoryDurations.get(cat) || 0) + (task.end - task.start))
          }
          let dominantCategory: keyof typeof GANTT_CATEGORY_COLORS = 'other'
          let maxDuration = 0
          for (const [cat, dur] of categoryDurations) {
            if (dur > maxDuration) {
              maxDuration = dur
              dominantCategory = cat as keyof typeof GANTT_CATEGORY_COLORS
            }
          }
          const totalDur = Array.from(categoryDurations.values()).reduce((a, b) => a + b, 0)
          const dominantRatio = totalDur > 0 ? maxDuration / totalDur : 1

          // 合并连续同类型段
          if (currentSegment && currentCategory === dominantCategory) {
            currentSegment.endTime = start + (bin + 1) * binWidth
            for (const task of binTasks) {
              if (!currentSegment.tasks.includes(task)) {
                currentSegment.tasks.push(task)
                currentSegment.taskCount++
              }
            }
            currentSegment.dominantRatio = (currentSegment.dominantRatio + dominantRatio) / 2
          } else {
            // 保存前一段
            if (currentSegment) {
              segments.push(currentSegment)
            }
            // 开始新段
            currentSegment = {
              resourceIndex,
              startTime: start + bin * binWidth,
              endTime: start + (bin + 1) * binWidth,
              category: dominantCategory,
              taskCount: binTasks.length,
              tasks: [...binTasks],
              dominantRatio,
            }
            currentCategory = dominantCategory
          }
        } else {
          if (currentSegment) {
            segments.push(currentSegment)
            currentSegment = null
            currentCategory = null
          }
        }
      }

      if (currentSegment) {
        segments.push(currentSegment)
      }
    }

    return segments
  }, [data, visibleTimeRange, innerWidth, resourceIndexMap, visibleCategories])

  // 格式化时间
  const formatTime = (ms: number): string => {
    if (ms < 0.001) return `${(ms * 1000000).toFixed(0)}ns`
    if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`
    if (ms < 1000) return `${ms.toFixed(2)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  // Canvas 渲染
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data || containerWidth === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = chartWidth * dpr
    canvas.height = chartHeight * dpr
    ctx.scale(dpr, dpr)

    // 清空画布
    ctx.clearRect(0, 0, chartWidth, chartHeight)

    // 绘制背景
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, chartWidth, chartHeight)

    // 时间刻度
    const { start, end } = visibleTimeRange
    const duration = end - start
    const numTicks = Math.min(8, Math.floor(innerWidth / 80))
    ctx.strokeStyle = '#f0f0f0'
    ctx.setLineDash([3, 3])
    for (let i = 0; i <= numTicks; i++) {
      const t = start + (duration / numTicks) * i
      const x = MARGIN.left + ((t - start) / duration) * innerWidth
      ctx.beginPath()
      ctx.moveTo(x, MARGIN.top)
      ctx.lineTo(x, chartHeight - MARGIN.bottom)
      ctx.stroke()
    }
    ctx.setLineDash([])

    // Prefill/Decode 分界线
    if (data.phaseTransition && data.phaseTransition >= start && data.phaseTransition <= end) {
      const x = MARGIN.left + ((data.phaseTransition - start) / duration) * innerWidth
      ctx.strokeStyle = '#ff4d4f'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      ctx.moveTo(x, MARGIN.top - 5)
      ctx.lineTo(x, chartHeight - MARGIN.bottom)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.lineWidth = 1

      // 标签
      ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif'
      ctx.fillStyle = '#ff4d4f'
      ctx.textAlign = 'right'
      ctx.fillText('Prefill', x - 8, MARGIN.top - 10)
      ctx.fillStyle = '#1890ff'
      ctx.textAlign = 'left'
      ctx.fillText('Decode', x + 8, MARGIN.top - 10)
    }

    // 资源行背景和标签
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif'
    ctx.textAlign = 'right'
    for (let i = 0; i < data.resources.length; i++) {
      const y = MARGIN.top + i * ROW_HEIGHT
      // 交替背景
      ctx.fillStyle = i % 2 === 0 ? '#fafafa' : '#fff'
      ctx.fillRect(MARGIN.left, y + (ROW_HEIGHT - BAR_HEIGHT) / 2, innerWidth, BAR_HEIGHT)
      // 标签
      ctx.fillStyle = '#333'
      ctx.fillText(data.resources[i].name, MARGIN.left - 8, y + ROW_HEIGHT / 2 + 4)
    }

    // 绘制任务段
    for (const segment of aggregatedData) {
      const x = MARGIN.left + ((segment.startTime - start) / duration) * innerWidth
      const endX = MARGIN.left + ((segment.endTime - start) / duration) * innerWidth
      const width = Math.max(2, endX - x)
      const y = MARGIN.top + segment.resourceIndex * ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2

      const colors = GANTT_CATEGORY_COLORS[segment.category]
      const alpha = 0.6 + segment.dominantRatio * 0.4

      // 绘制圆角矩形
      const radius = 3
      ctx.beginPath()
      ctx.moveTo(x + radius, y)
      ctx.lineTo(x + width - radius, y)
      ctx.quadraticCurveTo(x + width, y, x + width, y + radius)
      ctx.lineTo(x + width, y + BAR_HEIGHT - radius)
      ctx.quadraticCurveTo(x + width, y + BAR_HEIGHT, x + width - radius, y + BAR_HEIGHT)
      ctx.lineTo(x + radius, y + BAR_HEIGHT)
      ctx.quadraticCurveTo(x, y + BAR_HEIGHT, x, y + BAR_HEIGHT - radius)
      ctx.lineTo(x, y + radius)
      ctx.quadraticCurveTo(x, y, x + radius, y)
      ctx.closePath()

      ctx.globalAlpha = alpha
      ctx.fillStyle = colors.primary
      ctx.fill()
      ctx.globalAlpha = 1
    }

    // 框选区域
    if (isSelecting && selectionStart && selectionEnd) {
      const sx = Math.min(selectionStart.x, selectionEnd.x)
      const sy = Math.min(selectionStart.y, selectionEnd.y)
      const sw = Math.abs(selectionEnd.x - selectionStart.x)
      const sh = Math.abs(selectionEnd.y - selectionStart.y)

      ctx.fillStyle = 'rgba(96, 165, 250, 0.1)'
      ctx.fillRect(sx, sy, sw, sh)
      ctx.strokeStyle = '#60A5FA'
      ctx.lineWidth = 1
      ctx.strokeRect(sx, sy, sw, sh)
    }

    // 时间轴
    ctx.strokeStyle = '#d9d9d9'
    ctx.beginPath()
    ctx.moveTo(MARGIN.left, chartHeight - MARGIN.bottom)
    ctx.lineTo(chartWidth - MARGIN.right, chartHeight - MARGIN.bottom)
    ctx.stroke()

    ctx.fillStyle = '#666'
    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif'
    ctx.textAlign = 'center'
    for (let i = 0; i <= numTicks; i++) {
      const t = start + (duration / numTicks) * i
      const x = MARGIN.left + ((t - start) / duration) * innerWidth
      ctx.beginPath()
      ctx.moveTo(x, chartHeight - MARGIN.bottom)
      ctx.lineTo(x, chartHeight - MARGIN.bottom + 5)
      ctx.stroke()
      ctx.fillText(formatTime(t), x, chartHeight - MARGIN.bottom + 18)
    }

    // 滚动条
    if (zoom > 1) {
      const scrollbarY = chartHeight - 6
      const totalDuration = data.timeRange.end - data.timeRange.start
      const thumbWidth = Math.max(30, innerWidth / zoom)
      const thumbX = MARGIN.left + (panOffset / totalDuration) * (innerWidth - thumbWidth)

      ctx.fillStyle = '#f0f0f0'
      ctx.fillRect(MARGIN.left, scrollbarY, innerWidth, SCROLLBAR_HEIGHT)
      ctx.fillStyle = '#999'
      ctx.fillRect(thumbX, scrollbarY + 1, thumbWidth, SCROLLBAR_HEIGHT - 2)
    }
  }, [data, containerWidth, chartWidth, chartHeight, visibleTimeRange, aggregatedData, isSelecting, selectionStart, selectionEnd, zoom, panOffset, innerWidth])

  // 鼠标事件处理
  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !data) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    if (isDragging) {
      const dx = e.clientX - dragStartRef.current.x
      const totalDuration = data.timeRange.end - data.timeRange.start
      const visibleDuration = totalDuration / zoom
      const pxPerMs = innerWidth / visibleDuration
      const newOffset = dragStartRef.current.offset - dx / pxPerMs
      setPanOffset(Math.max(0, Math.min(newOffset, totalDuration - visibleDuration)))
      return
    }

    if (isSelecting && selectionStart) {
      setSelectionEnd({ x, y })
      return
    }

    // 检测悬停
    const { start, end } = visibleTimeRange
    const duration = end - start

    for (const segment of aggregatedData) {
      const segX = MARGIN.left + ((segment.startTime - start) / duration) * innerWidth
      const segEndX = MARGIN.left + ((segment.endTime - start) / duration) * innerWidth
      const segY = MARGIN.top + segment.resourceIndex * ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2

      if (x >= segX && x <= segEndX && y >= segY && y <= segY + BAR_HEIGHT) {
        setTooltip({ segment, x: e.clientX + 10, y: e.clientY + 10 })
        canvas.style.cursor = onTaskClick ? 'pointer' : 'default'
        return
      }
    }

    setTooltip(null)
    canvas.style.cursor = zoom > 1 ? 'grab' : 'default'
  }, [data, isDragging, isSelecting, selectionStart, visibleTimeRange, aggregatedData, zoom, innerWidth, onTaskClick])

  const handleCanvasMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // Alt + 点击开始框选
    if (e.altKey) {
      setIsSelecting(true)
      setSelectionStart({ x, y })
      setSelectionEnd({ x, y })
      return
    }

    // 普通拖拽
    if (zoom > 1) {
      setIsDragging(true)
      dragStartRef.current = { x: e.clientX, offset: panOffset }
      canvas.style.cursor = 'grabbing'
    }
  }, [zoom, panOffset])

  const handleCanvasMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !data) return

    if (isSelecting && selectionStart && selectionEnd) {
      // 完成框选，计算选区对应的时间范围并缩放
      const { start } = visibleTimeRange
      const duration = visibleTimeRange.end - start

      const sx = Math.min(selectionStart.x, selectionEnd.x)
      const ex = Math.max(selectionStart.x, selectionEnd.x)

      // 只有选区宽度大于 10px 才执行缩放
      if (ex - sx > 10) {
        const newStart = start + ((sx - MARGIN.left) / innerWidth) * duration
        const newEnd = start + ((ex - MARGIN.left) / innerWidth) * duration
        const totalDuration = data.timeRange.end - data.timeRange.start
        const newDuration = Math.max(0.001, newEnd - newStart)
        const newZoom = Math.min(50, totalDuration / newDuration)

        setZoom(newZoom)
        setPanOffset(Math.max(0, newStart - data.timeRange.start))
      }

      setIsSelecting(false)
      setSelectionStart(null)
      setSelectionEnd(null)
      return
    }

    if (isDragging) {
      setIsDragging(false)
      canvas.style.cursor = zoom > 1 ? 'grab' : 'default'
      return
    }

    // 点击事件
    if (tooltip && onTaskClick && tooltip.segment.tasks.length > 0) {
      const longestTask = tooltip.segment.tasks.reduce((prev, curr) =>
        (curr.end - curr.start) > (prev.end - prev.start) ? curr : prev
      )
      onTaskClick(longestTask)
    }
  }, [data, isSelecting, selectionStart, selectionEnd, isDragging, tooltip, onTaskClick, visibleTimeRange, innerWidth, zoom])

  // 滚轮事件处理（在 DOM 级别强制拦截）
  useEffect(() => {
    const container = containerRef.current
    if (!container || !data) return

    const handleWheel = (e: WheelEvent) => {
      const totalDuration = data.timeRange.end - data.timeRange.start

      if (e.ctrlKey || e.metaKey) {
        // 强制阻止浏览器缩放
        e.preventDefault()
        e.stopPropagation()
        e.stopImmediatePropagation()

        if (e.deltaY < 0) {
          setZoom(z => Math.min(z * 1.2, 50))
        } else {
          setZoom(z => {
            const newZoom = Math.max(z / 1.2, 1)
            const newVisibleDuration = totalDuration / newZoom
            setPanOffset(offset => Math.max(0, Math.min(offset, totalDuration - newVisibleDuration)))
            return newZoom
          })
        }
        return false
      }

      if (e.shiftKey && zoom > 1) {
        e.preventDefault()
        e.stopPropagation()
        const visibleDuration = totalDuration / zoom
        const panStep = visibleDuration * 0.1
        const delta = e.deltaY > 0 ? panStep : -panStep
        setPanOffset(offset => Math.max(0, Math.min(offset + delta, totalDuration - visibleDuration)))
        return false
      }
    }

    // 在容器上监听，使用捕获阶段 + passive: false
    container.addEventListener('wheel', handleWheel, { passive: false, capture: true })
    return () => container.removeEventListener('wheel', handleWheel, { capture: true })
  }, [data, zoom])

  // 缩放控制
  const handleZoomIn = useCallback(() => setZoom(z => Math.min(z * 1.5, 50)), [])
  const handleZoomOut = useCallback(() => {
    setZoom(z => Math.max(z / 1.5, 1))
    if (data) {
      const totalDuration = data.timeRange.end - data.timeRange.start
      const newVisibleDuration = totalDuration / Math.max(zoom / 1.5, 1)
      setPanOffset(offset => Math.min(offset, totalDuration - newVisibleDuration))
    }
  }, [data, zoom])
  const handleReset = useCallback(() => {
    setZoom(1)
    setPanOffset(0)
  }, [])

  // 类型过滤切换
  const toggleCategory = useCallback((category: string) => {
    setVisibleCategories(prev => {
      const next = new Set(prev)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      return next
    })
  }, [])

  if (!data || data.tasks.length === 0) {
    return (
      <div ref={containerRef} style={{ width: '100%' }}>
        <div className="flex flex-col items-center justify-center py-10 text-gray-400">
          <div className="text-sm">运行模拟以生成甘特图</div>
        </div>
      </div>
    )
  }

  if (containerWidth === 0) {
    return <div ref={containerRef} style={{ width: '100%', height: 100 }} />
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      style={{ width: '100%', outline: 'none', touchAction: 'none' }}
    >
      {/* 工具栏 */}
      <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="text-[10px] text-gray-500">
          Ctrl+滚轮: 缩放 | Shift+滚轮: 平移 | 拖拽: 平移 | Alt+拖拽: 框选放大
        </span>
        <div className="flex items-center gap-1">
          {/* 类型过滤 */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="icon" className="h-7 w-7">
                <Filter className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-48 p-3">
              <div className="text-xs font-medium mb-2">任务类型过滤</div>
              <div className="space-y-2">
                {TASK_CATEGORIES.map(cat => (
                  <label key={cat.key} className="flex items-center gap-2 cursor-pointer">
                    <Checkbox
                      checked={visibleCategories.has(cat.key)}
                      onCheckedChange={() => toggleCategory(cat.key)}
                    />
                    <span
                      className="w-3 h-3 rounded-sm"
                      style={{ backgroundColor: cat.color }}
                    />
                    <span className="text-xs">{cat.name}</span>
                  </label>
                ))}
              </div>
            </PopoverContent>
          </Popover>

          <span className="text-[11px] text-gray-500 mx-1">
            {zoom.toFixed(1)}x
          </span>
          <InfoTooltip content="放大 (Ctrl+滚轮↑)">
            <Button variant="outline" size="icon" className="h-7 w-7" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
          </InfoTooltip>
          <InfoTooltip content="缩小 (Ctrl+滚轮↓)">
            <Button variant="outline" size="icon" className="h-7 w-7" onClick={handleZoomOut} disabled={zoom <= 1}>
              <ZoomOut className="h-4 w-4" />
            </Button>
          </InfoTooltip>
          <InfoTooltip content="重置视图">
            <Button variant="outline" size="icon" className="h-7 w-7" onClick={handleReset} disabled={zoom === 1}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </InfoTooltip>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        style={{
          width: chartWidth,
          height: chartHeight + (zoom > 1 ? SCROLLBAR_HEIGHT + 6 : 0),
          cursor: isDragging ? 'grabbing' : zoom > 1 ? 'grab' : 'default',
          touchAction: 'none',
        }}
        onMouseMove={handleCanvasMouseMove}
        onMouseDown={handleCanvasMouseDown}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={() => {
          setTooltip(null)
          if (isDragging) {
            setIsDragging(false)
          }
        }}
      />

      {/* 悬浮提示框 */}
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x,
            top: tooltip.y,
            background: 'rgba(0, 0, 0, 0.9)',
            color: '#fff',
            padding: '10px 14px',
            borderRadius: 8,
            fontSize: 12,
            lineHeight: 1.6,
            pointerEvents: 'none',
            zIndex: 1000,
            maxWidth: 280,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 6, borderBottom: '1px solid rgba(255,255,255,0.2)', paddingBottom: 4 }}>
            {formatTime(tooltip.segment.startTime)} - {formatTime(tooltip.segment.endTime)}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                backgroundColor: GANTT_CATEGORY_COLORS[tooltip.segment.category].primary,
              }}
            />
            <span>{TASK_CATEGORIES.find(c => c.key === tooltip.segment.category)?.name || '其他'}</span>
            <span style={{ color: 'rgba(255,255,255,0.7)' }}>
              ({(tooltip.segment.dominantRatio * 100).toFixed(0)}%)
            </span>
          </div>
          <div style={{ color: 'rgba(255,255,255,0.7)' }}>
            任务数: {tooltip.segment.taskCount}
          </div>
        </div>
      )}

      {/* 图例和统计 */}
      <div style={{
        marginTop: 8,
        padding: '8px 0',
        borderTop: '1px solid #f0f0f0',
      }}>
        {showLegend && (
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 6 }}>
            {TASK_CATEGORIES.map(cat => (
              <div
                key={cat.key}
                className="flex items-center gap-1.5 cursor-pointer"
                style={{ opacity: visibleCategories.has(cat.key) ? 1 : 0.4 }}
                onClick={() => toggleCategory(cat.key)}
              >
                <span
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: 2,
                    backgroundColor: cat.color,
                  }}
                />
                <span style={{ fontSize: 11, color: '#666' }}>{cat.name}</span>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 12, fontSize: 10, color: '#666', flexWrap: 'wrap' }}>
          <span>总时长: {formatTime(data.timeRange.end - data.timeRange.start)}</span>
          {data.phaseTransition && (
            <>
              <span>Prefill: {formatTime(data.phaseTransition)}</span>
              <span>Decode: {formatTime(data.timeRange.end - data.phaseTransition)}</span>
            </>
          )}
          <span>任务数: {data.tasks.length.toLocaleString()}</span>
          <span>显示: {aggregatedData.length} 段</span>
        </div>
      </div>
    </div>
  )
}

export default GanttChartCanvas
