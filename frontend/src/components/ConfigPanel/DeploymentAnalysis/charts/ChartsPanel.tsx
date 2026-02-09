/**
 * 图表面板 - 整合所有图表的容器组件（CSS Grid 响应式布局）
 * 使用统一主题配置
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { MemoryPieChart } from './MemoryPieChart'
import { RooflineChart } from './RooflineChart'
import { GanttChart } from './GanttChart'
import { LayerWaterfallChart } from './LayerWaterfallChart'
import { OperatorTimeBreakdownChart } from './OperatorTimeBreakdownChart'
import { TopologyTrafficChart } from './TopologyTrafficChart'
import { TaskDetailDrawer } from '../TaskDetailDrawer'
import { BaseCard } from "@/components/common/BaseCard"
import {
  CHART_CARD_STYLE,
  CHART_TITLE_STYLE,
  BOTTLENECK_COLORS,
} from './chartTheme'
import {
  PlanAnalysisResult,
  HardwareConfig,
  LLMModelConfig,
  InferenceConfig,
  GanttChartData,
  GanttTask,
  SimulationStats,
  SimulationResult,
} from '../../../../utils/llmDeployment/types'
import { HierarchicalTopology } from '../../../../types'

/** 后端模拟 API 地址（通过 vite proxy 转发）*/
const SIMULATION_API_URL = '/api/simulate'

/** 后端模拟结果 */
interface BackendSimulationResult {
  ganttChart: GanttChartData
  stats: SimulationStats
  timestamp: number
}

interface ChartsPanelProps {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  hardware: HardwareConfig
  model: LLMModelConfig
  inference?: InferenceConfig
  topology?: HierarchicalTopology | null
  /** 外部提供的甘特图数据（从后端评估结果中获取），优先使用 */
  externalGanttData?: GanttChartData | null
  /** 外部提供的统计数据（从后端评估结果中获取），优先使用 */
  externalStats?: SimulationStats | null
}

export const ChartsPanel: React.FC<ChartsPanelProps> = ({
  result,
  topKPlans,
  hardware,
  model,
  inference,
  topology,
  externalGanttData,
  externalStats,
}) => {
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)

  // 判断是否有外部数据（来自后端评估结果）
  const hasExternalData = !!(externalGanttData || externalStats)
  const [selectedTask, setSelectedTask] = useState<GanttTask | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)

  // 记录上次运行的result id，避免重复运行
  const lastResultIdRef = useRef<string | null>(null)

  // 运行后端模拟
  const runBackendSimulation = useCallback(async () => {
    if (!result || !inference || !topology) return

    setIsSimulating(true)
    try {
      const requestBody = {
        topology: topology,
        model: {
          model_name: model.model_name,
          model_type: model.model_type,
          hidden_size: model.hidden_size,
          num_layers: model.num_layers,
          num_attention_heads: model.num_attention_heads,
          num_kv_heads: model.num_kv_heads,
          intermediate_size: model.intermediate_size,
          vocab_size: model.vocab_size,
          weight_dtype: model.weight_dtype,
          activation_dtype: model.activation_dtype,
          max_seq_length: model.max_seq_length,
          attention_type: model.attention_type,
          norm_type: model.norm_type,
          moe_config: model.moe_config,
          mla_config: model.mla_config,
        },
        inference: {
          batch_size: inference.batch_size,
          input_seq_length: inference.input_seq_length,
          output_seq_length: inference.output_seq_length,
          max_seq_length: inference.max_seq_length,
        },
        parallelism: result.plan.parallelism,
        hardware: hardware,
        config: {
          maxSimulatedTokens: 1,
          enableDataTransferSimulation: true,
          enableDetailedTransformerOps: true,
          enableKVCacheAccessSimulation: true,
        },
      }

      const response = await fetch(SIMULATION_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`)
      }

      const backendResult: BackendSimulationResult = await response.json()

      // 设置模拟结果
      setSimulationResult(backendResult)
    } catch (error) {
      console.error('后端模拟失败:', error)
      toast.error('模拟失败，请检查后端服务是否启动')
    } finally {
      setIsSimulating(false)
    }
  }, [result, inference, topology, model, hardware])

  // 运行模拟
  const runSimulation = useCallback(async () => {
    if (!result || !inference || !topology) return
    await runBackendSimulation()
  }, [result, inference, topology, runBackendSimulation])

  // 当分析结果变化时自动运行模拟（仅在没有外部数据时）
  useEffect(() => {
    // 如果有外部数据，不需要运行模拟
    if (hasExternalData) {
      return
    }

    if (!result || !inference) {
      setSimulationResult(null)
      lastResultIdRef.current = null
      return
    }

    // 生成唯一标识
    const resultId = `${result.plan.parallelism.tp}-${result.plan.parallelism.pp}-${result.plan.parallelism.dp}-${result.plan.parallelism.ep}-${result.score.overall_score}`

    // 避免重复运行
    if (lastResultIdRef.current === resultId) return
    lastResultIdRef.current = resultId

    runSimulation()
  }, [result, inference, runSimulation, hasExternalData])

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-gray-400">
        <div className="text-sm">请先运行分析以查看图表</div>
      </div>
    )
  }

  // 获取瓶颈状态颜色
  const getBottleneckColor = (type: string) => {
    return BOTTLENECK_COLORS[type as keyof typeof BOTTLENECK_COLORS] || BOTTLENECK_COLORS.balanced
  }

  // 获取瓶颈状态文字
  const getBottleneckText = (type: string) => {
    switch (type) {
      case 'memory': return '带宽受限'
      case 'compute': return '算力受限'
      case 'communication': return '通信受限'
      default: return '均衡'
    }
  }

  return (
    <div>
      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 四、结果可视化 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <BaseCard
        title="结果可视化"
        className="mb-4"
      >
        {/* 第一行：Roofline + 内存占用（网格布局） */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* Roofline 图 */}
          <div style={CHART_CARD_STYLE} className="transition-shadow hover:shadow-md">
            <div style={CHART_TITLE_STYLE}>
              <span className="font-semibold">Roofline 性能分析</span>
              <span
                className="text-[11px] font-medium"
                style={{ color: getBottleneckColor(result.latency.bottleneck_type) }}
              >
                {getBottleneckText(result.latency.bottleneck_type)}
              </span>
            </div>
            <RooflineChart
              result={result}
              hardware={hardware}
              model={model}
              comparisonResults={topKPlans.slice(1, 4)}
              height={350}
              simulationStats={simulationResult?.stats}
            />
          </div>

          {/* 内存占用（简化版，无圆环图） */}
          <div style={CHART_CARD_STYLE} className="transition-shadow hover:shadow-md">
            <div style={CHART_TITLE_STYLE}>
              <span className="font-semibold">内存占用分解</span>
              <div className="flex items-center gap-2">
                {(result.memory as any)?.is_estimated && (
                  <span className="text-[10px] text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
                    估算值
                  </span>
                )}
                {(() => {
                  // 基于实际利用率判断内存状态（前端二次校验）
                  // 从 hardware.chips 中获取第一个芯片的容量
                  const chips = hardware?.chips || {}
                  const firstChipName = Object.keys(chips)[0]

                  if (!firstChipName || !chips[firstChipName]?.memory?.gmem?.capacity_gb) {
                    return (
                      <span className="text-[11px] text-red-500">
                        ⚠ 配置错误：无法获取芯片容量
                      </span>
                    )
                  }

                  const chipMemoryGB = chips[firstChipName].memory.gmem.capacity_gb
                  const utilization = (result.memory.total_per_chip_gb / chipMemoryGB) * 100
                  const statusColor = utilization > 100 ? '#ff4d4f' : utilization > 90 ? '#faad14' : '#52c41a'
                  const statusText = utilization > 100 ? '✗ 内存溢出' : utilization > 90 ? '⚠ 接近上限' : '✓ 内存充足'

                  return (
                    <span
                      className="text-[11px]"
                      style={{ color: statusColor }}
                    >
                      {statusText}
                    </span>
                  )
                })()}
              </div>
            </div>
            <MemoryPieChart memory={result.memory} height={350} />
          </div>
        </div>

        {/* 第二行：拓扑流量分析 + 层级时间分解（高度大） */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* 拓扑流量分析 */}
          <div style={{ ...CHART_CARD_STYLE, display: 'flex', flexDirection: 'column' }} className="transition-shadow hover:shadow-md">
            <div style={{ ...CHART_TITLE_STYLE, flexShrink: 0 }}>
              <span className="font-semibold">拓扑流量分析</span>
              <span className="text-[11px] text-gray-500">
                链路带宽利用率与通信热点可视化
              </span>
            </div>
            <div style={{ flex: 1, minHeight: 400 }}>
              <TopologyTrafficChart
                topology={topology ?? null}
                linkTrafficStats={
                  externalStats?.linkTrafficStats ??
                  simulationResult?.stats?.linkTrafficStats ??
                  null
                }
              />
            </div>
          </div>

          {/* 层级时间分解 */}
          <div style={{ ...CHART_CARD_STYLE, display: 'flex', flexDirection: 'column', maxHeight: 800 }} className="transition-shadow hover:shadow-md">
            <div style={{ ...CHART_TITLE_STYLE, flexShrink: 0 }}>
              <span className="font-semibold">层级时间分解</span>
              <span className="text-[11px] text-gray-500">每层的计算/访存/通信占比</span>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
              <LayerWaterfallChart
                data={externalGanttData ?? simulationResult?.ganttChart ?? null}
              />
            </div>
          </div>
        </div>

        {/* 第三行：算子时间分解 + 甘特图（高度小） */}
        <div className="flex gap-4 mb-4">
          {/* 算子时间分解 */}
          <div style={CHART_CARD_STYLE} className="flex-1 min-w-0">
            <div style={CHART_TITLE_STYLE}>
              <span className="font-semibold">算子时间分解</span>
              <span className="text-[11px] text-gray-500">不同算子类型的时间占比</span>
            </div>
            <div>
              <OperatorTimeBreakdownChart
                data={externalGanttData ?? simulationResult?.ganttChart ?? null}
                height={250}
              />
            </div>
          </div>

          {/* 时序甘特图 */}
          <div style={CHART_CARD_STYLE} className="flex-1 min-w-0">
            <div style={CHART_TITLE_STYLE}>
              <span className="font-semibold">Prefill + Decode 时序甘特图</span>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-gray-500">点击任务查看详情</span>
                {(externalStats || simulationResult?.stats) && (
                  <span className="text-[11px] text-gray-500">
                    | TTFT: {(externalStats?.ttft ?? simulationResult?.stats.ttft ?? 0).toFixed(2)}ms
                    | Avg TPOT: {(externalStats?.avgTpot ?? simulationResult?.stats.avgTpot ?? 0).toFixed(2)}ms
                    | 动态MFU: {((externalStats?.dynamicMfu ?? simulationResult?.stats.dynamicMfu ?? 0) * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center justify-center" style={{ height: '280px', overflowX: 'auto' }}>
              <GanttChart
                data={externalGanttData ?? simulationResult?.ganttChart ?? null}
                showLegend
                onTaskClick={(task) => {
                  setSelectedTask(task)
                  setIsDrawerOpen(true)
                }}
              />
            </div>
          </div>
        </div>
      </BaseCard>

      {/* 任务详情侧边栏 */}
      <TaskDetailDrawer
        task={selectedTask}
        open={isDrawerOpen}
        onOpenChange={setIsDrawerOpen}
      />

    </div>
  )
}
