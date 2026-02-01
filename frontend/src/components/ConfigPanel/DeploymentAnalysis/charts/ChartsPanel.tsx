/**
 * 图表面板 - 整合所有图表的容器组件（CSS Grid 布局）
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { toast } from 'sonner'
import { RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScoreRadarChart } from './ScoreRadarChart'
import { MetricsBarChart } from './MetricsBarChart'
import { MemoryPieChart } from './MemoryPieChart'
import { RooflineChart } from './RooflineChart'
import { GanttChart } from './GanttChart'
import { LayerWaterfallChart } from './LayerWaterfallChart'
import { CommunicationBreakdownChart } from './CommunicationBreakdownChart'
import { TaskDetailDrawer } from '../TaskDetailDrawer'
import { BaseCard } from "@/components/common/BaseCard"
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

/** 后端模拟 API 地址 */
const SIMULATION_API_URL = 'http://localhost:8001/api/simulate'

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

type MetricType = 'score' | 'ttft' | 'tpot' | 'throughput' | 'tps_per_batch' | 'tps_per_chip' | 'mfu' | 'mbu' | 'cost' | 'p99_ttft' | 'p99_tpot'

const chartCardStyle: React.CSSProperties = {
  background: '#fff',
  borderRadius: 12,
  padding: 16,
  border: '1px solid #E5E5E5',
  boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
}

const chartTitleStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: '#1a1a1a',
  marginBottom: 12,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
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
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('score')
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)

  // 判断是否有外部数据（来自后端评估结果）
  const hasExternalData = !!(externalGanttData || externalStats)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    charts: true,
    simulation: true,
    comparison: true,
  })
  const [activeSimTab, setActiveSimTab] = useState<'timeline' | 'layers' | 'communication'>('timeline')
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
        hardware: {
          chip: hardware.chip,
          board: hardware.board,
          rack: hardware.rack,
          pod: hardware.pod,
        },
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

  const metricOptions = [
    { value: 'score', label: '综合评分' },
    { value: 'ttft', label: 'TTFT (ms)' },
    { value: 'tpot', label: 'TPOT (ms)' },
    { value: 'throughput', label: '总吞吐 (tok/s)' },
    { value: 'tps_per_batch', label: 'TPS/Batch (tok/s)' },
    { value: 'tps_per_chip', label: 'TPS/Chip (tok/s)' },
    { value: 'mfu', label: 'MFU (%)' },
    { value: 'mbu', label: 'MBU (%)' },
    { value: 'cost', label: '成本 ($/M)' },
    { value: 'p99_ttft', label: 'TTFT P99 (ms)' },
    { value: 'p99_tpot', label: 'TPOT P99 (ms)' },
  ]

  return (
    <div>
      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 四、图表可视化 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <BaseCard collapsible
        title="图表可视化"
        open={expandedSections.charts}
        onOpenChange={(expanded) => setExpandedSections(prev => ({ ...prev, charts: expanded }))}
        className="mb-4"
      >
          <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: 16,
          }}
        >
          {/* 雷达图 - 左上 */}
          <div style={chartCardStyle}>
            <div style={chartTitleStyle}>
              <span className="font-semibold">多维评分分析</span>
              {topKPlans.length > 1 && (
                <span className="text-[11px] text-gray-500">
                  对比 Top-{Math.min(5, topKPlans.length)} 方案
                </span>
              )}
            </div>
            <ScoreRadarChart
              result={result}
              comparisonResults={topKPlans.slice(1, 5)}
              height={240}
            />
          </div>

          {/* 柱状图 - 右上 */}
          <div style={chartCardStyle}>
            <div style={chartTitleStyle}>
              <span className="font-semibold">多方案对比</span>
              <Select
                value={selectedMetric}
                onValueChange={(value: string) => setSelectedMetric(value as MetricType)}
              >
                <SelectTrigger className="w-[130px] h-7 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {metricOptions.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <MetricsBarChart
              plans={topKPlans}
              metric={selectedMetric}
              height={240}
            />
          </div>

          {/* 显存图 - 左下 */}
          <div style={chartCardStyle}>
            <div style={chartTitleStyle}>
              <span className="font-semibold">显存占用分解</span>
              <span
                className="text-[11px]"
                style={{ color: result.memory.is_memory_sufficient ? '#52c41a' : '#faad14' }}
              >
                {result.memory.is_memory_sufficient ? '✓ 显存充足' : '⚠ 显存不足'}
              </span>
            </div>
            <MemoryPieChart memory={result.memory} height={220} />
          </div>

          {/* Roofline 图 - 右下 */}
          <div style={chartCardStyle}>
            <div style={chartTitleStyle}>
              <span className="font-semibold">Roofline 性能分析</span>
              <span
                className="text-[11px]"
                style={{
                  color:
                    result.latency.bottleneck_type === 'memory'
                      ? '#1890ff'
                      : result.latency.bottleneck_type === 'compute'
                      ? '#52c41a'
                      : '#faad14',
                }}
              >
                {result.latency.bottleneck_type === 'memory'
                  ? '带宽受限'
                  : result.latency.bottleneck_type === 'compute'
                  ? '算力受限'
                  : '通信受限'}
              </span>
            </div>
            <RooflineChart
              result={result}
              hardware={hardware}
              model={model}
              comparisonResults={topKPlans.slice(1, 4)}
              height={220}
              simulationStats={simulationResult?.stats}
            />
          </div>
        </div>
      </BaseCard>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 五、推理时序模拟 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={{ marginBottom: 16 }}>
        <BaseCard
          title="推理时序模拟"
          accentColor="#faad14"
          collapsible
          expanded={expandedSections.simulation}
          onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, simulation: expanded }))}
          extra={
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {isSimulating ? (
                <span className="text-[11px] text-gray-500">模拟中...</span>
              ) : (externalStats || simulationResult?.stats) ? (
                <span className="text-[11px] text-gray-500">
                  TTFT: {(externalStats?.ttft ?? simulationResult?.stats.ttft ?? 0).toFixed(2)}ms |
                  Avg TPOT: {(externalStats?.avgTpot ?? simulationResult?.stats.avgTpot ?? 0).toFixed(2)}ms |
                  动态MFU: {((externalStats?.dynamicMfu ?? simulationResult?.stats.dynamicMfu ?? 0) * 100).toFixed(1)}%
                </span>
              ) : null}
              {/* 只有在没有外部数据时才显示刷新按钮 */}
              {!hasExternalData && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  disabled={!inference || isSimulating}
                  onClick={runSimulation}
                  title="重新运行模拟"
                >
                  <RefreshCw className={`h-4 w-4 ${isSimulating ? 'animate-spin' : ''}`} />
                </Button>
              )}
            </div>
          }
        >
          <Tabs value={activeSimTab} onValueChange={(v) => setActiveSimTab(v as typeof activeSimTab)}>
            <TabsList className="mb-4">
              <TabsTrigger value="timeline">时序甘特图</TabsTrigger>
              <TabsTrigger value="layers">层级分析</TabsTrigger>
              <TabsTrigger value="communication">通信分析</TabsTrigger>
            </TabsList>

            <TabsContent value="timeline">
              <div style={{ ...chartCardStyle, boxShadow: 'none', border: 'none' }}>
                <div style={chartTitleStyle}>
                  <span className="font-semibold">Prefill + Decode 时序甘特图</span>
                  <span className="text-[11px] text-gray-500">点击任务查看详情</span>
                </div>
                <GanttChart
                  data={externalGanttData ?? simulationResult?.ganttChart ?? null}
                  showLegend
                  onTaskClick={(task) => {
                    setSelectedTask(task)
                    setIsDrawerOpen(true)
                  }}
                />
              </div>
            </TabsContent>

            <TabsContent value="layers">
              <div style={{ ...chartCardStyle, boxShadow: 'none', border: 'none' }}>
                <div style={chartTitleStyle}>
                  <span className="font-semibold">层级时间分解</span>
                  <span className="text-[11px] text-gray-500">每层的计算/访存/通信占比</span>
                </div>
                <LayerWaterfallChart
                  data={externalGanttData ?? simulationResult?.ganttChart ?? null}
                  height={Math.max(300, ((externalGanttData ?? simulationResult?.ganttChart)?.tasks?.length ?? 0) > 100 ? 500 : 350)}
                />
              </div>
            </TabsContent>

            <TabsContent value="communication">
              <div style={{ ...chartCardStyle, boxShadow: 'none', border: 'none' }}>
                <div style={chartTitleStyle}>
                  <span className="font-semibold">通信开销分析</span>
                  <span className="text-[11px] text-gray-500">TP/PP/EP/SP 通信分解</span>
                </div>
                <CommunicationBreakdownChart
                  data={externalGanttData ?? simulationResult?.ganttChart ?? null}
                  height={400}
                />
              </div>
            </TabsContent>
          </Tabs>
        </BaseCard>
      </div>

      {/* 任务详情侧边栏 */}
      <TaskDetailDrawer
        task={selectedTask}
        open={isDrawerOpen}
        onOpenChange={setIsDrawerOpen}
      />

    </div>
  )
}
