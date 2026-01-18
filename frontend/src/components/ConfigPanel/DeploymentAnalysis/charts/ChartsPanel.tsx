/**
 * 图表面板 - 整合所有图表的容器组件（CSS Grid 布局）
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Typography, Select, Empty, Button, Tooltip, message } from 'antd'
import {
  ReloadOutlined,
  CloudServerOutlined,
  DesktopOutlined,
} from '@ant-design/icons'
import { ScoreRadarChart } from './ScoreRadarChart'
import { MetricsBarChart } from './MetricsBarChart'
import { MemoryPieChart } from './MemoryPieChart'
import { RooflineChart } from './RooflineChart'
import { GanttChart } from './GanttChart'
import { ComparisonTable } from './ComparisonTable'
import { BaseCard } from '../../../common/BaseCard'
import { compareFormulaAndSimulation } from '../../../../utils/llmDeployment/simulationScorer'
import {
  PlanAnalysisResult,
  HardwareConfig,
  LLMModelConfig,
  InferenceConfig,
  GanttChartData,
  SimulationStats,
} from '../../../../utils/llmDeployment/types'
import {
  runInferenceSimulation,
  type SimulationResult,
} from '../../../../utils/llmDeployment'
import { HierarchicalTopology } from '../../../../types'

const { Text } = Typography

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
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('score')
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [useBackend, setUseBackend] = useState(true)  // 默认使用后端模拟
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    charts: true,
    simulation: true,
    comparison: true,
  })

  // 记录上次运行的result id，避免重复运行
  const lastResultIdRef = useRef<string | null>(null)

  // 检查后端是否可用
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8001/health', { method: 'GET' })
        setBackendAvailable(response.ok)
      } catch {
        setBackendAvailable(false)
      }
    }
    checkBackend()
  }, [])

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
          node: hardware.node,
          cluster: hardware.cluster,
        },
        config: {
          maxSimulatedTokens: 16,
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

      // 转换为 SimulationResult 格式
      setSimulationResult({
        config: {
          granularity: 'layer',
          enableOverlap: true,
          enablePipeline: true,
          jitterFactor: 0.05,
          maxSimulatedTokens: 16,
        },
        events: [],
        commTrace: [],
        ganttChart: backendResult.ganttChart,
        stats: backendResult.stats,
        timestamp: backendResult.timestamp,
      })
    } catch (error) {
      console.error('后端模拟失败:', error)
      message.warning('后端模拟不可用，已切换到前端模拟')
      setUseBackend(false)
      runFrontendSimulation()
    } finally {
      setIsSimulating(false)
    }
  }, [result, inference, topology, model, hardware])

  // 运行前端模拟
  const runFrontendSimulation = useCallback(() => {
    if (!result || !inference) return

    setIsSimulating(true)
    setTimeout(() => {
      try {
        const simResult = runInferenceSimulation(
          model,
          inference,
          result.plan.parallelism,
          hardware,
          { maxSimulatedTokens: 16 }
        )
        setSimulationResult(simResult)
      } catch (error) {
        console.error('模拟失败:', error)
      } finally {
        setIsSimulating(false)
      }
    }, 50)
  }, [result, inference, model, hardware])

  // 运行模拟（根据设置选择后端或前端）
  const runSimulation = useCallback(() => {
    if (useBackend && topology && backendAvailable !== false) {
      runBackendSimulation()
    } else {
      runFrontendSimulation()
    }
  }, [useBackend, topology, backendAvailable, runBackendSimulation, runFrontendSimulation])

  // 当分析结果变化时自动运行模拟
  useEffect(() => {
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
  }, [result, inference, runSimulation])

  if (!result) {
    return (
      <Empty
        description="请先运行分析以查看图表"
        style={{ marginTop: 40 }}
      />
    )
  }

  const metricOptions = [
    { value: 'score', label: '综合评分' },
    { value: 'ttft', label: 'FTL (ms)' },
    { value: 'tpot', label: 'TPOT (ms)' },
    { value: 'throughput', label: '总吞吐 (tok/s)' },
    { value: 'tps_per_batch', label: 'TPS/Batch (tok/s)' },
    { value: 'tps_per_chip', label: 'TPS/Chip (tok/s)' },
    { value: 'mfu', label: 'MFU (%)' },
    { value: 'mbu', label: 'MBU (%)' },
    { value: 'cost', label: '成本 ($/M)' },
    { value: 'p99_ttft', label: 'FTL P99 (ms)' },
    { value: 'p99_tpot', label: 'TPOT P99 (ms)' },
  ]

  return (
    <div>
      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 四、图表可视化 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={{ marginBottom: 16 }}>
        <BaseCard
          title="图表可视化"
          accentColor="#eb2f96"
          collapsible
          expanded={expandedSections.charts}
          onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, charts: expanded }))}
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
              <Text strong>多维评分分析</Text>
              {topKPlans.length > 1 && (
                <Text type="secondary" style={{ fontSize: 11 }}>
                  对比 Top-{Math.min(5, topKPlans.length)} 方案
                </Text>
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
              <Text strong>多方案对比</Text>
              <Select
                size="small"
                value={selectedMetric}
                onChange={setSelectedMetric}
                options={metricOptions}
                style={{ width: 130 }}
              />
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
              <Text strong>显存占用分解</Text>
              <Text
                type="secondary"
                style={{
                  fontSize: 11,
                  color: result.memory.is_memory_sufficient ? '#52c41a' : '#faad14',
                }}
              >
                {result.memory.is_memory_sufficient ? '✓ 显存充足' : '⚠ 显存不足'}
              </Text>
            </div>
            <MemoryPieChart memory={result.memory} height={220} />
          </div>

          {/* Roofline 图 - 右下 */}
          <div style={chartCardStyle}>
            <div style={chartTitleStyle}>
              <Text strong>Roofline 性能分析</Text>
              <Text
                type="secondary"
                style={{
                  fontSize: 11,
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
              </Text>
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
      </div>

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
        >
          <div style={{ ...chartCardStyle, boxShadow: 'none', border: 'none' }}>
          <div style={chartTitleStyle}>
            <Text strong>Prefill + Decode 时序甘特图</Text>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {isSimulating ? (
                <Text type="secondary" style={{ fontSize: 11 }}>模拟中...</Text>
              ) : simulationResult ? (
                <Text type="secondary" style={{ fontSize: 11 }}>
                  TTFT: {simulationResult.stats.ttft.toFixed(2)}ms |
                  Avg TPOT: {simulationResult.stats.avgTpot.toFixed(2)}ms |
                  动态MFU: {(simulationResult.stats.dynamicMfu * 100).toFixed(1)}%
                </Text>
              ) : null}
              <Tooltip title={useBackend ? '使用后端模拟 (精细模式)' : '使用前端模拟 (快速模式)'}>
                <Button
                  type="text"
                  size="small"
                  icon={useBackend ? <CloudServerOutlined /> : <DesktopOutlined />}
                  onClick={() => {
                    setUseBackend(!useBackend)
                    lastResultIdRef.current = null  // 强制重新运行
                  }}
                  style={{ color: backendAvailable === false ? '#999' : undefined }}
                />
              </Tooltip>
              <Tooltip title="重新运行模拟">
                <Button
                  type="text"
                  size="small"
                  icon={<ReloadOutlined spin={isSimulating} />}
                  disabled={!inference || isSimulating}
                  onClick={runSimulation}
                />
              </Tooltip>
            </div>
          </div>
          <GanttChart
            data={simulationResult?.ganttChart ?? null}
            showLegend
          />
        </div>
        </BaseCard>
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 六、公式 vs 仿真结果 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {simulationResult && (
        <div style={{ marginBottom: 16 }}>
          <BaseCard
            title="公式 vs 仿真结果"
            accentColor="#1890ff"
            collapsible
            expanded={expandedSections.comparison}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, comparison: expanded }))}
          >
            <div style={{ ...chartCardStyle, boxShadow: 'none', border: 'none' }}>
            <div style={chartTitleStyle}>
              <Text strong>理论估算与事件驱动模拟对比</Text>
              <Text type="secondary" style={{ fontSize: 11 }}>
                误差分析与评分
              </Text>
            </div>
            <ComparisonTable
              comparison={compareFormulaAndSimulation(
                result,
                simulationResult.stats,
                undefined,  // 使用默认权重
                inference ? {
                  model,
                  inference,
                  parallelism: result.plan.parallelism,
                  hardware,
                } : undefined
              )}
            />
          </div>
          </BaseCard>
        </div>
      )}
    </div>
  )
}
