/**
 * Roofline 图 - 性能瓶颈分析
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  PlanAnalysisResult,
  HardwareConfig,
  LLMModelConfig,
  SimulationStats,
} from '../../../../utils/llmDeployment/types'

interface RooflineChartProps {
  result: PlanAnalysisResult
  hardware: HardwareConfig
  model: LLMModelConfig
  comparisonResults?: PlanAnalysisResult[]
  height?: number
  simulationStats?: SimulationStats
}

const COLORS = ['#5E6AD2', '#52c41a', '#faad14', '#722ed1']

export const RooflineChart: React.FC<RooflineChartProps> = ({
  result,
  hardware,
  model: _model,
  comparisonResults = [],
  height = 280,
  simulationStats,
}) => {
  const option: EChartsOption = useMemo(() => {
    const peakTflops = hardware.chip.compute_tflops_fp16
    const memoryBandwidthTBps = hardware.chip.memory_bandwidth_gbps / 1000
    const ridgePoint = peakTflops / memoryBandwidthTBps

    // 生成 Roofline 边界线数据点
    const rooflineData: [number, number][] = []
    const minOI = 0.1
    const maxOI = 1000

    for (let oi = minOI; oi <= maxOI; oi *= 1.5) {
      const memoryBoundPerf = oi * memoryBandwidthTBps
      const actualPerf = Math.min(peakTflops, memoryBoundPerf)
      rooflineData.push([oi, actualPerf])
    }

    // 生成带宽受限区域的填充数据（拐点左侧）
    const memoryBoundAreaData: [number, number][] = []
    for (let oi = minOI; oi <= ridgePoint; oi *= 1.3) {
      memoryBoundAreaData.push([oi, oi * memoryBandwidthTBps])
    }
    memoryBoundAreaData.push([ridgePoint, peakTflops])

    // 生成算力受限区域的填充数据（拐点右侧）
    const computeBoundAreaData: [number, number][] = [
      [ridgePoint, peakTflops],
    ]
    for (let oi = ridgePoint * 1.3; oi <= maxOI; oi *= 1.3) {
      computeBoundAreaData.push([oi, peakTflops])
    }
    computeBoundAreaData.push([maxOI, peakTflops])

    // 计算当前方案的工作点
    const calculateWorkPoint = (r: PlanAnalysisResult, isMainResult: boolean = false) => {
      // 优先使用后端仿真的精确值（仅对主方案）
      if (isMainResult && simulationStats && simulationStats.dynamicMfu > 0) {
        const achievedTflops = simulationStats.dynamicMfu * peakTflops
        // 根据 MBU 和 MFU 判断瓶颈类型
        const isMemoryBound = simulationStats.dynamicMbu > simulationStats.dynamicMfu
        const bottleneck = isMemoryBound ? 'memory' : 'compute'

        // 反推算术强度：如果是 memory-bound，AI = perf / bandwidth
        const operationalIntensity = isMemoryBound
          ? Math.max(0.1, achievedTflops / memoryBandwidthTBps)
          : ridgePoint * 1.2

        return {
          oi: operationalIntensity,
          perf: achievedTflops,
          planId: r.plan.plan_id,
          bottleneck: bottleneck as 'memory' | 'compute' | 'communication' | 'balanced',
          phase: 'decode' as const,
        }
      }

      // 使用 bottleneck_analysis 中的计算值（对比方案或无仿真结果时）
      if (r.latency.bottleneck_analysis) {
        const ba = r.latency.bottleneck_analysis
        const dominantPhase = ba.dominant_phase === 'prefill' ? ba.prefill : ba.decode
        const achievedTflops = dominantPhase.utilization * peakTflops

        return {
          oi: dominantPhase.arithmetic_intensity,
          perf: achievedTflops,
          planId: r.plan.plan_id,
          bottleneck: r.latency.bottleneck_type,
          phase: ba.dominant_phase,
        }
      }

      // 回退：简化估算
      const achievedTflops = r.throughput.model_flops_utilization * peakTflops
      const operationalIntensity = r.latency.bottleneck_type === 'memory'
        ? achievedTflops / memoryBandwidthTBps * 0.8
        : ridgePoint * 1.2

      return {
        oi: operationalIntensity,
        perf: achievedTflops,
        planId: r.plan.plan_id,
        bottleneck: r.latency.bottleneck_type,
        phase: 'decode' as const,
      }
    }

    const allResults = [result, ...comparisonResults.slice(0, 3)]
    const workPoints = allResults
      .filter((r) => r.is_feasible)
      .map((r, index) => calculateWorkPoint(r, index === 0))

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { seriesName: string; data: [number, number]; name?: string }
          if (p.seriesName === 'Roofline' || p.seriesName === '带宽受限区' || p.seriesName === '算力受限区') {
            return `算术强度: ${p.data[0].toFixed(1)} FLOP/Byte<br/>性能上限: ${p.data[1].toFixed(1)} TFLOPS`
          }
          const point = workPoints.find((pt) => pt.planId === p.name)
          if (point) {
            return `
              <div style="font-weight: bold;">${point.planId}</div>
              <div>算术强度: ${point.oi.toFixed(1)} FLOP/Byte</div>
              <div>实际性能: ${point.perf.toFixed(2)} TFLOPS</div>
              <div style="margin-top: 4px; padding-top: 4px; border-top: 1px solid #e8e8e8;">
                瓶颈: <span style="color: ${point.bottleneck === 'memory' ? '#1890ff' : point.bottleneck === 'compute' ? '#52c41a' : '#faad14'}; font-weight: 500;">
                  ${point.bottleneck === 'memory' ? '带宽受限' : point.bottleneck === 'compute' ? '算力受限' : '通信受限'}
                </span>
              </div>
            `
          }
          return ''
        },
      },
      legend: {
        show: workPoints.length > 1,
        bottom: 5,
        itemWidth: 10,
        itemHeight: 10,
        itemGap: 12,
        textStyle: { fontSize: 10 },
      },
      grid: {
        left: 60,
        right: 30,
        top: 30,
        bottom: workPoints.length > 1 ? 60 : 30,  // 为 legend 和 X 轴名称留出空间
      },
      xAxis: {
        type: 'log',
        name: '算术强度 (FLOP/Byte)',
        nameLocation: 'middle',
        nameGap: 25,
        nameTextStyle: { fontSize: 11, color: '#666' },
        min: 0.1,
        max: 1000,
        axisLabel: { fontSize: 10 },
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      yAxis: {
        type: 'log',
        name: '性能 (TFLOPS)',
        nameLocation: 'middle',
        nameGap: 40,
        nameTextStyle: { fontSize: 11, color: '#666' },
        min: 0.1,
        max: peakTflops * 1.5,
        axisLabel: { fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      series: [
        // 带宽受限区域填充（蓝色）
        {
          name: '带宽受限区',
          type: 'line',
          data: memoryBoundAreaData,
          smooth: false,
          symbol: 'none',
          lineStyle: { width: 0 },
          areaStyle: {
            color: 'rgba(24, 144, 255, 0.08)',
          },
          z: 1,
        },
        // 算力受限区域填充（绿色）
        {
          name: '算力受限区',
          type: 'line',
          data: computeBoundAreaData,
          smooth: false,
          symbol: 'none',
          lineStyle: { width: 0 },
          areaStyle: {
            color: 'rgba(82, 196, 26, 0.08)',
          },
          z: 1,
        },
        // Roofline 边界线
        {
          name: 'Roofline',
          type: 'line',
          data: rooflineData,
          smooth: false,
          symbol: 'none',
          lineStyle: {
            color: '#ff4d4f',
            width: 2.5,
          },
          z: 10,
        },
        // 工作点
        ...workPoints.map((point, index) => ({
          name: point.planId,
          type: 'scatter' as const,
          data: [[point.oi, point.perf]],
          symbolSize: index === 0 ? 16 : 12,
          itemStyle: {
            color: COLORS[index % COLORS.length],
            borderColor: '#fff',
            borderWidth: 2,
            shadowBlur: index === 0 ? 8 : 0,
            shadowColor: index === 0 ? 'rgba(94, 106, 210, 0.4)' : 'transparent',
          },
          label: {
            show: index === 0,
            position: 'top' as const,
            formatter: point.planId,
            fontSize: 11,
            fontWeight: 500,
            color: '#333',
          },
          z: 20,
        })),
        // 峰值性能线
        {
          name: '峰值算力',
          type: 'line',
          data: [[ridgePoint, peakTflops], [1000, peakTflops]],
          symbol: 'none',
          lineStyle: {
            color: '#52c41a',
            width: 1.5,
            type: 'dashed',
          },
          z: 5,
        },
        // 拐点标记
        {
          name: '拐点',
          type: 'scatter',
          data: [[ridgePoint, peakTflops]],
          symbolSize: 10,
          itemStyle: {
            color: '#52c41a',
            borderColor: '#fff',
            borderWidth: 2,
          },
          label: {
            show: true,
            position: 'right',
            formatter: `拐点: ${ridgePoint.toFixed(1)}`,
            fontSize: 10,
            fontWeight: 500,
            color: '#52c41a',
          },
          z: 15,
        },
      ],
      // 区域标签（使用 graphic）
      graphic: [
        // 带宽受限区标签
        {
          type: 'text',
          left: '15%',
          top: '20%',
          style: {
            text: '带宽受限区',
            fontSize: 11,
            fill: 'rgba(24, 144, 255, 0.6)',
            fontWeight: 500,
          },
          z: 5,
        },
        // 算力受限区标签
        {
          type: 'text',
          right: '15%',
          top: '20%',
          style: {
            text: '算力受限区',
            fontSize: 11,
            fill: 'rgba(82, 196, 26, 0.6)',
            fontWeight: 500,
          },
          z: 5,
        },
      ],
    }
  }, [result, hardware, comparisonResults, simulationStats])

  if (!result.is_feasible) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#999',
          fontSize: 12,
        }}
      >
        方案不可行，无法展示 Roofline 图
      </div>
    )
  }

  return (
    <ReactECharts
      option={option}
      style={{ height }}
      opts={{ renderer: 'svg' }}
    />
  )
}
