/**
 * 柱状图 - 多方案指标对比
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'

type MetricType = 'ttft' | 'tpot' | 'throughput' | 'tps_per_batch' | 'tps_per_chip' | 'mfu' | 'mbu' | 'cost' | 'p99_ttft' | 'p99_tpot' | 'score'

interface MetricsBarChartProps {
  plans: PlanAnalysisResult[]
  metric: MetricType
  height?: number
}

const METRIC_CONFIG: Record<MetricType, {
  name: string
  unit: string
  accessor: (p: PlanAnalysisResult) => number
  colorStart: string   // 渐变起始色（柱子底部/排名靠后）
  colorEnd: string     // 渐变结束色（柱子顶部/排名靠前）
  lowerIsBetter?: boolean
}> = {
  ttft: {
    name: 'FTL',
    unit: 'ms',
    accessor: (p) => p.latency.prefill_total_latency_ms,
    colorStart: '#69c0ff',
    colorEnd: '#1890ff',
    lowerIsBetter: true,
  },
  tpot: {
    name: 'TPOT',
    unit: 'ms',
    accessor: (p) => p.latency.decode_per_token_latency_ms,
    colorStart: '#5cdbd3',
    colorEnd: '#13c2c2',
    lowerIsBetter: true,
  },
  throughput: {
    name: '总吞吐',
    unit: 'tok/s',
    accessor: (p) => p.throughput.tokens_per_second,
    colorStart: '#95de64',
    colorEnd: '#52c41a',
  },
  tps_per_batch: {
    name: 'TPS/Batch',
    unit: 'tok/s',
    accessor: (p) => p.throughput.tps_per_batch,
    colorStart: '#7cb342',
    colorEnd: '#388e3c',
  },
  tps_per_chip: {
    name: 'TPS/Chip',
    unit: 'tok/s',
    accessor: (p) => p.throughput.tps_per_chip,
    colorStart: '#66bb6a',
    colorEnd: '#2e7d32',
  },
  mfu: {
    name: 'MFU',
    unit: '%',
    accessor: (p) => p.throughput.model_flops_utilization * 100,
    colorStart: '#ffd666',
    colorEnd: '#faad14',
  },
  mbu: {
    name: 'MBU',
    unit: '%',
    accessor: (p) => p.throughput.memory_bandwidth_utilization * 100,
    colorStart: '#b37feb',
    colorEnd: '#722ed1',
  },
  cost: {
    name: '成本',
    unit: '$/M',
    accessor: (p) => p.cost?.cost_per_million_tokens ?? 0,
    colorStart: '#ff9c6e',
    colorEnd: '#fa541c',
    lowerIsBetter: true,
  },
  p99_ttft: {
    name: 'FTL P99',
    unit: 'ms',
    accessor: (p) => p.latency.ttft_percentiles?.p99 ?? p.latency.prefill_total_latency_ms * 1.8,
    colorStart: '#85a5ff',
    colorEnd: '#2f54eb',
    lowerIsBetter: true,
  },
  p99_tpot: {
    name: 'TPOT P99',
    unit: 'ms',
    accessor: (p) => p.latency.tpot_percentiles?.p99 ?? p.latency.decode_per_token_latency_ms * 1.5,
    colorStart: '#87e8de',
    colorEnd: '#08979c',
    lowerIsBetter: true,
  },
  score: {
    name: '综合评分',
    unit: '分',
    accessor: (p) => p.score.overall_score,
    colorStart: '#8B93DC',
    colorEnd: '#5E6AD2',
  },
}

// 根据排名生成渐变色 - 排名越靠前颜色越深
const getRankGradient = (
  _rank: number,
  _total: number,
  colorStart: string,
  colorEnd: string
): { type: 'linear'; x: number; y: number; x2: number; y2: number; colorStops: { offset: number; color: string }[] } => {
  return {
    type: 'linear',
    x: 0,
    y: 1,
    x2: 0,
    y2: 0,
    colorStops: [
      { offset: 0, color: colorStart },
      { offset: 1, color: colorEnd },
    ],
  }
}

// 根据排名计算透明度
const getRankOpacity = (rank: number, total: number): number => {
  if (total <= 1) return 1
  // 第1名100%，最后一名50%
  return 1 - (rank / (total - 1)) * 0.5
}

export const MetricsBarChart: React.FC<MetricsBarChartProps> = ({
  plans,
  metric,
  height = 250,
}) => {
  const config = METRIC_CONFIG[metric]

  const option: EChartsOption = useMemo(() => {
    const feasiblePlans = plans.filter((p) => p.is_feasible)
    const labels = feasiblePlans.map((p) => p.plan.plan_id)
    const values = feasiblePlans.map((p) => config.accessor(p))

    // 计算排名（根据 lowerIsBetter 决定排序方向）
    const sortedIndices = values
      .map((v, i) => ({ value: v, index: i }))
      .sort((a, b) => config.lowerIsBetter ? a.value - b.value : b.value - a.value)
      .map((item, rank) => ({ ...item, rank }))

    // 创建索引到排名的映射
    const rankMap = new Map<number, number>()
    sortedIndices.forEach((item) => rankMap.set(item.index, item.rank))

    const total = values.length

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: unknown) => {
          const items = params as { name: string; value: number; dataIndex: number }[]
          const item = items[0]
          const rank = rankMap.get(item.dataIndex) ?? 0
          return `
            <div style="font-weight: 500;">${item.name}</div>
            <div style="margin-top: 4px;">${config.name}: <b>${item.value.toFixed(2)}</b> ${config.unit}</div>
            <div style="color: ${rank === 0 ? '#52c41a' : '#666'}; margin-top: 2px;">排名: ${rank + 1}/${total}</div>
          `
        },
      },
      grid: {
        left: 50,
        right: 20,
        top: 20,
        bottom: 40,
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          rotate: labels.length > 4 ? 30 : 0,
          fontSize: 10,
          interval: 0,
        },
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: `${config.name} (${config.unit})`,
        nameTextStyle: { fontSize: 10, color: '#666' },
        axisLabel: { fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#f0f0f0' } },
      },
      series: [
        {
          type: 'bar',
          data: values.map((v, i) => {
            const rank = rankMap.get(i) ?? 0
            const opacity = getRankOpacity(rank, total)
            return {
              value: v,
              itemStyle: {
                color: getRankGradient(rank, total, config.colorStart, config.colorEnd),
                opacity,
                borderRadius: [6, 6, 0, 0],
                // 微光效果：排名第一的方案添加阴影
                shadowColor: rank === 0 ? config.colorEnd : 'transparent',
                shadowBlur: rank === 0 ? 8 : 0,
                shadowOffsetY: rank === 0 ? 2 : 0,
              },
            }
          }),
          barMaxWidth: 50,
          label: {
            show: true,
            position: 'top',
            fontSize: 10,
            fontWeight: 500,
            color: '#333',
            formatter: (params: unknown) => {
              const p = params as { value: number; dataIndex: number }
              const rank = rankMap.get(p.dataIndex) ?? 0
              // 排名第一添加标记
              return rank === 0 ? `★ ${p.value.toFixed(1)}` : p.value.toFixed(1)
            },
          },
        },
      ],
    }
  }, [plans, metric, config])

  if (plans.filter((p) => p.is_feasible).length === 0) {
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
        无可行方案
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
