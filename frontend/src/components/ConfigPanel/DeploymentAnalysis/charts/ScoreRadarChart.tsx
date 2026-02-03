/**
 * 雷达图 - 六维评分对比
 * 使用统一的评分计算器和主题配置
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'
import {
  calculateScores,
  scoresToRadarData,
  SCORE_RULES,
  ScoreInput,
} from '../../../../utils/llmDeployment/scoreCalculator'
import {
  CHART_SERIES_COLORS,
  RADAR_DIMENSIONS,
  getScoreColor,
} from './chartTheme'

interface ScoreRadarChartProps {
  result: PlanAnalysisResult
  comparisonResults?: PlanAnalysisResult[]
  height?: number
  /** 芯片显存容量 (GB)，用于显存评分计算 */
  chipMemoryGB?: number
}

/** 从 PlanAnalysisResult 提取评分输入 */
function extractScoreInput(result: PlanAnalysisResult, chipMemoryGB: number): ScoreInput {
  return {
    ttft: result.latency.prefill_total_latency_ms,
    tpot: result.latency.decode_per_token_latency_ms,
    tps: result.throughput.tokens_per_second,
    tpsPerChip: result.throughput.tps_per_chip || result.throughput.tokens_per_second / (result.plan.total_chips || 1),
    mfu: result.throughput.model_flops_utilization,
    mbu: result.throughput.memory_bandwidth_utilization,
    memoryUsedGB: result.memory.total_per_chip_gb,
    memoryCapacityGB: chipMemoryGB,
    prefillCommLatency: result.latency.prefill_comm_latency_ms,
    prefillComputeLatency: result.latency.prefill_compute_latency_ms,
    decodeCommLatency: result.latency.decode_comm_latency_ms,
    decodeComputeLatency: result.latency.decode_compute_latency_ms,
  }
}

export const ScoreRadarChart: React.FC<ScoreRadarChartProps> = ({
  result,
  comparisonResults = [],
  height = 280,
  chipMemoryGB = 80,
}) => {
  const option = useMemo((): EChartsOption => {
    const allResults = [result, ...comparisonResults.slice(0, 4)]

    const seriesData = allResults.map((r, index) => {
      // 使用统一的评分计算器
      const scoreInput = extractScoreInput(r, chipMemoryGB)
      const scores = calculateScores(scoreInput)
      const radarData = scoresToRadarData(scores)

      return {
        name: r.plan.plan_id,
        value: radarData,
        symbol: 'circle',
        symbolSize: index === 0 ? 6 : 4,
        lineStyle: {
          width: index === 0 ? 2.5 : 1.5,
        },
        areaStyle: {
          opacity: index === 0 ? 0.25 : 0.08,
        },
      }
    })

    return {
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        borderColor: 'transparent',
        padding: [12, 16],
        textStyle: {
          color: '#fff',
          fontSize: 12,
        },
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number[] }
          const dims = RADAR_DIMENSIONS.map(d => d.name)

          return `
            <div style="font-weight: bold; margin-bottom: 10px; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 8px;">
              ${p.name}
            </div>
            ${dims.map((dim, i) => {
              const ruleKey = `${RADAR_DIMENSIONS[i].key}Score` as keyof typeof SCORE_RULES
              const rule = SCORE_RULES[ruleKey]
              const score = p.value[i]?.toFixed(1) ?? '-'
              const scoreColor = getScoreColor(Number(score))
              return `
                <div style="margin: 8px 0; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500;">${dim}</span>
                    <span style="font-weight: 600; color: ${scoreColor};">${score}分</span>
                  </div>
                  <div style="font-size: 10px; color: rgba(255,255,255,0.6); margin-top: 4px;">
                    ${rule?.rule || ''}
                  </div>
                </div>
              `
            }).join('')}
            <div style="font-size: 10px; color: rgba(255,255,255,0.5); margin-top: 8px; text-align: center;">
              悬停维度查看详细规则
            </div>
          `
        },
      },
      legend: {
        show: allResults.length > 1,
        bottom: 0,
        itemWidth: 12,
        itemHeight: 8,
        textStyle: { fontSize: 10, color: '#666' },
      },
      radar: {
        indicator: RADAR_DIMENSIONS.map(d => ({
          name: d.name,
          max: 100,
        })),
        shape: 'polygon',
        radius: '60%',
        center: ['50%', allResults.length > 1 ? '45%' : '50%'],
        splitNumber: 4,
        axisName: {
          color: '#666',
          fontSize: 11,
          formatter: (name?: string) => {
            if (!name) return ''
            const dim = RADAR_DIMENSIONS.find(d => d.name === name)
            return dim ? `{name|${name}}\n{tip|${dim.tip}}` : name
          },
          rich: {
            name: {
              fontSize: 12,
              fontWeight: 500,
              color: '#333',
              lineHeight: 16,
            },
            tip: {
              fontSize: 9,
              color: '#999',
              lineHeight: 12,
            },
          },
        },
        splitLine: {
          lineStyle: { color: '#e8e8e8' },
        },
        splitArea: {
          areaStyle: {
            color: ['#fff', '#fafafa', '#f5f5f5', '#f0f0f0'],
          },
        },
        axisLine: {
          lineStyle: { color: '#d9d9d9' },
        },
      },
      series: [
        {
          type: 'radar',
          data: seriesData,
        },
      ],
      color: [...CHART_SERIES_COLORS],
    }
  }, [result, comparisonResults, chipMemoryGB])

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
        方案不可行，无法展示评分
      </div>
    )
  }

  return (
    <ReactECharts
      option={option}
      style={{ height }}
      opts={{ renderer: 'canvas' }}
    />
  )
}
