/**
 * 雷达图 - 六维评分对比
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'

interface ScoreRadarChartProps {
  result: PlanAnalysisResult
  comparisonResults?: PlanAnalysisResult[]
  height?: number
}

const COLORS = ['#5E6AD2', '#52c41a', '#faad14', '#722ed1', '#eb2f96']

// 评分规则说明
const SCORING_RULES: Record<string, { name: string; rule: string; tip: string }> = {
  '延迟': {
    name: '延迟评分',
    rule: 'TTFT < 100ms → 100分',
    tip: 'TTFT 越低越好，>1000ms 则 0 分',
  },
  '吞吐': {
    name: '吞吐评分',
    rule: 'MFU ≥ 50% → 100分',
    tip: 'MFU 越高越好',
  },
  '效率': {
    name: '效率评分',
    rule: '(计算 + 显存利用率) / 2',
    tip: '综合资源利用效率',
  },
  '均衡': {
    name: '均衡评分',
    rule: '负载均衡度 × 100',
    tip: 'TP/PP/EP 均匀切分时得分高',
  },
  '显存': {
    name: '显存评分',
    rule: '60-80% 利用率 → 100分',
    tip: '过高或过低都会扣分',
  },
  '通信': {
    name: '通信评分',
    rule: '(1 - 通信占比) × 100',
    tip: '通信开销越小越好',
  },
}

export const ScoreRadarChart: React.FC<ScoreRadarChartProps> = ({
  result,
  comparisonResults = [],
  height = 280,
}) => {
  const option = useMemo((): EChartsOption => {
    const allResults = [result, ...comparisonResults.slice(0, 4)]

    // 计算显存利用率评分 (60-80% 最优，超出则降分)
    const calcMemoryScore = (r: PlanAnalysisResult): number => {
      const utilization = r.memory.total_per_chip_gb / 80 // 假设 80GB 显存
      if (utilization <= 0.6) return utilization / 0.6 * 80  // 低于60%，线性增长到80分
      if (utilization <= 0.8) return 80 + (utilization - 0.6) / 0.2 * 20  // 60-80%，增长到100分
      if (utilization <= 0.95) return 100 - (utilization - 0.8) / 0.15 * 30  // 80-95%，降到70分
      return Math.max(0, 70 - (utilization - 0.95) / 0.05 * 70)  // >95%，急剧下降
    }

    // 计算通信效率评分 (通信占比越低越好)
    const calcCommScore = (r: PlanAnalysisResult): number => {
      const commRatio = r.latency.prefill_comm_latency_ms /
        (r.latency.prefill_compute_latency_ms + r.latency.prefill_comm_latency_ms + 0.01)
      return Math.max(0, Math.min(100, (1 - commRatio) * 100))
    }

    const seriesData = allResults.map((r, index) => ({
      name: r.plan.plan_id,
      value: [
        r.score.latency_score,
        r.score.throughput_score,
        r.score.efficiency_score,
        r.score.balance_score,
        calcMemoryScore(r),
        calcCommScore(r),
      ],
      symbol: 'circle',
      symbolSize: index === 0 ? 6 : 4,
      lineStyle: {
        width: index === 0 ? 2.5 : 1,
      },
      areaStyle: {
        opacity: index === 0 ? 0.25 : 0.08,
      },
    }))

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number[] }
          const dims = ['延迟', '吞吐', '效率', '均衡', '显存', '通信']
          return `
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px; border-bottom: 1px solid #e8e8e8; padding-bottom: 6px;">${p.name}</div>
            ${dims.map((dim, i) => {
              const rule = SCORING_RULES[dim]
              const score = p.value[i]?.toFixed(1) ?? '-'
              const scoreColor = Number(score) >= 80 ? '#52c41a' : Number(score) >= 60 ? '#faad14' : '#ff4d4f'
              return `
                <div style="margin: 6px 0; padding: 4px 0; border-bottom: 1px solid #f5f5f5;">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500;">${dim}</span>
                    <span style="font-weight: 600; color: ${scoreColor};">${score}分</span>
                  </div>
                  <div style="font-size: 11px; color: #999; margin-top: 2px;">${rule?.rule || ''}</div>
                </div>
              `
            }).join('')}
            <div style="font-size: 10px; color: #bbb; margin-top: 6px; text-align: center;">点击维度查看详细规则</div>
          `
        },
      },
      legend: {
        show: allResults.length > 1,
        bottom: 0,
        itemWidth: 12,
        itemHeight: 8,
        textStyle: { fontSize: 10 },
      },
      radar: {
        indicator: [
          { name: '延迟', max: 100 },
          { name: '吞吐', max: 100 },
          { name: '效率', max: 100 },
          { name: '均衡', max: 100 },
          { name: '显存', max: 100 },
          { name: '通信', max: 100 },
        ],
        shape: 'polygon',
        radius: '60%',
        center: ['50%', allResults.length > 1 ? '45%' : '50%'],
        splitNumber: 4,
        axisName: {
          color: '#666',
          fontSize: 11,
          formatter: (name?: string) => {
            if (!name) return ''
            const rule = SCORING_RULES[name]
            return rule ? `{name|${name}}\n{tip|${rule.tip}}` : name
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
      color: COLORS,
    }
  }, [result, comparisonResults])

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
      opts={{ renderer: 'svg' }}
    />
  )
}
