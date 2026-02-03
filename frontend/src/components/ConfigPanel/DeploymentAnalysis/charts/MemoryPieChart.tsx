/**
 * 显存占用可视化 - 堆叠柱状图
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { MemoryAnalysis } from '../../../../utils/llmDeployment/types'

interface MemoryPieChartProps {
  memory: MemoryAnalysis
  height?: number
  chipMemoryGB?: number  // 芯片显存容量，默认 80GB
}

const MEMORY_COMPONENTS = [
  { key: 'model', name: '模型参数', color: '#5E6AD2' },
  { key: 'kv', name: 'KV Cache', color: '#52c41a' },
  { key: 'activation', name: '激活值', color: '#faad14' },
] as const

export const MemoryPieChart: React.FC<MemoryPieChartProps> = React.memo(({
  memory,
  height = 220,
  chipMemoryGB = 80,
}) => {
  const option: EChartsOption = useMemo(() => {
    const data = [
      { key: 'model', value: memory.model_memory_gb },
      { key: 'kv', value: memory.kv_cache_memory_gb },
      { key: 'activation', value: memory.activation_memory_gb },
    ]

    const total = memory.total_per_chip_gb
    const utilization = (total / chipMemoryGB) * 100
    const warningThreshold = chipMemoryGB * 0.9  // 90% 警戒线
    // 动态调整x轴最大值：如果总内存超过芯片容量，则扩展x轴以显示完整内存
    const xAxisMax = Math.max(chipMemoryGB, total * 1.05)  // 留5%余量

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.componentSubType === 'bar') {
            const lines = data.map((d, i) => {
              const comp = MEMORY_COMPONENTS[i]
              const percent = ((d.value / chipMemoryGB) * 100).toFixed(1)
              return `<div style="display: flex; align-items: center; margin: 2px 0;">
                <span style="display: inline-block; width: 10px; height: 10px; background: ${comp.color}; border-radius: 2px; margin-right: 8px;"></span>
                <span style="flex: 1;">${comp.name}</span>
                <span style="font-weight: 500; margin-left: 12px;">${d.value.toFixed(2)} GB</span>
                <span style="color: #999; margin-left: 8px;">(${percent}%)</span>
              </div>`
            }).join('')

            return `
              <div style="font-weight: 600; margin-bottom: 8px; font-size: 13px;">内存分解</div>
              ${lines}
              <div style="border-top: 1px solid #e8e8e8; margin-top: 8px; padding-top: 8px;">
                <div style="display: flex; justify-content: space-between;">
                  <span>总计</span>
                  <span style="font-weight: 600;">${total.toFixed(2)} GB / ${chipMemoryGB} GB</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                  <span>利用率</span>
                  <span style="color: ${utilization > 90 ? '#ff4d4f' : utilization > 80 ? '#faad14' : '#52c41a'}; font-weight: 600;">
                    ${utilization.toFixed(1)}%
                  </span>
                </div>
              </div>
            `
          }
          return ''
        },
      },
      // 柱状图 grid（占满整个宽度）
      grid: {
        left: '8%',
        right: '8%',
        top: '15%',
        bottom: '20%',
      },
      xAxis: {
        type: 'value',
        max: xAxisMax,
        axisLabel: {
          formatter: (value: number) => `${value}GB`,
          fontSize: 12,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      yAxis: {
        type: 'category',
        data: ['内存'],
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: false },
      },
      series: [
        // 堆叠柱状图 - 各内存分量（确保所有组件都显示）
        ...MEMORY_COMPONENTS.map((comp, i) => ({
          name: comp.name,
          type: 'bar' as const,
          stack: 'memory',
          barWidth: 35,
          data: [data[i].value],
          itemStyle: {
            color: comp.color,
            borderRadius: i === 0 ? [6, 0, 0, 6] : i === MEMORY_COMPONENTS.length - 1 ? [0, 6, 6, 0] : 0,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0,0,0,0.15)',
            },
          },
          // 即使值为0也显示
          showBackground: false,
        })),
      ],
      // 警戒线和标注
      graphic: [
        // 90% 警戒线
        {
          type: 'line',
          left: `${8 + ((warningThreshold / xAxisMax) * 84)}%`,
          top: '15%',
          shape: {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: height * 0.65,
          },
          style: {
            stroke: '#faad14',
            lineDash: [5, 5],
            lineWidth: 2,
          },
        },
        // 90% 警戒线标签
        {
          type: 'text',
          left: `${8 + ((warningThreshold / xAxisMax) * 84)}%`,
          top: '10%',
          style: {
            text: '90%',
            fontSize: 11,
            fill: '#faad14',
            textAlign: 'center',
            fontWeight: 600,
          },
        },
        // 100% 容量线（当内存超出时显示）
        ...(total > chipMemoryGB ? [
          {
            type: 'line',
            left: `${8 + ((chipMemoryGB / xAxisMax) * 84)}%`,
            top: '15%',
            shape: {
              x1: 0,
              y1: 0,
              x2: 0,
              y2: height * 0.65,
            },
            style: {
              stroke: '#ff4d4f',
              lineDash: [4, 4],
              lineWidth: 2.5,
            },
          },
          {
            type: 'text',
            left: `${8 + ((chipMemoryGB / xAxisMax) * 84)}%`,
            top: '10%',
            style: {
              text: '100%',
              fontSize: 11,
              fill: '#ff4d4f',
              textAlign: 'center',
              fontWeight: 700,
            },
          },
        ] : []),
      ],
      // 图例
      legend: {
        bottom: '3%',
        left: 'center',
        itemWidth: 12,
        itemHeight: 10,
        itemGap: 16,
        textStyle: { fontSize: 12 },
        data: MEMORY_COMPONENTS.map(c => c.name),
      },
    }
  }, [memory, chipMemoryGB, height])

  return (
    <ReactECharts
      option={option}
      style={{ height }}
      opts={{ renderer: 'canvas' }}
      notMerge={false}
      lazyUpdate={true}
    />
  )
})
