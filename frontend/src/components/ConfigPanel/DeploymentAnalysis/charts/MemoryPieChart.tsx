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
  { key: 'overhead', name: '其他开销', color: '#ff7a45' },
] as const

export const MemoryPieChart: React.FC<MemoryPieChartProps> = ({
  memory,
  height = 220,
  chipMemoryGB = 80,
}) => {
  const option: EChartsOption = useMemo(() => {
    const data = [
      { key: 'model', value: memory.model_memory_gb },
      { key: 'kv', value: memory.kv_cache_memory_gb },
      { key: 'activation', value: memory.activation_memory_gb },
      { key: 'overhead', value: memory.overhead_gb },
    ]

    const total = memory.total_per_chip_gb
    const utilization = (total / chipMemoryGB) * 100
    const warningThreshold = chipMemoryGB * 0.9  // 90% 警戒线

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: () => {
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
            <div style="font-weight: 600; margin-bottom: 8px; font-size: 13px;">显存分解</div>
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
        },
      },
      grid: {
        left: 80,
        right: 50,
        top: 40,
        bottom: 50,  // 为 legend 留出空间
      },
      xAxis: {
        type: 'value',
        max: chipMemoryGB,
        axisLabel: {
          formatter: (value: number) => `${value}GB`,
          fontSize: 10,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      yAxis: {
        type: 'category',
        data: ['显存'],
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: false },
      },
      series: [
        // 堆叠柱状图 - 各内存分量
        ...MEMORY_COMPONENTS.map((comp, i) => ({
          name: comp.name,
          type: 'bar' as const,
          stack: 'memory',
          barWidth: 32,
          data: [data[i].value],
          itemStyle: {
            color: comp.color,
            borderRadius: i === 0 ? [4, 0, 0, 4] : i === MEMORY_COMPONENTS.length - 1 ? [0, 4, 4, 0] : 0,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0,0,0,0.15)',
            },
          },
        })),
      ],
      // 警戒线和标注
      graphic: [
        // 90% 警戒线
        {
          type: 'line',
          left: `${80 + ((warningThreshold / chipMemoryGB) * (100 - 80 - 50) / 100) * 100}%`,
          top: 30,
          shape: {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: height - 60,
          },
          style: {
            stroke: '#ff4d4f',
            lineDash: [4, 4],
            lineWidth: 1.5,
          },
        },
        // 警戒线标签
        {
          type: 'text',
          left: `${80 + ((warningThreshold / chipMemoryGB) * (100 - 80 - 50) / 100) * 100}%`,
          top: 12,
          style: {
            text: '90%',
            fontSize: 10,
            fill: '#ff4d4f',
            textAlign: 'center',
          },
        },
        // 总量标注
        {
          type: 'group',
          right: 10,
          top: height / 2 - 20,
          children: [
            {
              type: 'text',
              style: {
                text: `${total.toFixed(1)}`,
                fontSize: 20,
                fontWeight: 'bold',
                fill: utilization > 90 ? '#ff4d4f' : '#333',
                textAlign: 'right',
              },
            },
            {
              type: 'text',
              top: 24,
              style: {
                text: `/ ${chipMemoryGB} GB`,
                fontSize: 11,
                fill: '#999',
                textAlign: 'right',
              },
            },
          ],
        },
      ],
      // 图例
      legend: {
        bottom: 5,
        itemWidth: 12,
        itemHeight: 8,
        itemGap: 16,
        textStyle: { fontSize: 10 },
        data: MEMORY_COMPONENTS.map(c => c.name),
      },
    }
  }, [memory, chipMemoryGB, height])

  return (
    <ReactECharts
      option={option}
      style={{ height }}
      opts={{ renderer: 'svg' }}
    />
  )
}
