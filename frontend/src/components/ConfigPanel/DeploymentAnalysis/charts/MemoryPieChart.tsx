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

    // 饼图数据
    const pieData = MEMORY_COMPONENTS.map((comp, i) => ({
      name: comp.name,
      value: data[i].value,
      itemStyle: { color: comp.color },
    }))

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
          } else {
            // 饼图 tooltip
            const percent = ((params.value / total) * 100).toFixed(1)
            const absPercent = ((params.value / chipMemoryGB) * 100).toFixed(1)
            return `
              <div style="font-weight: 600; margin-bottom: 4px;">${params.name}</div>
              <div>占总量: <span style="font-weight: 500;">${percent}%</span></div>
              <div>绝对值: <span style="font-weight: 500;">${params.value.toFixed(2)} GB</span></div>
              <div>占容量: <span style="font-weight: 500;">${absPercent}%</span></div>
            `
          }
        },
      },
      // 左侧 grid（柱状图）
      grid: {
        left: '5%',
        right: '52%',
        top: '12%',
        bottom: '18%',
      },
      xAxis: {
        type: 'value',
        max: chipMemoryGB,
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
        // 右侧饼图
        {
          name: '内存占比',
          type: 'pie',
          radius: ['48%', '78%'],
          center: ['73%', '42%'],
          data: pieData.filter(d => d.value > 0),  // 过滤掉0值
          label: {
            formatter: '{b}',  // 只显示名称
            fontSize: 11,
            color: '#666',
          },
          labelLine: {
            length: 12,
            length2: 8,
            lineStyle: {
              width: 1,
            },
          },
          emphasis: {
            label: {
              fontSize: 13,
              fontWeight: 'bold',
            },
            itemStyle: {
              shadowBlur: 15,
              shadowColor: 'rgba(0, 0, 0, 0.3)',
            },
          },
        },
      ],
      // 警戒线和标注
      graphic: [
        // 90% 警戒线（左侧柱状图）
        {
          type: 'line',
          left: `${5 + ((warningThreshold / chipMemoryGB) * 43)}%`,
          top: '15%',
          shape: {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: height * 0.65,
          },
          style: {
            stroke: '#ff4d4f',
            lineDash: [5, 5],
            lineWidth: 2,
          },
        },
        // 警戒线标签
        {
          type: 'text',
          left: `${5 + ((warningThreshold / chipMemoryGB) * 43)}%`,
          top: '10%',
          style: {
            text: '90%',
            fontSize: 11,
            fill: '#ff4d4f',
            textAlign: 'center',
            fontWeight: 600,
          },
        },
        // 饼图中心文字（总量和利用率）
        {
          type: 'text',
          left: '73%',
          top: '35%',
          style: {
            text: `${total.toFixed(1)} GB`,
            fontSize: 22,
            fontWeight: 'bold',
            fill: utilization > 90 ? '#ff4d4f' : '#333',
            textAlign: 'center',
            x: -50,
          },
        },
        {
          type: 'text',
          left: '73%',
          top: '42%',
          style: {
            text: `/ ${chipMemoryGB} GB`,
            fontSize: 11,
            fill: '#999',
            textAlign: 'center',
            x: -32,
          },
        },
        {
          type: 'text',
          left: '73%',
          top: '48%',
          style: {
            text: `利用率`,
            fontSize: 11,
            fill: '#666',
            textAlign: 'center',
            x: -18,
          },
        },
        {
          type: 'text',
          left: '73%',
          top: '52%',
          style: {
            text: `${utilization.toFixed(1)}%`,
            fontSize: 16,
            fontWeight: 'bold',
            fill: utilization > 90 ? '#ff4d4f' : utilization > 80 ? '#faad14' : '#52c41a',
            textAlign: 'center',
            x: -30,
          },
        },
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
