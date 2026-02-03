/**
 * 瀑布图 - 任务时间线可视化
 * 展示并行任务的依赖关系和时间序列
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { CHART_SERIES_COLORS, ECHARTS_COMMON_CONFIG } from './chartTheme'

export interface WaterfallTask {
  name: string
  start: number
  duration: number
  deps?: string[]
}

interface WaterfallChartProps {
  data: WaterfallTask[]
  height?: number
  title?: string
}

export const WaterfallChart: React.FC<WaterfallChartProps> = ({
  data,
  height = 400,
  title = '任务时间线',
}) => {
  const option: EChartsOption = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'center',
          textStyle: { color: '#999', fontSize: 12 },
        },
      }
    }

    // 准备数据：[任务名, 开始时间, 结束时间, 持续时间]
    const categories = data.map((task) => task.name)
    const seriesData = data.map((task, index) => ({
      name: task.name,
      value: [index, task.start, task.start + task.duration, task.duration],
      itemStyle: {
        color: CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length],
      },
    }))

    // 计算时间轴范围
    const maxTime = Math.max(...data.map((task) => task.start + task.duration))

    return {
      title: {
        text: title,
        left: 'left',
        textStyle: {
          fontSize: 14,
          fontWeight: 600,
          color: '#1a1a1a',
        },
      },
      tooltip: {
        ...ECHARTS_COMMON_CONFIG.tooltip,
        formatter: (params: any) => {
          const { name, value } = params
          const start = value[1]
          const end = value[2]
          const duration = value[3]
          return `
            <div style="font-size: 12px;">
              <strong>${name}</strong><br/>
              开始: ${start.toFixed(1)} ms<br/>
              结束: ${end.toFixed(1)} ms<br/>
              持续: ${duration.toFixed(1)} ms
            </div>
          `
        },
      },
      grid: {
        left: 120,
        right: 40,
        top: 50,
        bottom: 40,
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: '时间 (ms)',
        min: 0,
        max: maxTime * 1.05,
        axisLabel: {
          formatter: '{value} ms',
          color: '#666',
          fontSize: 11,
        },
        splitLine: {
          lineStyle: {
            color: '#f0f0f0',
            type: 'dashed',
          },
        },
      },
      yAxis: {
        type: 'category',
        data: categories,
        axisLine: {
          show: true,
          lineStyle: { color: '#d9d9d9' },
        },
        axisTick: {
          show: false,
        },
        axisLabel: {
          color: '#333',
          fontSize: 11,
          fontWeight: 500,
        },
      },
      series: [
        {
          type: 'custom',
          renderItem: (params: any, api: any) => {
            const categoryIndex = api.value(0)
            const start = api.coord([api.value(1), categoryIndex])
            const end = api.coord([api.value(2), categoryIndex])
            const height = api.size([0, 1])[1] * 0.6

            return {
              type: 'rect',
              shape: {
                x: start[0],
                y: start[1] - height / 2,
                width: end[0] - start[0],
                height: height,
              },
              style: {
                fill: api.visual('color'),
                stroke: '#fff',
                lineWidth: 1,
              },
            }
          },
          encode: {
            x: [1, 2],
            y: 0,
          },
          data: seriesData,
        },
      ],
    }
  }, [data, title])

  return (
    <ReactECharts
      option={option}
      style={{ height: `${height}px`, width: '100%' }}
      opts={{ renderer: 'canvas' }}
    />
  )
}
