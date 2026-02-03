/**
 * 时间分解图 - 时间占比分析
 * 使用 Treemap 展示各模块的时间分层结构
 * 注: 这是矩形面积占比图,不是传统的横向堆叠火焰图
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { CHART_SERIES_COLORS, ECHARTS_COMMON_CONFIG } from './chartTheme'

export interface TimeBreakdownNode {
  name: string
  value: number
  children?: TimeBreakdownNode[]
}

interface TimeBreakdownChartProps {
  data: TimeBreakdownNode
  height?: number
  title?: string
}

export const TimeBreakdownChart: React.FC<TimeBreakdownChartProps> = ({
  data,
  height = 400,
  title = '时间占比分析',
}) => {
  const option: EChartsOption = useMemo(() => {
    if (!data || !data.name) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'center',
          textStyle: { color: '#999', fontSize: 12 },
        },
      }
    }

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
          const { name, value, treePathInfo } = params
          // 计算占总时间的百分比
          const rootValue = treePathInfo[0]?.value || value
          const percentage = ((value / rootValue) * 100).toFixed(1)
          return `
            <div style="font-size: 12px;">
              <strong>${name}</strong><br/>
              时间: ${value.toFixed(2)} ms<br/>
              占比: ${percentage}%
            </div>
          `
        },
      },
      series: [
        {
          type: 'treemap',
          data: [data],
          // 禁用缩放和点击下钻
          roam: false,
          nodeClick: false,
          // 面包屑导航
          breadcrumb: {
            show: false,
          },
          // 标签配置
          label: {
            show: true,
            formatter: (params: any) => {
              const { name, value } = params
              return `{name|${name}}\n{value|${value.toFixed(1)}ms}`
            },
            rich: {
              name: {
                fontSize: 11,
                fontWeight: 600,
                color: '#333',
              },
              value: {
                fontSize: 10,
                color: '#666',
              },
            },
          },
          // 层级配置
          levels: [
            {
              // 根节点
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 2,
                gapWidth: 2,
              },
              upperLabel: {
                show: false,
              },
            },
            {
              // 第一层
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 1,
                gapWidth: 1,
              },
              color: CHART_SERIES_COLORS.slice(0, 3),
              colorMappingBy: 'index',
            },
            {
              // 第二层
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 1,
                gapWidth: 1,
              },
              colorSaturation: [0.5, 0.8],
              colorAlpha: [0.8, 1],
            },
          ],
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
