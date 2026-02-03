/**
 * 成本/资源分解图 - 分层占比可视化
 * 使用 Treemap 展示成本/时间/资源的分层占比
 * 注: 这是矩形面积占比图(Treemap),不是节点-连线的树形图(Tree)
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { CHART_SERIES_COLORS, ECHARTS_COMMON_CONFIG } from './chartTheme'

export interface ProportionNode {
  name: string
  value: number
  children?: ProportionNode[]
}

interface CostBreakdownChartProps {
  data: ProportionNode
  height?: number
  title?: string
  unit?: string // 单位，如 "$" 或 "ms"
}

export const CostBreakdownChart: React.FC<CostBreakdownChartProps> = ({
  data,
  height = 400,
  title = '分层占比分析',
  unit = '',
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

    // 计算根节点的总值用于百分比计算
    const rootValue = data.value

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
          // 计算百分比
          const percentage = ((value / rootValue) * 100).toFixed(1)
          // 格式化数值
          const formattedValue =
            unit === '$'
              ? `$${value.toLocaleString()}`
              : unit === 'ms'
              ? `${value.toFixed(1)} ms`
              : value.toLocaleString()

          // 构建路径（面包屑）
          const path = treePathInfo
            .slice(1)
            .map((item: any) => item.name)
            .join(' > ')

          return `
            <div style="font-size: 12px;">
              ${path ? `<div style="color: #999; margin-bottom: 4px;">${path}</div>` : ''}
              <strong>${name}</strong><br/>
              数值: ${formattedValue}<br/>
              占比: ${percentage}%
            </div>
          `
        },
      },
      series: [
        {
          type: 'treemap',
          data: [data],
          // 禁用缩放和点击下钻（保持静态展示）
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
              const percentage = ((value / rootValue) * 100).toFixed(0)

              // 格式化数值
              let formattedValue
              if (unit === '$') {
                formattedValue = value >= 1000 ? `$${(value / 1000).toFixed(0)}K` : `$${value}`
              } else if (unit === 'ms') {
                formattedValue = `${value.toFixed(0)}ms`
              } else {
                formattedValue = value.toLocaleString()
              }

              // 根据区块大小决定显示内容
              if (percentage >= 5) {
                return `{name|${name}}\n{value|${formattedValue}}\n{percent|${percentage}%}`
              } else if (percentage >= 2) {
                return `{name|${name}}\n{percent|${percentage}%}`
              } else {
                return ''
              }
            },
            rich: {
              name: {
                fontSize: 11,
                fontWeight: 600,
                color: '#fff',
                lineHeight: 16,
              },
              value: {
                fontSize: 10,
                color: 'rgba(255,255,255,0.9)',
                lineHeight: 14,
              },
              percent: {
                fontSize: 9,
                color: 'rgba(255,255,255,0.8)',
                lineHeight: 12,
              },
            },
          },
          // 层级样式配置
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
                borderWidth: 2,
                gapWidth: 2,
              },
              color: CHART_SERIES_COLORS.slice(0, 4),
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
              colorAlpha: [0.85, 1],
            },
            {
              // 第三层及更深
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 1,
                gapWidth: 1,
              },
              colorSaturation: [0.3, 0.6],
              colorAlpha: [0.7, 0.9],
            },
          ],
          // 视觉映射
          visualDimension: 0,
          visualMin: 0,
          visualMax: rootValue,
        },
      ],
    }
  }, [data, title, unit])

  return (
    <ReactECharts
      option={option}
      style={{ height: `${height}px`, width: '100%' }}
      opts={{ renderer: 'canvas' }}
    />
  )
}
