/**
 * 单参数敏感度曲线图
 * 显示参数变化对性能指标的影响（均值曲线 + 范围阴影）
 */

import { useMemo, useRef } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { CHART_COLORS } from '@/components/ConfigPanel/DeploymentAnalysis/charts/chartTheme'
import { getMetricDecimals } from '@/utils/formatters'
import type { SensitivityDataPoint } from '../utils/parameterAnalysis'

interface SingleParamLineChartProps {
  /** 数据点 */
  data: SensitivityDataPoint[]
  /** 参数名称 */
  paramName: string
  /** 指标名称 */
  metricName: string
  /** 指标单位 */
  metricUnit?: string
  /** 指标 key（如 'tps', 'mfu'），用于统一精度 */
  metricKey?: string
  /** 图表高度 */
  height?: number
}

export function SingleParamLineChart({
  data,
  paramName,
  metricName,
  metricUnit = '',
  metricKey,
  height = 400,
}: SingleParamLineChartProps) {
  const chartRef = useRef<ReactECharts>(null)

  const option = useMemo<EChartsOption>(() => {
    if (data.length === 0) {
      return {}
    }

    // 使用统一精度配置，未指定 metricKey 时回退到基于范围的动态计算
    const values = data.map(d => d.mean_performance)
    const range = Math.max(...values) - Math.min(...values)
    const decimalPlaces = metricKey
      ? getMetricDecimals(metricKey)
      : (range > 100 ? 0 : range > 10 ? 1 : 2)

    return {
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: '#e8e8e8',
        borderWidth: 1,
        textStyle: { color: '#333' },
        formatter: (params: any) => {
          const xValue = params[0].name
          const dataPoint = data.find(d => String(d.value) === xValue)
          if (!dataPoint) return ''

          return `
            <div style="font-weight:600;margin-bottom:4px">${paramName} = ${xValue}</div>
            <div>均值: <span style="color:${CHART_COLORS.primary};font-weight:500">${dataPoint.mean_performance.toFixed(decimalPlaces)}</span> ${metricUnit}</div>
            <div>最大: ${dataPoint.max_performance.toFixed(decimalPlaces)} ${metricUnit}</div>
            <div>最小: ${dataPoint.min_performance.toFixed(decimalPlaces)} ${metricUnit}</div>
            <div style="color:#999;font-size:11px;margin-top:4px">样本数: ${dataPoint.count}</div>
          `
        },
      },
      xAxis: {
        type: 'category',
        data: data.map(d => String(d.value)),
        name: paramName,
        nameLocation: 'middle',
        nameGap: 30,
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        axisTick: { lineStyle: { color: '#d9d9d9' } },
        axisLabel: { color: '#666', fontSize: 11 },
        nameTextStyle: { color: '#333', fontSize: 12, fontWeight: 500 },
      },
      yAxis: {
        type: 'value',
        name: `${metricName}${metricUnit ? ` (${metricUnit})` : ''}`,
        scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: '#666',
          fontSize: 11,
          formatter: (value: number) => value.toFixed(decimalPlaces),
        },
        nameTextStyle: { color: '#333', fontSize: 12, fontWeight: 500 },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      series: [
        // 系列1: 最小值（不可见，用于堆叠基准）
        {
          name: '最小值',
          type: 'line',
          data: data.map(d => d.min_performance),
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          stack: 'range',
          symbol: 'none',
          smooth: true,
          silent: true,
        },
        // 系列2: 范围（最大值-最小值，半透明阴影）
        {
          name: '范围',
          type: 'line',
          data: data.map(d => d.max_performance - d.min_performance),
          lineStyle: { opacity: 0 },
          areaStyle: {
            opacity: 0.15,
            color: CHART_COLORS.primary,
          },
          stack: 'range',
          symbol: 'none',
          smooth: true,
          silent: true,
        },
        // 系列3: 均值（实线，带标记点）
        {
          name: '均值',
          type: 'line',
          data: data.map(d => d.mean_performance),
          lineStyle: { color: CHART_COLORS.primary, width: 2 },
          itemStyle: { color: CHART_COLORS.primary },
          symbol: 'circle',
          symbolSize: 6,
          smooth: true,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: CHART_COLORS.primary,
            },
          },
        },
      ],
      grid: {
        left: 60,
        right: 20,
        top: 40,
        bottom: 50,
      },
    }
  }, [data, paramName, metricName, metricUnit, metricKey])

  // 导出图片
  const handleExport = () => {
    if (chartRef.current) {
      const echartsInstance = chartRef.current.getEchartsInstance()
      const url = echartsInstance.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#fff',
      })
      const link = document.createElement('a')
      link.href = url
      link.download = `${paramName}_${metricName}.png`
      link.click()
    }
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        暂无数据
      </div>
    )
  }

  return (
    <div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  )
}
