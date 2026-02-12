/**
 * 双参数热力图
 * 显示两个参数组合对性能的影响，支持基准值调整和下降比例显示
 */

import { useState, useMemo, useRef, useCallback } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Button } from '@/components/ui/button'
import { getMetricDecimals } from '@/utils/formatters'
import type { HeatmapData } from '../utils/parameterAnalysis'

interface DualParamHeatmapProps {
  /** 热力图数据 */
  data: HeatmapData
  /** 指标名称 */
  metricName: string
  /** 指标单位 */
  metricUnit?: string
  /** 指标 key（如 'tps', 'mfu'），用于统一精度 */
  metricKey?: string
}

export function DualParamHeatmap({
  data,
  metricName,
  metricUnit = '',
  metricKey,
}: DualParamHeatmapProps) {
  const chartRef = useRef<ReactECharts>(null)
  const [baseValue, setBaseValue] = useState<number | null>(null)

  // 转换为 ECharts 热力图数据格式 [x_index, y_index, value]
  const chartData = useMemo(() => {
    return data.data
      .map(item => {
        const xIndex = data.x_values.indexOf(item.x_value)
        const yIndex = data.y_values.indexOf(item.y_value)
        if (xIndex >= 0 && yIndex >= 0) {
          return [xIndex, yIndex, item.mean_performance]
        }
        return null
      })
      .filter((item): item is [number, number, number] => item !== null)
  }, [data])

  // 计算数值范围
  const { minVal, maxVal, effectiveBaseValue, decimalPlaces } = useMemo(() => {
    const values = chartData.map(d => d[2])
    const min = Math.min(...values)
    const max = Math.max(...values)
    const effectiveBase = baseValue !== null ? baseValue : max
    const range = max - min
    const decimals = metricKey
      ? getMetricDecimals(metricKey)
      : (range > 100 ? 0 : range > 10 ? 1 : 2)

    return {
      minVal: min,
      maxVal: max,
      effectiveBaseValue: effectiveBase,
      decimalPlaces: decimals,
    }
  }, [chartData, baseValue, metricKey])

  // 计算单元格大小和图表尺寸
  const { cellSize, chartWidth, chartHeight, fontSize } = useMemo(() => {
    const xCount = data.x_values.length
    const yCount = data.y_values.length
    const cell = 60 // 固定单元格大小
    const gridLeft = 100
    const gridRight = 120
    const gridTop = 30
    const gridBottom = 60

    return {
      cellSize: cell,
      chartWidth: xCount * cell + gridLeft + gridRight,
      chartHeight: yCount * cell + gridTop + gridBottom,
      fontSize: Math.max(8, Math.min(12, cell / 5)),
    }
  }, [data.x_values.length, data.y_values.length])

  // 计算下降比例
  const formatDropPercent = useCallback(
    (val: number) => {
      if (effectiveBaseValue === 0) return ''
      const dropPercent = ((effectiveBaseValue - val) / effectiveBaseValue) * 100
      if (dropPercent <= 0) return '' // 等于或超过基准值不显示
      return `-${dropPercent.toFixed(1)}%`
    },
    [effectiveBaseValue]
  )

  const option = useMemo<EChartsOption>(() => {
    return {
      tooltip: {
        position: 'top',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: '#e8e8e8',
        borderWidth: 1,
        textStyle: { color: '#333' },
        formatter: (params: any) => {
          const [xIdx, yIdx, val] = params.data
          const dropPercent =
            effectiveBaseValue > 0
              ? ((effectiveBaseValue - val) / effectiveBaseValue) * 100
              : 0
          return `
            <div style="font-weight:600;margin-bottom:4px">
              ${data.param_x}: ${data.x_values[xIdx]}<br/>
              ${data.param_y}: ${data.y_values[yIdx]}
            </div>
            <div>${metricName}: <span style="font-weight:500">${val.toFixed(decimalPlaces)}</span> ${metricUnit}</div>
            <div style="color:#999;font-size:11px;margin-top:2px">
              相对基准下降: ${dropPercent > 0 ? dropPercent.toFixed(1) : 0}%
            </div>
          `
        },
      },
      xAxis: {
        type: 'category',
        data: data.x_values.map(String),
        name: data.param_x,
        nameLocation: 'middle',
        nameGap: 40,
        splitArea: { show: true },
        axisLabel: {
          rotate: data.x_values.length > 8 ? 45 : 0,
          fontSize: 11,
          color: '#666',
        },
        nameTextStyle: { color: '#333', fontSize: 12, fontWeight: 500 },
      },
      yAxis: {
        type: 'category',
        data: data.y_values.map(String),
        name: data.param_y,
        splitArea: { show: true },
        axisLabel: {
          fontSize: 11,
          color: '#666',
        },
        nameTextStyle: { color: '#333', fontSize: 12, fontWeight: 500 },
      },
      visualMap: {
        type: 'continuous',
        min: minVal,
        max: maxVal,
        calculable: true,
        orient: 'vertical',
        right: 10,
        top: 'center',
        realtime: false, // 拖动结束后触发
        inRange: {
          color: ['#f5f5f5', '#91d5ff', '#1890ff', '#ff7875', '#ff4d4f'],
        },
        textStyle: {
          color: '#666',
          fontSize: 11,
        },
      },
      series: [
        {
          type: 'heatmap',
          data: chartData,
          label: {
            show: true,
            formatter: (params: any) => {
              const val = params.data[2]
              const valueStr = val.toFixed(decimalPlaces)
              const dropStr = formatDropPercent(val)
              return dropStr ? `${valueStr}\n{drop|${dropStr}}` : valueStr
            },
            fontSize: fontSize,
            color: '#333',
            align: 'center',
            verticalAlign: 'middle',
            rich: {
              drop: {
                fontSize: fontSize - 1,
                color: '#666',
                lineHeight: fontSize + 2,
                align: 'center',
              },
            },
          },
          itemStyle: {
            borderColor: '#fff',
            borderWidth: 1,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
      grid: {
        left: 100,
        right: 120,
        top: 30,
        bottom: 60,
      },
    }
  }, [
    chartData,
    data.param_x,
    data.param_y,
    data.x_values,
    data.y_values,
    minVal,
    maxVal,
    effectiveBaseValue,
    decimalPlaces,
    fontSize,
    formatDropPercent,
    metricName,
    metricUnit,
  ])

  // 监听 visualMap 事件
  const onChartReady = useCallback((chart: any) => {
    const handleVisualMapChange = (params: any) => {
      if (params.selected !== undefined) {
        const [, selectedMax] = params.selected
        if (typeof selectedMax === 'number') {
          setBaseValue(selectedMax)
        }
      }
    }

    chart.off('datarangeselected')
    chart.on('datarangeselected', handleVisualMapChange)
  }, [])

  // 重置基准值
  const handleResetBase = () => {
    setBaseValue(null)
  }

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
      link.download = `${data.param_x}_${data.param_y}_${metricName}.png`
      link.click()
    }
  }

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        暂无数据
      </div>
    )
  }

  return (
    <div>
      {/* 基准值提示 */}
      <div className="mb-2 text-sm text-gray-600 flex items-center gap-2 flex-wrap">
        <span>
          基准值:{' '}
          <span className="font-bold text-blue-600">
            {effectiveBaseValue.toFixed(decimalPlaces)}
          </span>
          {metricUnit && ` ${metricUnit}`}
        </span>
        {baseValue !== null && (
          <>
            <span className="text-orange-500">
              (已调整，原最高值: {maxVal.toFixed(decimalPlaces)})
            </span>
            <Button size="sm" variant="outline" onClick={handleResetBase}>
              重置
            </Button>
          </>
        )}
        <span className="text-gray-400">| 拖动右侧色条可调整基准值</span>
      </div>

      {/* 热力图 */}
      <div className="overflow-x-auto overflow-y-hidden flex justify-center">
        <ReactECharts
          ref={chartRef}
          option={option}
          onChartReady={onChartReady}
          style={{
            width: chartWidth,
            height: chartHeight,
            flexShrink: 0,
          }}
          opts={{ renderer: 'canvas' }}
        />
      </div>
    </div>
  )
}
