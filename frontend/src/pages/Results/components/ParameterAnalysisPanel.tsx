/**
 * 参数分析面板
 * 协调参数选择、图表渲染、配置管理
 */

import { useState, useMemo, useEffect } from 'react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Save, BarChart3, ChevronDown } from 'lucide-react'
import { ParameterTreeSelect } from './ParameterTreeSelect'
import { SingleParamLineChart } from './SingleParamLineChart'
import { DualParamHeatmap } from './DualParamHeatmap'
import {
  aggregateSensitivityData,
  aggregateHeatmapData,
  extractParametersFromResults,
  getMetricLabel,
  type MetricType,
  type SensitivityDataPoint,
  type HeatmapData,
} from '../utils/parameterAnalysis'
import { classifyParameters, getParamDisplayInfo } from '../utils/parameterClassifier'
import type { EvaluationTask } from '@/api/results'

interface ParameterAnalysisPanelProps {
  /** 实验 ID */
  experimentId: number
  /** 实验的所有任务 */
  tasks: EvaluationTask[]
}

type ChartType = 'line' | 'heatmap'

/** 图表配置（用于保存/加载） */
interface ChartConfig {
  id: string
  name: string
  type: ChartType
  parameters: string[]
  metrics: MetricType[]
  createdAt: number
}

/** 可选的性能指标列表（分组） */
const METRIC_OPTIONS: { value: MetricType; label: string; group: string }[] = [
  // 吞吐量
  { value: 'tps', label: '集群吞吐量 (TPS)', group: '吞吐量' },
  { value: 'tps_per_chip', label: '单芯片吞吐量', group: '吞吐量' },
  { value: 'tps_per_batch', label: '单请求吞吐量', group: '吞吐量' },
  // 延迟
  { value: 'tpot', label: '每Token延迟 (TPOT)', group: '延迟' },
  { value: 'ttft', label: '首Token延迟 (TTFT)', group: '延迟' },
  { value: 'end_to_end_latency', label: '端到端延迟', group: '延迟' },
  // 利用率
  { value: 'mfu', label: '算力利用率 (MFU)', group: '利用率' },
  { value: 'mbu', label: '带宽利用率 (MBU)', group: '利用率' },
  // 资源
  { value: 'score', label: '综合得分', group: '资源' },
  { value: 'chips', label: '芯片数量', group: '资源' },
  { value: 'dram_occupy', label: '显存占用 (GB)', group: '资源' },
  { value: 'flops', label: '计算量 (TFLOPs)', group: '资源' },
  // 成本
  { value: 'cost_total', label: '总成本', group: '成本' },
  { value: 'cost_server', label: '服务器成本', group: '成本' },
  { value: 'cost_interconnect', label: '互联成本', group: '成本' },
  { value: 'cost_per_chip', label: '单芯片成本', group: '成本' },
  { value: 'cost_dfop', label: 'DFOP ($/TPS)', group: '成本' },
]

export function ParameterAnalysisPanel({ experimentId, tasks }: ParameterAnalysisPanelProps) {
  // 状态管理
  const [chartType, setChartType] = useState<ChartType>('line')
  const [selectedParams, setSelectedParams] = useState<string[]>([])
  const [selectedMetrics, setSelectedMetrics] = useState<MetricType[]>(['tps'])
  const [savedConfigs, setSavedConfigs] = useState<ChartConfig[]>([])

  // 性能指标选择对话框
  const [metricDialogOpen, setMetricDialogOpen] = useState(false)
  const [tempSelectedMetrics, setTempSelectedMetrics] = useState<MetricType[]>(['tps'])

  // 保存配置对话框
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [configName, setConfigName] = useState('')

  // 删除确认对话框
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [configToDelete, setConfigToDelete] = useState<string | null>(null)

  // 提取可用参数
  const { paramTree, hasParams } = useMemo(() => {
    const parametersMap = extractParametersFromResults(tasks)
    const tree = classifyParameters(parametersMap)
    return {
      paramTree: tree,
      hasParams: parametersMap.size > 0,
    }
  }, [tasks])

  // 加载保存的配置
  useEffect(() => {
    const key = `tier6_chart_configs_${experimentId}`
    const stored = localStorage.getItem(key)
    if (stored) {
      try {
        const configs = JSON.parse(stored)
        // 兼容旧格式（单个 metric）
        const migratedConfigs = configs.map((c: any) => ({
          ...c,
          metrics: c.metrics || (c.metric ? [c.metric] : ['tps']),
        }))
        setSavedConfigs(migratedConfigs)
      } catch (e) {
        console.error('Failed to load chart configs:', e)
      }
    }
  }, [experimentId])

  // 数据聚合 - 为每个选中的指标生成数据（自动分析）
  const chartsData = useMemo(() => {
    if (selectedMetrics.length === 0) return null
    // 检查参数数量是否满足要求
    const requiredParamCount = chartType === 'line' ? 1 : 2
    if (selectedParams.length !== requiredParamCount) return null

    const result: { metric: MetricType; data: SensitivityDataPoint[] | HeatmapData | null }[] = []

    for (const metric of selectedMetrics) {
      if (chartType === 'line') {
        const data = aggregateSensitivityData(tasks, selectedParams[0], metric)
        result.push({ metric, data })
      } else if (chartType === 'heatmap') {
        const data = aggregateHeatmapData(tasks, selectedParams[0], selectedParams[1], metric)
        result.push({ metric, data })
      }
    }

    return result
  }, [tasks, selectedParams, selectedMetrics, chartType])

  // 切换临时指标选择
  const handleToggleTempMetric = (metric: MetricType) => {
    setTempSelectedMetrics(prev => {
      if (prev.includes(metric)) {
        // 至少保留一个指标
        if (prev.length === 1) return prev
        return prev.filter(m => m !== metric)
      } else {
        return [...prev, metric]
      }
    })
  }

  // 打开指标选择对话框
  const handleOpenMetricDialog = () => {
    setTempSelectedMetrics([...selectedMetrics])
    setMetricDialogOpen(true)
  }

  // 确认指标选择
  const handleConfirmMetrics = () => {
    setSelectedMetrics([...tempSelectedMetrics])
    setMetricDialogOpen(false)
  }

  // 保存配置
  const handleSaveConfig = () => {
    if (!configName.trim()) return

    const config: ChartConfig = {
      id: Date.now().toString(),
      name: configName.trim(),
      type: chartType,
      parameters: selectedParams,
      metrics: selectedMetrics,
      createdAt: Date.now(),
    }

    const newConfigs = [...savedConfigs, config]
    setSavedConfigs(newConfigs)

    const key = `tier6_chart_configs_${experimentId}`
    localStorage.setItem(key, JSON.stringify(newConfigs))

    // 关闭对话框并重置
    setSaveDialogOpen(false)
    setConfigName('')
  }

  // 加载配置
  const handleLoadConfig = (config: ChartConfig) => {
    setChartType(config.type)
    setSelectedParams(config.parameters)
    setSelectedMetrics(config.metrics || ['tps'])
  }

  // 打开删除确认对话框
  const handleOpenDeleteDialog = (configId: string) => {
    setConfigToDelete(configId)
    setDeleteDialogOpen(true)
  }

  // 确认删除配置
  const handleConfirmDelete = () => {
    if (!configToDelete) return

    const newConfigs = savedConfigs.filter(c => c.id !== configToDelete)
    setSavedConfigs(newConfigs)

    const key = `tier6_chart_configs_${experimentId}`
    localStorage.setItem(key, JSON.stringify(newConfigs))

    setDeleteDialogOpen(false)
    setConfigToDelete(null)
  }

  // 渲染图表
  const renderCharts = () => {
    // 检查参数选择状态
    const requiredParamCount = chartType === 'line' ? 1 : 2
    if (selectedParams.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center py-16 text-gray-500">
          <BarChart3 className="h-12 w-12 mb-4 opacity-20" />
          <p>请选择要分析的参数</p>
        </div>
      )
    }

    if (selectedParams.length !== requiredParamCount) {
      return (
        <div className="flex flex-col items-center justify-center py-16 text-gray-500">
          <BarChart3 className="h-12 w-12 mb-4 opacity-20" />
          <p>{chartType === 'line' ? '请选择 1 个参数' : '请选择 2 个参数'}</p>
        </div>
      )
    }

    if (!chartsData || chartsData.length === 0) {
      return (
        <div className="flex items-center justify-center py-16 text-gray-500">
          数据不足，无法生成图表
        </div>
      )
    }

    const paramInfo = getParamDisplayInfo(selectedParams[0])

    // 渲染多个图表（网格布局）
    return (
      <div className={`grid gap-6 ${chartsData.length === 1 ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2'}`}>
        {chartsData.map(({ metric, data }) => {
          if (!data) return null

          const metricLabel = getMetricLabel(metric)

          if (chartType === 'line' && Array.isArray(data)) {
            return (
              <Card key={metric} className="shadow-sm">
                <CardHeader className="pb-2">
                  <h4 className="text-sm font-medium text-gray-700">{metricLabel.name}</h4>
                </CardHeader>
                <CardContent>
                  <SingleParamLineChart
                    data={data}
                    paramName={paramInfo?.title || selectedParams[0]}
                    metricName={metricLabel.name}
                    metricUnit={metricLabel.unit}
                    height={300}
                  />
                </CardContent>
              </Card>
            )
          }

          if (chartType === 'heatmap' && 'data' in data) {
            return (
              <Card key={metric} className="shadow-sm">
                <CardHeader className="pb-2">
                  <h4 className="text-sm font-medium text-gray-700">{metricLabel.name}</h4>
                </CardHeader>
                <CardContent>
                  <DualParamHeatmap
                    data={data}
                    metricName={metricLabel.name}
                    metricUnit={metricLabel.unit}
                  />
                </CardContent>
              </Card>
            )
          }

          return null
        })}
      </div>
    )
  }

  // 性能指标多选显示文本
  const metricsDisplayText = useMemo(() => {
    if (selectedMetrics.length === 0) return '选择指标...'
    if (selectedMetrics.length === 1) {
      return METRIC_OPTIONS.find(m => m.value === selectedMetrics[0])?.label || selectedMetrics[0]
    }
    return `已选 ${selectedMetrics.length} 个指标`
  }, [selectedMetrics])

  if (!hasParams) {
    return (
      <Card>
        <CardContent className="py-16">
          <div className="text-center text-gray-500">
            <p>该实验暂无可分析的参数变化</p>
            <p className="text-sm mt-2">需要至少有一个参数存在多个不同取值</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* 保存的配置列表 */}
      {savedConfigs.length > 0 && (
        <Card>
          <CardHeader>
            <h3 className="text-sm font-medium">已保存的配置</h3>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {savedConfigs.map(config => (
                <div
                  key={config.id}
                  className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 rounded hover:bg-gray-200 cursor-pointer"
                  onClick={() => handleLoadConfig(config)}
                >
                  <span className="text-sm">{config.name}</span>
                  <span className="text-xs text-gray-500">
                    ({config.type === 'line' ? '曲线' : '热力图'})
                  </span>
                  <button
                    onClick={e => {
                      e.stopPropagation()
                      handleOpenDeleteDialog(config.id)
                    }}
                    className="text-red-500 hover:text-red-700 ml-1"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 控制面板 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">参数分析</h3>
            <div className="flex gap-2">
              {chartsData && chartsData.length > 0 && (
                <Button onClick={() => setSaveDialogOpen(true)} size="sm" variant="outline">
                  <Save className="h-4 w-4 mr-1" />
                  保存配置
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {/* 图表类型 */}
            <div>
              <label className="text-sm font-medium mb-2 block">图表类型</label>
              <Select value={chartType} onValueChange={(v) => {
                setChartType(v as ChartType)
                setSelectedParams([])
              }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="line">单参数曲线</SelectItem>
                  <SelectItem value="heatmap">双参数热力图</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* 性能指标（多选） */}
            <div>
              <label className="text-sm font-medium mb-2 block">性能指标</label>
              <Button
                variant="outline"
                className="w-full justify-between font-normal"
                onClick={handleOpenMetricDialog}
              >
                <span className="truncate">{metricsDisplayText}</span>
                <ChevronDown className="h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </div>

            {/* 参数选择 */}
            <div>
              <label className="text-sm font-medium mb-2 block">
                选择参数 ({chartType === 'line' ? '选择1个' : '选择2个'})
              </label>
              <ParameterTreeSelect
                tree={paramTree}
                value={selectedParams}
                onChange={setSelectedParams}
                maxSelection={chartType === 'line' ? 1 : 2}
                placeholder={chartType === 'line' ? '选择1个参数...' : '选择2个参数...'}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 图表区域 */}
      <Card>
        <CardContent className="pt-6">{renderCharts()}</CardContent>
      </Card>

      {/* 性能指标选择对话框 */}
      <Dialog open={metricDialogOpen} onOpenChange={setMetricDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>选择性能指标</DialogTitle>
          </DialogHeader>
          <div className="max-h-80 overflow-y-auto py-4">
            {['吞吐量', '延迟', '利用率', '资源', '成本'].map(group => {
              const groupOptions = METRIC_OPTIONS.filter(o => o.group === group)
              if (groupOptions.length === 0) return null
              return (
                <div key={group} className="mb-4 last:mb-0">
                  <div className="text-xs font-medium text-gray-400 px-2 py-1 uppercase tracking-wide">
                    {group}
                  </div>
                  {groupOptions.map(option => (
                    <label
                      key={option.value}
                      className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-blue-50 cursor-pointer transition-colors"
                    >
                      <Checkbox
                        checked={tempSelectedMetrics.includes(option.value)}
                        onCheckedChange={() => handleToggleTempMetric(option.value)}
                      />
                      <span className="text-sm">{option.label}</span>
                    </label>
                  ))}
                </div>
              )
            })}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setMetricDialogOpen(false)}>
              取消
            </Button>
            <Button onClick={handleConfirmMetrics}>
              确定
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 保存配置对话框 */}
      <Dialog open={saveDialogOpen} onOpenChange={setSaveDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>保存配置</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">配置名称</label>
            <Input
              value={configName}
              onChange={(e) => setConfigName(e.target.value)}
              placeholder="请输入配置名称"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && configName.trim()) {
                  handleSaveConfig()
                }
              }}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => {
              setSaveDialogOpen(false)
              setConfigName('')
            }}>
              取消
            </Button>
            <Button onClick={handleSaveConfig} disabled={!configName.trim()}>
              确定
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* 删除确认对话框 */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除配置</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除此配置吗？此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setConfigToDelete(null)}>
              取消
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmDelete}>
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
