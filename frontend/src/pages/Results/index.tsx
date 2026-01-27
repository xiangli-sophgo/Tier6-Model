/**
 * 结果汇总页面
 * 显示所有实验和评估任务的列表
 */

import React, { useEffect, useState, useCallback } from 'react'
import { toast } from 'sonner'
import {
  RefreshCw,
  Trash2,
  BarChart3,
  Loader2,
  ArrowLeft,
  Pencil,
  Save,
  X,
  Download,
  Upload,
  Inbox,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
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
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { listExperiments, deleteExperiment, deleteExperimentsBatch, getExperimentDetail, getTaskResults, updateExperiment, downloadExperimentJSON, checkImportFile, executeImport, Experiment, EvaluationTask, TaskResultsResponse } from '@/api/results'
import { AnalysisResultDisplay } from '@/components/ConfigPanel/DeploymentAnalysis/AnalysisResultDisplay'
import { ChartsPanel } from '@/components/ConfigPanel/DeploymentAnalysis/charts'
import { PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig } from '@/utils/llmDeployment/types'
import TaskTable from './components/TaskTable'

// 分页组件
const Pagination: React.FC<{
  currentPage: number
  totalPages: number
  pageSize: number
  total: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}> = ({ currentPage, totalPages, pageSize, total, onPageChange, onPageSizeChange }) => {
  return (
    <div className="flex items-center justify-between px-2 py-4">
      <span className="text-sm text-gray-500">共 {total} 个实验</span>
      <div className="flex items-center gap-2">
        <select
          value={pageSize}
          onChange={(e) => onPageSizeChange(Number(e.target.value))}
          className="h-8 rounded-md border border-gray-300 px-2 text-sm"
        >
          <option value={10}>10条/页</option>
          <option value={20}>20条/页</option>
          <option value={50}>50条/页</option>
        </select>
        <Button
          variant="outline"
          size="sm"
          disabled={currentPage <= 1}
          onClick={() => onPageChange(currentPage - 1)}
        >
          上一页
        </Button>
        <span className="text-sm">
          {currentPage} / {totalPages || 1}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={currentPage >= totalPages}
          onClick={() => onPageChange(currentPage + 1)}
        >
          下一页
        </Button>
        <Input
          type="number"
          min={1}
          max={totalPages || 1}
          className="h-8 w-16"
          placeholder="页码"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const value = parseInt((e.target as HTMLInputElement).value)
              if (value >= 1 && value <= totalPages) {
                onPageChange(value)
              }
            }
          }}
        />
      </div>
    </div>
  )
}

export const Results: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedExperimentId, setSelectedExperimentId] = useState<number | null>(null)
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null)

  // 任务分析相关状态
  const [selectedTask, setSelectedTask] = useState<EvaluationTask | null>(null)
  const [taskResults, setTaskResults] = useState<TaskResultsResponse | null>(null)
  const [taskResultsLoading, setTaskResultsLoading] = useState(false)

  // 编辑状态
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editingName, setEditingName] = useState('')
  const [editingDescription, setEditingDescription] = useState('')
  const [editingLoading, setEditingLoading] = useState(false)

  // 批量选择状态
  const [selectedExperimentIds, setSelectedExperimentIds] = useState<number[]>([])

  // 分页状态
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)

  // 导入导出状态
  const [exportModalVisible, setExportModalVisible] = useState(false)
  const [importModalVisible, setImportModalVisible] = useState(false)
  const [importStep, setImportStep] = useState<'upload' | 'config' | 'importing' | 'result'>('upload')
  const [, setImportFile] = useState<File | null>(null)
  const [importCheckResult, setImportCheckResult] = useState<any>(null)
  const [importConfig, setImportConfig] = useState<Map<string, any>>(new Map())
  const [importLoading, setImportLoading] = useState(false)
  const [importResult, setImportResult] = useState<any>(null)

  // 加载实验列表
  const loadExperiments = async () => {
    setLoading(true)
    try {
      const data = await listExperiments()
      setExperiments(data || [])
    } catch (error) {
      toast.error('加载实验列表失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 首次加载
  useEffect(() => {
    loadExperiments()
  }, [])

  // 加载实验详情
  const loadExperimentDetail = async (id: number) => {
    try {
      const data = await getExperimentDetail(id)
      setSelectedExperiment(data)
      setSelectedExperimentId(id)
    } catch (error) {
      toast.error('加载实验详情失败')
      console.error(error)
    }
  }

  // 删除实验
  const handleDelete = async (id: number) => {
    try {
      await deleteExperiment(id)
      toast.success('实验已删除')
      loadExperiments()
      // 如果删除的是当前查看的实验，返回列表
      if (selectedExperimentId === id) {
        setSelectedExperimentId(null)
        setSelectedExperiment(null)
      }
    } catch (error) {
      toast.error('删除失败')
      console.error(error)
    }
  }

  // 开始编辑
  const handleStartEdit = (record: Experiment) => {
    setEditingId(record.id)
    setEditingName(record.name)
    setEditingDescription(record.description || '')
  }

  // 保存编辑
  const handleSaveEdit = async (id: number) => {
    if (!editingName.trim()) {
      toast.error('实验名称不能为空')
      return
    }
    setEditingLoading(true)
    try {
      await updateExperiment(id, {
        name: editingName.trim(),
        description: editingDescription.trim() || undefined,
      })
      toast.success('实验已更新')
      loadExperiments()
      setEditingId(null)
    } catch (error) {
      toast.error('更新失败')
      console.error(error)
    } finally {
      setEditingLoading(false)
    }
  }

  // 取消编辑
  const handleCancelEdit = () => {
    setEditingId(null)
    setEditingName('')
    setEditingDescription('')
  }

  // 批量删除
  const handleBatchDelete = async () => {
    if (selectedExperimentIds.length === 0) {
      toast.warning('请先选择要删除的实验')
      return
    }
    try {
      await deleteExperimentsBatch(selectedExperimentIds)
      toast.success(`成功删除 ${selectedExperimentIds.length} 个实验`)
      setSelectedExperimentIds([])
      loadExperiments()
    } catch (error) {
      toast.error('批量删除失败')
      console.error(error)
    }
  }

  // 导出实验
  const handleExport = async () => {
    try {
      const exportIds = selectedExperimentIds.length > 0 ? selectedExperimentIds : undefined
      await downloadExperimentJSON(exportIds)
      toast.success('导出成功')
      setExportModalVisible(false)
      setSelectedExperimentIds([])
    } catch (error) {
      toast.error('导出失败')
      console.error(error)
    }
  }

  // 处理导入文件上传
  const handleImportFile = async (file: File) => {
    setImportFile(file)
    setImportLoading(true)
    try {
      const result = await checkImportFile(file)
      if (result.valid && result.experiments) {
        setImportCheckResult(result)
        // 初始化导入配置
        const config = new Map()
        for (const exp of result.experiments) {
          config.set(exp.name, {
            original_name: exp.name,
            action: exp.conflict ? 'rename' : 'rename',
            new_name: exp.conflict ? `${exp.name}_imported` : exp.name,
          })
        }
        setImportConfig(config)
        setImportStep('config')
      } else {
        toast.error(result.error || '导入文件无效')
      }
    } catch (error) {
      toast.error('检查导入文件失败')
      console.error(error)
    } finally {
      setImportLoading(false)
    }
  }

  // 执行导入
  const handleExecuteImport = async () => {
    if (!importCheckResult?.temp_file_id) {
      toast.error('导入会话已过期，请重新上传')
      return
    }

    setImportLoading(true)
    setImportStep('importing')
    try {
      const configs = Array.from(importConfig.values())
      const result = await executeImport(importCheckResult.temp_file_id, configs)
      setImportResult(result)
      setImportStep('result')
      if (result.success) {
        toast.success(result.message)
        loadExperiments()
      }
    } catch (error) {
      toast.error('导入失败')
      console.error(error)
    } finally {
      setImportLoading(false)
    }
  }

  // 重置导入对话框
  const resetImportModal = () => {
    setImportModalVisible(false)
    setImportStep('upload')
    setImportFile(null)
    setImportCheckResult(null)
    setImportConfig(new Map())
    setImportResult(null)
  }

  // 加载任务结果
  const loadTaskResults = async (task: EvaluationTask) => {
    setSelectedTask(task)
    setTaskResultsLoading(true)
    try {
      const results = await getTaskResults(task.task_id)
      setTaskResults(results)
    } catch (error) {
      toast.error('加载任务结果失败')
      console.error(error)
      setTaskResults(null)
    } finally {
      setTaskResultsLoading(false)
    }
  }

  // 返回任务列表
  const handleBackToTasks = () => {
    setSelectedTask(null)
    setTaskResults(null)
  }

  // 将 API 返回的 top_k_plans 转换为 PlanAnalysisResult[]
  const convertToAnalysisResults = (results: TaskResultsResponse | null): PlanAnalysisResult[] => {
    if (!results || !results.top_k_plans) return []
    return results.top_k_plans.map(plan => ({
      is_feasible: plan.is_feasible,
      plan: {
        plan_id: `plan_${plan.chips}_${plan.parallelism.tp}_${plan.parallelism.ep}`,
        total_chips: plan.chips,
        parallelism: plan.parallelism,
      },
      latency: {
        prefill_total_latency_ms: plan.ttft,
        decode_per_token_latency_ms: plan.tpot,
        end_to_end_latency_ms: plan.ttft + plan.tpot * 100,
        bottleneck_type: 'balanced' as const,
      },
      throughput: {
        tokens_per_second: plan.throughput,
        tps_per_chip: plan.tps_per_chip,
        tps_per_batch: plan.throughput,
        model_flops_utilization: plan.mfu,
        memory_bandwidth_utilization: plan.mbu,
      },
      memory: {
        total_per_chip_gb: plan.dram_occupy ? plan.dram_occupy / (1024 * 1024 * 1024) : 0,  // 字节转 GB
        is_memory_sufficient: true,
      },
      communication: {},
      utilization: {},
      score: {
        overall_score: plan.score,
        latency_score: 0,
        throughput_score: 0,
        efficiency_score: 0,
        balance_score: 0,
      },
      suggestions: [],
    } as unknown as PlanAnalysisResult))
  }

  // 处理全选
  const handleSelectAll = useCallback((checked: boolean) => {
    if (checked) {
      setSelectedExperimentIds(experiments.map(e => e.id))
    } else {
      setSelectedExperimentIds([])
    }
  }, [experiments])

  // 处理单选
  const handleSelectOne = useCallback((id: number, checked: boolean) => {
    if (checked) {
      setSelectedExperimentIds(prev => [...prev, id])
    } else {
      setSelectedExperimentIds(prev => prev.filter(i => i !== id))
    }
  }, [])

  // 分页数据
  const paginatedExperiments = experiments.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  )
  const totalPages = Math.ceil(experiments.length / pageSize)

  // 如果选中了实验，显示详情视图
  if (selectedExperiment) {
    const progress =
      selectedExperiment.total_tasks > 0
        ? Math.round((selectedExperiment.completed_tasks / selectedExperiment.total_tasks) * 100)
        : 0

    // 如果选中了任务，显示任务分析视图
    if (selectedTask) {
      // 从任务的 config_snapshot 中提取配置
      const modelConfig = selectedTask.config_snapshot?.model as Record<string, unknown> || {}
      const inferenceConfig = selectedTask.config_snapshot?.inference as Record<string, unknown> || {}
      const topology = selectedTask.config_snapshot?.topology as Record<string, unknown> || {}

      // 从拓扑中提取硬件配置（简化版）
      const hardwareConfig: HardwareConfig | undefined = (() => {
        const pods = (topology.pods as any[]) || []
        if (pods.length > 0 && pods[0].racks && pods[0].racks[0].boards && pods[0].racks[0].boards[0].chips) {
          const chip = pods[0].racks[0].boards[0].chips[0]
          return {
            chip: {
              chip_type: chip.name || 'Unknown',
              compute_tflops_fp16: chip.compute_tflops_fp16 || 0,
              memory_gb: chip.memory_gb || 0,
              memory_bandwidth_gbps: chip.memory_bandwidth_gbps || 0,
              memory_bandwidth_utilization: chip.memory_bandwidth_utilization || 0.9,
            },
            node: {
              chips_per_node: 8, // 默认值
              intra_node_bandwidth_gbps: 900,
            },
            cluster: {
              inter_node_bandwidth_gbps: 400,
            },
          } as HardwareConfig
        }
        return undefined
      })()

      const analysisResults = convertToAnalysisResults(taskResults)
      const bestResult = analysisResults.length > 0 ? analysisResults[0] : null

      return (
        <TooltipProvider>
          <div className="h-full w-full flex flex-col bg-gray-50">
            {/* 标题栏 */}
            <div className="px-6 py-4 border-b border-gray-200 bg-white flex justify-between items-center">
              <div className="flex items-center gap-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleBackToTasks}
                >
                  <ArrowLeft className="h-4 w-4 mr-1" />
                  返回任务列表
                </Button>
                <div>
                  <div className="text-lg font-semibold text-gray-900">
                    任务分析结果
                  </div>
                  <span className="text-sm text-gray-500">
                    任务 ID: {selectedTask.task_id.slice(0, 8)}... ·
                    共 {taskResults?.top_k_plans?.length || 0} 个评估结果
                  </span>
                </div>
              </div>
            </div>

            {/* 内容区 - 使用 AnalysisResultDisplay + ChartsPanel */}
            <div className="flex-1 overflow-auto p-6">
              <div className="w-full">
                <AnalysisResultDisplay
                  result={bestResult}
                  topKPlans={analysisResults}
                  loading={taskResultsLoading}
                  viewMode="detail"
                  hardware={hardwareConfig}
                  model={modelConfig as unknown as LLMModelConfig}
                  inference={inferenceConfig as unknown as InferenceConfig}
                  configSnapshot={selectedTask.config_snapshot}
                  benchmarkName={selectedTask.benchmark_name}
                  topologyConfigName={selectedTask.topology_config_name}
                  onSelectPlan={(plan) => {
                    // 切换选中的方案
                    const idx = analysisResults.findIndex(p => p.plan?.plan_id === plan.plan?.plan_id)
                    if (idx >= 0) {
                      // 可以在这里添加选中效果
                      console.log('Selected plan:', idx)
                    }
                  }}
                />
                {/* 图表可视化面板 - 包含雷达图、柱状图、饼图、Roofline、甘特图 */}
                {bestResult && (
                  <div className="mt-4">
                    <ChartsPanel
                      result={bestResult}
                      topKPlans={analysisResults}
                      hardware={hardwareConfig!}
                      model={modelConfig as unknown as LLMModelConfig}
                      inference={inferenceConfig as unknown as InferenceConfig}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </TooltipProvider>
      )
    }

    return (
      <TooltipProvider>
        <div className="h-full w-full flex flex-col bg-gray-50">
          {/* 标题栏 */}
          <div className="px-6 py-4 border-b border-gray-200 bg-white flex justify-between items-center">
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSelectedExperimentId(null)
                  setSelectedExperiment(null)
                }}
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                返回
              </Button>
              <div>
                <div className="text-lg font-semibold text-gray-900">
                  {selectedExperiment.name}
                </div>
                <span className="text-sm text-gray-500">{selectedExperiment.description || '无描述'}</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (selectedExperimentId) {
                        loadExperimentDetail(selectedExperimentId)
                      }
                    }}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>刷新</TooltipContent>
              </Tooltip>
            </div>
          </div>

          {/* 内容区 */}
          <div className="flex-1 overflow-auto p-6">
            <div className="w-full">
              {/* 进度提示 */}
              {progress < 100 && (
                <Alert className="mb-4">
                  <AlertDescription>
                    实验进度: {progress}% ({selectedExperiment.completed_tasks}/{selectedExperiment.total_tasks} 任务完成)
                  </AlertDescription>
                </Alert>
              )}

              {/* 任务列表表格 */}
              <Card className="mb-4">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-base">任务列表 ({selectedExperiment.tasks?.length || 0})</CardTitle>
                      <span className="text-xs text-gray-500 font-normal">
                        双击任务查看详细分析结果
                      </span>
                    </div>
                    <Button size="sm" variant="outline" onClick={() => loadExperimentDetail(selectedExperimentId!)}>
                      <RefreshCw className="h-3 w-3 mr-1" /> 刷新
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <TaskTable
                    tasks={selectedExperiment.tasks || []}
                    loading={false}
                    experimentId={selectedExperiment.id}
                    onTaskSelect={(task) => {
                      if (task.status === 'completed') {
                        loadTaskResults(task)
                      } else {
                        toast.info('仅已完成的任务可查看详细结果')
                      }
                    }}
                    onTasksDelete={async (taskIds) => {
                      // TODO: 实现批量删除任务的API调用
                      console.log('删除任务:', taskIds)
                      toast.info('批量删除功能待实现')
                    }}
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </TooltipProvider>
    )
  }

  return (
    <TooltipProvider>
      <div className="h-full w-full flex flex-col bg-gray-50">
        {/* 内容区 */}
        <div className="flex-1 overflow-auto p-6">
          {/* 实验列表 */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">实验列表</CardTitle>
                <div className="flex items-center gap-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button size="sm" variant="outline" onClick={() => setExportModalVisible(true)}>
                        <Download className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>导出</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button size="sm" variant="outline" onClick={() => setImportModalVisible(true)}>
                        <Upload className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>导入</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button size="sm" variant="outline" onClick={loadExperiments}>
                        <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>刷新</TooltipContent>
                  </Tooltip>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
                </div>
              ) : (
                <>
                  {selectedExperimentIds.length > 0 && (
                    <div className="mb-4 p-3 bg-blue-50 rounded-lg flex items-center gap-3">
                      <span className="text-sm">已选择 {selectedExperimentIds.length} 个实验</span>
                      <Button
                        variant="link"
                        size="sm"
                        onClick={() => setSelectedExperimentIds([])}
                      >
                        取消选择
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="destructive" size="sm">
                            删除选中
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>确定删除选中的实验吗？</AlertDialogTitle>
                            <AlertDialogDescription>
                              将删除 {selectedExperimentIds.length} 个实验，此操作无法恢复
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction onClick={handleBatchDelete}>删除</AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  )}
                  {experiments.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                      <Inbox className="h-12 w-12 mb-3" />
                      <span>暂无实验数据</span>
                    </div>
                  ) : (
                    <>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="w-10">
                              <Checkbox
                                checked={selectedExperimentIds.length === experiments.length && experiments.length > 0}
                                onCheckedChange={(checked) => handleSelectAll(checked as boolean)}
                              />
                            </TableHead>
                            <TableHead className="w-[280px]">实验名称</TableHead>
                            <TableHead className="w-20 text-center">任务数</TableHead>
                            <TableHead className="w-40 text-center">创建时间</TableHead>
                            <TableHead className="w-[200px]">描述</TableHead>
                            <TableHead className="w-40 text-center">操作</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {paginatedExperiments.map((record) => (
                            <TableRow key={record.id}>
                              <TableCell>
                                <Checkbox
                                  checked={selectedExperimentIds.includes(record.id)}
                                  onCheckedChange={(checked) => handleSelectOne(record.id, checked as boolean)}
                                />
                              </TableCell>
                              <TableCell>
                                {editingId === record.id ? (
                                  <Input
                                    value={editingName}
                                    onChange={(e) => setEditingName(e.target.value)}
                                    placeholder="实验名称"
                                    autoFocus
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter') {
                                        handleSaveEdit(record.id)
                                      }
                                    }}
                                  />
                                ) : (
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <span
                                        className="text-blue-500 cursor-pointer hover:underline"
                                        onClick={() => loadExperimentDetail(record.id)}
                                        onContextMenu={(e) => {
                                          e.preventDefault()
                                          handleStartEdit(record)
                                        }}
                                      >
                                        {record.name}
                                      </span>
                                    </TooltipTrigger>
                                    <TooltipContent>点击查看详情，右键编辑</TooltipContent>
                                  </Tooltip>
                                )}
                              </TableCell>
                              <TableCell className="text-center">
                                {(() => {
                                  const taskCount = record.tasks?.length || 0
                                  const completedCount = record.tasks?.filter((t: EvaluationTask) => t.status === 'completed').length || 0
                                  return `${completedCount}/${taskCount}`
                                })()}
                              </TableCell>
                              <TableCell className="text-center">
                                {record.created_at ? new Date(record.created_at).toLocaleString('zh-CN') : '-'}
                              </TableCell>
                              <TableCell>
                                {editingId === record.id ? (
                                  <Textarea
                                    value={editingDescription}
                                    onChange={(e) => setEditingDescription(e.target.value)}
                                    placeholder="实验描述"
                                    rows={2}
                                  />
                                ) : (
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <span className="truncate block max-w-[180px]">{record.description || '-'}</span>
                                    </TooltipTrigger>
                                    <TooltipContent>{record.description || '无描述'}</TooltipContent>
                                  </Tooltip>
                                )}
                              </TableCell>
                              <TableCell className="text-center">
                                {editingId === record.id ? (
                                  <div className="flex items-center justify-center gap-1">
                                    <Button
                                      size="sm"
                                      disabled={editingLoading}
                                      onClick={() => handleSaveEdit(record.id)}
                                    >
                                      {editingLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      onClick={handleCancelEdit}
                                    >
                                      <X className="h-3 w-3" />
                                    </Button>
                                  </div>
                                ) : (
                                  <div className="flex items-center justify-center gap-1">
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={() => loadExperimentDetail(record.id)}
                                        >
                                          <BarChart3 className="h-4 w-4" />
                                        </Button>
                                      </TooltipTrigger>
                                      <TooltipContent>查看详情</TooltipContent>
                                    </Tooltip>
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={() => handleStartEdit(record)}
                                        >
                                          <Pencil className="h-4 w-4" />
                                        </Button>
                                      </TooltipTrigger>
                                      <TooltipContent>编辑</TooltipContent>
                                    </Tooltip>
                                    <AlertDialog>
                                      <AlertDialogTrigger asChild>
                                        <Button variant="ghost" size="sm" className="text-red-500 hover:text-red-600">
                                          <Trash2 className="h-4 w-4" />
                                        </Button>
                                      </AlertDialogTrigger>
                                      <AlertDialogContent>
                                        <AlertDialogHeader>
                                          <AlertDialogTitle>确定删除此实验吗？</AlertDialogTitle>
                                          <AlertDialogDescription>删除后将无法恢复</AlertDialogDescription>
                                        </AlertDialogHeader>
                                        <AlertDialogFooter>
                                          <AlertDialogCancel>取消</AlertDialogCancel>
                                          <AlertDialogAction onClick={() => handleDelete(record.id)}>删除</AlertDialogAction>
                                        </AlertDialogFooter>
                                      </AlertDialogContent>
                                    </AlertDialog>
                                  </div>
                                )}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      <Pagination
                        currentPage={currentPage}
                        totalPages={totalPages}
                        pageSize={pageSize}
                        total={experiments.length}
                        onPageChange={setCurrentPage}
                        onPageSizeChange={(size) => {
                          setPageSize(size)
                          setCurrentPage(1)
                        }}
                      />
                    </>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* 导出模态框 */}
        <Dialog open={exportModalVisible} onOpenChange={setExportModalVisible}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>导出实验</DialogTitle>
            </DialogHeader>
            <div className="mb-4">
              {selectedExperimentIds.length > 0 ? (
                <div>
                  <div>已选择 {selectedExperimentIds.length} 个实验</div>
                  <Button
                    variant="link"
                    size="sm"
                    onClick={() => setSelectedExperimentIds([])}
                  >
                    导出全部实验
                  </Button>
                </div>
              ) : (
                <div>将导出所有 {experiments.length} 个实验的配置信息</div>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setExportModalVisible(false)}>取消</Button>
              <Button onClick={handleExport}>导出</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 导入模态框 */}
        <Dialog open={importModalVisible} onOpenChange={(open) => !open && resetImportModal()}>
          <DialogContent className="max-w-3xl">
            <DialogHeader>
              <DialogTitle>导入实验</DialogTitle>
            </DialogHeader>

            {importStep === 'upload' && (
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${importLoading ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-400 cursor-pointer'}`}
                onDragOver={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                }}
                onDrop={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (importLoading) return
                  const file = e.dataTransfer.files[0]
                  if (file && file.name.endsWith('.json')) {
                    handleImportFile(file)
                  } else {
                    toast.error('请上传 JSON 文件')
                  }
                }}
                onClick={() => {
                  if (importLoading) return
                  const input = document.createElement('input')
                  input.type = 'file'
                  input.accept = '.json'
                  input.onchange = (e) => {
                    const file = (e.target as HTMLInputElement).files?.[0]
                    if (file) {
                      handleImportFile(file)
                    }
                  }
                  input.click()
                }}
              >
                <Inbox className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600 mb-2">点击或拖拽 JSON 文件到此区域上传</p>
                <p className="text-gray-400 text-sm">支持导出的实验配置文件</p>
              </div>
            )}

            {importStep === 'config' && importCheckResult && (
              <div>
                <div className="mb-4">
                  <div>检测到 {importCheckResult.experiments?.length || 0} 个实验</div>
                </div>
                <div className="max-h-[400px] overflow-y-auto mb-4 space-y-3">
                  {importCheckResult.experiments?.map((exp: any, idx: number) => (
                    <Card key={idx}>
                      <CardContent className="pt-4">
                        <div className="mb-2">
                          <strong>实验名称：</strong> {exp.name}
                        </div>
                        {exp.description && (
                          <div className="mb-2">
                            <strong>描述：</strong> {exp.description}
                          </div>
                        )}
                        <div className="mb-2">
                          <strong>任务数：</strong> {exp.completed_tasks}/{exp.total_tasks}
                        </div>
                        {exp.conflict && (
                          <Alert variant="destructive" className="mb-3">
                            <AlertTitle>名称冲突</AlertTitle>
                            <AlertDescription>与现有实验 "{exp.name}" 重名</AlertDescription>
                          </Alert>
                        )}
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Checkbox
                              checked={importConfig.get(exp.name)?.action === 'rename'}
                              onCheckedChange={() => {
                                const config = new Map(importConfig)
                                config.set(exp.name, {
                                  original_name: exp.name,
                                  action: 'rename',
                                  new_name: exp.conflict ? `${exp.name}_imported` : exp.name,
                                })
                                setImportConfig(config)
                              }}
                            />
                            <span>重命名导入</span>
                            {exp.conflict && (
                              <Input
                                placeholder="新名称"
                                className="w-48 ml-2"
                                value={importConfig.get(exp.name)?.new_name || ''}
                                onChange={(e) => {
                                  const config = new Map(importConfig)
                                  const item = config.get(exp.name) || {}
                                  item.new_name = e.target.value
                                  config.set(exp.name, item)
                                  setImportConfig(config)
                                }}
                              />
                            )}
                          </div>
                          {!exp.conflict && (
                            <div className="flex items-center gap-2">
                              <Checkbox
                                checked={importConfig.get(exp.name)?.action === 'skip'}
                                onCheckedChange={() => {
                                  const config = new Map(importConfig)
                                  config.set(exp.name, {
                                    original_name: exp.name,
                                    action: 'skip',
                                  })
                                  setImportConfig(config)
                                }}
                              />
                              <span>跳过</span>
                            </div>
                          )}
                          {exp.conflict && (
                            <div className="flex items-center gap-2">
                              <Checkbox
                                checked={importConfig.get(exp.name)?.action === 'overwrite'}
                                onCheckedChange={() => {
                                  const config = new Map(importConfig)
                                  config.set(exp.name, {
                                    original_name: exp.name,
                                    action: 'overwrite',
                                  })
                                  setImportConfig(config)
                                }}
                              />
                              <span>覆盖现有实验</span>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={() => setImportStep('upload')}>返回</Button>
                  <Button onClick={handleExecuteImport} disabled={importLoading}>
                    {importLoading && <Loader2 className="h-4 w-4 mr-1 animate-spin" />}
                    导入
                  </Button>
                </div>
              </div>
            )}

            {importStep === 'importing' && (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-400" />
                <p>正在导入...</p>
              </div>
            )}

            {importStep === 'result' && importResult && (
              <div>
                <Alert variant={importResult.success ? 'default' : 'destructive'} className="mb-4">
                  <AlertTitle>{importResult.success ? '导入成功' : '导入失败'}</AlertTitle>
                  <AlertDescription>{importResult.message}</AlertDescription>
                </Alert>
                <div className="mb-4 space-y-1">
                  <div>导入成功：{importResult.imported_count} 个</div>
                  <div>跳过：{importResult.skipped_count} 个</div>
                  <div>覆盖：{importResult.overwritten_count} 个</div>
                </div>
                <Button onClick={resetImportModal} className="w-full">
                  完成
                </Button>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  )
}

export default Results
