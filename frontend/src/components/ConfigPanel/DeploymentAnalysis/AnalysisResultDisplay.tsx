/**
 * 分析结果展示组件
 *
 * - 概览显示历史记录列表
 * - 点击历史记录查看详情
 * - 支持返回历史记录列表
 */

import React, { useState, useCallback } from 'react'
import {
  Info,
  AlertTriangle,
  CheckCircle,
  History,
  Trash2,
  Trash,
  Download,
  Zap,
  Gauge,
  Clock,
  Target,
  StopCircle,
  XCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
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
import { PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig, DEFAULT_SCORE_WEIGHTS } from '../../../utils/llmDeployment/types'
import { InfeasibleResult } from '../../../utils/llmDeployment'
import { generateBenchmarkName, parseBenchmarkParts } from '../../../utils/llmDeployment/benchmarkNaming'
import { AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import { colors } from './ConfigSelectors'
import { BaseCard } from '../../common/BaseCard'
import { MetricDetailCard } from './components/MetricDetailCard'
import { ModelInfoCard } from './components/ModelInfoCard'
import { ParallelismInfo, ParallelismCard, type ParallelismType } from './components/ParallelismInfo'
import { ConfigSnapshotDisplay } from './components/ConfigSnapshotDisplay'

// ============================================
// 历史记录列表组件
// ============================================

interface HistoryListProps {
  history: AnalysisHistoryItem[]
  onLoad: (item: AnalysisHistoryItem) => void
  onDelete: (id: string) => void
  onClear: () => void
}

const HistoryList: React.FC<HistoryListProps> = ({
  history,
  onLoad,
  onDelete,
  onClear,
}) => {
  const [currentPage, setCurrentPage] = useState(1)
  const pageSize = 10

  // 导出JSON
  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `llm-deployment-history-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10">
        <div className="text-gray-400 text-sm mb-2">暂无历史记录</div>
        <span className="text-gray-400 text-xs">
          点击左侧"运行分析"开始第一次分析
        </span>
      </div>
    )
  }

  // 分页数据
  const totalPages = Math.ceil(history.length / pageSize)
  const paginatedData = history.slice((currentPage - 1) * pageSize, currentPage * pageSize)

  return (
    <div>
      {/* 标题栏 */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2">
          <History className="h-[18px] w-[18px]" style={{ color: colors.primary }} />
          <span className="font-semibold text-base">历史记录</span>
          <Badge variant="secondary" className="text-xs">{history.length}</Badge>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleExportJSON}
          >
            <Download className="h-3.5 w-3.5 mr-1" />
            导出
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" size="sm" className="text-red-500 hover:text-red-600">
                <Trash className="h-3.5 w-3.5 mr-1" />
                清空
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>清空所有历史记录？</AlertDialogTitle>
                <AlertDialogDescription>
                  此操作将删除所有分析历史记录，且无法恢复。
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction onClick={onClear} className="bg-red-500 hover:bg-red-600">
                  清空
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      {/* 历史记录表格 */}
      <div className="border rounded-lg mt-2">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[260px]">Benchmark</TableHead>
              <TableHead className="w-[160px]">并行策略</TableHead>
              <TableHead className="w-[120px] text-center">TPS/Chip</TableHead>
              <TableHead className="w-[90px] text-center">FTL</TableHead>
              <TableHead className="w-[40px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedData.map((record) => (
              <TableRow
                key={record.id}
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => onLoad(record)}
              >
                <TableCell className="font-semibold text-sm truncate max-w-[260px]">
                  {generateBenchmarkName(record.modelConfig, record.inferenceConfig)}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1 flex-wrap">
                    {record.parallelism.dp > 1 && (
                      <Badge className="text-xs m-0 bg-blue-100 text-blue-700 hover:bg-blue-100">DP{record.parallelism.dp}</Badge>
                    )}
                    {record.parallelism.tp > 1 && (
                      <Badge className="text-xs m-0 bg-green-100 text-green-700 hover:bg-green-100">TP{record.parallelism.tp}</Badge>
                    )}
                    {record.parallelism.ep > 1 && (
                      <Badge className="text-xs m-0 bg-purple-100 text-purple-700 hover:bg-purple-100">EP{record.parallelism.ep}</Badge>
                    )}
                  </div>
                </TableCell>
                <TableCell className="text-center text-sm">
                  {record.chips > 0 ? (record.throughput / record.chips).toFixed(0) : 0} tok/s
                </TableCell>
                <TableCell className="text-center text-sm">
                  {record.ttft.toFixed(1)} ms
                </TableCell>
                <TableCell>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 text-gray-400 hover:text-red-500"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent onClick={(e) => e.stopPropagation()}>
                      <AlertDialogHeader>
                        <AlertDialogTitle>删除此记录？</AlertDialogTitle>
                        <AlertDialogDescription>
                          此操作将删除该分析记录，且无法恢复。
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>取消</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={(e) => {
                            e.stopPropagation()
                            onDelete(record.id)
                          }}
                          className="bg-red-500 hover:bg-red-600"
                        >
                          删除
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="flex justify-end items-center gap-2 mt-3">
          <Button
            variant="outline"
            size="sm"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(p => p - 1)}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-gray-500">
            {currentPage} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={currentPage === totalPages}
            onClick={() => setCurrentPage(p => p + 1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}

      <div className="mt-3 p-2 px-3 bg-gray-100 rounded-md text-xs text-gray-500 text-center">
        💡 点击行查看详细分析结果
      </div>
    </div>
  )
}

// ============================================
// 分析结果展示组件
// ============================================

interface AnalysisResultDisplayProps {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  /** 不可行方案列表 */
  infeasiblePlans?: InfeasibleResult[]
  loading: boolean
  onSelectPlan?: (plan: PlanAnalysisResult) => void
  searchStats?: { evaluated: number; feasible: number; timeMs: number } | null
  searchProgress?: {
    stage: 'idle' | 'generating' | 'evaluating' | 'completed' | 'cancelled'
    totalCandidates: number
    currentEvaluating: number
    evaluated: number
  }
  /** 取消评估的回调 */
  onCancelEvaluation?: () => void
  errorMsg?: string | null
  // 视图模式（从父组件传入）
  viewMode?: AnalysisViewMode
  onViewModeChange?: (mode: AnalysisViewMode) => void
  // 历史记录相关
  history?: AnalysisHistoryItem[]
  onLoadFromHistory?: (item: AnalysisHistoryItem) => void
  onDeleteHistory?: (id: string) => void
  onClearHistory?: () => void
  // 详情视图功能按钮
  canMapToTopology?: boolean
  onMapToTopology?: () => void
  onClearTraffic?: () => void
  // HeroKPIPanel 需要的数据
  hardware?: HardwareConfig
  model?: LLMModelConfig
  inference?: InferenceConfig
  // 配置快照（从任务中获取）
  configSnapshot?: {
    model: Record<string, unknown>
    inference: Record<string, unknown>
    topology: Record<string, unknown>
  }
  benchmarkName?: string
  topologyConfigName?: string
}

type MetricType = 'ttft' | 'tpot' | 'throughput' | 'tps_batch' | 'tps_chip' | 'mfu' | 'mbu' | 'cost' | 'percentiles' | 'bottleneck' | 'e2e' | 'chips' | 'memory' | null

export const AnalysisResultDisplay: React.FC<AnalysisResultDisplayProps> = ({
  result,
  topKPlans,
  infeasiblePlans = [],
  loading,
  onSelectPlan,
  searchStats,
  searchProgress,
  onCancelEvaluation,
  errorMsg,
  viewMode = 'history',
  onViewModeChange: _onViewModeChange,
  history = [],
  onLoadFromHistory,
  onDeleteHistory,
  onClearHistory,
  canMapToTopology,
  onMapToTopology,
  onClearTraffic,
  hardware: _hardware,
  model,
  inference,
  configSnapshot,
  benchmarkName,
  topologyConfigName,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>(null)
  const [showScoreDetails, setShowScoreDetails] = useState(false)
  const [showModelArchitecture, setShowModelArchitecture] = useState(false)
  const [selectedParallelism, setSelectedParallelism] = useState<ParallelismType | null>(null)

  // 各章节折叠状态
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    deployment: true,
    performance: true,
    suggestions: true,
    candidates: true,
  })

  // 从历史记录加载（父组件会自动切换到详情视图）
  const handleLoadFromHistory = useCallback((item: AnalysisHistoryItem) => {
    onLoadFromHistory?.(item)
  }, [onLoadFromHistory])

  // 搜索进度卡片组件（独立提取）
  const SearchProgressCard = () => {
    if (!loading && (!searchProgress || searchProgress.stage === 'idle')) {
      return null
    }

    return (
      <BaseCard
        title="搜索与评估"
        style={{ marginBottom: 16 }}
      >
        {loading ? (
          <div className="flex flex-col gap-3">
            {searchProgress && searchProgress.stage !== 'idle' ? (
              <>
                {/* 阶段 1: 生成候选方案 */}
                <div className="flex items-center gap-2">
                  {searchProgress.stage === 'generating' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <CheckCircle className="h-4 w-4" style={{ color: colors.success }} />
                  )}
                  <span className="text-[13px]">
                    生成候选方案: <strong>{searchProgress.totalCandidates}</strong> 个
                  </span>
                </div>

                {/* 阶段 2: 后端评估 */}
                {searchProgress.stage !== 'generating' && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {searchProgress.stage === 'evaluating' ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <CheckCircle className="h-4 w-4" style={{ color: colors.success }} />
                      )}
                      <span className="text-[13px]">
                        后端评估: <strong>{searchProgress.evaluated}</strong> / <strong>{searchProgress.totalCandidates}</strong>
                        {searchProgress.stage === 'evaluating' && (
                          <span className="text-gray-400 text-[11px] ml-2">（5 并发）</span>
                        )}
                      </span>
                    </div>
                    {/* 取消按钮 */}
                    {searchProgress.stage === 'evaluating' && onCancelEvaluation && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-500 hover:text-red-600 text-xs"
                        onClick={onCancelEvaluation}
                      >
                        <StopCircle className="h-3.5 w-3.5 mr-1" />
                        取消
                      </Button>
                    )}
                  </div>
                )}

                {/* 阶段 3: 排序结果 */}
                {searchProgress.stage === 'completed' && (
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4" style={{ color: colors.success }} />
                    <span className="text-[13px]">排序并显示结果</span>
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-gray-500 text-[13px]">正在搜索最优方案...</span>
              </div>
            )}
          </div>
        ) : searchProgress?.stage === 'cancelled' ? (
          // 搜索已取消
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4" style={{ color: colors.success }} />
              <span className="text-[13px]">
                生成候选方案: <strong>{searchProgress.totalCandidates}</strong> 个
              </span>
            </div>
            <div className="flex items-center gap-2">
              <XCircle className="h-4 w-4 text-amber-500" />
              <span className="text-[13px] text-amber-500">
                评估已取消: 已完成 <strong className="text-amber-500">{searchProgress.evaluated}</strong> / <strong className="text-amber-500">{searchProgress.totalCandidates}</strong>
              </span>
            </div>
          </div>
        ) : (
          // 搜索完成后显示最后一次的统计
          <div className="flex items-center gap-2">
            <CheckCircle className="h-4 w-4" style={{ color: colors.success }} />
            <span className="text-[13px]">
              最近搜索: 评估了 <strong>{searchProgress?.totalCandidates || 0}</strong> 个方案
            </span>
          </div>
        )}
      </BaseCard>
    )
  }

  // 错误提示组件
  const ErrorAlert = () => {
    if (!errorMsg) return null

    return (
      <div className="mb-4">
        <div className="text-center p-5 bg-red-50 rounded-lg border border-red-200">
          <AlertTriangle className="h-6 w-6 text-red-500 mx-auto mb-2" />
          <div className="text-red-500 font-medium">{errorMsg}</div>
        </div>
        {searchStats && (
          <div className="mt-3 p-2 bg-gray-100 rounded-md">
            <span className="text-gray-500 text-[11px]">
              搜索统计: 评估 {searchStats.evaluated} 个方案，{searchStats.feasible} 个可行，耗时 {searchStats.timeMs.toFixed(0)}ms
            </span>
          </div>
        )}
      </div>
    )
  }

  // 不可行方案列表组件
  const InfeasiblePlansList = () => {
    const [expanded, setExpanded] = useState(false)

    if (infeasiblePlans.length === 0) return null

    // 按错误原因分组统计
    const reasonCounts: Record<string, number> = {}
    infeasiblePlans.forEach(plan => {
      const reason = plan.reason || '未知原因'
      reasonCounts[reason] = (reasonCounts[reason] || 0) + 1
    })

    return (
      <BaseCard
        title={
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <span>不可行方案 ({infeasiblePlans.length})</span>
          </div>
        }
        style={{ marginBottom: 16 }}
      >
        {/* 错误原因统计 */}
        <div className={expanded ? 'mb-3' : ''}>
          {Object.entries(reasonCounts).map(([reason, count]) => (
            <div key={reason} className="flex justify-between items-center py-1 border-b border-dashed border-gray-200">
              <span className="text-gray-500 text-xs">{reason}</span>
              <Badge className="text-[11px] m-0 bg-orange-100 text-orange-700 hover:bg-orange-100">{count} 个</Badge>
            </div>
          ))}
        </div>

        {/* 展开详细列表 */}
        <div
          className="cursor-pointer text-center py-2 text-blue-500 text-xs"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? '收起详情 ▲' : '展开详情 ▼'}
        </div>

        {expanded && (
          <div className="max-h-[300px] overflow-y-auto mt-2">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[150px]">并行策略</TableHead>
                  <TableHead className="w-[60px]">芯片数</TableHead>
                  <TableHead>失败原因</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {infeasiblePlans.map((plan, index) => (
                  <TableRow key={index}>
                    <TableCell className="text-[11px] font-mono">
                      DP={plan.parallelism.dp} TP={plan.parallelism.tp} EP={plan.parallelism.ep}
                    </TableCell>
                    <TableCell className="text-[11px]">
                      {plan.parallelism.dp * plan.parallelism.tp * plan.parallelism.ep}
                    </TableCell>
                    <TableCell>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="text-red-500 text-[11px] truncate block max-w-[200px]">{plan.reason}</span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{plan.reason}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </BaseCard>
    )
  }

  // 历史列表视图
  if (viewMode === 'history') {
    return (
      <div className="p-1">
        <SearchProgressCard />
        <ErrorAlert />
        <InfeasiblePlansList />
        <HistoryList
          history={history}
          onLoad={handleLoadFromHistory}
          onDelete={onDeleteHistory || (() => {})}
          onClear={onClearHistory || (() => {})}
        />
      </div>
    )
  }

  // 详情视图但没有结果（回退到历史列表）
  if (!result) {
    return (
      <div className="p-1">
        <SearchProgressCard />
        <ErrorAlert />
        <InfeasiblePlansList />
        <HistoryList
          history={history}
          onLoad={handleLoadFromHistory}
          onDelete={onDeleteHistory || (() => {})}
          onClear={onClearHistory || (() => {})}
        />
      </div>
    )
  }

  const { plan, memory, latency, throughput, score, suggestions, is_feasible, infeasibility_reason } = result

  // 统一的可点击卡片样式（适用于所有交互卡片）
  // 使用固定2px边框避免布局抖动，未选中时使用透明边框
  const clickableCardStyle = (isSelected: boolean): React.CSSProperties => ({
    padding: '12px 16px',
    background: isSelected ? colors.interactiveLight : colors.cardBg,
    borderRadius: 8,
    cursor: 'pointer',
    border: `2px solid ${isSelected ? colors.interactive : 'transparent'}`,
    outline: isSelected ? 'none' : `1px solid ${colors.border}`,
    outlineOffset: '-2px',
    transition: 'background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease',
    boxShadow: isSelected ? `0 2px 8px ${colors.interactiveShadow}` : 'none',
  })

  // 性能指标卡片样式（保持紧凑布局）
  const metricCardStyle = (isSelected: boolean): React.CSSProperties => ({
    ...clickableCardStyle(isSelected),
    padding: '12px 10px',
  })

  return (
    <TooltipProvider>
      <div>
        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 一、部署方案 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        <div className="mb-4">
          <BaseCard
            title="部署方案"
            accentColor="#5E6AD2"
            collapsible
            expanded={expandedSections.deployment}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, deployment: expanded }))}
          >
            {/* 1. Benchmark 参数卡片（默认显示） */}
            {inference && model && (
              <div className="mb-4">
                <div className="text-[13px] font-medium mb-2" style={{ color: colors.text }}>Benchmark</div>

                {/* Benchmark 参数卡片 */}
                <div className="flex flex-wrap gap-3">
                  {parseBenchmarkParts(model, inference).map((part, idx) => (
                    <div
                      key={idx}
                      style={{
                        ...(idx === 0 ? clickableCardStyle(showModelArchitecture) : {
                          padding: '12px 16px',
                          background: colors.cardBg,
                          borderRadius: 8,
                          border: '2px solid transparent',
                          outline: `1px solid ${colors.border}`,
                          outlineOffset: '-2px',
                        }),
                        minWidth: 100,
                        cursor: idx === 0 ? 'pointer' : 'default',
                        position: 'relative',
                      }}
                      onClick={() => idx === 0 && setShowModelArchitecture(!showModelArchitecture)}
                    >
                      {/* 右上角标记：仅第一个卡片显示 */}
                      {idx === 0 && (
                        <Info
                          className="absolute top-2 right-2 h-3 w-3 cursor-pointer"
                          style={{ color: showModelArchitecture ? colors.interactive : '#d9d9d9' }}
                        />
                      )}
                      <div className="text-center text-lg font-semibold mb-1" style={{ color: colors.primary }}>
                        {part.key}
                      </div>
                      <div className="text-[13px]">
                        <span className="text-gray-400">{part.label}：</span>
                        <span className="font-medium" style={{ color: colors.text }}>{part.value}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* 模型架构详细信息 */}
                {showModelArchitecture && (
                  <div className="mt-4 pt-4 border-t border-dashed border-gray-200">
                    <div className="text-[13px] font-medium mb-3" style={{ color: colors.text }}>模型架构</div>
                    <ModelInfoCard model={model} inference={inference} />
                  </div>
                )}
              </div>
            )}

            {/* 分割线 */}
            <div
              className="h-px mb-4"
              style={{ background: 'linear-gradient(to right, transparent, #e8e8e8, transparent)' }}
            />

            {/* 2. 并行策略卡片 */}
            <div className="mb-4">
              <div className="text-[13px] font-medium mb-2" style={{ color: colors.text }}>并行策略</div>
              <div className="flex gap-2">
                <ParallelismCard
                  type="dp"
                  value={plan.parallelism.dp}
                  selected={selectedParallelism === 'dp'}
                  onClick={() => setSelectedParallelism(selectedParallelism === 'dp' ? null : 'dp')}
                />
                <ParallelismCard
                  type="tp"
                  value={plan.parallelism.tp}
                  selected={selectedParallelism === 'tp'}
                  onClick={() => setSelectedParallelism(selectedParallelism === 'tp' ? null : 'tp')}
                />
                {/* 不显示 PP */}
                {plan.parallelism.ep > 1 && (
                  <ParallelismCard
                    type="ep"
                    value={plan.parallelism.ep}
                    selected={selectedParallelism === 'ep'}
                    onClick={() => setSelectedParallelism(selectedParallelism === 'ep' ? null : 'ep')}
                  />
                )}
                {plan.parallelism.moe_tp && plan.parallelism.moe_tp > 1 && (
                  <ParallelismCard
                    type="moe_tp"
                    value={plan.parallelism.moe_tp}
                    selected={selectedParallelism === 'moe_tp'}
                    onClick={() => setSelectedParallelism(selectedParallelism === 'moe_tp' ? null : 'moe_tp')}
                  />
                )}
                {plan.parallelism.sp > 1 && (
                  <ParallelismCard
                    type="sp"
                    value={plan.parallelism.sp}
                    selected={selectedParallelism === 'sp'}
                    onClick={() => setSelectedParallelism(selectedParallelism === 'sp' ? null : 'sp')}
                  />
                )}
              </div>

              {/* 并行策略详细介绍 - 显示在并行策略小节内 */}
              {selectedParallelism && (
                <div className="mt-3">
                  <ParallelismInfo type={selectedParallelism} />
                </div>
              )}
            </div>

            {/* 分割线 */}
            <div
              className="h-px mb-4"
              style={{ background: 'linear-gradient(to right, transparent, #e8e8e8, transparent)' }}
            />

            {/* 3. 芯片数量配置 */}
            <div className="mb-3">
              <div className="text-[13px] font-medium mb-2" style={{ color: colors.text }}>芯片数量配置</div>

              {/* 总芯片数和搜索统计 */}
              <div className="text-[13px] mb-3" style={{ color: colors.textSecondary }}>
                <span>总芯片数: <b style={{ color: colors.text }}>{plan.total_chips}</b></span>
                {searchStats && (
                  <span className="ml-4">
                    搜索: {searchStats.evaluated} 方案 · {searchStats.feasible} 可行 · {searchStats.timeMs.toFixed(0)}ms
                  </span>
                )}
              </div>

              {/* 硬件拓扑配置 */}
              {_hardware && (
                <div className="p-2.5 px-3 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex flex-wrap gap-4 text-xs">
                    {/* Chip配置 */}
                    <div>
                      <span className="text-gray-400">Chip: </span>
                      <b style={{ color: colors.text }}>{_hardware.chip.chip_type}</b>
                      <span className="text-gray-300 ml-1">
                        ({_hardware.chip.compute_tflops_fp16} TFLOPs, {_hardware.chip.memory_gb}GB, {_hardware.chip.memory_bandwidth_gbps} GB/s)
                      </span>
                    </div>
                    {/* Board配置 */}
                    <div>
                      <span className="text-gray-400">Board: </span>
                      <b style={{ color: colors.text }}>{_hardware.node.chips_per_node} Chips/Board</b>
                      <span className="text-gray-300 ml-1">
                        (NVLink {_hardware.node.intra_node_bandwidth_gbps} GB/s)
                      </span>
                    </div>
                    {/* 总Board数：根据总芯片数和每Board芯片数计算 */}
                    <div>
                      <span className="text-gray-400">总计: </span>
                      <b style={{ color: colors.text }}>{Math.ceil(plan.total_chips / _hardware.node.chips_per_node)} Boards</b>
                      <span className="text-gray-300 ml-1">
                        (Board间 {_hardware.cluster.inter_node_bandwidth_gbps} GB/s)
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* 拓扑映射操作 */}
            {canMapToTopology && (
              <div
                className="mt-3 pt-3 flex justify-between items-center"
                style={{ borderTop: `1px dashed ${colors.borderLight}` }}
              >
                <span className="text-[11px]" style={{ color: colors.textSecondary }}>
                  将并行策略映射到拓扑视图，查看通信流量分布
                </span>
                <div className="flex gap-1.5">
                  <Button
                    size="sm"
                    onClick={onMapToTopology}
                    className="text-[11px]"
                  >
                    映射到拓扑
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onClearTraffic}
                    className="text-[11px]"
                  >
                    清除映射
                  </Button>
                </div>
              </div>
            )}

            {/* 配置快照展示 */}
            {configSnapshot && (
              <div className="mt-4">
                <div className="text-[13px] font-medium mb-3" style={{ color: colors.text }}>
                  完整配置快照
                </div>
                <ConfigSnapshotDisplay
                  configSnapshot={configSnapshot}
                  benchmarkName={benchmarkName}
                  topologyConfigName={topologyConfigName}
                />
              </div>
            )}
          </BaseCard>
        </div>

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 二、性能分析 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        <div className="mb-4">
          <BaseCard
            title="性能分析"
            accentColor="#52c41a"
            collapsible
            expanded={expandedSections.performance}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, performance: expanded }))}
          >
          <>
          {/* 延迟指标 */}
          <span className="text-[13px] font-medium block mb-2" style={{ color: colors.text }}>延迟</span>
          <div className="grid grid-cols-4 gap-2 mb-3">
            <div style={{ ...metricCardStyle(selectedMetric === 'ttft'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'ttft' ? null : 'ttft')}>
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'ttft' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>FTL</span>
              <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                {latency?.prefill_total_latency_ms?.toFixed(1) || '0.0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>ms</span>
              </div>
            </div>
            <div style={{ ...metricCardStyle(selectedMetric === 'tpot'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'tpot' ? null : 'tpot')}>
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'tpot' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>TPOT</span>
              <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                {latency?.decode_per_token_latency_ms?.toFixed(2) || '0.00'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>ms</span>
              </div>
            </div>
            <div style={{ ...metricCardStyle(selectedMetric === 'e2e'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'e2e' ? null : 'e2e')}>
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'e2e' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>E2E</span>
              <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                {((latency?.end_to_end_latency_ms || 0) / 1000).toFixed(2)} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>s</span>
              </div>
            </div>
            <div style={{ ...metricCardStyle(selectedMetric === 'percentiles'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'percentiles' ? null : 'percentiles')}>
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'percentiles' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>P99</span>
              <div className="text-lg font-semibold mt-1" style={{ color: latency.ttft_percentiles && latency.ttft_percentiles.p99 > 450 ? colors.error : colors.text }}>
                {latency.ttft_percentiles ? latency.ttft_percentiles.p99.toFixed(0) : '-'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>ms</span>
              </div>
            </div>
          </div>

          {/* 吞吐与效率 */}
          <span className="text-[13px] font-medium block mb-2" style={{ color: colors.text }}>吞吐与效率</span>
          <div className="grid grid-cols-3 gap-2 mb-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <div style={{ ...metricCardStyle(selectedMetric === 'throughput'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'throughput' ? null : 'throughput')}>
                  <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'throughput' ? colors.interactive : '#d9d9d9' }} />
                  <span className="text-[13px]" style={{ color: colors.textSecondary }}>Total TPS</span>
                  <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                    {throughput?.tokens_per_second?.toFixed(0) || '0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>tok/s</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>Total TPS = TPS_chip × NumChips，集群总吞吐</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <div style={{ ...metricCardStyle(selectedMetric === 'tps_batch'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'tps_batch' ? null : 'tps_batch')}>
                  <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'tps_batch' ? colors.interactive : '#d9d9d9' }} />
                  <span className="text-[13px]" style={{ color: colors.textSecondary }}>TPS/Batch</span>
                  <div className="text-lg font-semibold mt-1" style={{ color: (throughput?.tps_per_batch || 0) >= 10 ? colors.text : colors.error }}>
                    {throughput?.tps_per_batch?.toFixed(1) || '0.0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>tok/s</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>TPS per Batch = 1000 / TPOT(ms)，用户体验指标，SLO约束 ≥10</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <div style={{ ...metricCardStyle(selectedMetric === 'tps_chip'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'tps_chip' ? null : 'tps_chip')}>
                  <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'tps_chip' ? colors.interactive : '#d9d9d9' }} />
                  <span className="text-[13px]" style={{ color: colors.textSecondary }}>TPS/Chip</span>
                  <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                    {throughput?.tps_per_chip?.toFixed(0) || '0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>tok/s</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>TPS per Chip = B × TPS_batch，成本效益优化目标</TooltipContent>
            </Tooltip>
          </div>
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div style={{ ...metricCardStyle(selectedMetric === 'mfu'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'mfu' ? null : 'mfu')}>
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'mfu' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>MFU</span>
              <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                {((throughput?.model_flops_utilization || 0) * 100).toFixed(1)} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>%</span>
              </div>
            </div>
            <div style={{ ...metricCardStyle(selectedMetric === 'mbu'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'mbu' ? null : 'mbu')}>
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'mbu' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>MBU</span>
              <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                {((throughput?.memory_bandwidth_utilization || 0) * 100).toFixed(1)} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>%</span>
              </div>
            </div>
          </div>

          {/* 资源利用 */}
          <span className="text-[13px] font-medium block mb-2" style={{ color: colors.text }}>资源利用</span>
          <div className="grid grid-cols-2 gap-2 mb-3">
            {/* 显存占用 */}
            <div
              style={{ ...metricCardStyle(selectedMetric === 'memory'), textAlign: 'center', position: 'relative' }}
              onClick={() => setSelectedMetric(selectedMetric === 'memory' ? null : 'memory')}
            >
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'memory' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>显存占用</span>
              <div className="text-lg font-semibold mt-1" style={{ color: memory?.is_memory_sufficient ? colors.text : colors.error }}>
                {memory?.total_per_chip_gb?.toFixed(1) || '0.0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>/ 80G</span>
              </div>
            </div>
            {/* 推理成本 */}
            <div
              style={{ ...metricCardStyle(selectedMetric === 'cost'), textAlign: 'center', position: 'relative' }}
              onClick={() => setSelectedMetric(selectedMetric === 'cost' ? null : 'cost')}
            >
              <Info className="absolute top-2 right-2 h-2.5 w-2.5" style={{ color: selectedMetric === 'cost' ? colors.interactive : '#d9d9d9' }} />
              <span className="text-[13px]" style={{ color: colors.textSecondary }}>推理成本</span>
              <div className="text-lg font-semibold mt-1" style={{ color: colors.text }}>
                ${result.cost ? result.cost.cost_per_million_tokens.toFixed(3) : '-'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>/M</span>
              </div>
            </div>
          </div>

          {/* 综合评分 + 瓶颈分析 */}
          <div className="flex items-stretch gap-3 mt-4 pt-4" style={{ borderTop: `1px dashed ${colors.borderLight}` }}>
            {/* 综合评分 */}
            <div
              className="relative flex items-center gap-3 px-5 py-3 rounded-lg cursor-pointer transition-shadow"
              style={{
                background: is_feasible ? '#f6ffed' : '#fff2f0',
                border: `2px solid ${is_feasible ? '#b7eb8f' : '#ffccc7'}`,
                boxShadow: showScoreDetails ? (is_feasible ? '0 2px 8px rgba(82, 196, 26, 0.15)' : '0 2px 8px rgba(255, 77, 79, 0.15)') : 'none',
              }}
              onClick={() => setShowScoreDetails(!showScoreDetails)}
            >
              {/* 右上角标记 */}
              <Info
                className="absolute top-2 right-2 h-2.5 w-2.5"
                style={{ color: showScoreDetails ? (is_feasible ? colors.success : colors.error) : '#d9d9d9' }}
              />
              {is_feasible ? (
                <CheckCircle className="h-[18px] w-[18px]" style={{ color: colors.success }} />
              ) : (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <AlertTriangle className="h-[18px] w-[18px]" style={{ color: colors.error }} />
                  </TooltipTrigger>
                  <TooltipContent>{infeasibility_reason}</TooltipContent>
                </Tooltip>
              )}
              <div>
                <span className="text-2xl font-bold leading-none" style={{ color: is_feasible ? colors.success : colors.error }}>
                  {score?.overall_score?.toFixed(1) || '0.0'}
                </span>
                <span className="text-[13px] ml-1" style={{ color: colors.textSecondary }}>分</span>
              </div>
              <div className="text-xs" style={{ color: colors.textSecondary }}>
                综合评分
              </div>
            </div>

            {/* 瓶颈分析 */}
            <div
              style={{
                ...clickableCardStyle(selectedMetric === 'bottleneck'),
                flex: 1,
              }}
              onClick={() => setSelectedMetric(selectedMetric === 'bottleneck' ? null : 'bottleneck')}
            >
              {/* 右上角标记 */}
              <Info
                className="absolute top-2 right-2 h-2.5 w-2.5"
                style={{ color: selectedMetric === 'bottleneck' ? colors.interactive : '#d9d9d9' }}
              />
              <div className="flex items-center justify-between mb-1.5">
                <Badge
                  className="m-0"
                  style={{
                    background: latency.bottleneck_type === 'compute' ? '#fff7e6' :
                              latency.bottleneck_type === 'memory' ? '#e6f7ff' :
                              latency.bottleneck_type === 'communication' ? '#f9f0ff' :
                              latency.bottleneck_type === 'balanced' ? '#f6ffed' : '#f5f5f5',
                    color: latency.bottleneck_type === 'compute' ? '#fa8c16' :
                           latency.bottleneck_type === 'memory' ? '#1890ff' :
                           latency.bottleneck_type === 'communication' ? '#722ed1' :
                           latency.bottleneck_type === 'balanced' ? '#52c41a' : '#666',
                    border: 'none',
                  }}
                >
                  {latency.bottleneck_type === 'compute' ? '算力瓶颈' :
                   latency.bottleneck_type === 'memory' ? '访存瓶颈' :
                   latency.bottleneck_type === 'communication' ? '通信瓶颈' :
                   latency.bottleneck_type === 'balanced' ? '均衡状态' : latency.bottleneck_type}
                </Badge>
                {latency.bottleneck_analysis && (
                  <span className="text-[11px]" style={{ color: colors.textSecondary }}>
                    {latency.bottleneck_analysis.dominant_phase === 'prefill' ? 'Prefill主导' : 'Decode主导'}
                  </span>
                )}
              </div>
              {latency.bottleneck_analysis && (
                <>
                  <div className="flex h-1.5 rounded-sm overflow-hidden bg-gray-200">
                    {(() => {
                      const analysis = latency.bottleneck_analysis.dominant_phase === 'prefill'
                        ? latency.bottleneck_analysis.prefill
                        : latency.bottleneck_analysis.decode;
                      return (
                        <>
                          <div style={{ width: `${analysis.compute_ratio * 100}%`, background: '#faad14' }} />
                          <div style={{ width: `${analysis.memory_ratio * 100}%`, background: '#1890ff' }} />
                          <div style={{ width: `${analysis.comm_ratio * 100}%`, background: '#722ed1' }} />
                        </>
                      );
                    })()}
                  </div>
                  <div className="flex gap-3 mt-1 text-[10px]" style={{ color: colors.textSecondary }}>
                    {(() => {
                      const analysis = latency.bottleneck_analysis.dominant_phase === 'prefill'
                        ? latency.bottleneck_analysis.prefill
                        : latency.bottleneck_analysis.decode;
                      return (
                        <>
                          <span><span className="inline-block w-1.5 h-1.5 rounded-sm mr-0.5 align-middle" style={{ background: '#faad14' }} />计算{(analysis.compute_ratio * 100).toFixed(0)}%</span>
                          <span><span className="inline-block w-1.5 h-1.5 rounded-sm mr-0.5 align-middle" style={{ background: '#1890ff' }} />访存{(analysis.memory_ratio * 100).toFixed(0)}%</span>
                          <span><span className="inline-block w-1.5 h-1.5 rounded-sm mr-0.5 align-middle" style={{ background: '#722ed1' }} />通信{(analysis.comm_ratio * 100).toFixed(0)}%</span>
                        </>
                      );
                    })()}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* 评分详情展开区域 */}
          {showScoreDetails && (
            <div className="mt-3 p-3 bg-gray-50 rounded-lg">
              <div className="grid grid-cols-4 gap-2 mb-3">
                <div className="text-center p-2 bg-blue-50 rounded-md">
                  <Clock className="h-3.5 w-3.5 text-blue-500 mx-auto" />
                  <div className="text-base font-semibold text-blue-500 my-1">{score?.latency_score?.toFixed(0) || '0'}</div>
                  <div className="text-[10px]" style={{ color: colors.textSecondary }}>延迟 {(DEFAULT_SCORE_WEIGHTS.latency * 100).toFixed(0)}%</div>
                </div>
                <div className="text-center p-2 bg-green-50 rounded-md">
                  <Zap className="h-3.5 w-3.5 text-green-500 mx-auto" />
                  <div className="text-base font-semibold text-green-500 my-1">{score?.throughput_score?.toFixed(0) || '0'}</div>
                  <div className="text-[10px]" style={{ color: colors.textSecondary }}>吞吐 {(DEFAULT_SCORE_WEIGHTS.throughput * 100).toFixed(0)}%</div>
                </div>
                <div className="text-center p-2 bg-orange-50 rounded-md">
                  <Gauge className="h-3.5 w-3.5 text-orange-500 mx-auto" />
                  <div className="text-base font-semibold text-orange-500 my-1">{score?.efficiency_score?.toFixed(0) || '0'}</div>
                  <div className="text-[10px]" style={{ color: colors.textSecondary }}>效率 {(DEFAULT_SCORE_WEIGHTS.efficiency * 100).toFixed(0)}%</div>
                </div>
                <div className="text-center p-2 bg-purple-50 rounded-md">
                  <Target className="h-3.5 w-3.5 text-purple-500 mx-auto" />
                  <div className="text-base font-semibold text-purple-500 my-1">{score?.balance_score?.toFixed(0) || '0'}</div>
                  <div className="text-[10px]" style={{ color: colors.textSecondary }}>均衡 {(DEFAULT_SCORE_WEIGHTS.balance * 100).toFixed(0)}%</div>
                </div>
              </div>
              <div className="text-[11px] text-center font-mono" style={{ color: colors.textSecondary }}>
                综合 = {(DEFAULT_SCORE_WEIGHTS.latency * 100).toFixed(0)}%×延迟 + {(DEFAULT_SCORE_WEIGHTS.throughput * 100).toFixed(0)}%×吞吐 + {(DEFAULT_SCORE_WEIGHTS.efficiency * 100).toFixed(0)}%×效率 + {(DEFAULT_SCORE_WEIGHTS.balance * 100).toFixed(0)}%×均衡
              </div>
            </div>
          )}

          {/* 指标详情展示 - 内嵌在性能分析中 */}
          {selectedMetric && (
            <div className="mt-4 pt-4" style={{ borderTop: `1px dashed ${colors.borderLight}` }}>
              <MetricDetailCard metric={selectedMetric} result={result} />
            </div>
          )}
          </>
          </BaseCard>
        </div>

        {/* 优化建议 */}
        {suggestions.length > 0 && (
          <div className="mb-4">
            <BaseCard
              title="优化建议"
              accentColor="#722ed1"
              collapsible
              expanded={expandedSections.suggestions}
              onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, suggestions: expanded }))}
            >
              {suggestions.slice(0, 3).map((s, i) => (
                <div key={i} className="p-2.5 bg-white rounded-lg mb-2 border" style={{
                  borderLeft: `3px solid ${s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary}`,
                  borderColor: colors.border,
                }}>
                  <div className="flex justify-between items-start">
                    <span className="text-xs flex-1" style={{ color: colors.text }}>{s.description}</span>
                    <Badge
                      className="ml-2 text-[9px] px-1.5 py-0 rounded border-none"
                      style={{
                        background: s.priority <= 2 ? colors.errorLight : s.priority <= 3 ? colors.warningLight : colors.primaryLight,
                        color: s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary,
                      }}
                    >
                      P{s.priority}
                    </Badge>
                  </div>
                  <span className="text-[10px] mt-1 block" style={{ color: colors.textSecondary }}>预期: {s.expected_improvement}</span>
                </div>
              ))}
            </BaseCard>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 八、候选方案 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {topKPlans.length > 1 && (
          <div className="mb-4">
            <BaseCard
              title="候选方案"
              subtitle={`${topKPlans.length}个`}
              accentColor="#1890ff"
              collapsible
              expanded={expandedSections.candidates}
              onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, candidates: expanded }))}
            >
              <div className="max-h-[200px] overflow-auto">
              {topKPlans.map((p, i) => {
                const isSelected = p.plan.plan_id === result?.plan.plan_id
                return (
                  <div
                    key={p.plan.plan_id}
                    onClick={() => onSelectPlan?.(p)}
                    style={{
                      ...clickableCardStyle(isSelected),
                      padding: 10,
                      marginBottom: 6,
                      position: 'relative',
                    }}
                  >
                    {/* 右上角标记 */}
                    <Info
                      className="absolute top-2 right-2 h-2.5 w-2.5"
                      style={{ color: isSelected ? colors.interactive : '#d9d9d9' }}
                    />
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="text-[11px] font-semibold min-w-[20px]"
                          style={{ color: isSelected ? colors.interactive : colors.textSecondary }}
                        >
                          #{i + 1}
                        </span>
                        <div className="flex gap-0.5">
                          <span className="text-[10px]" style={{ color: colors.textSecondary }}>DP{p.plan.parallelism.dp}</span>
                          <span className="text-[10px]" style={{ color: colors.textSecondary }}>·</span>
                          <span className="text-[10px]" style={{ color: colors.textSecondary }}>TP{p.plan.parallelism.tp}</span>
                          {/* 不显示 PP */}
                          {p.plan.parallelism.ep > 1 && (
                            <>
                              <span className="text-[10px]" style={{ color: colors.textSecondary }}>·</span>
                              <span className="text-[10px]" style={{ color: colors.textSecondary }}>EP{p.plan.parallelism.ep}</span>
                            </>
                          )}
                        </div>
                      </div>
                      <span className="text-sm font-semibold" style={{ color: isSelected ? colors.interactive : colors.text }}>
                        {p.score?.overall_score?.toFixed(1) || '0.0'}
                      </span>
                    </div>
                    <div className="flex justify-between mt-1.5 text-[10px]" style={{ color: colors.textSecondary }}>
                      <span>{p.latency?.prefill_total_latency_ms?.toFixed(1) || '0.0'}ms</span>
                      <span>{p.throughput?.tokens_per_second?.toFixed(0) || '0'} tok/s</span>
                      <span>{((p.throughput?.model_flops_utilization || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )
              })}
              </div>
            </BaseCard>
          </div>
        )}

      </div>
    </TooltipProvider>
  )
}

export default AnalysisResultDisplay
