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
  StopCircle,
  XCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { InfoTooltip, conditionalTooltip } from '@/components/ui/info-tooltip'
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
import { PlanAnalysisResult } from '../../../utils/llmDeployment/types'
import { InfeasibleResult } from '../../../utils/llmDeployment'
import { generateBenchmarkName } from '../../../utils/llmDeployment/benchmarkNaming'
import { AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import { colors } from './ConfigSelectors'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BaseCard } from '@/components/common/BaseCard'
import { formatNumber, formatPercent } from '../../../utils/formatters'

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
              <TableHead className="w-[90px] text-center">TTFT</TableHead>
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
                  {record.chips > 0 ? formatNumber(record.tps / record.chips, 0) : 0} tok/s
                </TableCell>
                <TableCell className="text-center text-sm">
                  {formatNumber(record.ttft, 1)} ms
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
}


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
}) => {

  // 各章节折叠状态
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    performance: false,
    suggestions: false,
    candidates: false,
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
      <Card className="mb-4">
        <CardHeader className="py-2.5 px-3 border-b border-gray-100">
          <CardTitle className="text-sm font-semibold text-gray-700">搜索与评估</CardTitle>
        </CardHeader>
        <CardContent className="p-3">
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
        </CardContent>
      </Card>
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
              搜索统计: 评估 {searchStats.evaluated} 个方案，{searchStats.feasible} 个可行，耗时 {formatNumber(searchStats.timeMs, 0)}ms
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
      <Card className="mb-4">
        <CardHeader className="py-2.5 px-3 border-b border-gray-100">
          <CardTitle className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <span>不可行方案 ({infeasiblePlans.length})</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3">
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
                      <InfoTooltip content={<p>{plan.reason}</p>}>
                        <span className="text-red-500 text-[11px] truncate block max-w-[200px]">{plan.reason}</span>
                      </InfoTooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
        </CardContent>
      </Card>
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

  const { memory, latency, throughput, score, suggestions, is_feasible, infeasibility_reason } = result

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

  // 性能指标卡片样式 - 紧凑版 + 渐变背景（移除交互）
  const metricCardStyle: React.CSSProperties = {
    padding: '8px',
    textAlign: 'center',
    background: 'linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%)',
    borderRadius: '6px',
    border: `1px solid ${colors.border}`,
    transition: 'all 0.2s ease',
  }

  const metricCardClassName = 'hover:shadow-md hover:border-blue-300'

  return (
      <div>
        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 二、性能分析 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        <BaseCard collapsible
          title="性能分析"
          expanded={expandedSections.performance}
          onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, performance: expanded }))}
          gradient
        >
          <>
          {/* 所有指标统一展示 */}
          <div className="grid grid-cols-6 gap-1.5 mb-2.5">
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>TTFT</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                {formatNumber(latency?.prefill_total_latency_ms, 1) || '0.0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>ms</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>TPOT</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                {formatNumber(latency?.decode_per_token_latency_ms, 2) || '0.00'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>ms</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>Total TPS</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                {formatNumber(throughput?.tokens_per_second, 0) || '0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>tok/s</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>TPS/Batch</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: (throughput?.tps_per_batch || 0) >= 10 ? colors.text : colors.error }}>
                {formatNumber(throughput?.tps_per_batch, 1) || '0.0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>tok/s</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>TPS/Chip</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                {formatNumber(throughput?.tps_per_chip, 0) || '0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>tok/s</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>MFU</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                {formatNumber((throughput?.model_flops_utilization || 0) * 100, 1)} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>%</span>
              </div>
            </div>

            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>MBU</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                {formatNumber((throughput?.memory_bandwidth_utilization || 0) * 100, 1)} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>%</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>显存占用</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: memory?.is_memory_sufficient ? colors.text : colors.error }}>
                {formatNumber(memory?.total_per_chip_gb, 1) || '0.0'} <span className="text-xs font-normal" style={{ color: colors.textSecondary }}>/ 80G</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>硬件成本</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                ${result.cost ? formatNumber(result.cost.total_cost / 1000, 0) : '-'}<span className="text-xs font-normal" style={{ color: colors.textSecondary }}>K</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>服务器成本</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                ${result.cost ? formatNumber(result.cost.server_cost / 1000, 0) : '-'}<span className="text-xs font-normal" style={{ color: colors.textSecondary }}>K</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>互联成本</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                ${result.cost ? formatNumber(result.cost.interconnect_cost / 1000, 0) : '-'}<span className="text-xs font-normal" style={{ color: colors.textSecondary }}>K</span>
              </div>
            </div>
            <div style={metricCardStyle} className={metricCardClassName}>
              <span className="text-xs" style={{ color: colors.textSecondary }}>单芯片成本</span>
              <div className="text-xl font-bold mt-0.5" style={{ color: colors.text }}>
                ${result.cost ? formatNumber(result.cost.cost_per_chip / 1000, 1) : '-'}<span className="text-xs font-normal" style={{ color: colors.textSecondary }}>K</span>
              </div>
            </div>
          </div>

          </>
        </BaseCard>

        {/* 优化建议 */}
        {suggestions.length > 0 && (
          <BaseCard collapsible
            title="优化建议"
            expanded={expandedSections.suggestions}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, suggestions: expanded }))}
            gradient
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
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* 八、候选方案 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {topKPlans.length > 1 && (
          <BaseCard collapsible
            title={<>候选方案 <span className="text-xs font-normal text-gray-400 ml-2">({topKPlans.length}个)</span></>}
            expanded={expandedSections.candidates}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, candidates: expanded }))}
            gradient
          >
              <div className="max-h-[400px] overflow-auto">
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
                        {formatNumber(p.score?.overall_score, 1) || '0.0'}
                      </span>
                    </div>
                    <div className="flex justify-between mt-1.5 text-[10px]" style={{ color: colors.textSecondary }}>
                      <span>{formatNumber(p.latency?.prefill_total_latency_ms, 1) || '0.0'}ms</span>
                      <span>{formatNumber(p.throughput?.tokens_per_second, 0) || '0'} tok/s</span>
                      <span>{formatNumber((p.throughput?.model_flops_utilization || 0) * 100, 1)}%</span>
                    </div>
                  </div>
                )
              })}
              </div>
          </BaseCard>
        )}

      </div>
  )
}

export default AnalysisResultDisplay
