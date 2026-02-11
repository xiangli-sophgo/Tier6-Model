/**
 * 任务表格组件 - 使用Handsontable实现Excel-like功能
 * 基于CrossRing项目的ResultTable.tsx改造
 */

import { useState, useEffect, useMemo, useRef } from 'react'
import { Download, Settings, Trash2, Search, GripVertical, ChevronRight, ChevronDown, Save, Plus, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { HotTable, HotTableClass } from '@handsontable/react'
import { registerAllModules } from 'handsontable/registry'
import 'handsontable/dist/handsontable.full.min.css'
import type { EvaluationTask } from '@/api/results'
import { getColumnPresetsByExperiment, addColumnPreset, deleteColumnPreset, type ColumnPreset } from '@/api/results'
import { classifyTaskFieldsWithHierarchy, extractTaskFields } from '../utils/taskFieldClassifier'
import { formatNumber } from '@/utils/formatters'
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { InfoTooltip } from '@/components/ui/info-tooltip'
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
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'

// 注册所有Handsontable模块
registerAllModules()

// 存储键
const getStorageKey = (experimentId: number) => `task_table_visible_columns_${experimentId}`
const getFixedColumnsKey = (experimentId: number) => `task_table_fixed_columns_${experimentId}`
const getColumnOrderKey = (experimentId: number) => `task_table_column_order_${experimentId}`
const getRowOrderKey = (experimentId: number) => `task_table_row_order_${experimentId}`

// 列标题映射
const columnNameMap: Record<string, string> = {
  'benchmark_name': 'Benchmark',
  'topology_config_name': '拓扑配置',
  'status': '任务状态',
  'tps': 'TPS',
  'tps_per_chip': 'TPS/Chip',
  'tps_per_batch': 'TPS/Batch',
  'tpot': 'TPOT (ms)',
  'ttft': 'TTFT (ms)',
  'mfu': 'MFU (%)',
  'mbu': 'MBU (%)',
  'score': '综合得分',
  'chips': '芯片数',
  'dram_occupy': '显存占用 (GB)',
  'flops': '计算量 (TFLOPs)',
  'end_to_end_latency': 'E2E延迟 (ms)',
  'parallelism_dp': 'DP',
  'parallelism_tp': 'TP',
  'parallelism_pp': 'PP',
  'parallelism_ep': 'EP',
  'parallelism_sp': 'SP',
  'parallelism_moe_tp': 'MoE_TP',
  'created_at': '创建时间',
  'result_rank': '排名',
  'result_id': '结果ID',
  // 成本字段
  'cost_total': '总成本 ($)',
  'cost_server': '服务器成本 ($)',
  'cost_interconnect': '互联成本 ($)',
  'cost_per_chip': '单芯成本 ($)',
  'cost_dfop': 'DFOP ($/TPS)',
}

// 统一的列名格式化函数
const getColumnDisplayName = (col: string): string => {
  // 优先使用中文映射
  if (columnNameMap[col]) {
    return columnNameMap[col]
  }
  // 搜索统计字段
  if (col.startsWith('search_stats_')) {
    const key = col.replace('search_stats_', '')
    return `搜索统计: ${key.replace(/_/g, ' ')}`
  }
  // 性能指标字段
  if (col.startsWith('best_')) {
    return col.replace(/^best_/, '').replace(/_/g, ' ')
  }
  // 默认处理：替换下划线
  return col.replace(/_/g, ' ')
}

// 树节点类型
interface TreeNode {
  title: string
  key: string
  children?: TreeNode[]
}

// 可拖拽的列项组件
interface SortableColumnItemProps {
  id: string
  isFixed: boolean
  onToggleFixed: (col: string) => void
}

function SortableColumnItem({ id, isFixed, onToggleFixed }: SortableColumnItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id })

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      className={`flex items-center p-2 mb-1 rounded border cursor-grab ${
        isDragging ? 'bg-blue-50 border-blue-200' : 'bg-gray-50 border-gray-200'
      } ${isDragging ? 'opacity-80' : ''}`}
    >
      <GripVertical {...listeners} className="h-4 w-4 mr-2 text-gray-400 cursor-grab" />
      <Checkbox
        checked={isFixed}
        onCheckedChange={() => onToggleFixed(id)}
        className="mr-2"
      />
      <span className="flex-1 truncate text-sm">{getColumnDisplayName(id)}</span>
    </div>
  )
}

// 自定义树组件
interface CustomTreeProps {
  data: TreeNode[]
  checkedKeys: string[]
  expandedKeys: string[]
  onCheck: (keys: string[]) => void
  onExpand: (keys: string[]) => void
}

function CustomTree({ data, checkedKeys, expandedKeys, onCheck, onExpand }: CustomTreeProps) {
  const toggleExpand = (key: string) => {
    if (expandedKeys.includes(key)) {
      onExpand(expandedKeys.filter((k) => k !== key))
    } else {
      onExpand([...expandedKeys, key])
    }
  }

  const toggleCheck = (key: string, isCategory: boolean, childKeys?: string[]) => {
    if (isCategory && childKeys) {
      const allChecked = childKeys.every((k) => checkedKeys.includes(k))
      if (allChecked) {
        onCheck(checkedKeys.filter((k) => !childKeys.includes(k)))
      } else {
        onCheck([...new Set([...checkedKeys, ...childKeys])])
      }
    } else {
      if (checkedKeys.includes(key)) {
        onCheck(checkedKeys.filter((k) => k !== key))
      } else {
        onCheck([...checkedKeys, key])
      }
    }
  }

  const renderNode = (node: TreeNode, level: number = 0) => {
    const isCategory = node.key.startsWith('category_')
    const isExpanded = expandedKeys.includes(node.key)
    const childKeys = node.children?.map((c) => c.key) || []
    const isChecked = isCategory
      ? childKeys.length > 0 && childKeys.every((k) => checkedKeys.includes(k))
      : checkedKeys.includes(node.key)
    const isIndeterminate = isCategory && childKeys.some((k) => checkedKeys.includes(k)) && !isChecked

    return (
      <div key={node.key}>
        <div
          className={`flex items-center py-1 px-1 hover:bg-gray-100 rounded cursor-pointer`}
          style={{ paddingLeft: level * 16 }}
        >
          {node.children && node.children.length > 0 ? (
            <button
              className="p-0.5 hover:bg-gray-200 rounded mr-1"
              onClick={() => toggleExpand(node.key)}
            >
              {isExpanded ? (
                <ChevronDown className="h-3.5 w-3.5 text-gray-500" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-gray-500" />
              )}
            </button>
          ) : (
            <span className="w-5" />
          )}
          <Checkbox
            checked={isChecked}
            data-state={isIndeterminate ? 'indeterminate' : isChecked ? 'checked' : 'unchecked'}
            className={isIndeterminate ? 'data-[state=indeterminate]:bg-primary/50' : ''}
            onCheckedChange={() => toggleCheck(node.key, isCategory, childKeys)}
          />
          <span className="ml-2 text-sm">{node.title}</span>
        </div>
        {isExpanded && node.children && (
          <div>{node.children.map((child) => renderNode(child, level + 1))}</div>
        )}
      </div>
    )
  }

  return <div className="space-y-0.5">{data.map((node) => renderNode(node))}</div>
}

interface Props {
  tasks: EvaluationTask[]
  loading: boolean
  experimentId: number
  onTaskSelect: (task: EvaluationTask) => void
  onTaskDoubleClick?: (task: EvaluationTask) => void
  onResultsDelete?: (resultIds: number[]) => Promise<void>
}

export default function TaskTable({
  tasks,
  loading,
  experimentId,
  onTaskSelect,
  onTaskDoubleClick,
  onResultsDelete,
}: Props) {
  // 显示所有有结果的任务（包括取消前已保存结果的任务）
  const completedTasks = useMemo(() => {
    return tasks.filter(task => task.result !== undefined && task.result !== null)
  }, [tasks])

  // 提取所有可用字段
  const allFieldKeys = useMemo(() => extractTaskFields(completedTasks), [completedTasks])

  // 可见列状态
  const [visibleColumns, setVisibleColumns] = useState<string[]>([])

  // 列顺序
  const [columnOrder, setColumnOrder] = useState<string[]>([])

  // 固定列
  const [fixedColumns, setFixedColumns] = useState<string[]>([])

  // 批量选中的行索引集合
  const [selectedRowIndices, setSelectedRowIndices] = useState<Set<number>>(new Set())

  // 选择模式
  const [selectMode, setSelectMode] = useState(false)

  // 行顺序（存储 task_id）
  const [rowOrder, setRowOrder] = useState<string[]>([])

  // 配置方案相关状态
  const [presets, setPresets] = useState<ColumnPreset[]>([])
  const [presetsLoading, setPresetsLoading] = useState(false)
  const [savePresetModalVisible, setSavePresetModalVisible] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')
  const [savingPreset, setSavingPreset] = useState(false)

  // Handsontable ref
  const hotTableRef = useRef<HotTableClass>(null)

  // 分类数据
  const classifiedFields = useMemo(() => classifyTaskFieldsWithHierarchy(allFieldKeys), [allFieldKeys])

  // 生成树形数据用于列选择器
  const treeData = useMemo((): TreeNode[] => {
    const nodes: TreeNode[] = []
    const classified = classifiedFields

    if (classified.important.length > 0) {
      nodes.push({
        title: `重要信息 (${classified.important.length})`,
        key: 'category_important',
        children: classified.important.map((col) => ({
          title: getColumnDisplayName(col),
          key: col,
        })),
      })
    }

    if (classified.config.length > 0) {
      nodes.push({
        title: `配置参数 (${classified.config.length})`,
        key: 'category_config',
        children: classified.config.map((col) => ({
          title: getColumnDisplayName(col),
          key: col,
        })),
      })
    }

    if (classified.stats.length > 0) {
      nodes.push({
        title: `搜索统计 (${classified.stats.length})`,
        key: 'category_stats',
        children: classified.stats.map((col) => ({
          title: getColumnDisplayName(col),
          key: col,
        })),
      })
    }

    if (classified.performance.length > 0) {
      nodes.push({
        title: `性能指标 (${classified.performance.length})`,
        key: 'category_performance',
        children: classified.performance.map((col) => ({
          title: getColumnDisplayName(col),
          key: col,
        })),
      })
    }

    if (classified.time.length > 0) {
      nodes.push({
        title: `时间信息 (${classified.time.length})`,
        key: 'category_time',
        children: classified.time.map((col) => ({
          title: getColumnDisplayName(col),
          key: col,
        })),
      })
    }

    return nodes
  }, [classifiedFields])

  // 初始化可见列
  useEffect(() => {
    if (allFieldKeys.length === 0) return
    const defaultColumns = [
      'benchmark_name',
      'topology_config_name',
      'status',
      'tps',
      'tps_per_chip',
      'tps_per_batch',
      'tpot',
      'ttft',
      'mfu',
      'mbu',
      'score',
      'chips',
      'dram_occupy',
      'created_at'
    ].filter(col => allFieldKeys.includes(col))

    const saved = localStorage.getItem(getStorageKey(experimentId))
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed) && parsed.length > 0) {
          const validColumns = parsed.filter((col: string) => allFieldKeys.includes(col))
          if (validColumns.length > 0) {
            setVisibleColumns(validColumns)
            return
          }
        }
      } catch {
        // ignore
      }
    }
    setVisibleColumns(defaultColumns)
  }, [allFieldKeys, experimentId])

  // 初始化列顺序和固定列
  useEffect(() => {
    const savedOrder = localStorage.getItem(getColumnOrderKey(experimentId))
    if (savedOrder) {
      try {
        const parsed = JSON.parse(savedOrder)
        if (Array.isArray(parsed)) {
          setColumnOrder(parsed)
        }
      } catch {
        setColumnOrder([])
      }
    } else {
      setColumnOrder([])
    }

    const savedFixed = localStorage.getItem(getFixedColumnsKey(experimentId))
    if (savedFixed) {
      try {
        const parsed = JSON.parse(savedFixed)
        if (Array.isArray(parsed)) {
          setFixedColumns(parsed)
        }
      } catch {
        setFixedColumns([])
      }
    } else {
      setFixedColumns([])
    }
  }, [experimentId])

  // 保存可见列
  useEffect(() => {
    if (visibleColumns.length > 0) {
      localStorage.setItem(getStorageKey(experimentId), JSON.stringify(visibleColumns))
    }
  }, [visibleColumns, experimentId])

  // 保存固定列
  useEffect(() => {
    localStorage.setItem(getFixedColumnsKey(experimentId), JSON.stringify(fixedColumns))
  }, [fixedColumns, experimentId])

  // 保存列顺序
  useEffect(() => {
    if (columnOrder.length > 0) {
      localStorage.setItem(getColumnOrderKey(experimentId), JSON.stringify(columnOrder))
    }
  }, [columnOrder, experimentId])

  // 生成唯一行键（task_id + result_id，因为同一个 task 可能有多个 result）
  const getRowKey = (task: EvaluationTask): string => {
    const resultId = (task as any).result_id
    if (resultId !== undefined && resultId !== null) {
      return `${task.task_id}_${resultId}`
    }
    return task.task_id
  }

  // 初始化行顺序
  useEffect(() => {
    if (completedTasks.length === 0) {
      setRowOrder([])
      return
    }
    const saved = localStorage.getItem(getRowOrderKey(experimentId))
    if (saved) {
      try {
        const savedOrder: string[] = JSON.parse(saved)
        const currentIds = new Set(completedTasks.map((t) => getRowKey(t)))
        const validOrder = savedOrder.filter((id) => currentIds.has(id))
        const newIds = completedTasks.filter((t) => !savedOrder.includes(getRowKey(t))).map((t) => getRowKey(t))
        setRowOrder([...validOrder, ...newIds])
        return
      } catch {
        // ignore
      }
    }
    setRowOrder(completedTasks.map((t) => getRowKey(t)))
  }, [completedTasks, experimentId])

  // 保存行顺序
  useEffect(() => {
    if (rowOrder.length > 0) {
      localStorage.setItem(getRowOrderKey(experimentId), JSON.stringify(rowOrder))
    }
  }, [rowOrder, experimentId])

  // 从服务器加载配置方案
  const loadPresetsFromServer = async () => {
    setPresetsLoading(true)
    try {
      const response = await getColumnPresetsByExperiment(experimentId)
      setPresets(response.presets || [])
    } catch (error) {
      console.error('加载配置方案失败:', error)
      toast.error('加载配置方案失败')
    } finally {
      setPresetsLoading(false)
    }
  }

  // 初始化时加载配置方案
  useEffect(() => {
    loadPresetsFromServer()
  }, [experimentId])

  // 保存当前配置为新方案
  const handleSavePreset = async () => {
    if (!newPresetName.trim()) {
      toast.warning('请输入配置名称')
      return
    }

    const preset: ColumnPreset = {
      name: newPresetName.trim(),
      experiment_id: experimentId,
      visible_columns: [...visibleColumns],
      column_order: [...columnOrder],
      fixed_columns: [...fixedColumns],
      created_at: new Date().toISOString(),
    }

    setSavingPreset(true)
    try {
      const response = await addColumnPreset(preset)
      toast.success(response.message)
      await loadPresetsFromServer()
      setSavePresetModalVisible(false)
      setNewPresetName('')
    } catch (error) {
      console.error('保存配置方案失败:', error)
      toast.error('保存配置方案失败')
    } finally {
      setSavingPreset(false)
    }
  }

  // 加载配置方案
  const handleLoadPreset = (preset: ColumnPreset) => {
    // 验证列是否存在
    const validVisible = preset.visible_columns.filter(col => allFieldKeys.includes(col))
    const validOrder = preset.column_order.filter(col => allFieldKeys.includes(col))
    const validFixed = preset.fixed_columns.filter(col => allFieldKeys.includes(col))

    if (validVisible.length === 0) {
      toast.warning('该配置方案中的列在当前数据中不存在')
      return
    }

    // 销毁 Handsontable 实例避免渲染错误
    if (hotTableRef.current?.hotInstance) {
      hotTableRef.current.hotInstance.deselectCell()
    }

    // 批量更新状态
    setVisibleColumns(validVisible)
    setColumnOrder(validOrder)
    setFixedColumns(validFixed)

    // 延迟刷新表格
    setTimeout(() => {
      hotTableRef.current?.hotInstance?.render()
    }, 0)

    toast.success(`已加载配置方案「${preset.name}」`)
  }

  // 删除配置方案
  const handleDeletePreset = async (presetName: string) => {
    try {
      const response = await deleteColumnPreset(experimentId, presetName)
      toast.success(response.message)
      await loadPresetsFromServer()
    } catch (error) {
      console.error('删除配置方案失败:', error)
      toast.error('删除配置方案失败')
    }
  }

  // 切换固定列
  const toggleFixedColumn = (col: string) => {
    setFixedColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    )
  }

  // 拖拽传感器
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

  // 根据用户自定义顺序排列可见列
  const orderedVisibleColumns = useMemo(() => {
    if (columnOrder.length === 0) {
      return [...visibleColumns].sort((a, b) => a.localeCompare(b, 'zh-CN'))
    }

    const manualCols = columnOrder.filter((col) => visibleColumns.includes(col))
    const newCols = visibleColumns.filter((col) => !columnOrder.includes(col))

    return [...manualCols, ...newCols]
  }, [visibleColumns, columnOrder])

  // 生成列配置（固定列在前）
  const allColumns = useMemo(() => {
    const fixedCols = orderedVisibleColumns.filter((col) => fixedColumns.includes(col))
    const nonFixedCols = orderedVisibleColumns.filter((col) => !fixedColumns.includes(col))
    return [...fixedCols, ...nonFixedCols]
  }, [orderedVisibleColumns, fixedColumns])

  // 固定列数量
  const fixedColumnCount = useMemo(() => {
    return allColumns.filter((col) => fixedColumns.includes(col)).length
  }, [allColumns, fixedColumns])

  // 列头
  const colHeaders = useMemo(() => {
    return allColumns.map((col) => getColumnDisplayName(col))
  }, [allColumns])

  // 拖拽结束处理
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      const oldIndex = orderedVisibleColumns.indexOf(active.id as string)
      const newIndex = orderedVisibleColumns.indexOf(over.id as string)
      const newOrder = arrayMove(orderedVisibleColumns, oldIndex, newIndex)
      setColumnOrder(newOrder)
    }
  }

  // Tree 展开状态
  const [expandedKeys, setExpandedKeys] = useState<string[]>([])
  const allCategoryKeys = useMemo(() => treeData.map((node) => node.key), [treeData])
  const expandAll = () => setExpandedKeys(allCategoryKeys)
  const collapseAll = () => setExpandedKeys([])

  // 搜索状态
  const [searchValue, setSearchValue] = useState<string>('')

  // 过滤后的树数据
  const filteredTreeData = useMemo((): TreeNode[] => {
    if (!searchValue.trim()) {
      return treeData
    }
    const searchLower = searchValue.toLowerCase()

    const filterNode = (node: TreeNode): TreeNode | null => {
      if (!node.children || node.children.length === 0) {
        const matches =
          String(node.title).toLowerCase().includes(searchLower) ||
          String(node.key).toLowerCase().includes(searchLower)
        return matches ? node : null
      }

      const filteredChildren: TreeNode[] = []
      for (const child of node.children) {
        const filtered = filterNode(child)
        if (filtered) {
          filteredChildren.push(filtered)
        }
      }

      if (filteredChildren.length > 0) {
        return {
          ...node,
          title: `${String(node.title).split(' (')[0]} (${filteredChildren.length})`,
          children: filteredChildren,
        }
      }

      return null
    }

    const filtered: TreeNode[] = []
    for (const category of treeData) {
      const result = filterNode(category)
      if (result) {
        filtered.push(result)
      }
    }
    return filtered
  }, [treeData, searchValue])

  // 处理列选择
  const handleColumnCheck = (checkedKeys: string[]) => {
    const newCheckedColumns = checkedKeys.filter((key) => !key.startsWith('category_'))
    setVisibleColumns(newCheckedColumns)
  }

  const checkedKeys = useMemo(() => visibleColumns, [visibleColumns])

  // 根据行顺序排列的任务
  const sortedTasks = useMemo(() => {
    if (completedTasks.length === 0) return []
    if (rowOrder.length === 0) return completedTasks
    const taskMap = new Map(completedTasks.map((t) => [getRowKey(t), t]))
    const sorted = rowOrder.filter((id) => taskMap.has(id)).map((id) => taskMap.get(id)!)
    const orderedIds = new Set(rowOrder)
    const remaining = completedTasks.filter((t) => !orderedIds.has(getRowKey(t)))
    return [...sorted, ...remaining]
  }, [completedTasks, rowOrder])

  // 从任务对象中提取字段值
  const getTaskFieldValue = (task: EvaluationTask, field: string): string | number => {
    // 新增的结果字段（result_id, result_rank）
    if (field === 'result_id' || field === 'result_rank') {
      const value = (task as any)[field]
      if (value === undefined || value === null) return '-'
      return value
    }

    // 状态字段特殊处理（中文显示）
    if (field === 'status') {
      const status = task.status
      const statusMap: Record<string, string> = {
        'pending': '等待中',
        'running': '运行中',
        'completed': '已完成',
        'failed': '失败',
        'cancelled': '已取消',
      }
      return statusMap[status] || status
    }

    // 基础字段（benchmark_name, topology_config_name, created_at 等）
    if (field in task) {
      const value = (task as any)[field]
      if (value === undefined || value === null) return '-'
      return value
    }

    // 搜索统计字段
    if (field.startsWith('search_stats_')) {
      const statKey = field.replace('search_stats_', '')
      return (task.search_stats as any)?.[statKey] ?? '-'
    }

    // 性能指标字段（从 result 中提取）
    const performanceFields = ['tps', 'tps_per_chip', 'tps_per_batch', 'tpot', 'ttft', 'mfu', 'mbu', 'score', 'chips']
    if (performanceFields.includes(field)) {
      const result = (task as any).result
      if (!result) return '-'
      return result[field] ?? '-'
    }

    // 显存占用字段（从 result 转换为 GB）
    if (field === 'dram_occupy') {
      const result = (task as any).result
      if (!result || !result.dram_occupy) return '-'
      return result.dram_occupy / (1024 * 1024 * 1024) // 字节转 GB
    }

    // 计算量字段（从 result 转换为 TFLOPs）
    if (field === 'flops') {
      const result = (task as any).result
      if (!result || !result.flops) return '-'
      return result.flops / 1e12 // FLOPs 转 TFLOPs
    }

    // E2E延迟（通过 ttft + tpot * output_length 计算，这里简化为 ttft + tpot * 100）
    if (field === 'end_to_end_latency') {
      const result = (task as any).result
      if (!result || !result.ttft || !result.tpot) return '-'
      // 假设输出长度为100个token（可根据实际配置调整）
      return result.ttft + result.tpot * 100
    }

    // 并行策略字段
    if (field.startsWith('parallelism_')) {
      const result = (task as any).result
      if (!result || !result.parallelism) return '-'
      const strategyKey = field.replace('parallelism_', '')
      return result.parallelism[strategyKey] ?? '-'
    }

    // 成本字段（从 result.cost 中提取）
    if (field.startsWith('cost_')) {
      const result = (task as any).result
      if (!result || !result.cost) return '-'

      const costKey = field.replace('cost_', '')
      const costMap: Record<string, keyof typeof result.cost> = {
        'total': 'total_cost',
        'server': 'server_cost',
        'interconnect': 'interconnect_cost',
        'per_chip': 'cost_per_chip',
        'dfop': 'dfop',
      }

      const mappedKey = costMap[costKey]
      if (mappedKey && result.cost[mappedKey] !== undefined) {
        return result.cost[mappedKey]
      }
      return '-'
    }

    return '-'
  }

  // 表格数据
  const tableData = useMemo(() => {
    if (!sortedTasks.length) return []
    return sortedTasks.map((task) => {
      return allColumns.map((col) => {
        const value = getTaskFieldValue(task, col)
        if (typeof value === 'number') {
          // MFU/MBU 百分比字段
          if (col === 'mfu' || col === 'mbu') {
            return `${formatNumber(value * 100, 2)}%`
          }
          // 浮点数保留两位
          return Number.isInteger(value) ? value : formatNumber(value, 2)
        }
        // 时间字段格式化
        if (col.endsWith('_at') && typeof value === 'string' && value !== '-') {
          return new Date(value).toLocaleString('zh-CN')
        }
        return value
      })
    })
  }, [sortedTasks, allColumns])

  // 数据变化时强制重新渲染
  useEffect(() => {
    if (hotTableRef.current?.hotInstance) {
      setTimeout(() => {
        hotTableRef.current?.hotInstance?.render()
      }, 0)
    }
  }, [tableData])

  // 监听滚动事件，滚动时移除表格焦点防止自动回滚
  useEffect(() => {
    const handleScroll = () => {
      // 当页面滚动时，移除 Handsontable 的选中状态和焦点
      if (hotTableRef.current?.hotInstance) {
        hotTableRef.current.hotInstance.deselectCell()
      }
      // 移除任何获得焦点的表格单元格
      if (document.activeElement instanceof HTMLElement) {
        const activeElement = document.activeElement
        // 只在焦点在表格内时才移除
        if (activeElement.closest('.task-hot-table')) {
          activeElement.blur()
        }
      }
    }

    // 监听父容器或 window 的滚动
    window.addEventListener('scroll', handleScroll, true)

    return () => {
      window.removeEventListener('scroll', handleScroll, true)
    }
  }, [])

  // 双击行显示详情
  const handleRowDoubleClick = (row: number) => {
    if (row >= 0 && sortedTasks[row]) {
      onTaskDoubleClick?.(sortedTasks[row])
    }
  }

  // 切换行选中状态
  const toggleRowSelection = (rowIndex: number) => {
    setSelectedRowIndices((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(rowIndex)) {
        newSet.delete(rowIndex)
      } else {
        newSet.add(rowIndex)
      }
      return newSet
    })
  }

  // 全选/取消全选
  const toggleSelectAll = () => {
    if (selectedRowIndices.size === sortedTasks.length) {
      setSelectedRowIndices(new Set())
    } else {
      setSelectedRowIndices(new Set(sortedTasks.map((_, idx) => idx)))
    }
  }

  // 批量删除结果
  const [batchDeleting, setBatchDeleting] = useState(false)
  const handleBatchDelete = async () => {
    if (selectedRowIndices.size === 0 || !onResultsDelete) return
    setBatchDeleting(true)
    try {
      // 获取选中行的 result_id 列表（过滤掉没有 result_id 的行）
      const resultIds = Array.from(selectedRowIndices)
        .map((idx) => sortedTasks[idx].result_id)
        .filter((id): id is number => id !== undefined && id !== null)
      if (resultIds.length === 0) {
        toast.warning('选中的行没有有效的结果ID')
        return
      }
      await onResultsDelete(resultIds)
      toast.success('批量删除成功')
      setSelectedRowIndices(new Set())
    } catch {
      toast.error('批量删除失败')
    } finally {
      setBatchDeleting(false)
    }
  }

  // 导出 CSV
  const handleExport = () => {
    if (!sortedTasks.length) {
      toast.warning('没有数据可导出')
      return
    }
    const headers = allColumns
    const rows = sortedTasks.map((task) =>
      allColumns.map((col) => getTaskFieldValue(task, col)).join(',')
    )
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `tasks_experiment_${experimentId}.csv`
    link.click()
    toast.success('导出成功')
  }

  // 行拖拽处理
  const handleBeforeRowMove = (movedRows: number[], finalIndex: number) => {
    const currentIds = sortedTasks.map((t) => getRowKey(t))
    const movedIds = movedRows.map((idx) => currentIds[idx])
    const remaining = currentIds.filter((_, idx) => !movedRows.includes(idx))
    remaining.splice(finalIndex, 0, ...movedIds)
    setRowOrder(remaining)
    return false // 阻止 Handsontable 内部移动，由 React 状态控制
  }

  // 列设置标签页状态
  const [columnSettingTab, setColumnSettingTab] = useState<'select' | 'order' | 'preset'>('select')

  // 列选择器
  const columnSelector = (
    <div className="max-h-[500px] overflow-auto min-w-[360px]">
      <div className="mb-3 flex gap-2">
        <Button
          variant={columnSettingTab === 'select' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setColumnSettingTab('select')}
        >
          选择列
        </Button>
        <Button
          variant={columnSettingTab === 'order' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setColumnSettingTab('order')}
        >
          排序列 ({orderedVisibleColumns.length})
        </Button>
        <Button
          variant={columnSettingTab === 'preset' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setColumnSettingTab('preset')}
        >
          配置方案 ({presets.length})
        </Button>
      </div>

      {columnSettingTab === 'select' ? (
        <>
          <div className="mb-2 relative">
            <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="搜索列名..."
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              className="pl-8 h-8"
            />
          </div>
          <div className="mb-2 flex justify-between items-center">
            <strong className="text-sm">显示列</strong>
            <div className="flex gap-2">
              <Button variant="ghost" size="sm" onClick={expandAll}>全部展开</Button>
              <Button variant="ghost" size="sm" onClick={collapseAll}>全部折叠</Button>
            </div>
          </div>
          <CustomTree
            data={filteredTreeData}
            checkedKeys={checkedKeys}
            expandedKeys={searchValue ? filteredTreeData.map((n) => n.key) : expandedKeys}
            onExpand={setExpandedKeys}
            onCheck={handleColumnCheck}
          />
        </>
      ) : columnSettingTab === 'order' ? (
        <>
          <div className="mb-2 text-xs text-gray-500">
            拖拽调整列顺序，勾选复选框固定列到左侧
          </div>
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={orderedVisibleColumns}
              strategy={verticalListSortingStrategy}
            >
              <div className="max-h-[400px] overflow-auto">
                {orderedVisibleColumns.map((col) => (
                  <SortableColumnItem
                    key={col}
                    id={col}
                    isFixed={fixedColumns.includes(col)}
                    onToggleFixed={toggleFixedColumn}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>
        </>
      ) : (
        <>
          {/* 配置方案标签页 */}
          <div className="mb-3">
            <Button
              className="w-full bg-blue-600 hover:bg-blue-700"
              size="sm"
              onClick={() => setSavePresetModalVisible(true)}
            >
              <Save className="h-4 w-4 mr-2" />
              保存当前配置为方案
            </Button>
          </div>

          {presetsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
            </div>
          ) : presets.length === 0 ? (
            <div className="text-center py-8 text-gray-400 text-sm">
              暂无保存的配置方案
            </div>
          ) : (
            <div className="space-y-2 max-h-[400px] overflow-auto">
              {presets.map((preset) => (
                <div
                  key={preset.name}
                  className="p-3 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-sm mb-1">{preset.name}</div>
                      <div className="text-xs text-gray-500">
                        {preset.visible_columns.length} 列 · {preset.fixed_columns.length} 固定
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        {new Date(preset.created_at).toLocaleString('zh-CN')}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 ml-2">
                      <InfoTooltip content="加载配置">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleLoadPreset(preset)}
                          className="h-7 w-7 p-0"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </Button>
                      </InfoTooltip>
                      <AlertDialog>
                        <InfoTooltip content="删除配置">
                          <AlertDialogTrigger asChild>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-red-500 hover:text-red-600 hover:bg-red-50"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </AlertDialogTrigger>
                        </InfoTooltip>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>删除配置方案</AlertDialogTitle>
                            <AlertDialogDescription>
                              确定要删除配置方案「{preset.name}」吗？此操作不可恢复。
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction
                              className="bg-red-600 hover:bg-red-700"
                              onClick={() => handleDeletePreset(preset.name)}
                            >
                              删除
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )

  return (
    <div>
        <div className="mb-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Popover modal={true}>
              <PopoverTrigger asChild>
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  列显示设置 ({visibleColumns.length}/{allFieldKeys.length})
                </Button>
              </PopoverTrigger>
              <PopoverContent
                align="start"
                className="w-auto p-4 z-[9999]"
                onInteractOutside={(e) => {
                  // 防止点击内部元素时关闭
                  const target = e.target as HTMLElement
                  if (target.closest('.popover-content-inner')) {
                    e.preventDefault()
                  }
                }}
              >
                <div className="popover-content-inner">
                  {columnSelector}
                </div>
              </PopoverContent>
            </Popover>
            <span className="text-gray-500 text-xs">
              {selectMode ? '选择模式：单击行选中，再次点击取消' : '双击行查看详情 | 双击列头排序 | 拖拽行调整顺序'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <InfoTooltip content="进入选择模式后单击行可选中，用于批量删除">
              <Button
                variant={selectMode ? 'destructive' : 'outline'}
                onClick={() => {
                  setSelectMode(!selectMode)
                  if (selectMode) {
                    setSelectedRowIndices(new Set())
                  }
                }}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                {selectMode ? '退出选择' : '选择模式'}
              </Button>
            </InfoTooltip>
            {selectMode && onResultsDelete && (
              <>
                <InfoTooltip content="选中/取消选中所有行">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      checked={selectedRowIndices.size === sortedTasks.length && sortedTasks.length > 0}
                      onCheckedChange={toggleSelectAll}
                    />
                    <span className="text-sm">全选</span>
                  </div>
                </InfoTooltip>
                <AlertDialog>
                  <InfoTooltip content="删除所有选中的任务">
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="destructive"
                        disabled={selectedRowIndices.size === 0 || batchDeleting}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        删除 ({selectedRowIndices.size})
                      </Button>
                    </AlertDialogTrigger>
                  </InfoTooltip>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>批量删除</AlertDialogTitle>
                      <AlertDialogDescription>
                        确定要删除选中的 {selectedRowIndices.size} 个任务吗？此操作不可恢复。
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>取消</AlertDialogCancel>
                      <AlertDialogAction
                        className="bg-red-600 hover:bg-red-700"
                        onClick={handleBatchDelete}
                      >
                        删除
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </>
            )}
            <InfoTooltip content="将任务数据导出为CSV文件">
              <Button variant="outline" onClick={handleExport} disabled={!sortedTasks.length}>
                <Download className="h-4 w-4 mr-2" />
                导出CSV
              </Button>
            </InfoTooltip>
          </div>
        </div>

        <div className="task-table-container">
          {loading ? (
            <div className="text-center py-12 bg-gray-50">加载中...</div>
          ) : tableData.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 text-gray-500">暂无任务数据</div>
          ) : (
            <HotTable
              ref={hotTableRef}
              data={tableData}
              colHeaders={colHeaders}
              rowHeaders={true}
              width="100%"
              height="auto"
              fixedColumnsStart={fixedColumnCount}
              manualColumnResize={true}
              manualRowMove={true}
              columnSorting={{ headerAction: false, indicator: true }}
              licenseKey="non-commercial-and-evaluation"
              stretchH="all"
              selectionMode="multiple"
              outsideClickDeselects={true}
              beforeRowMove={handleBeforeRowMove}
              afterOnCellMouseDown={(event, coords) => {
                if (selectMode && coords.row >= 0) {
                  toggleRowSelection(coords.row)
                  return
                }
                // 双击列头触发排序
                if (event.detail === 2 && coords.row === -1 && coords.col >= 0) {
                  const hot = hotTableRef.current?.hotInstance
                  if (hot) {
                    const columnSortingPlugin = hot.getPlugin('columnSorting')
                    const currentSort = columnSortingPlugin.getSortConfig()
                    const sortArray = Array.isArray(currentSort) ? currentSort : (currentSort ? [currentSort] : [])
                    const currentColSort = sortArray.find((s: { column: number }) => s.column === coords.col)

                    let newOrder: 'asc' | 'desc' | undefined
                    if (!currentColSort) {
                      newOrder = 'asc'
                    } else if (currentColSort.sortOrder === 'asc') {
                      newOrder = 'desc'
                    } else {
                      newOrder = undefined // 取消排序
                    }

                    if (newOrder) {
                      columnSortingPlugin.sort({ column: coords.col, sortOrder: newOrder })
                    } else {
                      columnSortingPlugin.clearSort()
                    }
                  }
                  return
                }
                if (event.detail === 2 && coords.row >= 0) {
                  handleRowDoubleClick(coords.row)
                }
              }}
              readOnly={true}
              cells={(row) => ({
                className: `htCenter htMiddle${selectedRowIndices.has(row) ? ' selected-row' : ''}`,
              })}
              className="task-hot-table"
            />
          )}
        </div>

        <style>{`
          .selected-row {
            background-color: #e6f7ff !important;
          }
          .task-hot-table .htCore td {
            text-align: center;
            vertical-align: middle;
          }
        `}</style>

        {/* 保存配置方案对话框 */}
        <Dialog open={savePresetModalVisible} onOpenChange={setSavePresetModalVisible}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>保存配置方案</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <label className="text-sm font-medium mb-2 block">配置名称</label>
                <Input
                  placeholder="输入配置方案名称"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !savingPreset) {
                      handleSavePreset()
                    }
                  }}
                  autoFocus
                />
              </div>
              <div className="text-sm text-gray-500 space-y-1">
                <div>当前配置:</div>
                <div>• 显示列: {visibleColumns.length} 列</div>
                <div>• 固定列: {fixedColumns.length} 列</div>
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => {
                  setSavePresetModalVisible(false)
                  setNewPresetName('')
                }}
                disabled={savingPreset}
              >
                取消
              </Button>
              <Button
                onClick={handleSavePreset}
                disabled={savingPreset || !newPresetName.trim()}
              >
                {savingPreset && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                保存
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
    </div>
  )
}
