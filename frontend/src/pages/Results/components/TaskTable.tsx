/**
 * 任务表格组件 - 使用Handsontable实现Excel-like功能
 * 基于CrossRing项目的ResultTable.tsx改造
 */

import { useState, useEffect, useMemo, useRef } from 'react'
import { Download, Settings, Trash2, Search, GripVertical, ChevronRight, ChevronDown } from 'lucide-react'
import { toast } from 'sonner'
import { HotTable, HotTableClass } from '@handsontable/react'
import { registerAllModules } from 'handsontable/registry'
import 'handsontable/dist/handsontable.full.min.css'
import type { EvaluationTask } from '@/api/results'
import { classifyTaskFieldsWithHierarchy, extractTaskFields } from '../utils/taskFieldClassifier'
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
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
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
      <span className="flex-1 truncate text-sm">{id.replace(/_/g, ' ')}</span>
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
  onTasksDelete?: (taskIds: string[]) => Promise<void>
}

export default function TaskTable({
  tasks,
  loading,
  experimentId,
  onTaskSelect,
  onTasksDelete,
}: Props) {
  // 只显示已完成的任务
  const completedTasks = useMemo(() => {
    return tasks.filter(task => task.status === 'completed')
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
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      })
    }

    if (classified.config.length > 0) {
      nodes.push({
        title: `配置参数 (${classified.config.length})`,
        key: 'category_config',
        children: classified.config.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      })
    }

    if (classified.stats.length > 0) {
      nodes.push({
        title: `搜索统计 (${classified.stats.length})`,
        key: 'category_stats',
        children: classified.stats.map((col) => ({
          title: col.replace(/^search_stats_/, '').replace(/_/g, ' '),
          key: col,
        })),
      })
    }

    if (classified.performance.length > 0) {
      nodes.push({
        title: `性能指标 (${classified.performance.length})`,
        key: 'category_performance',
        children: classified.performance.map((col) => ({
          title: col.replace(/^best_/, '').replace(/_/g, ' '),
          key: col,
        })),
      })
    }

    if (classified.time.length > 0) {
      nodes.push({
        title: `时间信息 (${classified.time.length})`,
        key: 'category_time',
        children: classified.time.map((col) => ({
          title: col.replace(/_/g, ' '),
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
      'throughput',
      'tps_per_chip',
      'tpot',
      'ttft',
      'mfu',
      'score',
      'chips',
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

  // 中文列标题映射
  const columnNameMap: Record<string, string> = {
    'benchmark_name': '基准名称',
    'topology_config_name': '拓扑配置',
    'throughput': '吞吐量 (tokens/s)',
    'tps_per_chip': '单卡吞吐量',
    'tpot': 'TPOT (ms)',
    'ttft': 'TTFT (ms)',
    'mfu': 'MFU (%)',
    'score': '综合得分',
    'chips': '芯片数',
    'parallelism_dp': 'DP',
    'parallelism_tp': 'TP',
    'parallelism_pp': 'PP',
    'parallelism_ep': 'EP',
    'parallelism_sp': 'SP',
    'created_at': '创建时间',
  }

  // 列头
  const colHeaders = useMemo(() => {
    return allColumns.map((col) => {
      // 优先使用中文映射
      if (columnNameMap[col]) {
        return columnNameMap[col]
      }
      // 搜索统计字段
      if (col.startsWith('search_stats_')) {
        const key = col.replace('search_stats_', '')
        return `搜索统计: ${key.replace(/_/g, ' ')}`
      }
      // 默认处理：替换下划线
      return col.replace(/_/g, ' ')
    })
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

  // 从任务对象中提取字段值
  const getTaskFieldValue = (task: EvaluationTask, field: string): string | number => {
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
    const performanceFields = ['throughput', 'tps_per_chip', 'tpot', 'ttft', 'mfu', 'score', 'chips']
    if (performanceFields.includes(field)) {
      const result = (task as any).result
      if (!result) return '-'
      return result[field] ?? '-'
    }

    // 并行策略字段
    if (field.startsWith('parallelism_')) {
      const result = (task as any).result
      if (!result || !result.parallelism) return '-'
      const strategyKey = field.replace('parallelism_', '')
      return result.parallelism[strategyKey] ?? '-'
    }

    return '-'
  }

  // 表格数据
  const tableData = useMemo(() => {
    if (!completedTasks.length) return []
    return completedTasks.map((task) => {
      return allColumns.map((col) => {
        const value = getTaskFieldValue(task, col)
        if (typeof value === 'number') {
          // MFU 百分比字段
          if (col === 'mfu') {
            return `${(value * 100).toFixed(2)}%`
          }
          // 浮点数保留两位
          return Number.isInteger(value) ? value : value.toFixed(2)
        }
        // 时间字段格式化
        if (col.endsWith('_at') && typeof value === 'string' && value !== '-') {
          return new Date(value).toLocaleString('zh-CN')
        }
        return value
      })
    })
  }, [completedTasks, allColumns])

  // 数据变化时强制重新渲染
  useEffect(() => {
    if (hotTableRef.current?.hotInstance) {
      setTimeout(() => {
        hotTableRef.current?.hotInstance?.render()
      }, 0)
    }
  }, [tableData])

  // 双击行选择
  const handleRowDoubleClick = (row: number) => {
    if (row >= 0 && completedTasks[row]) {
      onTaskSelect(completedTasks[row])
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
    if (selectedRowIndices.size === completedTasks.length) {
      setSelectedRowIndices(new Set())
    } else {
      setSelectedRowIndices(new Set(completedTasks.map((_, idx) => idx)))
    }
  }

  // 批量删除任务
  const [batchDeleting, setBatchDeleting] = useState(false)
  const handleBatchDelete = async () => {
    if (selectedRowIndices.size === 0 || !onTasksDelete) return
    setBatchDeleting(true)
    try {
      const taskIds = Array.from(selectedRowIndices).map((idx) => completedTasks[idx].task_id)
      await onTasksDelete(taskIds)
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
    if (!completedTasks.length) {
      toast.warning('没有数据可导出')
      return
    }
    const headers = allColumns
    const rows = completedTasks.map((task) =>
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

  // 列设置标签页状态
  const [columnSettingTab, setColumnSettingTab] = useState<'select' | 'order'>('select')

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
      ) : (
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
      )}
    </div>
  )

  return (
    <TooltipProvider>
      <div>
        <div className="mb-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline">
                  <Settings className="h-4 w-4 mr-2" />
                  列显示设置 ({visibleColumns.length}/{allFieldKeys.length})
                </Button>
              </PopoverTrigger>
              <PopoverContent align="start" className="w-auto p-4">
                {columnSelector}
              </PopoverContent>
            </Popover>
            <span className="text-gray-500 text-xs">
              {selectMode ? '选择模式：单击行选中，再次点击取消' : '双击行查看详情'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
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
              </TooltipTrigger>
              <TooltipContent>进入选择模式后单击行可选中，用于批量删除</TooltipContent>
            </Tooltip>
            {selectMode && onTasksDelete && (
              <>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        checked={selectedRowIndices.size === completedTasks.length && completedTasks.length > 0}
                        onCheckedChange={toggleSelectAll}
                      />
                      <span className="text-sm">全选</span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>选中/取消选中所有行</TooltipContent>
                </Tooltip>
                <AlertDialog>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <AlertDialogTrigger asChild>
                        <Button
                          variant="destructive"
                          disabled={selectedRowIndices.size === 0 || batchDeleting}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          删除 ({selectedRowIndices.size})
                        </Button>
                      </AlertDialogTrigger>
                    </TooltipTrigger>
                    <TooltipContent>删除所有选中的任务</TooltipContent>
                  </Tooltip>
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
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" onClick={handleExport} disabled={!completedTasks.length}>
                  <Download className="h-4 w-4 mr-2" />
                  导出CSV
                </Button>
              </TooltipTrigger>
              <TooltipContent>将任务数据导出为CSV文件</TooltipContent>
            </Tooltip>
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
              licenseKey="non-commercial-and-evaluation"
              stretchH="all"
              selectionMode="multiple"
              outsideClickDeselects={false}
              afterOnCellMouseDown={(event, coords) => {
                if (selectMode && coords.row >= 0) {
                  toggleRowSelection(coords.row)
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
      </div>
    </TooltipProvider>
  )
}
