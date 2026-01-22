/**
 * 任务表格组件 - 使用Handsontable实现Excel-like功能
 * 基于CrossRing项目的ResultTable.tsx改造
 */

import { useState, useEffect, useMemo, useRef } from 'react'
import { Button, message, Space, Tree, Popover, Checkbox, Popconfirm, Input, Tooltip } from 'antd'
import { DownloadOutlined, SettingOutlined, DeleteOutlined, SearchOutlined, HolderOutlined } from '@ant-design/icons'
import type { DataNode } from 'antd/es/tree'
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

// 注册所有Handsontable模块
registerAllModules()

// 存储键
const getStorageKey = (experimentId: number) => `task_table_visible_columns_${experimentId}`
const getFixedColumnsKey = (experimentId: number) => `task_table_fixed_columns_${experimentId}`
const getColumnOrderKey = (experimentId: number) => `task_table_column_order_${experimentId}`

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
    display: 'flex',
    alignItems: 'center',
    padding: '4px 8px',
    marginBottom: 4,
    background: isDragging ? '#e6f7ff' : '#fafafa',
    border: '1px solid #d9d9d9',
    borderRadius: 4,
    cursor: 'grab',
    opacity: isDragging ? 0.8 : 1,
  }

  return (
    <div ref={setNodeRef} style={style} {...attributes}>
      <HolderOutlined {...listeners} style={{ marginRight: 8, color: '#999', cursor: 'grab' }} />
      <Checkbox
        checked={isFixed}
        onChange={() => onToggleFixed(id)}
        style={{ marginRight: 8 }}
      />
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {id.replace(/_/g, ' ')}
      </span>
    </div>
  )
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
  // 提取所有可用字段
  const allFieldKeys = useMemo(() => extractTaskFields(tasks), [tasks])

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
  const treeData = useMemo((): DataNode[] => {
    const nodes: DataNode[] = []
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
      'task_id', 'status', 'progress',
      'benchmark_name', 'topology_config_name',
      'search_stats_feasible',
      'best_throughput', 'best_ttft', 'best_tpot', 'best_score',
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

  // 列头
  const colHeaders = useMemo(() => {
    return allColumns.map((col) => col.replace(/_/g, ' '))
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
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([])
  const allCategoryKeys = useMemo(() => treeData.map((node) => node.key), [treeData])
  const expandAll = () => setExpandedKeys(allCategoryKeys)
  const collapseAll = () => setExpandedKeys([])

  // 搜索状态
  const [searchValue, setSearchValue] = useState<string>('')

  // 过滤后的树数据
  const filteredTreeData = useMemo((): DataNode[] => {
    if (!searchValue.trim()) {
      return treeData
    }
    const searchLower = searchValue.toLowerCase()

    const filterNode = (node: DataNode): DataNode | null => {
      if (!node.children || node.children.length === 0) {
        const matches =
          String(node.title).toLowerCase().includes(searchLower) ||
          String(node.key).toLowerCase().includes(searchLower)
        return matches ? node : null
      }

      const filteredChildren: DataNode[] = []
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

    const filtered: DataNode[] = []
    for (const category of treeData) {
      const result = filterNode(category)
      if (result) {
        filtered.push(result)
      }
    }
    return filtered
  }, [treeData, searchValue])

  // 处理列选择
  const handleColumnCheck = (checkedKeys: React.Key[]) => {
    const newCheckedColumns = (checkedKeys as string[]).filter((key) => !key.startsWith('category_'))
    setVisibleColumns(newCheckedColumns)
  }

  const checkedKeys = useMemo(() => visibleColumns, [visibleColumns])

  // 从任务对象中提取字段值
  const getTaskFieldValue = (task: EvaluationTask, field: string): string | number => {
    // 基础字段
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

    // 最佳方案字段（从 results[0] 提取，completed 状态下应该有数据）
    if (field.startsWith('best_')) {
      // 这里简化处理，实际应该从 top_k_plans 或 results 中获取
      return '-'
    }

    return '-'
  }

  // 表格数据
  const tableData = useMemo(() => {
    if (!tasks.length) return []
    return tasks.map((task) => {
      return allColumns.map((col) => {
        const value = getTaskFieldValue(task, col)
        if (typeof value === 'number') {
          // 百分比字段
          if (col === 'progress') {
            return `${value.toFixed(1)}%`
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
  }, [tasks, allColumns])

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
    if (row >= 0 && tasks[row]) {
      onTaskSelect(tasks[row])
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
    if (selectedRowIndices.size === tasks.length) {
      setSelectedRowIndices(new Set())
    } else {
      setSelectedRowIndices(new Set(tasks.map((_, idx) => idx)))
    }
  }

  // 批量删除任务
  const [batchDeleting, setBatchDeleting] = useState(false)
  const handleBatchDelete = async () => {
    if (selectedRowIndices.size === 0 || !onTasksDelete) return
    setBatchDeleting(true)
    try {
      const taskIds = Array.from(selectedRowIndices).map((idx) => tasks[idx].task_id)
      await onTasksDelete(taskIds)
      message.success('批量删除成功')
      setSelectedRowIndices(new Set())
    } catch {
      message.error('批量删除失败')
    } finally {
      setBatchDeleting(false)
    }
  }

  // 导出 CSV
  const handleExport = () => {
    if (!tasks.length) {
      message.warning('没有数据可导出')
      return
    }
    const headers = allColumns
    const rows = tasks.map((task) =>
      allColumns.map((col) => getTaskFieldValue(task, col)).join(',')
    )
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `tasks_experiment_${experimentId}.csv`
    link.click()
    message.success('导出成功')
  }

  // 列设置标签页状态
  const [columnSettingTab, setColumnSettingTab] = useState<'select' | 'order'>('select')

  // 列选择器
  const columnSelector = (
    <div style={{ maxHeight: 500, overflow: 'auto', minWidth: 360 }}>
      <div style={{ marginBottom: 12, display: 'flex', gap: 8 }}>
        <Button
          type={columnSettingTab === 'select' ? 'primary' : 'default'}
          size="small"
          onClick={() => setColumnSettingTab('select')}
        >
          选择列
        </Button>
        <Button
          type={columnSettingTab === 'order' ? 'primary' : 'default'}
          size="small"
          onClick={() => setColumnSettingTab('order')}
        >
          排序列 ({orderedVisibleColumns.length})
        </Button>
      </div>

      {columnSettingTab === 'select' ? (
        <>
          <div style={{ marginBottom: 8 }}>
            <Input
              placeholder="搜索列名..."
              prefix={<SearchOutlined />}
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              allowClear
              size="small"
            />
          </div>
          <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong>显示列</strong>
            <Space size="small">
              <Button size="small" onClick={expandAll}>全部展开</Button>
              <Button size="small" onClick={collapseAll}>全部折叠</Button>
            </Space>
          </div>
          <Tree
            checkable
            selectable={false}
            treeData={filteredTreeData}
            checkedKeys={checkedKeys}
            expandedKeys={searchValue ? filteredTreeData.map((n) => n.key) : expandedKeys}
            onExpand={(keys) => setExpandedKeys(keys)}
            onCheck={(checked) => handleColumnCheck(checked as React.Key[])}
          />
        </>
      ) : (
        <>
          <div style={{ marginBottom: 8, color: '#666', fontSize: 12 }}>
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
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
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
    <div>
      <div
        style={{
          marginBottom: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Space>
          <Popover content={columnSelector} title="列显示设置" trigger="click" placement="bottomLeft">
            <Button icon={<SettingOutlined />}>
              列显示设置 ({visibleColumns.length}/{allFieldKeys.length})
            </Button>
          </Popover>
          <span style={{ color: '#888', fontSize: 12 }}>
            {selectMode ? '选择模式：单击行选中，再次点击取消' : '双击行查看详情'}
          </span>
        </Space>
        <Space>
          <Tooltip title="进入选择模式后单击行可选中，用于批量删除">
            <Button
              icon={<DeleteOutlined />}
              type={selectMode ? 'primary' : 'default'}
              danger={selectMode}
              onClick={() => {
                setSelectMode(!selectMode)
                if (selectMode) {
                  setSelectedRowIndices(new Set())
                }
              }}
            >
              {selectMode ? '退出选择' : '选择模式'}
            </Button>
          </Tooltip>
          {selectMode && onTasksDelete && (
            <>
              <Tooltip title="选中/取消选中所有行">
                <Checkbox
                  checked={selectedRowIndices.size === tasks.length && tasks.length > 0}
                  indeterminate={selectedRowIndices.size > 0 && selectedRowIndices.size < tasks.length}
                  onChange={toggleSelectAll}
                >
                  全选
                </Checkbox>
              </Tooltip>
              <Popconfirm
                title="批量删除"
                description={`确定要删除选中的 ${selectedRowIndices.size} 个任务吗？此操作不可恢复。`}
                onConfirm={handleBatchDelete}
                okText="删除"
                cancelText="取消"
                okButtonProps={{ danger: true }}
                disabled={selectedRowIndices.size === 0}
              >
                <Tooltip title="删除所有选中的任务">
                  <Button
                    danger
                    icon={<DeleteOutlined />}
                    loading={batchDeleting}
                    disabled={selectedRowIndices.size === 0}
                  >
                    删除 ({selectedRowIndices.size})
                  </Button>
                </Tooltip>
              </Popconfirm>
            </>
          )}
          <Tooltip title="将任务数据导出为CSV文件">
            <Button icon={<DownloadOutlined />} onClick={handleExport} disabled={!tasks.length}>
              导出CSV
            </Button>
          </Tooltip>
        </Space>
      </div>

      <div className="task-table-container">
        {loading ? (
          <div style={{ textAlign: 'center', padding: 50, background: '#fafafa' }}>加载中...</div>
        ) : tableData.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 50, background: '#fafafa', color: '#999' }}>暂无任务数据</div>
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
  )
}
