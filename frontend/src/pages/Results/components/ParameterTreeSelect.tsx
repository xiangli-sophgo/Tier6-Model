/**
 * 参数树形选择器
 * 基于 Popover + Collapsible 实现的树形多选组件
 */

import { useState, useMemo } from 'react'
import { ChevronDown, ChevronRight, Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import type { ParameterTreeNode } from '../utils/parameterClassifier'

interface ParameterTreeSelectProps {
  /** 树形数据 */
  tree: ParameterTreeNode[]
  /** 选中的参数路径 */
  value: string[]
  /** 选择变化回调 */
  onChange: (value: string[]) => void
  /** 最大选择数量 */
  maxSelection?: number
  /** 占位文本 */
  placeholder?: string
}

export function ParameterTreeSelect({
  tree,
  value,
  onChange,
  maxSelection = 999,
  placeholder = '选择参数...',
}: ParameterTreeSelectProps) {
  const [open, setOpen] = useState(false)
  const [searchKeyword, setSearchKeyword] = useState('')
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set())

  // 过滤树节点
  const filteredTree = useMemo(() => {
    if (!searchKeyword.trim()) return tree

    const keyword = searchKeyword.toLowerCase()
    return tree
      .map(categoryNode => {
        const matchedChildren = categoryNode.children?.filter(
          child =>
            child.title.toLowerCase().includes(keyword) ||
            child.key.toLowerCase().includes(keyword)
        )
        if (matchedChildren && matchedChildren.length > 0) {
          return { ...categoryNode, children: matchedChildren }
        }
        return null
      })
      .filter(Boolean) as ParameterTreeNode[]
  }, [tree, searchKeyword])

  // 展开/折叠分类
  const toggleExpand = (key: string) => {
    const newExpanded = new Set(expandedKeys)
    if (newExpanded.has(key)) {
      newExpanded.delete(key)
    } else {
      newExpanded.add(key)
    }
    setExpandedKeys(newExpanded)
  }

  // 处理参数选择
  const handleSelect = (paramKey: string) => {
    if (value.includes(paramKey)) {
      // 取消选择
      onChange(value.filter(k => k !== paramKey))
    } else {
      // 添加选择
      if (value.length >= maxSelection) {
        return // 已达到最大选择数
      }
      onChange([...value, paramKey])
    }
  }

  // 清空选择
  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation()
    onChange([])
  }

  // 显示标签
  const displayText = useMemo(() => {
    if (value.length === 0) return placeholder
    if (value.length <= 3) {
      return value.map(key => {
        // 从树中找到对应的节点标题
        for (const category of tree) {
          const found = category.children?.find(c => c.key === key)
          if (found) {
            // 提取显示名称（去掉范围信息）
            return found.title.split('[')[0].trim()
          }
        }
        return key
      }).join(', ')
    }
    return `已选择 ${value.length} 个参数`
  }, [value, tree, placeholder])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between"
        >
          <span className="truncate">{displayText}</span>
          <div className="ml-2 flex items-center gap-1">
            {value.length > 0 && (
              <X
                className="h-4 w-4 shrink-0 opacity-50 hover:opacity-100"
                onClick={handleClear}
              />
            )}
            <ChevronDown className="h-4 w-4 shrink-0 opacity-50" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[400px] p-0" align="start">
        {/* 搜索框 */}
        <div className="p-3 border-b">
          <Input
            placeholder="搜索参数..."
            value={searchKeyword}
            onChange={e => setSearchKeyword(e.target.value)}
            className="h-8"
          />
          {maxSelection < 999 && (
            <div className="text-xs text-gray-500 mt-2">
              最多选择 {maxSelection} 个参数 ({value.length}/{maxSelection})
            </div>
          )}
        </div>

        {/* 树形列表 */}
        <div className="max-h-[400px] overflow-y-auto p-2">
          {filteredTree.length === 0 ? (
            <div className="text-center text-gray-500 py-8 text-sm">
              未找到匹配的参数
            </div>
          ) : (
            filteredTree.map(categoryNode => (
              <Collapsible
                key={categoryNode.key}
                open={expandedKeys.has(categoryNode.key) || searchKeyword.trim() !== ''}
                onOpenChange={() => toggleExpand(categoryNode.key)}
              >
                <CollapsibleTrigger className="flex items-center w-full px-2 py-2 hover:bg-gray-100 rounded text-sm font-medium">
                  {expandedKeys.has(categoryNode.key) || searchKeyword.trim() !== '' ? (
                    <ChevronDown className="h-4 w-4 mr-1" />
                  ) : (
                    <ChevronRight className="h-4 w-4 mr-1" />
                  )}
                  <span>{categoryNode.title}</span>
                  {categoryNode.children && (
                    <span className="ml-1 text-gray-400">
                      ({categoryNode.children.length})
                    </span>
                  )}
                </CollapsibleTrigger>
                <CollapsibleContent className="pl-4">
                  {categoryNode.children?.map(paramNode => {
                    const isSelected = value.includes(paramNode.key)
                    const isDisabled = !isSelected && value.length >= maxSelection

                    return (
                      <div
                        key={paramNode.key}
                        onClick={() => !isDisabled && handleSelect(paramNode.key)}
                        className={`
                          flex items-center justify-between px-2 py-1.5 rounded text-sm cursor-pointer
                          ${isSelected ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100'}
                          ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}
                        `}
                      >
                        <span className="flex-1 truncate" title={paramNode.title}>
                          {paramNode.title}
                        </span>
                        {isSelected && (
                          <Check className="h-4 w-4 shrink-0 ml-2 text-blue-600" />
                        )}
                      </div>
                    )
                  })}
                </CollapsibleContent>
              </Collapsible>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}
