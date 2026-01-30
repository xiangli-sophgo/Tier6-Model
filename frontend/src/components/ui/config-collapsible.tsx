import * as React from 'react'
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ConfigCollapsibleProps {
  /** 标题内容 */
  title: React.ReactNode
  /** 子内容 */
  children: React.ReactNode
  /** 非受控模式：默认展开状态 */
  defaultOpen?: boolean
  /** 受控模式：展开状态 */
  open?: boolean
  /** 受控模式：状态变化回调 */
  onOpenChange?: (open: boolean) => void
  /** 容器额外样式 */
  className?: string
  /** 内容区额外样式 */
  contentClassName?: string
}

/**
 * 统一的配置面板折叠组件
 * - 统一的卡片样式和 hover 效果
 * - 自动的箭头图标（根据展开状态旋转）
 * - 支持受控和非受控两种模式
 */
export const ConfigCollapsible: React.FC<ConfigCollapsibleProps> = ({
  title,
  children,
  defaultOpen,
  open,
  onOpenChange,
  className,
  contentClassName,
}) => {
  return (
    <CollapsiblePrimitive.Root
      defaultOpen={defaultOpen}
      open={open}
      onOpenChange={onOpenChange}
      className={cn(
        'mb-3 rounded-lg transition-shadow duration-200 hover:shadow-md hover:shadow-gray-200/60',
        className
      )}
    >
      <CollapsiblePrimitive.Trigger className="group flex items-center justify-between w-full p-2.5 rounded-lg bg-gray-100 text-sm font-medium border border-gray-200/50 cursor-pointer">
        <span>{title}</span>
        <ChevronDown className="h-4 w-4 text-gray-500 transition-transform duration-200 group-data-[state=open]:rotate-180" />
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content>
        <div
          className={cn(
            'p-3 bg-white border border-t-0 border-gray-200/50 rounded-b-lg',
            contentClassName
          )}
        >
          {children}
        </div>
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  )
}
