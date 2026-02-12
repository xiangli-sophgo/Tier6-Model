import React from 'react'
import { cn } from '@/lib/utils'

interface PageHeaderProps {
  title: string
  children?: React.ReactNode
  className?: string
}

/**
 * 统一的页面标题栏组件
 * 封装渐变背景、阴影和渐变文字样式
 */
export const PageHeader: React.FC<PageHeaderProps> = ({ title, children, className }) => {
  return (
    <div
      className={cn(
        'px-8 py-6 border-b border-blue-100 bg-gradient-to-r from-blue-50 to-white flex justify-between items-center flex-shrink-0',
        className
      )}
      style={{ boxShadow: '0 2px 12px rgba(37, 99, 235, 0.08)' }}
    >
      <h3 className="m-0 bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-2xl font-bold text-transparent">
        {title}
      </h3>
      {children}
    </div>
  )
}
