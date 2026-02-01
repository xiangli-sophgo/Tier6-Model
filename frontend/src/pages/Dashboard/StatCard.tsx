/**
 * 统计卡片组件
 */

import React from 'react'
import { Card } from "@/components/ui/card"
import { cn } from '@/lib/utils'

interface StatCardProps {
  title: string
  value: number
  icon: React.ReactNode
  color: string
  bgColor: string
  suffix?: string
  onClick?: () => void
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon,
  color,
  bgColor,
  suffix,
  onClick,
}) => {
  return (
    <Card
      className={cn(
        'cursor-default',
        onClick && 'cursor-pointer hover:shadow-md'
      )}
      onClick={onClick}
    >
      <div className="flex items-start gap-4 p-5">
        <div
          className="flex h-14 w-14 items-center justify-center rounded-lg text-2xl"
          style={{
            backgroundColor: bgColor,
            color: color,
          }}
        >
          {icon}
        </div>
        <div className="flex-1">
          <p className="text-sm text-text-secondary">{title}</p>
          <div className="mt-1">
            <span className="text-[28px] font-semibold" style={{ color }}>
              {value}
            </span>
            {suffix && (
              <span className="ml-1 text-sm text-text-muted">{suffix}</span>
            )}
          </div>
        </div>
      </div>
    </Card>
  )
}
