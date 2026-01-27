/**
 * 快速操作按钮组件
 */

import React from 'react'
import { Card } from '@/components/ui/card'

interface QuickActionProps {
  icon: React.ReactNode
  title: string
  description: string
  color: string
  onClick: () => void
}

export const QuickAction: React.FC<QuickActionProps> = ({
  icon,
  title,
  description,
  color,
  onClick,
}) => {
  return (
    <Card
      className="cursor-pointer text-center hover:shadow-md"
      onClick={onClick}
    >
      <div className="px-4 py-6">
        <div
          className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-lg text-[22px] text-white"
          style={{
            background: `linear-gradient(135deg, ${color} 0%, ${color}cc 100%)`,
          }}
        >
          {icon}
        </div>
        <h5 className="mb-1 font-display text-[15px] font-semibold">
          {title}
        </h5>
        <p className="text-xs text-text-secondary">{description}</p>
      </div>
    </Card>
  )
}
