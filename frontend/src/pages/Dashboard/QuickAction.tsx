/**
 * 快速操作按钮组件
 */

import React from 'react'
import { Card } from "@/components/ui/card"

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
      className="cursor-pointer text-center transition-all duration-300 hover:shadow-md"
      onClick={onClick}
    >
      <div className="px-4 py-8">
        <div
          className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl text-[24px] transition-transform duration-300 hover:scale-110"
          style={{
            background: `linear-gradient(135deg, ${color}15 0%, ${color}08 100%)`,
            color: color,
          }}
        >
          {icon}
        </div>
        <h5 className="mb-2 font-display text-[15px] font-semibold text-text-primary">
          {title}
        </h5>
        <p className="text-xs text-text-muted leading-relaxed">{description}</p>
      </div>
    </Card>
  )
}
