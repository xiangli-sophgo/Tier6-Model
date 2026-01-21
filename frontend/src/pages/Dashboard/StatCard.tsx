/**
 * 统计卡片组件
 */

import React from 'react'
import { Card, Typography } from 'antd'

const { Text } = Typography

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
      hoverable={!!onClick}
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
      bodyStyle={{ padding: 20 }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: 12,
            backgroundColor: bgColor,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: color,
            fontSize: 24,
          }}
        >
          {icon}
        </div>
        <div style={{ flex: 1 }}>
          <Text type="secondary" style={{ fontSize: 14 }}>
            {title}
          </Text>
          <div style={{ marginTop: 4 }}>
            <span style={{ fontSize: 28, fontWeight: 600, color }}>{value}</span>
            {suffix && (
              <span style={{ fontSize: 14, color: 'rgba(0,0,0,0.45)', marginLeft: 4 }}>
                {suffix}
              </span>
            )}
          </div>
        </div>
      </div>
    </Card>
  )
}
