/**
 * 快速操作按钮组件
 */

import React from 'react'
import { Card, Typography } from 'antd'

const { Title, Text } = Typography

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
      hoverable
      onClick={onClick}
      style={{ textAlign: 'center', borderRadius: 12 }}
      styles={{ body: { padding: '24px 16px' } }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: 12,
          background: `linear-gradient(135deg, ${color} 0%, ${color}cc 100%)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 12px',
          color: '#fff',
          fontSize: 22,
        }}
      >
        {icon}
      </div>
      <Title level={5} style={{ marginBottom: 4, fontSize: 15 }}>
        {title}
      </Title>
      <Text type="secondary" style={{ fontSize: 12 }}>
        {description}
      </Text>
    </Card>
  )
}
