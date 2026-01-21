/**
 * 结果分析页面
 * 展示详细的性能指标和图表
 */

import React from 'react'
import { Card, Typography, Alert } from 'antd'

const { Title, Text } = Typography

export const ResultAnalysis: React.FC = () => {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 标题栏 */}
      <div
        style={{
          padding: '16px 24px',
          borderBottom: '1px solid #f0f0f0',
          background: '#fff',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 20, fontWeight: 600, color: '#1a1a1a' }}>
            结果分析
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c' }}>
            性能指标详细分析
          </span>
        </div>
      </div>

      {/* 内容区 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
      <Card>
        <Alert
          message="功能开发中"
          description="此页面将展示详细的性能指标、图表和分析报告。"
          type="info"
          showIcon
        />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">TODO: 实现以下功能</Text>
          <ul>
            <li>性能指标图表（MFU、MBU、TTFT、TPOT）</li>
            <li>甘特图（Gantt Chart）</li>
            <li>通信开销分析</li>
            <li>气泡率统计</li>
            <li>方案对比</li>
            <li>导出报告</li>
          </ul>
        </div>
      </Card>
      </div>
    </div>
  )
}
