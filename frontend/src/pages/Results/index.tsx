/**
 * 结果汇总页面
 * 显示历史评估结果的数据库视图
 */

import React from 'react'
import { Card, Typography, Alert } from 'antd'

const { Title, Text } = Typography

export const Results: React.FC = () => {
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
            结果汇总
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c' }}>
            查看历史评估结果
          </span>
        </div>
      </div>

      {/* 内容区 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
      <Card>
        <Alert
          message="功能开发中"
          description="此页面将展示实验和评估任务的完整历史记录。"
          type="info"
          showIcon
        />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">TODO: 实现以下功能</Text>
          <ul>
            <li>实验列表（Experiment 表）</li>
            <li>任务列表（EvaluationTask 表）</li>
            <li>结果列表（EvaluationResult 表）</li>
            <li>筛选和搜索</li>
            <li>导出功能</li>
          </ul>
        </div>
      </Card>
      </div>
    </div>
  )
}
