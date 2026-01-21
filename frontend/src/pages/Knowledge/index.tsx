/**
 * 知识网络页面
 * 分布式计算知识图谱
 */

import React from 'react'
import { KnowledgeGraph } from '@/components/KnowledgeGraph'

export const Knowledge: React.FC = () => {
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
            知识网络
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c' }}>
            分布式计算知识图谱
          </span>
        </div>
      </div>

      {/* 内容区 */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <KnowledgeGraph />
      </div>
    </div>
  )
}
