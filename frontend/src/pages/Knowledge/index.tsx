/**
 * 知识网络页面
 * 分布式计算知识图谱
 */

import React, { useState, useRef } from 'react'
import { KnowledgeGraph, KnowledgeNodeCards } from '@/components/KnowledgeGraph'
import { useWorkbench } from '@/contexts/WorkbenchContext'

const CARD_WIDTH_KEY = 'knowledge_card_width'
const MIN_CARD_WIDTH = 250
const MAX_CARD_WIDTH = 600
const DEFAULT_CARD_WIDTH = 340

export const Knowledge: React.FC = () => {
  const { ui } = useWorkbench()
  const [cardWidth, setCardWidth] = useState(() => {
    const cached = localStorage.getItem(CARD_WIDTH_KEY)
    return cached ? Math.max(MIN_CARD_WIDTH, Math.min(MAX_CARD_WIDTH, parseInt(cached, 10))) : DEFAULT_CARD_WIDTH
  })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartX = useRef(0)
  const dragStartWidth = useRef(0)

  const handleDragStart = (e: React.MouseEvent) => {
    setIsDragging(true)
    dragStartX.current = e.clientX
    dragStartWidth.current = cardWidth
  }

  const handleDragMove = (e: MouseEvent) => {
    if (!isDragging) return
    const delta = e.clientX - dragStartX.current
    const newWidth = Math.max(MIN_CARD_WIDTH, Math.min(MAX_CARD_WIDTH, dragStartWidth.current + delta))
    setCardWidth(newWidth)
    localStorage.setItem(CARD_WIDTH_KEY, String(newWidth))
  }

  const handleDragEnd = () => {
    setIsDragging(false)
  }

  React.useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleDragMove)
      window.addEventListener('mouseup', handleDragEnd)
      return () => {
        window.removeEventListener('mousemove', handleDragMove)
        window.removeEventListener('mouseup', handleDragEnd)
      }
    }
  }, [isDragging])

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

      {/* 内容区 - 分为左侧卡片和右侧图形 */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
        {/* 左侧 - 节点详情卡片 */}
        {ui.knowledgeSelectedNodes.length > 0 && (
          <>
            <div style={{
              width: cardWidth,
              borderRight: '1px solid #f0f0f0',
              background: '#fff',
              display: 'flex',
              flexDirection: 'column',
              padding: 12,
              overflow: 'hidden',
            }}>
              <KnowledgeNodeCards
                nodes={ui.knowledgeSelectedNodes}
                onClose={(nodeId) => ui.removeKnowledgeSelectedNode(nodeId)}
                onNodeClick={(node) => ui.addKnowledgeSelectedNode(node)}
              />
            </div>

            {/* 拖拽分割线 */}
            <div
              onMouseDown={handleDragStart}
              style={{
                width: 4,
                cursor: isDragging ? 'col-resize' : 'ew-resize',
                background: isDragging ? '#5e6ad2' : '#f0f0f0',
                transition: isDragging ? 'none' : 'background 0.2s ease',
                flexShrink: 0,
              }}
              title="拖拽调整卡片宽度"
            />
          </>
        )}

        {/* 右侧 - 知识图谱 */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <KnowledgeGraph />
        </div>
      </div>
    </div>
  )
}
