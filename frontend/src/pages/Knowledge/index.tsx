/**
 * 知识网络页面
 * 分布式计算知识图谱
 */

import React, { useState, useRef } from 'react'
import { KnowledgeGraph, KnowledgeNodeCards } from '@/components/KnowledgeGraph'
import { useWorkbench } from '@/contexts/WorkbenchContext'

export const Knowledge: React.FC = () => {
  const { ui } = useWorkbench()
  const [cardWidth, setCardWidth] = useState(320)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = () => {
    setIsDragging(true)
  }

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return

      const container = containerRef.current
      const rect = container.getBoundingClientRect()
      const newWidth = e.clientX - rect.left

      // 限制宽度范围：200-600px
      if (newWidth >= 200 && newWidth <= 600) {
        setCardWidth(newWidth)
      }
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
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

      {/* 内容区 - 左侧卡片 + 右侧知识图谱 */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
        {/* 左侧信息卡片 */}
        {ui.knowledgeSelectedNodes.length > 0 && (
          <>
            <div style={{
              width: cardWidth,
              borderRight: '1px solid #f0f0f0',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              background: '#fff',
              minWidth: '200px',
              maxWidth: '600px',
            }}>
              <KnowledgeNodeCards
                nodes={ui.knowledgeSelectedNodes}
                onClose={ui.removeKnowledgeSelectedNode}
                onNodeClick={ui.addKnowledgeSelectedNode}
              />
            </div>

            {/* 可拖动分隔符 */}
            <div
              onMouseDown={handleMouseDown}
              onMouseEnter={(e) => {
                if (!isDragging) {
                  (e.currentTarget as HTMLDivElement).style.background = '#e5e5e5'
                }
              }}
              onMouseLeave={(e) => {
                if (!isDragging) {
                  (e.currentTarget as HTMLDivElement).style.background = 'transparent'
                }
              }}
              style={{
                width: 6,
                background: isDragging ? '#1890ff' : 'transparent',
                cursor: 'col-resize',
                transition: isDragging ? 'none' : 'background 0.2s',
                userSelect: 'none',
              }}
            />
          </>
        )}

        {/* 右侧知识图谱 */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <KnowledgeGraph />
        </div>
      </div>
    </div>
  )
}
