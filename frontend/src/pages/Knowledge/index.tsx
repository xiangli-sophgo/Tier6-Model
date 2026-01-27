/**
 * 知识网络页面
 * 分布式计算知识图谱
 */

import React, { useState, useRef } from 'react'
import { KnowledgeGraph, KnowledgeNodeCards } from '@/components/KnowledgeGraph'
import { useWorkbench } from '@/contexts/WorkbenchContext'

export const Knowledge: React.FC = () => {
  const { knowledge } = useWorkbench()
  const [cardWidth, setCardWidth] = useState(400)  // 增加默认宽度 320 -> 400
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
    <div className="h-full w-full bg-white flex flex-col overflow-hidden">
      {/* 标题栏 */}
      <div
        style={{
          width: '100%',
          padding: '16px 24px',
          borderBottom: '1px solid #f0f0f0',
          background: '#fff',
          flexShrink: 0
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

      {/* 工具栏 - 独立层级，横跨整个宽度，在卡片上方 */}
      <div style={{ width: '100%', minWidth: 0, flexShrink: 0, overflow: 'hidden' }}>
        <KnowledgeGraph renderMode="toolbar-only" />
      </div>

      {/* 内容区 - 知识图谱画布 + 悬浮卡片 */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {/* 知识图谱画布 - 占满整个容器 */}
        <div style={{ width: '100%', height: '100%' }}>
          <KnowledgeGraph renderMode="canvas-only" />
        </div>

        {/* 左侧悬浮信息卡片 */}
        {knowledge.knowledgeSelectedNodes.length > 0 && (
          <>
            <div style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: cardWidth,
              borderRight: '1px solid #f0f0f0',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              background: '#fff',
              minWidth: '200px',
              maxWidth: '600px',
              boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
              zIndex: 10
            }}>
              <KnowledgeNodeCards
                nodes={knowledge.knowledgeSelectedNodes}
                onClose={knowledge.removeKnowledgeSelectedNode}
                onNodeClick={knowledge.addKnowledgeSelectedNode}
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
                position: 'absolute',
                left: cardWidth,
                top: 0,
                bottom: 0,
                width: 6,
                background: isDragging ? '#1890ff' : 'transparent',
                cursor: 'col-resize',
                transition: isDragging ? 'none' : 'background 0.2s',
                userSelect: 'none',
                zIndex: 11
              }}
            />
          </>
        )}
      </div>
    </div>
  )
}
