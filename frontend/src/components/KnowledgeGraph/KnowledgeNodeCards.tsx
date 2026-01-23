/**
 * 知识节点详情卡片组件
 * 显示选中节点的定义、相关概念和参考资料
 */
import React from 'react'
import { Card, Tag } from 'antd'
import { BookOutlined, LinkOutlined, FileTextOutlined } from '@ant-design/icons'
import { ForceKnowledgeNode, CATEGORY_COLORS, CATEGORY_NAMES } from './types'
import knowledgeData from '../../data/knowledge-graph'

interface KnowledgeNodeCardsProps {
  nodes: ForceKnowledgeNode[]
  onClose: (nodeId: string) => void
  onNodeClick: (node: ForceKnowledgeNode) => void
}

export const KnowledgeNodeCards: React.FC<KnowledgeNodeCardsProps> = ({ nodes, onClose, onNodeClick }) => {
  // 只显示最近选中的一个节点
  const node = nodes[0]
  if (!node) return null

  // 获取相关节点
  const getRelatedNodes = (nodeId: string) => {
    const relatedIds = new Set<string>()
    const data = knowledgeData as any
    data.relations.forEach((r: any) => {
      if (r.source === nodeId) relatedIds.add(r.target)
      if (r.target === nodeId) relatedIds.add(r.source)
    })
    return data.nodes.filter((n: any) => relatedIds.has(n.id))
  }

  // 渲染定义文本（支持 Markdown 格式：**加粗**、\n换行）
  const renderDefinition = (text: string) => {
    // 1. 按换行符拆分
    const lines = text.split('\n')

    return lines.map((line, lineIndex) => {
      // 2. 处理每行中的加粗标记 **text**
      const segments: React.ReactNode[] = []
      const boldRegex = /\*\*([^*]+)\*\*/g
      let lastIndex = 0
      let match

      while ((match = boldRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          segments.push(line.slice(lastIndex, match.index))
        }
        segments.push(
          <strong key={`${lineIndex}-${match.index}`} style={{ color: '#4f46e5' }}>
            {match[1]}
          </strong>
        )
        lastIndex = boldRegex.lastIndex
      }
      if (lastIndex < line.length) {
        segments.push(line.slice(lastIndex))
      }

      // 3. 判断是否是分点项（以数字+标点开头）
      const isListItem = /^\d+[）\.\)]\s*/.test(line)

      return (
        <span
          key={lineIndex}
          style={{
            display: lineIndex > 0 || isListItem ? 'block' : undefined,
            marginTop: lineIndex > 0 ? 4 : 0,
            paddingLeft: isListItem ? 12 : 0,
          }}
        >
          {segments.length > 0 ? segments : line}
        </span>
      )
    })
  }

  // 分区样式
  const sectionStyle: React.CSSProperties = {
    background: '#f9fafb',
    border: '1px solid #f3f4f6',
    borderRadius: 6,
    padding: '10px 12px',
    marginBottom: 10,
  }
  const sectionTitleStyle: React.CSSProperties = {
    fontSize: 14,
    fontWeight: 600,
    color: '#374151',
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  }

  const relatedNodes = getRelatedNodes(node.id)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Card
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag color={CATEGORY_COLORS[node.category]} style={{ margin: 0, fontSize: 12, flexShrink: 0 }}>
              {CATEGORY_NAMES[node.category]}
            </Tag>
            <span style={{ fontSize: 16, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              {node.fullName || node.name}
            </span>
          </div>
        }
        size="small"
        style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
        styles={{ body: { flex: 1, overflow: 'auto', minHeight: 0 } }}
        extra={<a onClick={() => onClose(node.id)}>关闭</a>}
      >
        {/* 定义区域 */}
        <div style={sectionStyle}>
          <div style={sectionTitleStyle}>
            <BookOutlined style={{ color: '#6366f1' }} />
            <span>定义</span>
          </div>
          <div style={{
            fontSize: 15,
            lineHeight: 1.8,
            color: '#1f2937',
          }}>
            {renderDefinition(node.definition)}
          </div>
        </div>

        {/* 相关概念区域 */}
        {relatedNodes.length > 0 && (
          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>
              <LinkOutlined style={{ color: '#10b981' }} />
              <span>相关概念 ({relatedNodes.length})</span>
            </div>
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 6,
            }}>
              {relatedNodes.map((n: any) => (
                <Tag
                  key={n.id}
                  color={CATEGORY_COLORS[n.category as keyof typeof CATEGORY_COLORS]}
                  style={{ cursor: 'pointer', margin: 0, fontSize: 13 }}
                  onClick={() => onNodeClick(n as ForceKnowledgeNode)}
                >
                  {n.name}
                </Tag>
              ))}
            </div>
          </div>
        )}

        {/* 参考资料区域（仅当有 source 时显示）*/}
        {node.source && (
          <div style={{ ...sectionStyle, marginBottom: 0 }}>
            <div style={sectionTitleStyle}>
              <FileTextOutlined style={{ color: '#8b5cf6' }} />
              <span>参考资料</span>
            </div>
            <div style={{ fontSize: 13, color: '#6b7280', lineHeight: 1.6 }}>
              {node.source}
              {(node as any).url && (
                <a
                  href={(node as any).url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ marginLeft: 8, color: '#6366f1' }}
                >
                  <LinkOutlined /> 链接
                </a>
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
