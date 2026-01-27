/**
 * 知识节点详情卡片组件
 * 显示选中节点的定义、相关概念和参考资料
 */
import React from 'react'
import { Card, CardHeader, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Book, Link2, FileText } from 'lucide-react'
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
          <strong key={`${lineIndex}-${match.index}`} className="text-primary">
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
          className={`${lineIndex > 0 || isListItem ? 'block' : ''} ${lineIndex > 0 ? 'mt-1' : ''} ${isListItem ? 'pl-3' : ''}`}
        >
          {segments.length > 0 ? segments : line}
        </span>
      )
    })
  }

  const relatedNodes = getRelatedNodes(node.id)

  return (
    <div className="flex flex-1 flex-col min-h-0">
      <Card className="flex flex-1 flex-col min-h-0">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Badge
              variant="default"
              className="m-0 text-xs flex-shrink-0"
              style={{
                backgroundColor: CATEGORY_COLORS[node.category],
              }}
            >
              {CATEGORY_NAMES[node.category]}
            </Badge>
            <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap text-base">
              {node.fullName || node.name}
            </span>
            <a
              onClick={() => onClose(node.id)}
              className="cursor-pointer text-sm text-primary hover:underline"
            >
              关闭
            </a>
          </div>
        </CardHeader>

        <CardContent className="flex-1 overflow-auto min-h-0">
          {/* 定义区域 */}
          <div className="mb-2.5 rounded-md border border-[#f3f4f6] bg-[#f9fafb] p-3">
            <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-[#374151]">
              <Book className="h-4 w-4 text-[#6366f1]" />
              <span>定义</span>
            </div>
            <div className="text-[15px] leading-relaxed text-[#1f2937]">
              {renderDefinition(node.definition)}
            </div>
          </div>

          {/* 相关概念区域 */}
          {relatedNodes.length > 0 && (
            <div className="mb-2.5 rounded-md border border-[#f3f4f6] bg-[#f9fafb] p-3">
              <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-[#374151]">
                <Link2 className="h-4 w-4 text-success" />
                <span>相关概念 ({relatedNodes.length})</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {relatedNodes.map((n: any) => (
                  <Badge
                    key={n.id}
                    variant="default"
                    className="m-0 cursor-pointer text-[13px]"
                    style={{
                      backgroundColor: CATEGORY_COLORS[n.category as keyof typeof CATEGORY_COLORS],
                    }}
                    onClick={() => onNodeClick(n as ForceKnowledgeNode)}
                  >
                    {n.name}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* 参考资料区域（仅当有 source 时显示）*/}
          {node.source && (
            <div className="rounded-md border border-[#f3f4f6] bg-[#f9fafb] p-3">
              <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-[#374151]">
                <FileText className="h-4 w-4 text-[#8b5cf6]" />
                <span>参考资料</span>
              </div>
              <div className="text-[13px] leading-relaxed text-[#6b7280]">
                {node.source}
                {(node as any).url && (
                  <a
                    href={(node as any).url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-2 text-[#6366f1] hover:underline"
                  >
                    <Link2 className="inline h-3 w-3" /> 链接
                  </a>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
