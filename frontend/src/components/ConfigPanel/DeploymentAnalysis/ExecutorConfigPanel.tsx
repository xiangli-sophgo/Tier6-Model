/**
 * 执行器配置面板
 *
 * 允许用户查看和调整最大并发任务数量
 */

import React, { useState, useEffect } from 'react'
import { Settings, RefreshCw, Info } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { BaseCard } from '@/components/common/BaseCard'
import { getExecutorConfig, updateExecutorConfig, ExecutorConfig } from '../../../api/tasks'

// 统计项组件
const StatItem: React.FC<{ title: string; value: number | string; suffix?: string; valueColor?: string }> = ({
  title,
  value,
  suffix,
  valueColor,
}) => (
  <div>
    <div className="text-gray-400 text-xs mb-1">{title}</div>
    <div className="text-xl font-semibold" style={valueColor ? { color: valueColor } : undefined}>
      {value}
      {suffix && <span className="text-sm text-gray-400 ml-1">{suffix}</span>}
    </div>
  </div>
)

interface ExecutorConfigPanelProps {
  /** 是否展开显示详细信息 */
  compact?: boolean
}

export const ExecutorConfigPanel: React.FC<ExecutorConfigPanelProps> = ({ compact = false }) => {
  const [config, setConfig] = useState<ExecutorConfig | null>(null)
  const [maxWorkers, setMaxWorkers] = useState<number>(4)
  const [saving, setSaving] = useState(false)

  // 加载配置
  const loadConfig = async () => {
    try {
      const data = await getExecutorConfig()
      setConfig(data)
      setMaxWorkers(data.max_workers)
    } catch (error) {
      console.error('加载配置失败:', error)
    }
  }

  // 初始加载
  useEffect(() => {
    loadConfig()
  }, [])

  // 修改并发数（失焦时自动保存）
  const handleChange = async (value: number) => {
    if (value < 1 || value > 16) return

    setMaxWorkers(value)
    setSaving(true)

    try {
      await updateExecutorConfig({ max_workers: value })
      toast.success(`已设置为 ${value} 个并发任务（重启服务后生效）`)
      await loadConfig()
    } catch (error) {
      console.error('更新配置失败:', error)
      toast.error('更新失败')
      // 恢复原值
      if (config) {
        setMaxWorkers(config.max_workers)
      }
    } finally {
      setSaving(false)
    }
  }

  if (compact) {
    // 紧凑模式：只显示输入框，修改后自动保存
    return (
      <Input
        type="number"
        min={1}
        max={16}
        value={maxWorkers}
        onChange={(e) => {
          const v = parseInt(e.target.value, 10)
          if (!isNaN(v)) handleChange(v)
        }}
        disabled={saving}
        className="w-full"
        placeholder="并发任务数"
      />
    )
  }

  // 完整模式：显示详细卡片
  return (
    <TooltipProvider>
      <BaseCard
        title={
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            <span>执行器配置</span>
          </div>
        }
        extra={
          <Button
            variant="outline"
            size="sm"
            onClick={loadConfig}
            disabled={saving}
          >
            <RefreshCw className={`h-3.5 w-3.5 mr-1 ${saving ? 'animate-spin' : ''}`} />
            刷新
          </Button>
        }
      >
        <div className="grid grid-cols-3 gap-4 mb-4">
          <StatItem
            title="当前最大并发数"
            value={config?.max_workers || 0}
            suffix="个任务"
          />
          <StatItem
            title="活跃任务"
            value={config?.active_tasks || 0}
            valueColor={config?.active_tasks ? '#3f8600' : '#8c8c8c'}
          />
          <StatItem
            title="运行中任务"
            value={config?.running_tasks || 0}
          />
        </div>

        <div className="mt-4">
          <div className="mb-2 flex items-center">
            <span className="mr-2 text-sm">最大并发数:</span>
            <Input
              type="number"
              min={1}
              max={16}
              value={maxWorkers}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10)
                if (!isNaN(v)) handleChange(v)
              }}
              disabled={saving}
              className="w-24 h-8"
            />
            <span className="ml-2 text-sm text-gray-500">个任务</span>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 ml-2 text-gray-400 cursor-help" />
              </TooltipTrigger>
              <TooltipContent>建议根据服务器 CPU 核心数设置，通常为 2-8</TooltipContent>
            </Tooltip>
          </div>

          <Alert className="mt-2">
            <AlertDescription>修改后需要重启服务生效</AlertDescription>
          </Alert>
        </div>
      </BaseCard>
    </TooltipProvider>
  )
}
