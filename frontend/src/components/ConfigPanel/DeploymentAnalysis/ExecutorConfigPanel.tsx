/**
 * 执行器配置面板
 *
 * 允许用户查看和调整最大并发任务数量
 */

import React, { useState, useEffect } from 'react'
import { Card, InputNumber, Button, Space, message, Tooltip, Statistic, Row, Col, Alert } from 'antd'
import { SettingOutlined, ReloadOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { getExecutorConfig, updateExecutorConfig, ExecutorConfig } from '../../../api/tasks'

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
  const handleChange = async (value: number | null) => {
    if (!value || value < 1 || value > 16) return

    setMaxWorkers(value)
    setSaving(true)

    try {
      await updateExecutorConfig({ max_workers: value })
      message.success(`已设置为 ${value} 个并发任务（重启服务后生效）`, 2)
      await loadConfig()
    } catch (error) {
      console.error('更新配置失败:', error)
      message.error('更新失败')
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
      <InputNumber
        min={1}
        max={16}
        value={maxWorkers}
        onChange={handleChange}
        loading={saving}
        style={{ width: '100%' }}
        placeholder="并发任务数"
      />
    )
  }

  // 完整模式：显示详细卡片
  return (
    <Card
      title={
        <Space>
          <SettingOutlined />
          <span>执行器配置</span>
        </Space>
      }
      bordered={false}
      size="small"
      extra={
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={saving}
          onClick={loadConfig}
        >
          刷新
        </Button>
      }
    >
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Statistic
            title="当前最大并发数"
            value={config?.max_workers || 0}
            suffix="个任务"
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="活跃任务"
            value={config?.active_tasks || 0}
            valueStyle={{ color: config?.active_tasks ? '#3f8600' : '#8c8c8c' }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="运行中任务"
            value={config?.running_tasks || 0}
          />
        </Col>
      </Row>

      <div style={{ marginTop: 16 }}>
        <div style={{ marginBottom: 8 }}>
          <span style={{ marginRight: 8 }}>最大并发数:</span>
          <InputNumber
            min={1}
            max={16}
            value={maxWorkers}
            onChange={handleChange}
            loading={saving}
            style={{ width: 120 }}
            addonAfter="个任务"
          />
          <Tooltip title="建议根据服务器 CPU 核心数设置，通常为 2-8">
            <InfoCircleOutlined style={{ marginLeft: 8, color: '#8c8c8c' }} />
          </Tooltip>
        </div>

        <Alert
          message="修改后需要重启服务生效"
          type="info"
          showIcon
          style={{ marginTop: 8 }}
        />
      </div>
    </Card>
  )
}
