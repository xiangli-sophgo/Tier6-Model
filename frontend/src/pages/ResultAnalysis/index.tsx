/**
 * 结果分析页面
 * 展示实验详情和结果详细分析
 */

import React, { useEffect, useState } from 'react'
import {
  Card,
  Tabs,
  Button,
  Space,
  message,
  Typography,
  Tooltip,
  Spin,
  Statistic,
  Row,
  Col,
  Alert,
} from 'antd'
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  DownloadOutlined,
} from '@ant-design/icons'
import { getExperimentDetail, Experiment } from '@/api/results'

const { Text } = Typography

// 统计项组件
interface StatItemProps {
  label: string
  value: string | number
  precision?: number
}

const StatItem: React.FC<StatItemProps> = ({ label, value, precision = 2 }) => (
  <div style={{ padding: 16, background: '#fafafa', borderRadius: 4 }}>
    <div style={{ color: '#8c8c8c', fontSize: 12, marginBottom: 4 }}>{label}</div>
    <div style={{ fontSize: 18, fontWeight: 600, color: '#1a1a1a' }}>
      {typeof value === 'number' ? value.toFixed(precision) : value}
    </div>
  </div>
)

interface ResultAnalysisProps {
  experimentId?: number
  onBack?: () => void
}

export const ResultAnalysis: React.FC<ResultAnalysisProps> = ({ experimentId, onBack }) => {
  void onBack  // Props may not always be provided
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // 加载实验详情
  const loadExperiment = async () => {
    if (!experimentId) return
    setLoading(true)
    try {
      const data = await getExperimentDetail(experimentId)
      setExperiment(data)
    } catch (error) {
      message.error('加载实验详情失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 首次加载
  useEffect(() => {
    loadExperiment()
  }, [experimentId])

  // 导出结果
  const handleExport = () => {
    message.info('导出功能开发中...')
  }

  if (!experiment) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid #f0f0f0',
            background: '#fff',
          }}
        >
          {onBack && (
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={onBack}
            >
              返回
            </Button>
          )}
        </div>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Spin spinning={loading} />
        </div>
      </div>
    )
  }

  // 从第一个任务的 config_snapshot 中提取配置
  const firstTask = experiment.tasks && experiment.tasks.length > 0 ? experiment.tasks[0] : null
  const modelConfig = firstTask?.config_snapshot?.model as Record<string, unknown> || {}
  const inferenceConfig = firstTask?.config_snapshot?.inference as Record<string, unknown> || {}

  const progress =
    experiment.total_tasks > 0
      ? Math.round((experiment.completed_tasks / experiment.total_tasks) * 100)
      : 0

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 标题栏 */}
      <div
        style={{
          padding: '16px 24px',
          borderBottom: '1px solid #f0f0f0',
          background: '#fff',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Space>
          {onBack && (
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={onBack}
            >
              返回
            </Button>
          )}
          <div>
            <div style={{ fontSize: 18, fontWeight: 600, color: '#1a1a1a' }}>
              {experiment.name}
            </div>
            <Text type="secondary">{experiment.description || '无描述'}</Text>
          </div>
        </Space>
        <Space>
          <Tooltip title="刷新">
            <Button icon={<ReloadOutlined />} onClick={loadExperiment} />
          </Tooltip>
          <Button icon={<DownloadOutlined />} onClick={handleExport}>
            导出
          </Button>
        </Space>
      </div>

      {/* 内容区 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
        {/* 进度提示 */}
        {progress < 100 && (
          <Alert
            message={`实验进度: ${progress}% (${experiment.completed_tasks}/${experiment.total_tasks} 任务完成)`}
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* 标签页 */}
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'overview',
              label: '概览',
              children: (
                <div style={{ paddingTop: 16 }}>
                  {/* 任务进度 */}
                  <Card title="任务进度" style={{ marginBottom: 16 }}>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Statistic
                          title="已完成任务"
                          value={experiment.completed_tasks}
                          suffix={`/ ${experiment.total_tasks}`}
                        />
                      </Col>
                      <Col span={12}>
                        <div style={{ padding: '16px 0' }}>
                          <div style={{ marginBottom: 8 }}>进度: {progress}%</div>
                          <div
                            style={{
                              width: '100%',
                              height: 8,
                              backgroundColor: '#f0f0f0',
                              borderRadius: 4,
                            }}
                          >
                            <div
                              style={{
                                width: `${progress}%`,
                                height: '100%',
                                backgroundColor: progress === 100 ? '#52c41a' : '#1890ff',
                                borderRadius: 4,
                                transition: 'width 0.3s',
                              }}
                            />
                          </div>
                        </div>
                      </Col>
                    </Row>
                  </Card>

                  {/* 模型配置 */}
                  <Card title="模型配置" style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 16]}>
                      <Col span={8}>
                        <StatItem
                          label="模型名称"
                          value={modelConfig.model_name as string || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="隐藏层尺寸"
                          value={modelConfig.hidden_size as number || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="层数"
                          value={modelConfig.num_layers as number || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="注意力头数"
                          value={modelConfig.num_attention_heads as number || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="KV 头数"
                          value={modelConfig.num_kv_heads as number || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="数据类型"
                          value={modelConfig.dtype as string || '-'}
                        />
                      </Col>
                    </Row>
                  </Card>

                  {/* 推理配置 */}
                  <Card title="推理配置" style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 16]}>
                      <Col span={8}>
                        <StatItem
                          label="批次大小"
                          value={inferenceConfig.batch_size as number || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="输入序列长度"
                          value={inferenceConfig.input_seq_length as number || '-'}
                        />
                      </Col>
                      <Col span={8}>
                        <StatItem
                          label="输出序列长度"
                          value={inferenceConfig.output_seq_length as number || '-'}
                        />
                      </Col>
                    </Row>
                  </Card>

                  {/* 创建时间 */}
                  <Card title="元数据" style={{ marginBottom: 16 }}>
                    <Row gutter={[16, 16]}>
                      <Col span={12}>
                        <StatItem
                          label="创建时间"
                          value={new Date(experiment.created_at).toLocaleString('zh-CN')}
                        />
                      </Col>
                      <Col span={12}>
                        <StatItem
                          label="最后更新"
                          value={new Date(experiment.updated_at).toLocaleString('zh-CN')}
                        />
                      </Col>
                    </Row>
                  </Card>
                </div>
              ),
            },
            {
              key: 'tasks',
              label: `任务列表 (${experiment.tasks?.length || 0})`,
              children: (
                <div style={{ paddingTop: 16 }}>
                  {experiment.tasks && experiment.tasks.length > 0 ? (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 16 }}>
                      {experiment.tasks.map((task) => (
                        <Card
                          key={task.task_id}
                          size="small"
                          hoverable
                          onClick={() => {
                            // 点击任务卡片可以展示任务详情
                            message.info(`任务 ID: ${task.task_id}`)
                          }}
                        >
                          <div style={{ marginBottom: 8 }}>
                            <Text strong>{task.task_id}</Text>
                          </div>
                          <div style={{ marginBottom: 4 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              状态:
                            </Text>
                            <span style={{ marginLeft: 8, color: '#1890ff' }}>
                              {task.status}
                            </span>
                          </div>
                          <div style={{ marginBottom: 4 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              进度:
                            </Text>
                            <span style={{ marginLeft: 8 }}>{task.progress.toFixed(1)}%</span>
                          </div>
                          <div style={{ marginBottom: 4 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              创建时间:
                            </Text>
                            <span style={{ marginLeft: 8, fontSize: 12 }}>
                              {new Date(task.created_at).toLocaleString('zh-CN')}
                            </span>
                          </div>
                          {task.message && (
                            <div>
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                消息:
                              </Text>
                              <div style={{ marginTop: 4, fontSize: 12, color: '#666' }}>
                                {task.message}
                              </div>
                            </div>
                          )}
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <Alert message="暂无任务" type="info" />
                  )}
                </div>
              ),
            },
            {
              key: 'results',
              label: '结果分析',
              children: (
                <div style={{ paddingTop: 16 }}>
                  <Alert
                    message="结果分析功能开发中"
                    description="将展示详细的性能指标、甘特图、通信开销分析等内容"
                    type="info"
                    showIcon
                  />
                </div>
              ),
            },
          ]}
        />
      </div>
    </div>
  )
}
