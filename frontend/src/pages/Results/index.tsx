/**
 * 结果汇总页面
 * 显示所有实验和评估任务的列表
 */

import React, { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  message,
  Typography,
  Tooltip,
  Popconfirm,
  Empty,
  Spin,
  Row,
  Col,
  Statistic,
  Alert,
} from 'antd'
import {
  ReloadOutlined,
  DeleteOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
  ArrowLeftOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { listExperiments, deleteExperiment, getExperimentDetail, getTaskResults, Experiment, EvaluationTask, TaskResultsResponse } from '@/api/results'
import { AnalysisResultDisplay } from '@/components/ConfigPanel/DeploymentAnalysis/AnalysisResultDisplay'
import { ChartsPanel } from '@/components/ConfigPanel/DeploymentAnalysis/charts'
import { PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig } from '@/utils/llmDeployment/types'

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

// 实验状态配置
const statusConfig: Record<string, { color: string; text: string; icon: React.ReactNode }> = {
  completed: {
    color: 'success',
    text: '已完成',
    icon: <CheckCircleOutlined />,
  },
  running: {
    color: 'processing',
    text: '运行中',
    icon: <SyncOutlined spin />,
  },
  failed: {
    color: 'error',
    text: '失败',
    icon: <CloseCircleOutlined />,
  },
  pending: {
    color: 'warning',
    text: '待运行',
    icon: <ClockCircleOutlined />,
  },
}

// 颜色配置
const colors = {
  primary: '#1890ff',
  success: '#52c41a',
  warning: '#faad14',
  error: '#ff4d4f',
  border: '#e8e8e8',
  textSecondary: '#8c8c8c',
}

export const Results: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedExperimentId, setSelectedExperimentId] = useState<number | null>(null)
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null)

  // 任务分析相关状态
  const [selectedTask, setSelectedTask] = useState<EvaluationTask | null>(null)
  const [taskResults, setTaskResults] = useState<TaskResultsResponse | null>(null)
  const [taskResultsLoading, setTaskResultsLoading] = useState(false)

  // 加载实验列表
  const loadExperiments = async () => {
    setLoading(true)
    try {
      const data = await listExperiments()
      setExperiments(data || [])
    } catch (error) {
      message.error('加载实验列表失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 首次加载
  useEffect(() => {
    loadExperiments()
  }, [])

  // 计算实验状态
  const getExperimentStatus = (exp: Experiment): string => {
    if (exp.total_tasks === 0) return 'pending'
    if (exp.completed_tasks === 0) return 'running'
    if (exp.completed_tasks === exp.total_tasks) return 'completed'
    return 'running'
  }

  // 加载实验详情
  const loadExperimentDetail = async (id: number) => {
    try {
      const data = await getExperimentDetail(id)
      setSelectedExperiment(data)
      setSelectedExperimentId(id)

      // 如果只有一个已完成的任务，自动加载该任务的分析结果
      if (data.tasks && data.tasks.length === 1 && data.tasks[0].status === 'completed') {
        const task = data.tasks[0]
        setSelectedTask(task)
        setTaskResultsLoading(true)
        try {
          const results = await getTaskResults(task.task_id)
          setTaskResults(results)
        } catch (err) {
          console.error('自动加载任务结果失败:', err)
          setTaskResults(null)
        } finally {
          setTaskResultsLoading(false)
        }
      }
    } catch (error) {
      message.error('加载实验详情失败')
      console.error(error)
    }
  }

  // 删除实验
  const handleDelete = async (id: number) => {
    try {
      await deleteExperiment(id)
      message.success('实验已删除')
      loadExperiments()
      // 如果删除的是当前查看的实验，返回列表
      if (selectedExperimentId === id) {
        setSelectedExperimentId(null)
        setSelectedExperiment(null)
      }
    } catch (error) {
      message.error('删除失败')
      console.error(error)
    }
  }

  // 加载任务结果
  const loadTaskResults = async (task: EvaluationTask) => {
    setSelectedTask(task)
    setTaskResultsLoading(true)
    try {
      const results = await getTaskResults(task.task_id)
      setTaskResults(results)
    } catch (error) {
      message.error('加载任务结果失败')
      console.error(error)
      setTaskResults(null)
    } finally {
      setTaskResultsLoading(false)
    }
  }

  // 返回任务列表
  const handleBackToTasks = () => {
    setSelectedTask(null)
    setTaskResults(null)
  }

  // 将 API 返回的 top_k_plans 转换为 PlanAnalysisResult[]
  const convertToAnalysisResults = (results: TaskResultsResponse | null): PlanAnalysisResult[] => {
    if (!results || !results.top_k_plans) return []
    return results.top_k_plans.map(plan => ({
      is_feasible: plan.is_feasible,
      plan: {
        plan_id: `plan_${plan.chips}_${plan.parallelism.tp}_${plan.parallelism.ep}`,
        total_chips: plan.chips,
        parallelism: plan.parallelism,
      },
      latency: {
        prefill_total_latency_ms: plan.ttft,
        decode_per_token_latency_ms: plan.tpot,
        end_to_end_latency_ms: plan.ttft + plan.tpot * 100,
        bottleneck_type: 'balanced' as const,
      },
      throughput: {
        tokens_per_second: plan.throughput,
        tps_per_chip: plan.tps_per_chip,
        tps_per_batch: plan.throughput,
        model_flops_utilization: plan.mfu,
        memory_bandwidth_utilization: plan.mbu,
      },
      memory: {
        total_per_chip_gb: 0,
        is_memory_sufficient: true,
      },
      communication: {},
      utilization: {},
      score: {
        overall_score: plan.score,
        latency_score: 0,
        throughput_score: 0,
        efficiency_score: 0,
        balance_score: 0,
      },
      suggestions: [],
    } as unknown as PlanAnalysisResult))
  }

  // 表格列配置
  const columns: ColumnsType<Experiment> = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      width: 300,
      ellipsis: true,
      align: 'center',
      render: (text, record) => (
        <Tooltip title="点击查看详情">
          <span
            style={{ color: '#1890ff', cursor: 'pointer' }}
            onClick={() => loadExperimentDetail(record.id)}
          >
            {text}
          </span>
        </Tooltip>
      ),
    },
    {
      title: '任务进度',
      key: 'progress',
      width: 150,
      align: 'center',
      render: (_, record) => {
        const progress =
          record.total_tasks > 0
            ? Math.round((record.completed_tasks / record.total_tasks) * 100)
            : 0
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
            <div style={{ width: 60, height: 6, backgroundColor: '#f0f0f0', borderRadius: 3 }}>
              <div
                style={{
                  width: `${progress}%`,
                  height: '100%',
                  backgroundColor: progress === 100 ? '#52c41a' : '#1890ff',
                  borderRadius: 3,
                  transition: 'width 0.3s',
                }}
              />
            </div>
            <span style={{ fontSize: 12, minWidth: 35 }}>
              {progress}%
            </span>
          </div>
        )
      },
    },
    {
      title: '任务数',
      key: 'tasks',
      width: 100,
      align: 'center',
      render: (_, record) => (
        <span>
          {record.completed_tasks}/{record.total_tasks}
        </span>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      align: 'center',
      render: (text) =>
        text ? new Date(text).toLocaleString('zh-CN') : '-',
    },
    {
      title: '状态',
      key: 'status',
      width: 120,
      align: 'center',
      render: (_, record) => {
        const status = getExperimentStatus(record)
        const config = statusConfig[status] || { color: 'default', text: status, icon: null }
        return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>
      },
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      width: 250,
      align: 'center',
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text || '-'}>
          <span>{text || '-'}</span>
        </Tooltip>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      align: 'center',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="link"
              size="small"
              icon={<BarChartOutlined />}
              onClick={() => loadExperimentDetail(record.id)}
            />
          </Tooltip>
          <Popconfirm
            title="确定删除此实验吗？"
            description="删除后将无法恢复"
            onConfirm={() => handleDelete(record.id)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]


  // 如果选中了实验，显示详情视图
  if (selectedExperiment) {
    const modelConfig = selectedExperiment.model_config as Record<string, unknown> || {}
    const inferenceConfig = selectedExperiment.inference_config as Record<string, unknown> || {}
    const hardwareConfig = selectedExperiment.hardware_config as unknown as HardwareConfig | undefined
    const progress =
      selectedExperiment.total_tasks > 0
        ? Math.round((selectedExperiment.completed_tasks / selectedExperiment.total_tasks) * 100)
        : 0

    // 如果选中了任务，显示任务分析视图
    if (selectedTask) {
      const analysisResults = convertToAnalysisResults(taskResults)
      const bestResult = analysisResults.length > 0 ? analysisResults[0] : null

      return (
        <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
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
              <Button
                type="text"
                icon={<ArrowLeftOutlined />}
                onClick={handleBackToTasks}
              >
                返回任务列表
              </Button>
              <div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#1a1a1a' }}>
                  任务分析结果
                </div>
                <Text type="secondary">
                  任务 ID: {selectedTask.task_id.slice(0, 8)}... ·
                  共 {taskResults?.top_k_plans?.length || 0} 个评估结果
                </Text>
              </div>
            </Space>
          </div>

          {/* 内容区 - 使用 AnalysisResultDisplay + ChartsPanel */}
          <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
            <div style={{ maxWidth: 1600, margin: '0 auto', width: '100%' }}>
              <AnalysisResultDisplay
                result={bestResult}
                topKPlans={analysisResults}
                loading={taskResultsLoading}
                viewMode="detail"
                hardware={hardwareConfig}
                model={modelConfig as unknown as LLMModelConfig}
                inference={inferenceConfig as unknown as InferenceConfig}
                onSelectPlan={(plan) => {
                  // 切换选中的方案
                  const idx = analysisResults.findIndex(p => p.plan?.plan_id === plan.plan?.plan_id)
                  if (idx >= 0) {
                    // 可以在这里添加选中效果
                    console.log('Selected plan:', idx)
                  }
                }}
              />
              {/* 图表可视化面板 - 包含雷达图、柱状图、饼图、Roofline、甘特图 */}
              {bestResult && (
                <div style={{ marginTop: 16 }}>
                  <ChartsPanel
                    result={bestResult}
                    topKPlans={analysisResults}
                    hardware={hardwareConfig!}
                    model={modelConfig as unknown as LLMModelConfig}
                    inference={inferenceConfig as unknown as InferenceConfig}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )
    }

    return (
      <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
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
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={() => {
                setSelectedExperimentId(null)
                setSelectedExperiment(null)
              }}
            >
              返回
            </Button>
            <div>
              <div style={{ fontSize: 18, fontWeight: 600, color: '#1a1a1a' }}>
                {selectedExperiment.name}
              </div>
              <Text type="secondary">{selectedExperiment.description || '无描述'}</Text>
            </div>
          </Space>
          <Space>
            <Tooltip title="刷新">
              <Button
                icon={<ReloadOutlined />}
                onClick={() => {
                  if (selectedExperimentId) {
                    loadExperimentDetail(selectedExperimentId)
                  }
                }}
              />
            </Tooltip>
          </Space>
        </div>

        {/* 内容区 */}
        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          <div style={{ maxWidth: 1600, margin: '0 auto', width: '100%' }}>
          {/* 进度提示 */}
          {progress < 100 && (
            <Alert
              message={`实验进度: ${progress}% (${selectedExperiment.completed_tasks}/${selectedExperiment.total_tasks} 任务完成)`}
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {/* 任务进度卡片 */}
          <Card title="任务进度" style={{ marginBottom: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="已完成任务"
                  value={selectedExperiment.completed_tasks}
                  suffix={`/ ${selectedExperiment.total_tasks}`}
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

          {/* 任务列表 */}
          <Card
            title={
              <Space>
                <span>任务列表 ({selectedExperiment.tasks?.length || 0})</span>
                <Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                  点击任务查看详细分析结果
                </Text>
              </Space>
            }
            style={{ marginBottom: 16 }}
          >
            {selectedExperiment.tasks && selectedExperiment.tasks.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
                {selectedExperiment.tasks.map((task) => {
                  const taskStatusConfig = statusConfig[task.status] || { color: 'default', text: task.status, icon: null }
                  const searchStats = task.search_stats as { total?: number; feasible?: number; infeasible?: number } | undefined

                  return (
                    <Card
                      key={task.task_id}
                      size="small"
                      hoverable={task.status === 'completed'}
                      onClick={() => task.status === 'completed' && loadTaskResults(task)}
                      style={{
                        cursor: task.status === 'completed' ? 'pointer' : 'default',
                        borderColor: task.status === 'completed' ? colors.success : colors.border,
                        opacity: task.status === 'completed' ? 1 : 0.7,
                      }}
                    >
                      {/* 标题行：任务ID + 状态 */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <Text strong style={{ fontSize: 13 }}>
                          {task.task_id.slice(0, 8)}...
                        </Text>
                        <Tag color={taskStatusConfig.color} icon={taskStatusConfig.icon} style={{ margin: 0 }}>
                          {taskStatusConfig.text}
                        </Tag>
                      </div>

                      {/* 进度条 */}
                      <div style={{ marginBottom: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>进度</Text>
                          <Text style={{ fontSize: 11 }}>{task.progress.toFixed(0)}%</Text>
                        </div>
                        <div style={{ width: '100%', height: 4, backgroundColor: '#f0f0f0', borderRadius: 2 }}>
                          <div
                            style={{
                              width: `${task.progress}%`,
                              height: '100%',
                              backgroundColor: task.status === 'completed' ? colors.success : colors.primary,
                              borderRadius: 2,
                              transition: 'width 0.3s',
                            }}
                          />
                        </div>
                      </div>

                      {/* 搜索统计 */}
                      {searchStats && (
                        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                          {searchStats.total !== undefined && (
                            <Tag color="default" style={{ margin: 0, fontSize: 10 }}>
                              总计 {searchStats.total}
                            </Tag>
                          )}
                          {searchStats.feasible !== undefined && searchStats.feasible > 0 && (
                            <Tag color="success" style={{ margin: 0, fontSize: 10 }}>
                              可行 {searchStats.feasible}
                            </Tag>
                          )}
                          {searchStats.infeasible !== undefined && searchStats.infeasible > 0 && (
                            <Tag color="warning" style={{ margin: 0, fontSize: 10 }}>
                              不可行 {searchStats.infeasible}
                            </Tag>
                          )}
                        </div>
                      )}

                      {/* 创建时间 */}
                      <div style={{ fontSize: 11, color: colors.textSecondary }}>
                        {new Date(task.created_at).toLocaleString('zh-CN')}
                      </div>

                      {/* 查看分析提示 */}
                      {task.status === 'completed' && (
                        <div style={{
                          marginTop: 8,
                          paddingTop: 8,
                          borderTop: `1px dashed ${colors.border}`,
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center',
                          gap: 4,
                        }}>
                          <BarChartOutlined style={{ color: colors.primary, fontSize: 12 }} />
                          <Text style={{ color: colors.primary, fontSize: 11 }}>点击查看详细分析</Text>
                        </div>
                      )}

                      {/* 错误消息 */}
                      {task.error && (
                        <Alert
                          message={task.error}
                          type="error"
                          style={{ marginTop: 8, fontSize: 11 }}
                          showIcon
                        />
                      )}
                    </Card>
                  )
                })}
              </div>
            ) : (
              <Alert message="暂无任务" type="info" />
            )}
          </Card>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', background: '#fff' }}>
      {/* 内容区 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
        {/* 实验列表 */}
        <Card
          title={
            <Space>
              <span>实验列表</span>
            </Space>
          }
          extra={
            <Tooltip title="刷新">
              <Button icon={<ReloadOutlined />} onClick={loadExperiments} size="small" />
            </Tooltip>
          }
        >
          <Spin spinning={loading}>
            <Table
              columns={columns}
              dataSource={experiments}
              rowKey="id"
              loading={loading}
              pagination={{
                defaultPageSize: 20,
                pageSizeOptions: [10, 20, 50],
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => (
                  <Text type="secondary">共 {total} 个实验</Text>
                ),
              }}
              locale={{
                emptyText: (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description="暂无实验数据"
                  />
                ),
              }}
            />
          </Spin>
        </Card>
      </div>
    </div>
  )
}
