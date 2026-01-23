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
import TaskTable from './components/TaskTable'

const { Text } = Typography

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

// 颜色配置 (保留以供其他地方使用)
// const colors = {
//   primary: '#1890ff',
//   success: '#52c41a',
//   warning: '#faad14',
//   error: '#ff4d4f',
//   border: '#e8e8e8',
//   textSecondary: '#8c8c8c',
// }

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
    const progress =
      selectedExperiment.total_tasks > 0
        ? Math.round((selectedExperiment.completed_tasks / selectedExperiment.total_tasks) * 100)
        : 0

    // 如果选中了任务，显示任务分析视图
    if (selectedTask) {
      // 从任务的 config_snapshot 中提取配置
      const modelConfig = selectedTask.config_snapshot?.model as Record<string, unknown> || {}
      const inferenceConfig = selectedTask.config_snapshot?.inference as Record<string, unknown> || {}
      const topology = selectedTask.config_snapshot?.topology as Record<string, unknown> || {}

      // 从拓扑中提取硬件配置（简化版）
      const hardwareConfig: HardwareConfig | undefined = (() => {
        const pods = (topology.pods as any[]) || []
        if (pods.length > 0 && pods[0].racks && pods[0].racks[0].boards && pods[0].racks[0].boards[0].chips) {
          const chip = pods[0].racks[0].boards[0].chips[0]
          return {
            chip: {
              chip_type: chip.name || 'Unknown',
              compute_tflops_fp16: chip.compute_tflops_fp16 || 0,
              memory_gb: chip.memory_gb || 0,
              memory_bandwidth_gbps: chip.memory_bandwidth_gbps || 0,
              memory_bandwidth_utilization: chip.memory_bandwidth_utilization || 0.9,
            },
            node: {
              chips_per_node: 8, // 默认值
              intra_node_bandwidth_gbps: 900,
            },
            cluster: {
              inter_node_bandwidth_gbps: 400,
            },
          } as HardwareConfig
        }
        return undefined
      })()

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
            <div style={{ width: '100%' }}>
              <AnalysisResultDisplay
                result={bestResult}
                topKPlans={analysisResults}
                loading={taskResultsLoading}
                viewMode="detail"
                hardware={hardwareConfig}
                model={modelConfig as unknown as LLMModelConfig}
                inference={inferenceConfig as unknown as InferenceConfig}
                configSnapshot={selectedTask.config_snapshot}
                benchmarkName={selectedTask.benchmark_name}
                topologyConfigName={selectedTask.topology_config_name}
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
          <div style={{ width: '100%' }}>
          {/* 进度提示 */}
          {progress < 100 && (
            <Alert
              message={`实验进度: ${progress}% (${selectedExperiment.completed_tasks}/${selectedExperiment.total_tasks} 任务完成)`}
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {/* 任务列表表格 */}
          <Card
            title={
              <Space>
                <span>任务列表 ({selectedExperiment.tasks?.length || 0})</span>
                <Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                  双击任务查看详细分析结果
                </Text>
              </Space>
            }
            style={{ marginBottom: 16 }}
          >
            <TaskTable
              tasks={selectedExperiment.tasks || []}
              loading={false}
              experimentId={selectedExperiment.id}
              onTaskSelect={(task) => {
                if (task.status === 'completed') {
                  loadTaskResults(task)
                } else {
                  message.info('该任务尚未完成')
                }
              }}
              onTasksDelete={async (taskIds) => {
                // TODO: 实现批量删除任务的API调用
                console.log('删除任务:', taskIds)
                message.info('批量删除功能待实现')
              }}
            />
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
