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
  Row,
  Col,
  Statistic,
  Input,
  Modal,
  Upload,
  Divider,
  Checkbox,
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
  EditOutlined,
  SaveOutlined,
  CloseOutlined,
  FileTextOutlined,
  DownloadOutlined,
  UploadOutlined,
  InboxOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { listExperiments, deleteExperiment, deleteExperimentsBatch, getExperimentDetail, getTaskResults, updateExperiment, downloadExperimentJSON, checkImportFile, executeImport, Experiment, EvaluationTask, TaskResultsResponse } from '@/api/results'
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

  // 编辑状态
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editingName, setEditingName] = useState('')
  const [editingDescription, setEditingDescription] = useState('')
  const [editingLoading, setEditingLoading] = useState(false)

  // 批量选择状态
  const [selectedExperimentIds, setSelectedExperimentIds] = useState<number[]>([])

  // 导入导出状态
  const [exportModalVisible, setExportModalVisible] = useState(false)
  const [importModalVisible, setImportModalVisible] = useState(false)
  const [importStep, setImportStep] = useState<'upload' | 'config' | 'importing' | 'result'>('upload')
  const [importFile, setImportFile] = useState<File | null>(null)
  const [importCheckResult, setImportCheckResult] = useState<any>(null)
  const [importConfig, setImportConfig] = useState<Map<string, any>>(new Map())
  const [importLoading, setImportLoading] = useState(false)
  const [importResult, setImportResult] = useState<any>(null)

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

  // 开始编辑
  const handleStartEdit = (record: Experiment) => {
    setEditingId(record.id)
    setEditingName(record.name)
    setEditingDescription(record.description || '')
  }

  // 保存编辑
  const handleSaveEdit = async (id: number) => {
    if (!editingName.trim()) {
      message.error('实验名称不能为空')
      return
    }
    setEditingLoading(true)
    try {
      await updateExperiment(id, {
        name: editingName.trim(),
        description: editingDescription.trim() || undefined,
      })
      message.success('实验已更新')
      loadExperiments()
      setEditingId(null)
    } catch (error) {
      message.error('更新失败')
      console.error(error)
    } finally {
      setEditingLoading(false)
    }
  }

  // 取消编辑
  const handleCancelEdit = () => {
    setEditingId(null)
    setEditingName('')
    setEditingDescription('')
  }

  // 批量删除
  const handleBatchDelete = async () => {
    if (selectedExperimentIds.length === 0) {
      message.warning('请先选择要删除的实验')
      return
    }
    try {
      await deleteExperimentsBatch(selectedExperimentIds)
      message.success(`成功删除 ${selectedExperimentIds.length} 个实验`)
      setSelectedExperimentIds([])
      loadExperiments()
    } catch (error) {
      message.error('批量删除失败')
      console.error(error)
    }
  }

  // 导出实验
  const handleExport = async () => {
    try {
      const exportIds = selectedExperimentIds.length > 0 ? selectedExperimentIds : undefined
      await downloadExperimentJSON(exportIds)
      message.success('导出成功')
      setExportModalVisible(false)
      setSelectedExperimentIds([])
    } catch (error) {
      message.error('导出失败')
      console.error(error)
    }
  }

  // 处理导入文件上传
  const handleImportFile = async (file: File) => {
    setImportFile(file)
    setImportLoading(true)
    try {
      const result = await checkImportFile(file)
      if (result.valid && result.experiments) {
        setImportCheckResult(result)
        // 初始化导入配置
        const config = new Map()
        for (const exp of result.experiments) {
          config.set(exp.name, {
            original_name: exp.name,
            action: exp.conflict ? 'rename' : 'rename',
            new_name: exp.conflict ? `${exp.name}_imported` : exp.name,
          })
        }
        setImportConfig(config)
        setImportStep('config')
      } else {
        message.error(result.error || '导入文件无效')
      }
    } catch (error) {
      message.error('检查导入文件失败')
      console.error(error)
    } finally {
      setImportLoading(false)
    }
  }

  // 执行导入
  const handleExecuteImport = async () => {
    if (!importCheckResult?.temp_file_id) {
      message.error('导入会话已过期，请重新上传')
      return
    }

    setImportLoading(true)
    setImportStep('importing')
    try {
      const configs = Array.from(importConfig.values())
      const result = await executeImport(importCheckResult.temp_file_id, configs)
      setImportResult(result)
      setImportStep('result')
      if (result.success) {
        message.success(result.message)
        loadExperiments()
      }
    } catch (error) {
      message.error('导入失败')
      console.error(error)
    } finally {
      setImportLoading(false)
    }
  }

  // 重置导入对话框
  const resetImportModal = () => {
    setImportModalVisible(false)
    setImportStep('upload')
    setImportFile(null)
    setImportCheckResult(null)
    setImportConfig(new Map())
    setImportResult(null)
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
        total_per_chip_gb: plan.dram_occupy ? plan.dram_occupy / (1024 * 1024 * 1024) : 0,  // 字节转 GB
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
      width: 280,
      render: (text, record) => {
        if (editingId === record.id) {
          return (
            <Input
              value={editingName}
              onChange={(e) => setEditingName(e.target.value)}
              placeholder="实验名称"
              autoFocus
              onPressEnter={() => handleSaveEdit(record.id)}
              onBlur={() => {
                // 按 Escape 取消编辑
              }}
            />
          )
        }
        return (
          <Tooltip title="点击查看详情，右键编辑">
            <span
              style={{ color: '#1890ff', cursor: 'pointer' }}
              onClick={() => loadExperimentDetail(record.id)}
              onContextMenu={(e) => {
                e.preventDefault()
                handleStartEdit(record)
              }}
            >
              {text}
            </span>
          </Tooltip>
        )
      },
    },
    {
      title: '任务数',
      key: 'tasks',
      width: 80,
      align: 'center',
      render: (_, record) => {
        const taskCount = record.tasks?.length || 0
        const completedCount = record.tasks?.filter((t: EvaluationTask) => t.status === 'completed').length || 0
        return (
          <span>
            {completedCount}/{taskCount}
          </span>
        )
      },
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      align: 'center',
      render: (text) =>
        text ? new Date(text).toLocaleString('zh-CN') : '-',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      width: 200,
      ellipsis: true,
      render: (text, record) => {
        if (editingId === record.id) {
          return (
            <Input.TextArea
              value={editingDescription}
              onChange={(e) => setEditingDescription(e.target.value)}
              placeholder="实验描述"
              rows={2}
              autoFocus={editingName !== ''}
            />
          )
        }
        return (
          <Tooltip title={text || '无描述'}>
            <span>{text || '-'}</span>
          </Tooltip>
        )
      },
    },
    {
      title: '操作',
      key: 'action',
      width: 160,
      align: 'center',
      render: (_, record) => {
        if (editingId === record.id) {
          return (
            <Space size="small">
              <Button
                type="primary"
                size="small"
                icon={<SaveOutlined />}
                loading={editingLoading}
                onClick={() => handleSaveEdit(record.id)}
              />
              <Button
                size="small"
                icon={<CloseOutlined />}
                onClick={handleCancelEdit}
              />
            </Space>
          )
        }
        return (
          <Space size="small">
            <Tooltip title="查看详情">
              <Button
                type="link"
                size="small"
                icon={<BarChartOutlined />}
                onClick={() => loadExperimentDetail(record.id)}
              />
            </Tooltip>
            <Tooltip title="编辑">
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleStartEdit(record)}
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
        )
      },
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
            extra={
              <Space>
                <Button size="small" onClick={() => loadExperimentDetail(selectedExperimentId!)}>
                  <ReloadOutlined /> 刷新
                </Button>
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
                  message.info('仅已完成的任务可查看详细结果')
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
    <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
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
            <Space>
              <Tooltip title="导出">
                <Button icon={<DownloadOutlined />} onClick={() => setExportModalVisible(true)} size="small" />
              </Tooltip>
              <Tooltip title="导入">
                <Button icon={<UploadOutlined />} onClick={() => setImportModalVisible(true)} size="small" />
              </Tooltip>
              <Tooltip title="刷新">
                <Button icon={<ReloadOutlined />} onClick={loadExperiments} size="small" />
              </Tooltip>
            </Space>
          }
        >
          <Spin spinning={loading}>
            {selectedExperimentIds.length > 0 && (
              <div style={{ marginBottom: 16, padding: '8px 12px', background: '#e6f7ff', borderRadius: 4 }}>
                <Space>
                  <span>已选择 {selectedExperimentIds.length} 个实验</span>
                  <Button
                    type="link"
                    onClick={() => setSelectedExperimentIds([])}
                  >
                    取消选择
                  </Button>
                  <Popconfirm
                    title="确定删除选中的实验吗？"
                    description={`将删除 ${selectedExperimentIds.length} 个实验，此操作无法恢复`}
                    onConfirm={handleBatchDelete}
                  >
                    <Button danger type="primary" size="small">
                      删除选中
                    </Button>
                  </Popconfirm>
                </Space>
              </div>
            )}
            <Table
              columns={columns}
              dataSource={experiments}
              rowKey="id"
              loading={loading}
              rowSelection={{
                selectedRowKeys: selectedExperimentIds,
                onChange: (selectedKeys) => {
                  setSelectedExperimentIds(selectedKeys as number[])
                },
              }}
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

      {/* 导出模态框 */}
      <Modal
        title="导出实验"
        open={exportModalVisible}
        onOk={handleExport}
        onCancel={() => setExportModalVisible(false)}
        okText="导出"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          {selectedExperimentIds.length > 0 ? (
            <div>
              <div>已选择 {selectedExperimentIds.length} 个实验</div>
              <Button
                type="link"
                size="small"
                onClick={() => setSelectedExperimentIds([])}
              >
                导出全部实验
              </Button>
            </div>
          ) : (
            <div>将导出所有 {experiments.length} 个实验的配置信息</div>
          )}
        </div>
      </Modal>

      {/* 导入模态框 */}
      <Modal
        title="导入实验"
        open={importModalVisible}
        onCancel={resetImportModal}
        footer={null}
        width={800}
      >
        {importStep === 'upload' && (
          <div>
            <Upload.Dragger
              accept=".json"
              maxCount={1}
              beforeUpload={(file) => {
                handleImportFile(file)
                return false
              }}
              disabled={importLoading}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽 JSON 文件到此区域上传</p>
              <p className="ant-upload-hint">支持导出的实验配置文件</p>
            </Upload.Dragger>
          </div>
        )}

        {importStep === 'config' && importCheckResult && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <div>检测到 {importCheckResult.experiments?.length || 0} 个实验</div>
            </div>
            <div style={{ maxHeight: 400, overflowY: 'auto', marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                {importCheckResult.experiments?.map((exp: any, idx: number) => (
                  <Card key={idx} size="small">
                    <div style={{ marginBottom: 8 }}>
                      <strong>实验名称：</strong> {exp.name}
                    </div>
                    {exp.description && (
                      <div style={{ marginBottom: 8 }}>
                        <strong>描述：</strong> {exp.description}
                      </div>
                    )}
                    <div style={{ marginBottom: 8 }}>
                      <strong>任务数：</strong> {exp.completed_tasks}/{exp.total_tasks}
                    </div>
                    {exp.conflict && (
                      <Alert
                        message="名称冲突"
                        description={`与现有实验 "${exp.name}" 重名`}
                        type="warning"
                        style={{ marginBottom: 8 }}
                      />
                    )}
                    <div>
                      <Checkbox.Group value={[importConfig.get(exp.name)?.action || 'rename']}>
                        <Space direction="vertical">
                          <Checkbox
                            value="rename"
                            onChange={() => {
                              const config = new Map(importConfig)
                              config.set(exp.name, {
                                original_name: exp.name,
                                action: 'rename',
                                new_name: exp.conflict ? `${exp.name}_imported` : exp.name,
                              })
                              setImportConfig(config)
                            }}
                          >
                            重命名导入
                            {exp.conflict && (
                              <Input
                                placeholder="新名称"
                                style={{ width: 200, marginLeft: 8 }}
                                value={importConfig.get(exp.name)?.new_name || ''}
                                onChange={(e) => {
                                  const config = new Map(importConfig)
                                  const item = config.get(exp.name) || {}
                                  item.new_name = e.target.value
                                  config.set(exp.name, item)
                                  setImportConfig(config)
                                }}
                              />
                            )}
                          </Checkbox>
                          {!exp.conflict && (
                            <>
                              <Checkbox
                                value="skip"
                                onChange={() => {
                                  const config = new Map(importConfig)
                                  config.set(exp.name, {
                                    original_name: exp.name,
                                    action: 'skip',
                                  })
                                  setImportConfig(config)
                                }}
                              >
                                跳过
                              </Checkbox>
                            </>
                          )}
                          {exp.conflict && (
                            <Checkbox
                              value="overwrite"
                              onChange={() => {
                                const config = new Map(importConfig)
                                config.set(exp.name, {
                                  original_name: exp.name,
                                  action: 'overwrite',
                                })
                                setImportConfig(config)
                              }}
                            >
                              覆盖现有实验
                            </Checkbox>
                          )}
                        </Space>
                      </Checkbox.Group>
                    </div>
                  </Card>
                ))}
              </Space>
            </div>
            <Space>
              <Button onClick={() => setImportStep('upload')}>返回</Button>
              <Button
                type="primary"
                onClick={handleExecuteImport}
                loading={importLoading}
              >
                导入
              </Button>
            </Space>
          </div>
        )}

        {importStep === 'importing' && (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin tip="正在导入..." />
          </div>
        )}

        {importStep === 'result' && importResult && (
          <div>
            <Alert
              message={importResult.success ? '导入成功' : '导入失败'}
              description={importResult.message}
              type={importResult.success ? 'success' : 'error'}
              style={{ marginBottom: 16 }}
            />
            <div style={{ marginBottom: 16 }}>
              <div>导入成功：{importResult.imported_count} 个</div>
              <div>跳过：{importResult.skipped_count} 个</div>
              <div>覆盖：{importResult.overwritten_count} 个</div>
            </div>
            <Button type="primary" onClick={resetImportModal} block>
              完成
            </Button>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default Results
