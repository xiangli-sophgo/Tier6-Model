/**
 * 部署分析页面
 * 配置和运行 LLM 部署评估任务
 */

import React from 'react'
import { Row, Col, Card, Spin, Typography, Alert, Progress, Button } from 'antd'
import {
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  StopOutlined
} from '@ant-design/icons'
import { DeploymentAnalysisPanel } from '@/components/ConfigPanel/DeploymentAnalysis'
import { useWorkbench } from '@/contexts/WorkbenchContext'

const { Text } = Typography

export const DeploymentAnalysis: React.FC = () => {
  const { analysis, topology } = useWorkbench()

  // 运行状态
  const loading = analysis.deploymentAnalysisData?.loading ?? false
  const errorMsg = analysis.deploymentAnalysisData?.errorMsg ?? null
  const searchProgress = analysis.deploymentAnalysisData?.searchProgress
  const onCancelEvaluation = analysis.deploymentAnalysisData?.onCancelEvaluation

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
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
            部署分析
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c' }}>
            配置并运行 LLM 部署评估任务
          </span>
        </div>
      </div>

      {/* 主内容区 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
        <div style={{ maxWidth: 1400, margin: '0 auto' }}>
          {/* 配置面板 - 暂时保持原有布局，待后续重构 */}
          <div style={{ padding: 16, background: '#fff', borderRadius: 8 }}>
            <DeploymentAnalysisPanel
              topology={topology.topology}
              onTrafficResultChange={analysis.setTrafficResult}
              onAnalysisDataChange={analysis.setDeploymentAnalysisData}
              rackConfig={topology.rackConfig}
              podCount={topology.podCount}
              racksPerPod={topology.racksPerPod}
              history={analysis.analysisHistory}
              onAddToHistory={analysis.handleAddToHistory}
              onDeleteHistory={analysis.handleDeleteHistory}
              onClearHistory={analysis.handleClearHistory}
            />
          </div>

          {/* 下方：运行状态卡片 */}
          {(loading || errorMsg || (searchProgress && searchProgress.stage !== 'idle' && searchProgress.stage !== 'completed')) && (
            <Card
              title="运行状态"
              bordered={false}
              style={{ marginTop: 16 }}
            >
              {/* 错误提示 */}
              {errorMsg && (
                <Alert
                  message="分析失败"
                  description={errorMsg}
                  type="error"
                  showIcon
                  icon={<CloseCircleOutlined />}
                  style={{ marginBottom: 16 }}
                />
              )}

              {/* 搜索进度 */}
              {loading && searchProgress && searchProgress.stage !== 'idle' && (
                <div>
                  {/* 阶段 1: 生成候选方案 */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                    {searchProgress.stage === 'generating' ? (
                      <Spin size="small" indicator={<LoadingOutlined />} />
                    ) : (
                      <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 16 }} />
                    )}
                    <Text style={{ fontSize: 14 }}>
                      生成候选方案: <Text strong>{searchProgress.totalCandidates}</Text> 个
                    </Text>
                  </div>

                  {/* 阶段 2: 后端评估 */}
                  {searchProgress.stage !== 'generating' && (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          {searchProgress.stage === 'evaluating' ? (
                            <Spin size="small" indicator={<LoadingOutlined />} />
                          ) : (
                            <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 16 }} />
                          )}
                          <Text style={{ fontSize: 14 }}>
                            后端评估: <Text strong>{searchProgress.evaluated}</Text> / <Text strong>{searchProgress.totalCandidates}</Text>
                            {searchProgress.stage === 'evaluating' && (
                              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>（5 并发）</Text>
                            )}
                          </Text>
                        </div>
                        {/* 取消按钮 */}
                        {searchProgress.stage === 'evaluating' && onCancelEvaluation && (
                          <Button
                            danger
                            size="small"
                            icon={<StopOutlined />}
                            onClick={onCancelEvaluation}
                          >
                            取消
                          </Button>
                        )}
                      </div>
                      {searchProgress.stage === 'evaluating' && (
                        <Progress
                          percent={Math.round((searchProgress.evaluated / searchProgress.totalCandidates) * 100)}
                          status="active"
                          strokeColor="#1890ff"
                        />
                      )}
                    </div>
                  )}

                  {/* 已取消 */}
                  {searchProgress.stage === 'cancelled' && (
                    <Alert
                      message="评估已取消"
                      description={`已完成 ${searchProgress.evaluated} / ${searchProgress.totalCandidates} 个方案的评估`}
                      type="warning"
                      showIcon
                      style={{ marginTop: 12 }}
                    />
                  )}
                </div>
              )}

              {/* 简单加载中状态（无进度信息） */}
              {loading && (!searchProgress || searchProgress.stage === 'idle') && (
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <Spin size="large" />
                  <div style={{ marginTop: 12, color: '#8c8c8c' }}>正在分析...</div>
                </div>
              )}
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
