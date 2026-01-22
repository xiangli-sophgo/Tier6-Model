/**
 * 配置快照展示组件
 * 使用 Collapse + Descriptions 展示完整的配置信息
 */

import React from 'react'
import { Collapse, Descriptions, Tag, Typography, Empty } from 'antd'
import { FileTextOutlined, DatabaseOutlined, ApiOutlined, SettingOutlined } from '@ant-design/icons'

const { Panel } = Collapse
const { Text } = Typography

interface ConfigSnapshotDisplayProps {
  configSnapshot: {
    model: Record<string, unknown>
    inference: Record<string, unknown>
    topology: Record<string, unknown>
  }
  benchmarkName?: string
  topologyConfigName?: string
}

export const ConfigSnapshotDisplay: React.FC<ConfigSnapshotDisplayProps> = ({
  configSnapshot,
  benchmarkName,
  topologyConfigName,
}) => {
  if (!configSnapshot) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="配置快照不可用"
        style={{ padding: '24px 0' }}
      />
    )
  }

  const { model, inference, topology } = configSnapshot

  // 提取拓扑延迟配置
  const protocolConfig = (topology as any).protocol_config
  const networkConfig = (topology as any).network_config || (topology as any).network_infra_config
  const chipLatencyConfig = (topology as any).chip_latency_config

  // 计算总芯片数
  const calculateTotalChips = () => {
    const pods = (topology as any).pods || []
    let total = 0
    for (const pod of pods) {
      for (const rack of pod.racks || []) {
        for (const board of rack.boards || []) {
          total += (board.chips || []).length
        }
      }
    }
    return total
  }

  const totalChips = calculateTotalChips()

  return (
    <Collapse
      defaultActiveKey={['benchmark']}
      style={{
        background: '#fafafa',
        border: '1px solid #e8e8e8',
        borderRadius: 8,
      }}
    >
      {/* Benchmark 配置 */}
      <Panel
        header={
          <span>
            <FileTextOutlined style={{ marginRight: 8, color: '#1890ff' }} />
            <Text strong>Benchmark 配置</Text>
            {benchmarkName && (
              <Tag color="blue" style={{ marginLeft: 8 }}>
                {benchmarkName}
              </Tag>
            )}
          </span>
        }
        key="benchmark"
      >
        <Descriptions column={2} size="small" bordered>
          <Descriptions.Item label="模型名称" span={2}>
            {(model as any).model_name || '-'}
          </Descriptions.Item>
          <Descriptions.Item label="层数">{(model as any).num_layers || '-'}</Descriptions.Item>
          <Descriptions.Item label="隐藏维度">{(model as any).hidden_size || '-'}</Descriptions.Item>
          <Descriptions.Item label="注意力头数">{(model as any).num_attention_heads || '-'}</Descriptions.Item>
          <Descriptions.Item label="KV头数">{(model as any).num_key_value_heads || '-'}</Descriptions.Item>
          <Descriptions.Item label="中间层维度">{(model as any).intermediate_size || '-'}</Descriptions.Item>
          <Descriptions.Item label="词汇表大小">{(model as any).vocab_size || '-'}</Descriptions.Item>

          {/* 推理配置 */}
          <Descriptions.Item label="批次大小">{(inference as any).batch_size || '-'}</Descriptions.Item>
          <Descriptions.Item label="输入序列长度">{(inference as any).input_seq_length || '-'}</Descriptions.Item>
          <Descriptions.Item label="输出序列长度">{(inference as any).output_seq_length || '-'}</Descriptions.Item>
          <Descriptions.Item label="推理模式">{(inference as any).mode || 'decode'}</Descriptions.Item>
        </Descriptions>
      </Panel>

      {/* 拓扑配置 */}
      <Panel
        header={
          <span>
            <DatabaseOutlined style={{ marginRight: 8, color: '#52c41a' }} />
            <Text strong>拓扑配置</Text>
            {topologyConfigName && (
              <Tag color="green" style={{ marginLeft: 8 }}>
                {topologyConfigName}
              </Tag>
            )}
          </span>
        }
        key="topology"
      >
        <Descriptions column={2} size="small" bordered>
          <Descriptions.Item label="Pod 数量">{((topology as any).pods || []).length}</Descriptions.Item>
          <Descriptions.Item label="总芯片数">
            <Tag color="blue">{totalChips}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Rack 数量">
            {((topology as any).pods || []).reduce((sum: number, pod: any) => sum + (pod.racks || []).length, 0)}
          </Descriptions.Item>
          <Descriptions.Item label="Board 数量">
            {((topology as any).pods || []).reduce(
              (sum: number, pod: any) =>
                sum + (pod.racks || []).reduce((s: number, rack: any) => s + (rack.boards || []).length, 0),
              0
            )}
          </Descriptions.Item>
        </Descriptions>

        {/* 芯片硬件信息（如果有） */}
        {(() => {
          const pods = (topology as any).pods || []
          if (pods.length > 0 && pods[0].racks && pods[0].racks[0].boards && pods[0].racks[0].boards[0].chips) {
            const chip = pods[0].racks[0].boards[0].chips[0]
            return (
              <Descriptions column={2} size="small" bordered style={{ marginTop: 12 }}>
                <Descriptions.Item label="芯片型号" span={2}>
                  {chip.name || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="算力 (TFLOPS FP16)">
                  {chip.compute_tflops_fp16 || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="显存 (GB)">{chip.memory_gb || '-'}</Descriptions.Item>
                <Descriptions.Item label="显存带宽 (GB/s)">
                  {chip.memory_bandwidth_gbps || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="显存带宽利用率">
                  {chip.memory_bandwidth_utilization ? `${(chip.memory_bandwidth_utilization * 100).toFixed(0)}%` : '-'}
                </Descriptions.Item>
              </Descriptions>
            )
          }
          return null
        })()}
      </Panel>

      {/* 协议延迟配置 */}
      {protocolConfig && (
        <Panel
          header={
            <span>
              <ApiOutlined style={{ marginRight: 8, color: '#fa8c16' }} />
              <Text strong>协议延迟配置</Text>
            </span>
          }
          key="protocol"
        >
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="TP RTT (μs)">
              {protocolConfig.rtt_tp_us || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="EP RTT (μs)">
              {protocolConfig.rtt_ep_us || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="带宽利用率">
              {protocolConfig.bandwidth_utilization
                ? `${(protocolConfig.bandwidth_utilization * 100).toFixed(0)}%`
                : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="同步延迟 (μs)">
              {protocolConfig.sync_latency_us || '-'}
            </Descriptions.Item>
          </Descriptions>
        </Panel>
      )}

      {/* 网络配置 */}
      {networkConfig && (
        <Panel
          header={
            <span>
              <ApiOutlined style={{ marginRight: 8, color: '#13c2c2' }} />
              <Text strong>网络基础设施配置</Text>
            </span>
          }
          key="network"
        >
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="交换机延迟 (μs)">
              {networkConfig.switch_delay_us || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="线缆延迟 (μs)">
              {networkConfig.cable_delay_us || '-'}
            </Descriptions.Item>
            {networkConfig.link_delay_us && (
              <Descriptions.Item label="链路延迟 (μs)" span={2}>
                {networkConfig.link_delay_us}
              </Descriptions.Item>
            )}
          </Descriptions>
        </Panel>
      )}

      {/* 芯片延迟配置 */}
      {chipLatencyConfig && (
        <Panel
          header={
            <span>
              <SettingOutlined style={{ marginRight: 8, color: '#722ed1' }} />
              <Text strong>芯片延迟配置</Text>
            </span>
          }
          key="chip_latency"
        >
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="C2C 延迟 (μs)">
              {chipLatencyConfig.c2c_lat_us || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="DDR 读延迟 (μs)">
              {chipLatencyConfig.ddr_r_lat_us || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="DDR 写延迟 (μs)">
              {chipLatencyConfig.ddr_w_lat_us || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="NoC 延迟 (μs)">
              {chipLatencyConfig.noc_lat_us || '-'}
            </Descriptions.Item>
            {chipLatencyConfig.d2d_lat_us && (
              <Descriptions.Item label="D2D 延迟 (μs)" span={2}>
                {chipLatencyConfig.d2d_lat_us}
              </Descriptions.Item>
            )}
          </Descriptions>
        </Panel>
      )}
    </Collapse>
  )
}
