/**
 * 算子详情侧边栏
 *
 * 点击甘特图任务时显示完整性能信息
 */

import React from 'react'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import type { GanttTask, GanttTaskExtended } from '../../../utils/llmDeployment/types'
import {
  formatBytes,
  formatTime,
  formatFlops,
  formatPercent,
  formatGemmShape,
  formatTimeMs,
  formatPercentValue,
} from '../../../utils/formatters'
import {
  getTaskCategory,
  TIME_BREAKDOWN_COLORS,
} from '../../../utils/llmDeployment/ganttDataUtils'

interface TaskDetailDrawerProps {
  task: GanttTask | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** 任务类型名称映射 */
const TASK_TYPE_LABELS: Record<string, string> = {
  compute: '计算',
  embedding: 'Embedding',
  layernorm: 'LayerNorm',
  attention_qkv: 'QKV投影',
  attention_score: '注意力得分',
  attention_softmax: 'Softmax',
  attention_output: '注意力输出',
  ffn_gate: 'FFN Gate',
  ffn_up: 'FFN Up',
  ffn_down: 'FFN Down',
  lm_head: 'LM Head',
  pcie_h2d: 'PCIe H2D',
  pcie_d2h: 'PCIe D2H',
  hbm_write: 'HBM 写入',
  hbm_read: 'HBM 读取',
  weight_load: '权重加载',
  kv_cache_read: 'KV Cache 读取',
  kv_cache_write: 'KV Cache 写入',
  tp_comm: 'TP 通信',
  pp_comm: 'PP 通信',
  ep_comm: 'EP 通信',
  sp_allgather: 'SP AllGather',
  sp_reduce_scatter: 'SP ReduceScatter',
  dp_gradient_sync: 'DP 梯度同步',
  moe_gate: 'MoE 路由',
  moe_expert: 'MoE 专家',
  moe_shared_expert: 'MoE 共享专家',
  ep_dispatch: 'EP Dispatch',
  ep_combine: 'EP Combine',
  bubble: '气泡',
  idle: '空闲',
  rmsnorm_q_lora: 'RMSNorm Q LoRA',
  rmsnorm_kv_lora: 'RMSNorm KV LoRA',
  mm_q_lora_a: 'MM Q LoRA A',
  mm_q_lora_b: 'MM Q LoRA B',
  mm_kv_lora_a: 'MM KV LoRA A',
  attn_fc: 'Attention FC',
  bmm_qk: 'BMM QK',
  bmm_sv: 'BMM SV',
}

/** 阶段名称映射 */
const PHASE_LABELS: Record<string, string> = {
  prefill: 'Prefill',
  decode: 'Decode',
}

/** 任务类别徽章变体映射 */
const CATEGORY_BADGE_VARIANT: Record<string, 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning' | 'processing'> = {
  compute: 'success',
  memory: 'processing',
  tp: 'default',
  pp: 'destructive',
  ep: 'warning',
  sp: 'secondary',
  other: 'outline',
}

/** 信息行组件 */
const InfoRow: React.FC<{
  label: string
  value: React.ReactNode
  highlight?: boolean
}> = ({ label, value, highlight }) => (
  <div className="flex items-center justify-between py-1.5">
    <span className="text-sm text-text-muted">{label}</span>
    <span className={`text-sm font-medium ${highlight ? 'text-primary' : 'text-text-primary'}`}>
      {value}
    </span>
  </div>
)

/** 时间条形图组件 */
const TimeBar: React.FC<{
  computeTime: number
  memoryTime: number
  commTime: number
  totalTime: number
}> = ({ computeTime, memoryTime, commTime, totalTime }) => {
  if (totalTime === 0) return null

  const computePercent = (computeTime / totalTime) * 100
  const memoryPercent = (memoryTime / totalTime) * 100
  const commPercent = (commTime / totalTime) * 100

  return (
    <div className="space-y-2">
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-bg-surface">
        {computePercent > 0 && (
          <div
            className="h-full transition-all"
            style={{ width: `${computePercent}%`, backgroundColor: TIME_BREAKDOWN_COLORS.compute }}
            title={`计算: ${formatPercentValue(computePercent, 1)}`}
          />
        )}
        {memoryPercent > 0 && (
          <div
            className="h-full transition-all"
            style={{ width: `${memoryPercent}%`, backgroundColor: TIME_BREAKDOWN_COLORS.memory }}
            title={`访存: ${formatPercentValue(memoryPercent, 1)}`}
          />
        )}
        {commPercent > 0 && (
          <div
            className="h-full transition-all"
            style={{ width: `${commPercent}%`, backgroundColor: TIME_BREAKDOWN_COLORS.tp }}
            title={`通信: ${formatPercentValue(commPercent, 1)}`}
          />
        )}
      </div>
      <div className="flex gap-4 text-xs text-text-muted">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: TIME_BREAKDOWN_COLORS.compute }} />
          计算 {formatPercentValue(computePercent, 1)}
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: TIME_BREAKDOWN_COLORS.memory }} />
          访存 {formatPercentValue(memoryPercent, 1)}
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: TIME_BREAKDOWN_COLORS.tp }} />
          通信 {formatPercentValue(commPercent, 1)}
        </span>
      </div>
    </div>
  )
}

export const TaskDetailDrawer: React.FC<TaskDetailDrawerProps> = ({
  task,
  open,
  onOpenChange,
}) => {
  if (!task) return null

  const extTask = task as GanttTaskExtended
  const category = getTaskCategory(task.type)
  const duration = (task.end - task.start) * 1000 // ms to us

  // 计算时间分解
  const computeTime = extTask.compute_time_us ?? (category === 'compute' ? duration : 0)
  const memoryTime = extTask.memory_time_us ?? (category === 'memory' ? duration : 0)
  const commTime = extTask.comm_time_us ?? (['tp', 'pp', 'ep', 'sp'].includes(category) ? duration : 0)
  const totalTime = computeTime + memoryTime + commTime || duration

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[400px] overflow-y-auto sm:max-w-[450px]">
        <SheetHeader className="mb-4">
          <SheetTitle className="flex items-center gap-2">
            {TASK_TYPE_LABELS[task.type] || task.type}
            <Badge variant={CATEGORY_BADGE_VARIANT[category]}>
              {category === 'compute' ? '计算' :
               category === 'memory' ? '访存' :
               category === 'tp' ? 'TP通信' :
               category === 'pp' ? 'PP通信' :
               category === 'ep' ? 'EP通信' :
               category === 'sp' ? 'SP通信' : '其他'}
            </Badge>
          </SheetTitle>
          <SheetDescription>
            {task.name}
          </SheetDescription>
        </SheetHeader>

        {/* 基本信息 */}
        <div className="space-y-4">
          <div>
            <h4 className="mb-2 text-sm font-semibold text-text-primary">基本信息</h4>
            <div className="rounded-lg bg-bg-surface p-3">
              <InfoRow label="任务类型" value={task.type} />
              <InfoRow label="阶段" value={PHASE_LABELS[task.phase] || task.phase} />
              {task.layer !== undefined && (
                <InfoRow label="层索引" value={task.layer} />
              )}
              {task.ppStage !== undefined && (
                <InfoRow label="PP Stage" value={task.ppStage} />
              )}
              <InfoRow label="资源" value={task.resourceId} />
            </div>
          </div>

          <Separator />

          {/* 时间分解 */}
          <div>
            <h4 className="mb-2 text-sm font-semibold text-text-primary">时间分解</h4>
            <div className="rounded-lg bg-bg-surface p-3">
              <InfoRow
                label="开始时间"
                value={formatTimeMs(task.start)}
              />
              <InfoRow
                label="结束时间"
                value={formatTimeMs(task.end)}
              />
              <InfoRow
                label="总持续时间"
                value={formatTime(duration)}
                highlight
              />
              {extTask.compute_time_us !== undefined && (
                <InfoRow label="计算时间" value={formatTime(extTask.compute_time_us)} />
              )}
              {extTask.memory_time_us !== undefined && (
                <InfoRow label="访存时间" value={formatTime(extTask.memory_time_us)} />
              )}
              {extTask.comm_time_us !== undefined && (
                <InfoRow label="通信时间" value={formatTime(extTask.comm_time_us)} />
              )}
            </div>

            {/* 时间占比条形图 */}
            <div className="mt-3">
              <TimeBar
                computeTime={computeTime}
                memoryTime={memoryTime}
                commTime={commTime}
                totalTime={totalTime}
              />
            </div>
          </div>

          {/* 计算详情 (仅计算类型任务显示) */}
          {category === 'compute' && (extTask.flops || extTask.gemm_shape || extTask.best_tile) && (
            <>
              <Separator />
              <div>
                <h4 className="mb-2 text-sm font-semibold text-text-primary">计算详情</h4>
                <div className="rounded-lg bg-bg-surface p-3">
                  {extTask.flops !== undefined && (
                    <InfoRow label="计算量" value={formatFlops(extTask.flops)} highlight />
                  )}
                  {extTask.gemm_shape && (
                    <InfoRow label="GEMM 形状" value={formatGemmShape(extTask.gemm_shape)} />
                  )}
                  {extTask.best_tile && (
                    <InfoRow label="最优 Tile" value={extTask.best_tile} />
                  )}
                  {extTask.best_partition && (
                    <InfoRow label="最优分区" value={extTask.best_partition} />
                  )}
                </div>
              </div>
            </>
          )}

          {/* 利用率 */}
          {(extTask.arch_utilization !== undefined || extTask.effective_utilization !== undefined) && (
            <>
              <Separator />
              <div>
                <h4 className="mb-2 text-sm font-semibold text-text-primary">利用率</h4>
                <div className="rounded-lg bg-bg-surface p-3 space-y-3">
                  {extTask.arch_utilization !== undefined && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-text-muted">架构利用率</span>
                        <span className="text-sm font-medium text-text-primary">
                          {formatPercent(extTask.arch_utilization)}
                        </span>
                      </div>
                      <Progress value={extTask.arch_utilization * 100} />
                    </div>
                  )}
                  {extTask.effective_utilization !== undefined && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-text-muted">有效利用率</span>
                        <span className="text-sm font-medium text-text-primary">
                          {formatPercent(extTask.effective_utilization)}
                        </span>
                      </div>
                      <Progress value={extTask.effective_utilization * 100} />
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* 内存信息 */}
          {(extTask.dram_traffic_bytes !== undefined || extTask.dram_occupy_bytes !== undefined) && (
            <>
              <Separator />
              <div>
                <h4 className="mb-2 text-sm font-semibold text-text-primary">内存</h4>
                <div className="rounded-lg bg-bg-surface p-3">
                  {extTask.dram_traffic_bytes !== undefined && (
                    <InfoRow label="DRAM 流量" value={formatBytes(extTask.dram_traffic_bytes)} />
                  )}
                  {extTask.dram_occupy_bytes !== undefined && (
                    <InfoRow label="DRAM 占用" value={formatBytes(extTask.dram_occupy_bytes)} />
                  )}
                </div>
              </div>
            </>
          )}

          {/* 通信信息 (仅通信类型任务显示) */}
          {['tp', 'pp', 'ep', 'sp'].includes(category) && (
            <>
              <Separator />
              <div>
                <h4 className="mb-2 text-sm font-semibold text-text-primary">通信</h4>
                <div className="rounded-lg bg-bg-surface p-3">
                  {extTask.comm_size_bytes !== undefined && (
                    <InfoRow label="通信数据量" value={formatBytes(extTask.comm_size_bytes)} highlight />
                  )}
                  {extTask.comm_algorithm && (
                    <InfoRow label="通信算法" value={extTask.comm_algorithm} />
                  )}
                  {extTask.comm_group_size !== undefined && (
                    <InfoRow label="通信组大小" value={extTask.comm_group_size} />
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}

export default TaskDetailDrawer
