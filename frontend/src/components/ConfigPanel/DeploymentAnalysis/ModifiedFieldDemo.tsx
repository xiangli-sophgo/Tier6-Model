/**
 * 修改字段视觉方案演示组件
 * 展示三种不同的修改标识方案（匹配实际的网格布局）
 */

import React, { useState } from 'react'
import { NumberInput } from '@/components/ui/number-input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { BaseCard } from '@/components/common/BaseCard'
import { Badge } from '@/components/ui/badge'

export const ModifiedFieldDemo: React.FC = () => {
  // 方案1的状态
  const [batchSize1, setBatchSize1] = useState(16)
  const [inputLen1, setInputLen1] = useState(512)
  const [outputLen1, setOutputLen1] = useState(256)
  const [numLayers1, setNumLayers1] = useState(32)

  // 方案2的状态
  const [batchSize2, setBatchSize2] = useState(16)
  const [inputLen2, setInputLen2] = useState(512)
  const [outputLen2, setOutputLen2] = useState(256)
  const [numLayers2, setNumLayers2] = useState(32)

  // 方案3的状态
  const [batchSize3, setBatchSize3] = useState(16)
  const [inputLen3, setInputLen3] = useState(512)
  const [outputLen3, setOutputLen3] = useState(256)
  const [numLayers3, setNumLayers3] = useState(32)

  // 原始值
  const originalBatchSize = 8
  const originalInputLen = 512
  const originalOutputLen = 128
  const originalNumLayers = 32

  return (
    <div className="space-y-6 p-6 bg-gray-50">
      <div className="text-center mb-6">
        <h2 className="text-xl font-bold mb-2">修改字段视觉方案对比</h2>
        <p className="text-sm text-gray-600">原始配置: Batch Size = 8, Input Length = 512, Output Length = 128, Num Layers = 32</p>
        <p className="text-xs text-gray-500 mt-1">请修改下面的值，观察不同方案的视觉效果</p>
      </div>

      {/* 方案1：蓝色边框高亮 */}
      <BaseCard title="方案1：蓝色边框高亮（输入框级别）" gradient>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <div className="mb-1 text-[13px] text-gray-600">Batch Size</div>
            <NumberInput
              min={1}
              max={128}
              value={batchSize1}
              onChange={(v) => setBatchSize1(v || 8)}
              className="w-full"
              style={
                batchSize1 !== originalBatchSize
                  ? {
                      borderColor: '#3b82f6',
                      borderWidth: '2px',
                      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
                    }
                  : undefined
              }
            />
          </div>
          <div>
            <div className="mb-1 text-[13px] text-gray-600">Input Length</div>
            <NumberInput
              min={1}
              max={4096}
              value={inputLen1}
              onChange={(v) => setInputLen1(v || 512)}
              className="w-full"
              style={
                inputLen1 !== originalInputLen
                  ? {
                      borderColor: '#3b82f6',
                      borderWidth: '2px',
                      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
                    }
                  : undefined
              }
            />
          </div>
          <div>
            <div className="mb-1 text-[13px] text-gray-600">Output Length</div>
            <NumberInput
              min={1}
              max={4096}
              value={outputLen1}
              onChange={(v) => setOutputLen1(v || 128)}
              className="w-full"
              style={
                outputLen1 !== originalOutputLen
                  ? {
                      borderColor: '#3b82f6',
                      borderWidth: '2px',
                      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
                    }
                  : undefined
              }
            />
          </div>
          <div>
            <div className="mb-1 text-[13px] text-gray-600">Num Layers</div>
            <NumberInput
              min={1}
              max={256}
              value={numLayers1}
              onChange={(v) => setNumLayers1(v || 32)}
              className="w-full"
              style={
                numLayers1 !== originalNumLayers
                  ? {
                      borderColor: '#3b82f6',
                      borderWidth: '2px',
                      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)',
                    }
                  : undefined
              }
            />
          </div>
        </div>
        <div className="text-xs text-gray-500 pt-3 border-t mt-3">
          ✓ 视觉最直接，修改的输入框有明显蓝色边框<br/>
          ✓ 不占额外空间<br/>
          - 可能与表单验证错误样式混淆
        </div>
      </BaseCard>

      {/* 方案2：字段容器级别高亮 */}
      <BaseCard title="方案2：字段容器级别高亮（整个字段区域）" gradient>
        <div className="grid grid-cols-2 gap-2">
          <div
            className={`p-2 rounded -m-2 mb-0 ${
              batchSize2 !== originalBatchSize ? 'bg-blue-50/50 border-l-4 border-l-blue-500' : ''
            }`}
          >
            <div className="mb-1 text-[13px] text-gray-600">Batch Size</div>
            <NumberInput
              min={1}
              max={128}
              value={batchSize2}
              onChange={(v) => setBatchSize2(v || 8)}
              className="w-full"
            />
          </div>
          <div
            className={`p-2 rounded -m-2 mb-0 ${
              inputLen2 !== originalInputLen ? 'bg-blue-50/50 border-l-4 border-l-blue-500' : ''
            }`}
          >
            <div className="mb-1 text-[13px] text-gray-600">Input Length</div>
            <NumberInput
              min={1}
              max={4096}
              value={inputLen2}
              onChange={(v) => setInputLen2(v || 512)}
              className="w-full"
            />
          </div>
          <div
            className={`p-2 rounded -m-2 mb-0 ${
              outputLen2 !== originalOutputLen ? 'bg-blue-50/50 border-l-4 border-l-blue-500' : ''
            }`}
          >
            <div className="mb-1 text-[13px] text-gray-600">Output Length</div>
            <NumberInput
              min={1}
              max={4096}
              value={outputLen2}
              onChange={(v) => setOutputLen2(v || 128)}
              className="w-full"
            />
          </div>
          <div
            className={`p-2 rounded -m-2 mb-0 ${
              numLayers2 !== originalNumLayers ? 'bg-blue-50/50 border-l-4 border-l-blue-500' : ''
            }`}
          >
            <div className="mb-1 text-[13px] text-gray-600">Num Layers</div>
            <NumberInput
              min={1}
              max={256}
              value={numLayers2}
              onChange={(v) => setNumLayers2(v || 32)}
              className="w-full"
            />
          </div>
        </div>
        <div className="text-xs text-gray-500 pt-3 border-t mt-3">
          ✓ 优雅简洁，整个字段区域有淡蓝背景<br/>
          ✓ 左侧蓝色竖线作为强烈视觉提示<br/>
          ✓ 不干扰输入控件本身
        </div>
      </BaseCard>

      {/* 方案3：徽章 + 整体背景 */}
      <BaseCard title="方案3：徽章提示 + 整体淡蓝背景" gradient>
        <div className="grid grid-cols-2 gap-2">
          <div className={`p-2 rounded -m-2 mb-0 ${batchSize3 !== originalBatchSize ? 'bg-blue-50/50' : ''}`}>
            <div className="mb-1 text-[13px] text-gray-600 flex items-center gap-1.5">
              Batch Size
              {batchSize3 !== originalBatchSize && (
                <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">
                  已修改
                </Badge>
              )}
            </div>
            <NumberInput
              min={1}
              max={128}
              value={batchSize3}
              onChange={(v) => setBatchSize3(v || 8)}
              className="w-full"
            />
          </div>
          <div className={`p-2 rounded -m-2 mb-0 ${inputLen3 !== originalInputLen ? 'bg-blue-50/50' : ''}`}>
            <div className="mb-1 text-[13px] text-gray-600 flex items-center gap-1.5">
              Input Length
              {inputLen3 !== originalInputLen && (
                <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">
                  已修改
                </Badge>
              )}
            </div>
            <NumberInput
              min={1}
              max={4096}
              value={inputLen3}
              onChange={(v) => setInputLen3(v || 512)}
              className="w-full"
            />
          </div>
          <div className={`p-2 rounded -m-2 mb-0 ${outputLen3 !== originalOutputLen ? 'bg-blue-50/50' : ''}`}>
            <div className="mb-1 text-[13px] text-gray-600 flex items-center gap-1.5">
              Output Length
              {outputLen3 !== originalOutputLen && (
                <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">
                  已修改
                </Badge>
              )}
            </div>
            <NumberInput
              min={1}
              max={4096}
              value={outputLen3}
              onChange={(v) => setOutputLen3(v || 128)}
              className="w-full"
            />
          </div>
          <div className={`p-2 rounded -m-2 mb-0 ${numLayers3 !== originalNumLayers ? 'bg-blue-50/50' : ''}`}>
            <div className="mb-1 text-[13px] text-gray-600 flex items-center gap-1.5">
              Num Layers
              {numLayers3 !== originalNumLayers && (
                <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">
                  已修改
                </Badge>
              )}
            </div>
            <NumberInput
              min={1}
              max={256}
              value={numLayers3}
              onChange={(v) => setNumLayers3(v || 32)}
              className="w-full"
            />
          </div>
        </div>
        <div className="text-xs text-gray-500 pt-3 border-t mt-3">
          ✓ 信息最明确，标签旁直接显示"已修改"徽章<br/>
          ✓ 整个字段区域淡蓝背景，视觉清晰<br/>
          ✓ 输入框保持原样，不干扰交互
        </div>
      </BaseCard>

      {/* 对比总结 */}
      <BaseCard title="方案对比总结" gradient>
        <div className="space-y-2 text-sm">
          <div className="grid grid-cols-3 gap-4 font-semibold text-gray-700 pb-2 border-b">
            <div>方案1</div>
            <div>方案2</div>
            <div>方案3</div>
          </div>
          <div className="grid grid-cols-3 gap-4 text-xs">
            <div className="text-gray-600">
              <span className="text-green-600">✓</span> 视觉最直接
              <br />
              <span className="text-green-600">✓</span> 不占额外空间
              <br />
              <span className="text-yellow-600">-</span> 可能与验证错误样式混淆
            </div>
            <div className="text-gray-600">
              <span className="text-green-600">✓</span> 优雅简洁
              <br />
              <span className="text-green-600">✓</span> 视觉层次清晰
              <br />
              <span className="text-green-600">✓</span> 不干扰输入控件
            </div>
            <div className="text-gray-600">
              <span className="text-green-600">✓</span> 信息最丰富
              <br />
              <span className="text-yellow-600">-</span> 占用空间稍多
              <br />
              <span className="text-yellow-600">-</span> 视觉稍显复杂
            </div>
          </div>
        </div>
      </BaseCard>
    </div>
  )
}
