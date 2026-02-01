/**
 * BaseCard 功能展示页面
 * 展示统一卡片组件的各种用法
 */

import React, { useState } from 'react'
import { BaseCard } from '@/components/common/BaseCard'
import { Settings, Database } from 'lucide-react'

export default function CardPlayground() {
  const [expanded1, setExpanded1] = useState(true)
  const [expanded2, setExpanded2] = useState(false)

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">BaseCard 功能展示</h1>
          <p className="text-gray-600">统一卡片组件的所有用法示例</p>
        </div>

        {/* 1. 基础用法 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            1. 基础用法
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard title="标准卡片" subtitle="这是副标题">
              <p className="text-sm">这是一个基础的卡片内容区域。</p>
            </BaseCard>

            <BaseCard
              title="带图标的卡片"
              icon={<Settings className="h-4 w-4 text-blue-600" />}
            >
              <p className="text-sm">左侧显示图标</p>
            </BaseCard>
          </div>
        </div>

        {/* 2. 渐变模式 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            2. 渐变模式 (gradient)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard title="渐变卡片" gradient>
              <p className="text-sm">蓝色渐变背景标题</p>
              <p className="text-xs text-gray-500">bg-gradient-to-r from-blue-50 to-white</p>
            </BaseCard>

            <BaseCard
              title="带副标题"
              subtitle="副标题信息"
              gradient
              icon={<Database className="h-4 w-4 text-blue-600" />}
            >
              <p className="text-sm">渐变模式 + 图标 + 副标题</p>
            </BaseCard>
          </div>
        </div>

        {/* 3. 可折叠模式 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            3. 可折叠模式 (collapsible)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard
              title="非受控折叠"
              collapsible
              defaultExpanded={true}
              gradient
            >
              <p className="text-sm">点击标题栏可以折叠/展开</p>
              <p className="text-xs text-gray-500">使用 defaultExpanded</p>
            </BaseCard>

            <BaseCard
              title="受控折叠"
              collapsible
              expanded={expanded1}
              onExpandChange={setExpanded1}
              gradient
            >
              <p className="text-sm">受控模式，状态由外部管理</p>
              <p className="text-xs text-gray-500">expanded + onExpandChange</p>
            </BaseCard>
          </div>
        </div>

        {/* 4. 折叠计数 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            4. 折叠计数显示 (collapsibleCount)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard
              title="手动连接"
              collapsible
              defaultExpanded
              collapsibleCount={5}
              gradient
            >
              <p className="text-sm">标题后显示计数 (5)</p>
              <p className="text-xs text-gray-500">适合列表类内容</p>
            </BaseCard>

            <BaseCard
              title="当前连接"
              collapsible
              expanded={expanded2}
              onExpandChange={setExpanded2}
              collapsibleCount={12}
              gradient
            >
              <p className="text-sm">受控模式 + 计数显示</p>
              <p className="text-xs text-gray-500">collapsibleCount={12}</p>
            </BaseCard>
          </div>
        </div>

        {/* 5. 编辑按钮 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            5. 编辑按钮 (onEdit)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard
              title="集群规模"
              collapsible
              defaultExpanded
              collapsibleCount={4}
              onEdit={() => alert('编辑集群规模')}
              gradient
            >
              <p className="text-sm">右上角显示编辑按钮</p>
              <p className="text-xs text-gray-500">点击编辑按钮触发回调</p>
            </BaseCard>

            <BaseCard
              title="硬件配置"
              collapsible
              defaultExpanded
              collapsibleCount={8}
              onEdit={() => alert('编辑硬件配置')}
              editLabel="修改"
              gradient
            >
              <p className="text-sm">自定义编辑按钮文字</p>
              <p className="text-xs text-gray-500">editLabel="修改"</p>
            </BaseCard>
          </div>
        </div>

        {/* 6. Titleless模式 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            6. Titleless模式 - 完全自定义
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard
              titleless
              className="cursor-pointer hover:shadow-md"
            >
              <div className="px-4 py-8 text-center">
                <Settings className="h-12 w-12 text-blue-600 mx-auto mb-3" />
                <h3 className="font-semibold text-lg mb-2">快速操作</h3>
                <p className="text-sm text-gray-500">无标题栏，完全自定义布局</p>
              </div>
            </BaseCard>

            <BaseCard
              titleless
              glassmorphism
              className="cursor-pointer hover:shadow-md"
            >
              <div className="px-4 py-8 text-center">
                <Database className="h-12 w-12 text-purple-600 mx-auto mb-3" />
                <h3 className="font-semibold text-lg mb-2">毛玻璃效果</h3>
                <p className="text-sm text-gray-500">glassmorphism + titleless</p>
              </div>
            </BaseCard>
          </div>
        </div>

        {/* 7. 内容区自定义样式 */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            7. 内容区自定义 (contentClassName)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard
              title="滚动内容"
              collapsible
              defaultExpanded
              contentClassName="max-h-40 overflow-auto p-2"
              gradient
            >
              <div className="space-y-2">
                {[...Array(20)].map((_, i) => (
                  <p key={i} className="text-sm">条目 {i + 1}</p>
                ))}
              </div>
            </BaseCard>

            <BaseCard
              title="无内边距"
              collapsible
              defaultExpanded
              contentClassName="p-0"
              gradient
            >
              <div className="divide-y">
                <div className="px-4 py-2 hover:bg-gray-50">选项 1</div>
                <div className="px-4 py-2 hover:bg-gray-50">选项 2</div>
                <div className="px-4 py-2 hover:bg-gray-50">选项 3</div>
              </div>
            </BaseCard>
          </div>
        </div>

        {/* 8. Glassmorphism */}
        <div>
          <h2 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            8. 毛玻璃效果 (glassmorphism)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <BaseCard
              title="基础信息"
              collapsible
              defaultExpanded
              collapsibleCount={3}
              glassmorphism
              className="border-gray-100"
            >
              <p className="text-sm">半透明白色背景</p>
              <p className="text-xs text-gray-500">bg-white/80 backdrop-blur-sm</p>
            </BaseCard>

            <BaseCard
              title="高级配置"
              collapsible
              defaultExpanded
              glassmorphism
              gradient
              onEdit={() => alert('编辑')}
            >
              <p className="text-sm">毛玻璃 + 渐变 + 编辑按钮</p>
              <p className="text-xs text-gray-500">组合所有功能</p>
            </BaseCard>
          </div>
        </div>

        {/* Props 总结 */}
        <div className="mt-8 p-6 bg-white rounded-lg border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">BaseCard Props 总结</h2>
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-3 gap-4 font-semibold border-b pb-2">
              <div>Props</div>
              <div>类型</div>
              <div>说明</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">title</div>
              <div className="text-gray-600">ReactNode</div>
              <div className="text-gray-600">标题内容</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">subtitle</div>
              <div className="text-gray-600">string</div>
              <div className="text-gray-600">副标题</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">icon</div>
              <div className="text-gray-600">ReactNode</div>
              <div className="text-gray-600">标题图标</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">gradient</div>
              <div className="text-gray-600">boolean</div>
              <div className="text-gray-600">蓝色渐变标题</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">collapsible</div>
              <div className="text-gray-600">boolean</div>
              <div className="text-gray-600">可折叠</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">defaultExpanded</div>
              <div className="text-gray-600">boolean</div>
              <div className="text-gray-600">默认展开</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">expanded</div>
              <div className="text-gray-600">boolean</div>
              <div className="text-gray-600">受控展开状态</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b">
              <div className="font-mono text-xs">onExpandChange</div>
              <div className="text-gray-600">function</div>
              <div className="text-gray-600">展开状态回调</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b bg-blue-50">
              <div className="font-mono text-xs">contentClassName</div>
              <div className="text-gray-600">string</div>
              <div className="text-gray-600 font-semibold">内容区自定义样式 ⭐</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b bg-blue-50">
              <div className="font-mono text-xs">titleless</div>
              <div className="text-gray-600">boolean</div>
              <div className="text-gray-600 font-semibold">无标题模式 ⭐</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b bg-blue-50">
              <div className="font-mono text-xs">glassmorphism</div>
              <div className="text-gray-600">boolean</div>
              <div className="text-gray-600 font-semibold">毛玻璃效果 ⭐</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b bg-blue-50">
              <div className="font-mono text-xs">collapsibleCount</div>
              <div className="text-gray-600">number</div>
              <div className="text-gray-600 font-semibold">折叠区计数 ⭐</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2 border-b bg-blue-50">
              <div className="font-mono text-xs">onEdit</div>
              <div className="text-gray-600">function</div>
              <div className="text-gray-600 font-semibold">编辑按钮回调 ⭐</div>
            </div>
            <div className="grid grid-cols-3 gap-4 py-2">
              <div className="font-mono text-xs">editLabel</div>
              <div className="text-gray-600">string</div>
              <div className="text-gray-600">编辑按钮文字</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
