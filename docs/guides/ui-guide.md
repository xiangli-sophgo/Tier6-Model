# UI间距设计规范

## 概述
本文档定义了Tier6+Model项目中卡片组件的统一间距规范，确保整体UI的一致性和视觉和谐。

## 设计原则
1. **视觉层次**：不同层级的间距应该反映内容的层级关系
2. **呼吸感**：合理的间距让界面更加舒适，避免拥挤
3. **一致性**：相同层级的元素使用相同的间距

## 间距规范

### 1. 卡片间距

#### 大卡片间距（顶级容器）
- **间距值**：`16px` (Tailwind: `gap-4` 或 `space-y-4`)
- **CSS变量**：`--spacing-card-gap-large`
- **使用场景**：
  - ConfigPanel中的"拓扑汇总"、"层级配置"、"Switch配置"之间
  - 同级别的BaseCard组件之间
  - 主要内容区域的分组

**示例：**
```tsx
<div className="flex flex-col gap-4">
  <BaseCard title="拓扑汇总">...</BaseCard>
  <BaseCard title="层级配置">...</BaseCard>
  <BaseCard title="Switch配置">...</BaseCard>
</div>
```

#### 小卡片间距（嵌套容器）
- **间距值**：`12px` (Tailwind: `gap-3` 或 `space-y-3`)
- **CSS变量**：`--spacing-card-gap-small`
- **使用场景**：
  - TabsContent内部的BaseCard之间
  - 同一大卡片内的多个配置section
  - 列表项之间

**示例：**
```tsx
<TabsContent value="datacenter">
  <div className="space-y-3">
    <BaseCard title="节点配置">...</BaseCard>
    <BaseCard title="连接配置">...</BaseCard>
  </div>
</TabsContent>
```

### 2. 卡片内边距

#### 标准内边距
- **间距值**：`16px` (Tailwind: `p-4`)
- **CSS变量**：`--spacing-card-padding`
- **使用场景**：
  - BaseCard的contentClassName默认值
  - 大容器的内边距

#### 紧凑内边距
- **间距值**：`12px` (Tailwind: `p-3`)
- **CSS变量**：`--spacing-card-padding-compact`
- **使用场景**：
  - 空间受限的小卡片
  - 嵌套较深的内容区域

### 3. 内容元素间距

#### 表单行间距
- **间距值**：`12px` (Tailwind: `gap-3`)
- **使用场景**：表单字段之间、配置项之间

#### 文本段落间距
- **间距值**：`8px` (Tailwind: `gap-2` 或 `space-y-2`)
- **使用场景**：文本段落、描述信息

## 交互效果

### 悬浮效果（优化后）
```css
.card:hover {
  transform: translateY(-2px);  /* 轻微上移 2px（原4px过大） */
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08),
              0 2px 4px rgba(37, 99, 235, 0.04);  /* 柔和阴影 */
}
```

**优化说明：**
- ✅ 移动距离：从 `-4px` 减少到 `-2px`，更加微妙
- ✅ 阴影尺寸：从 `0 12px 24px` 减少到 `0 4px 12px`，更加柔和
- ✅ 过渡时间：从 `0.3s` 减少到 `0.2s`，响应更快

### 按钮悬浮效果
```css
button:hover:not(:disabled) {
  transform: translateY(-1px);  /* 按钮只需1px，比卡片更轻微 */
}
```

## 应用示例

### ConfigPanel布局
```tsx
// 外层：大卡片间距 gap-4
<div className="flex flex-col gap-4">
  {/* 顶级卡片 */}
  <BaseCard title="拓扑汇总">
    {/* 内容区：默认 p-2.5 (10px) */}
  </BaseCard>

  <BaseCard title="层级配置">
    {/* Tab内容：小卡片间距 space-y-3 */}
    <Tabs>
      <TabsContent value="datacenter">
        <div className="space-y-3">
          <BaseCard title="节点配置">...</BaseCard>
          <BaseCard title="连接配置">...</BaseCard>
        </div>
      </TabsContent>
    </Tabs>
  </BaseCard>
</div>
```

### 嵌套层级示例
```
外层容器 (flex flex-col gap-4)
├── 大卡片1 (BaseCard)
│   └── 内容 (p-2.5)
│       └── 小容器 (space-y-3)
│           ├── 小卡片1 (BaseCard)
│           │   └── 表单内容 (gap-3)
│           └── 小卡片2 (BaseCard)
├── [16px间距]
├── 大卡片2 (BaseCard)
└── [16px间距]
└── 大卡片3 (BaseCard)
```

## 快速参考

| 用途 | Tailwind类 | 像素值 | CSS变量 |
|-----|-----------|--------|---------|
| 大卡片间距 | `gap-4` / `space-y-4` | 16px | `--spacing-card-gap-large` |
| 小卡片间距 | `gap-3` / `space-y-3` | 12px | `--spacing-card-gap-small` |
| 卡片内边距 | `p-4` | 16px | `--spacing-card-padding` |
| 紧凑内边距 | `p-3` | 12px | `--spacing-card-padding-compact` |
| BaseCard默认 | `p-2.5` | 10px | - |
| 表单行间距 | `gap-3` | 12px | - |
| 文本段落 | `gap-2` | 8px | - |

## 更新记录

### 2026-02-01
- 创建间距规范文档
- 优化卡片悬浮效果（移动距离和阴影）
- 统一ConfigPanel的大卡片间距（gap-4）
- 统一TabsContent内部小卡片间距（space-y-3）
- 添加CSS变量支持
