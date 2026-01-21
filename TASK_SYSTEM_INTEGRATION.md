# 任务系统集成指南

本文档说明如何将新的任务管理系统集成到 Tier6-Model 项目中。

## 后端已完成的工作

✅ **数据库层** (`backend/llm_simulator/database.py`, `models.py`)
- SQLite 数据库（可轻松切换到 PostgreSQL）
- 三张表：`experiments`, `evaluation_tasks`, `evaluation_results`
- ORM 模型和会话管理

✅ **任务管理器** (`backend/llm_simulator/task_manager.py`)
- 线程池执行器（最多 4 个并发任务）
- 创建、提交、取消、删除任务
- 任务状态更新和结果保存
- WebSocket 广播回调

✅ **WebSocket 管理器** (`backend/llm_simulator/websocket_manager.py`)
- 管理活跃 WebSocket 连接
- 广播任务更新到所有客户端
- 自动清理失效连接

✅ **API 接口** (`backend/llm_simulator/api.py`)
- `POST /api/evaluation/submit` - 提交评估任务
- `GET /api/evaluation/tasks/{task_id}` - 获取任务状态
- `GET /api/evaluation/tasks/{task_id}/results` - 获取任务结果
- `POST /api/evaluation/tasks/{task_id}/cancel` - 取消任务
- `DELETE /api/evaluation/tasks/{task_id}` - 删除任务
- `GET /api/evaluation/tasks` - 获取任务列表
- `GET /api/evaluation/running` - 获取运行中任务
- `GET /api/evaluation/experiments` - 获取实验列表
- `WS /ws/tasks` - WebSocket 实时推送

✅ **依赖项** (`backend/requirements.txt`)
- sqlalchemy>=2.0.0
- websockets>=12.0

## 前端已完成的工作

✅ **组件**
- `TaskStatusCard.tsx` - 运行中任务状态卡片
- `TaskHistoryTable.tsx` - 历史任务表格

✅ **Hooks**
- `useTaskWebSocket.ts` - WebSocket 实时订阅

✅ **API 客户端** (`frontend/src/api/tasks.ts`)
- 封装所有任务管理相关的 HTTP 请求

## 前端集成步骤

### 步骤 1：在 ConfigPanel 中集成任务系统

修改 `frontend/src/components/ConfigPanel/index.tsx`（或部署分析相关组件）：

```tsx
import { useState } from 'react'
import { submitEvaluation, getTaskStatus } from '@/api/tasks'
import { useTaskWebSocket } from '@/hooks/useTaskWebSocket'
import { TaskStatusCard, type TaskStatus } from './DeploymentAnalysis/components/TaskStatusCard'
import { TaskHistoryTable, type TaskHistoryItem } from './DeploymentAnalysis/components/TaskHistoryTable'

export const ConfigPanel = () => {
  // 运行中的任务
  const [runningTasks, setRunningTasks] = useState<Map<string, TaskStatus>>(new Map())

  // 历史任务
  const [taskHistory, setTaskHistory] = useState<TaskHistoryItem[]>([])

  // WebSocket 订阅
  useTaskWebSocket({
    onTaskUpdate: (update) => {
      // 更新运行中任务
      setRunningTasks(prev => {
        const newMap = new Map(prev)
        if (newMap.has(update.task_id)) {
          newMap.set(update.task_id, { ...newMap.get(update.task_id)!, ...update })
        }
        return newMap
      })

      // 任务完成时，从运行中移除并添加到历史
      if (['completed', 'failed', 'cancelled'].includes(update.status)) {
        setTimeout(() => {
          setRunningTasks(prev => {
            const newMap = new Map(prev)
            newMap.delete(update.task_id)
            return newMap
          })
          loadTaskHistory() // 刷新历史列表
        }, 3000)
      }
    }
  })

  // 提交评估任务
  const handleSubmitEvaluation = async () => {
    try {
      const response = await submitEvaluation({
        experiment_name: 'My Experiment',
        description: 'Test evaluation',
        topology: topologyConfig,
        model: modelConfig,
        hardware: hardwareConfig,
        inference: inferenceConfig,
        search_mode: 'auto', // or 'manual'
        search_constraints: searchConstraints,
      })

      // 添加到运行中任务
      const initialTask: TaskStatus = {
        task_id: response.task_id,
        status: 'pending',
        progress: 0,
        message: response.message,
        experiment_name: 'My Experiment',
        created_at: new Date().toISOString(),
      }
      setRunningTasks(prev => new Map(prev).set(response.task_id, initialTask))

      message.success('评估任务已提交')
    } catch (error) {
      message.error('提交失败')
    }
  }

  // 加载任务历史
  const loadTaskHistory = async () => {
    const { tasks } = await getTasks({ limit: 50 })
    setTaskHistory(tasks)
  }

  return (
    <div>
      {/* 配置表单 */}
      <Button onClick={handleSubmitEvaluation}>运行分析</Button>

      {/* 运行中的任务卡片 */}
      {Array.from(runningTasks.values()).map(task => (
        <TaskStatusCard
          key={task.task_id}
          task={task}
          onCancel={() => cancelTask(task.task_id)}
        />
      ))}

      {/* 历史任务表格 */}
      <TaskHistoryTable
        tasks={taskHistory}
        loading={false}
        onRefresh={loadTaskHistory}
        onViewResult={(taskId) => {
          // 查看结果
          getTaskResults(taskId).then(result => {
            // 显示结果...
          })
        }}
        onDelete={(taskId) => deleteTask(taskId).then(loadTaskHistory)}
      />
    </div>
  )
}
```

### 步骤 2：在 App.tsx 中显示全局任务

在 `App.tsx` 的右下角添加一个浮动的任务通知区域：

```tsx
import { useTaskWebSocket } from '@/hooks/useTaskWebSocket'
import { TaskStatusCard } from '@/components/ConfigPanel/DeploymentAnalysis/components/TaskStatusCard'

const App = () => {
  const [globalTasks, setGlobalTasks] = useState<Map<string, TaskStatus>>(new Map())

  useTaskWebSocket({
    onTaskUpdate: (update) => {
      setGlobalTasks(prev => {
        const newMap = new Map(prev)
        newMap.set(update.task_id, { ...newMap.get(update.task_id), ...update })

        // 完成的任务 3 秒后自动消失
        if (['completed', 'failed'].includes(update.status)) {
          setTimeout(() => {
            setGlobalTasks(prev => {
              const newMap = new Map(prev)
              newMap.delete(update.task_id)
              return newMap
            })
          }, 3000)
        }

        return newMap
      })
    }
  })

  return (
    <div>
      {/* 主应用内容 */}

      {/* 全局任务通知（右下角） */}
      {globalTasks.size > 0 && (
        <div style={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          maxWidth: 400,
          zIndex: 1000,
        }}>
          {Array.from(globalTasks.values()).map(task => (
            <TaskStatusCard key={task.task_id} task={task} />
          ))}
        </div>
      )}
    </div>
  )
}
```

### 步骤 3：恢复运行中的任务

在应用启动时，恢复页面刷新前的运行中任务：

```tsx
useEffect(() => {
  const restoreRunningTasks = async () => {
    try {
      const { tasks } = await getRunningTasks()
      const taskMap = new Map()

      for (const task of tasks) {
        const fullStatus = await getTaskStatus(task.task_id)
        taskMap.set(task.task_id, fullStatus)
      }

      setRunningTasks(taskMap)
      if (taskMap.size > 0) {
        message.info(`已恢复 ${taskMap.size} 个运行中的任务`)
      }
    } catch (error) {
      console.error('恢复任务失败:', error)
    }
  }

  restoreRunningTasks()
}, [])
```

## 测试步骤

### 1. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动后端

```bash
cd backend
python main.py
```

启动后会看到：
- 初始化数据库（创建 `backend/data/llm_evaluations.db`）
- 设置 WebSocket 广播回调
- 服务运行在端口 8001

### 3. 测试 API

使用 curl 或 Postman 测试：

```bash
# 提交评估任务
curl -X POST http://localhost:8001/api/evaluation/submit \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "Test Experiment",
    "description": "测试评估",
    "topology": {...},
    "model": {...},
    "hardware": {...},
    "inference": {...},
    "search_mode": "auto",
    "search_constraints": {...}
  }'

# 获取任务状态
curl http://localhost:8001/api/evaluation/tasks/{task_id}

# 获取任务列表
curl http://localhost:8001/api/evaluation/tasks

# 获取运行中任务
curl http://localhost:8001/api/evaluation/running
```

### 4. 测试 WebSocket

使用浏览器控制台或 WebSocket 客户端：

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/tasks')
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  console.log('Task Update:', data)
}
```

### 5. 启动前端

```bash
cd frontend
pnpm dev
```

在浏览器中访问 http://localhost:3100，测试：
- 提交评估任务
- 查看任务状态卡片实时更新
- 刷新页面后任务恢复
- 查看历史任务列表

## 数据库查看

使用 SQLite 客户端查看数据库：

```bash
cd backend/data
sqlite3 llm_evaluations.db

# 查看表结构
.schema

# 查看实验
SELECT * FROM experiments;

# 查看任务
SELECT * FROM evaluation_tasks;

# 查看结果
SELECT * FROM evaluation_results;
```

## 故障排查

### 问题：WebSocket 连接失败

检查：
- 后端是否正常运行
- 端口是否正确（默认 8001）
- CORS 是否配置正确

### 问题：任务提交后没有反应

检查：
- `task_manager.py` 中的 `evaluate_deployment()` 函数是否正确导入
- 查看后端日志，确认任务是否开始执行
- 检查数据库中任务状态

### 问题：页面刷新后任务丢失

检查：
- 是否实现了 `restoreRunningTasks()` 恢复逻辑
- WebSocket 是否成功重连

## 下一步优化

1. **任务优先级**：支持高优先级任务优先执行
2. **任务队列可视化**：显示等待队列长度
3. **批量任务管理**：批量取消、删除任务
4. **实验结果对比**：支持多个实验结果对比
5. **导出功能**：导出任务结果为 JSON/CSV
6. **通知系统**：任务完成时桌面通知

## 总结

✅ **后端完成**：完整的任务管理系统、数据库持久化、WebSocket 实时推送
✅ **前端基础组件**：TaskStatusCard, TaskHistoryTable, useTaskWebSocket, API 客户端
⏳ **待集成**：将组件集成到现有的 ConfigPanel 和 App.tsx 中

按照本文档的步骤进行集成，即可实现完整的任务管理功能。
