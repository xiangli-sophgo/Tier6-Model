# 数据库重构总结 - 2层架构迁移

## 概述

将数据库从3层结构 (Experiment → Task → Result) 简化为2层结构 (Experiment → Result)

**迁移时间**: 2026-02-10
**版本**: v2.3.0

---

## 架构变化

### 旧架构 (3层)

```
Experiment (实验)
  └─ EvaluationTask (任务)
      └─ EvaluationResult (结果)
```

- **Experiment**: 实验元数据容器
- **Task**: 中间层，记录任务状态、配置、进度
- **Result**: 评估结果数据

**问题**:
- 中间层Task导致数据冗余和维护复杂度
- Task和Result之间的1:N关系实际使用中大多是1:1
- 删除操作需要同步多个层级，容易出现不一致

### 新架构 (2层)

```
Experiment (实验)
  └─ EvaluationResult (结果)
```

- **Experiment**: 实验元数据容器
- **Result**: 合并了Task字段，直接存储完整评估结果

**优势**:
- 简化数据模型，减少join操作
- 级联删除自动处理清理
- 数据一致性更强
- 更接近CrossRing项目的成熟架构

---

## 数据模型变化

### Experiment 表

**旧字段** (移除):
- `total_tasks` - 静态任务计数（容易不同步）
- `completed_tasks` - 静态完成计数（容易不同步）

**新字段**:
- 保持不变：`id`, `name`, `description`, `created_at`, `updated_at`
- 任务数量改为动态计算：`len(experiment.results)`

### EvaluationResult 表

**新增字段** (从Task迁移):
- `task_id` (String) - UUID字符串，标识任务来源（不再是外键）
- `config_snapshot` (JSON) - 配置快照
- `benchmark_name` (String) - Benchmark名称
- `topology_config_name` (String) - 拓扑配置名称
- `search_mode` (String) - 搜索模式 (manual/auto)
- `manual_parallelism` (JSON) - 手动并行策略
- `search_constraints` (JSON) - 搜索约束
- `search_stats` (JSON) - 搜索统计

**保留字段**:
- 并行策略：`dp`, `tp`, `pp`, `ep`, `sp`, `moe_tp`, `chips`
- 性能指标：`tps`, `tpot`, `ttft`, `mfu`, 等
- 其他：`full_result`, `score`, `is_feasible`, `infeasible_reason`

**关系变化**:
- `experiment_id` - 外键，直接指向 Experiment（不再经过Task）

### EvaluationTask 表

**完全移除**

---

## API 变化

### 修改的端点

#### `POST /api/evaluation/submit`
- 不再创建Task记录
- 结果完成后直接保存到EvaluationResult
- 相同实验名称自动分组

#### `GET /api/evaluation/experiments`
- `total_tasks` = `len(exp.results)`
- `completed_tasks` = `len(exp.results)` (保存到DB的都是完成的)

#### `GET /api/evaluation/experiments/{id}`
- 直接查询Result，不再查询Task
- `tasks` 字段保留（兼容前端），但内容来自Result
- 每个条目的 `status` 固定为 "completed"

#### `POST /api/evaluation/experiments/{id}/results/batch-delete`
- 简化逻辑：直接删除Result
- 不再需要检查"空任务"并删除

#### `GET /api/evaluation/experiments/export`
- 结果按 `task_id` 分组导出
- 导出格式版本升级到 "2.0"
- 保持向后兼容（仍使用 "tasks" 字段名）

#### `POST /api/evaluation/experiments/execute-import`
- 不再创建Task记录
- 直接导入Result，从task_data继承配置字段

### 不变的端点

以下端点操作内存中的任务队列，与数据库无关，**保持不变**:
- `GET /api/evaluation/tasks` - 列出运行中的任务
- `GET /api/evaluation/tasks/{id}` - 获取任务状态
- `POST /api/evaluation/tasks/{id}/cancel` - 取消任务
- `DELETE /api/evaluation/tasks/{id}` - 删除任务
- `GET /api/evaluation/running` - 获取运行中任务

---

## 代码变化

### 删除的函数
- `_update_task_status_in_db()` - 不再需要更新Task状态

### 修改的函数
- `_save_task_result_to_db()` - 直接创建Result，包含配置字段
- `list_experiments()` - 动态计算任务数
- `get_experiment()` - 直接查询Result
- `batch_delete_results()` - 简化删除逻辑
- `export_experiments()` - 按task_id分组
- `execute_import()` - 直接导入Result

### 数据库模型
- `database.py`:
  - 删除 `TaskStatus` 枚举
  - 删除 `EvaluationTask` 类
  - `Experiment.results` 直接关联 `EvaluationResult`
  - `EvaluationResult` 合并Task字段

---

## 迁移步骤

### 1. 数据备份
```bash
备份文件: backend/math_model/data/llm_evaluations_backup_YYYYMMDD_HHMMSS.db
```

### 2. 执行迁移
```bash
cd backend
python migrate_database.py
```

### 3. 迁移逻辑
1. 读取旧表数据 (experiments, evaluation_tasks, evaluation_results)
2. 删除旧表结构
3. 创建新表结构
4. 迁移数据:
   - Experiment: 直接复制（移除计数字段）
   - Result: 合并Task字段到Result

### 4. 验证
```python
from math_model.L0_entry.database import engine
from sqlalchemy import inspect

inspector = inspect(engine)
print('Tables:', inspector.get_table_names())
# 输出: ['evaluation_results', 'experiments']
```

---

## 兼容性说明

### 前端兼容性
- API响应格式保持兼容（使用 "tasks" 字段名）
- 所有字段保持一致，前端无需修改

### 导入导出
- 支持导入旧格式 (version 1.0)
- 导出新格式 (version 2.0)
- 向后兼容：新格式可以被旧版本导入（略有信息丢失）

### 内存任务管理
- TaskManager 和 TaskStatus 枚举保留（用于运行时任务）
- 与数据库独立，互不影响

---

## 注意事项

1. **备份重要**: 迁移前会自动备份，建议手动额外备份
2. **不可逆**: 迁移后无法直接回退，需从备份恢复
3. **空数据库**: 如果数据库为空，会直接创建新结构
4. **storage/目录**: `L0_entry/storage/database.py` 未使用，保留备用

---

## 后续优化

### 可选的改进
1. 添加Result的索引（task_id, experiment_id, score）
2. 实现Result的软删除（标记删除而非物理删除）
3. 添加Result的版本控制
4. 实现增量导出（仅导出新增结果）

### 性能优化
1. 大量Result时考虑分页加载
2. Result列表按score排序可以添加索引
3. 考虑缓存实验的result_count

---

## 相关文件

### 核心文件
- `backend/math_model/L0_entry/database.py` - 数据库模型
- `backend/math_model/L0_entry/api.py` - API端点
- `backend/migrate_database.py` - 迁移脚本

### 配置文件
- `backend/math_model/L0_entry/config_schema.py` - API响应模型
- `backend/math_model/L0_entry/tasks.py` - 内存任务管理（未改动）

### 文档
- `docs/database_refactoring_summary.md` - 本文档
- `CLAUDE.md` - 项目指导文档（需更新）

---

## 总结

通过这次重构，我们：
- ✅ 简化了数据模型，从3层降至2层
- ✅ 消除了数据不一致的风险
- ✅ 提升了代码可维护性
- ✅ 保持了完全的前端兼容性
- ✅ 提供了平滑的迁移路径
- ✅ 借鉴了CrossRing项目的成熟架构

新架构更简洁、更可靠、更易维护！
