"""L0: 入口与编排层

提供 API 接口、任务管理、数据存储等功能。

核心类型:
    - ConfigLoader: 配置加载器
    - TaskManager: 任务管理器
    - Database: 数据库管理
    - run_evaluation: 评估引擎入口
"""

from perf_model.L0_entry.config_loader import (
    ConfigLoader,
    get_config_loader,
    load_benchmark,
    load_chip,
    load_model,
    load_topology,
)
from perf_model.L0_entry.tasks import (
    TaskInfo,
    TaskManager,
    TaskStatus,
    get_task_manager,
)
from perf_model.L0_entry.engine import run_evaluation
from perf_model.L0_entry.api import router

__all__ = [
    # 配置加载
    "ConfigLoader",
    "get_config_loader",
    "load_chip",
    "load_model",
    "load_topology",
    "load_benchmark",
    # 任务管理
    "TaskStatus",
    "TaskInfo",
    "TaskManager",
    "get_task_manager",
    # 评估引擎
    "run_evaluation",
    # API 路由
    "router",
]
