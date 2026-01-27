"""
任务管理模块
"""

from .manager import (
    GlobalWorkerPool,
    create_and_submit_task,
    get_task_status,
    cancel_task,
    get_task_results,
    delete_task,
    get_running_tasks,
    get_executor_info,
    set_main_loop,
    subscribe_global,
    unsubscribe_global,
)
from .deployment import evaluate_deployment

__all__ = [
    'GlobalWorkerPool',
    'create_and_submit_task',
    'get_task_status',
    'cancel_task',
    'get_task_results',
    'delete_task',
    'get_running_tasks',
    'get_executor_info',
    'set_main_loop',
    'subscribe_global',
    'unsubscribe_global',
    'evaluate_deployment',
]
