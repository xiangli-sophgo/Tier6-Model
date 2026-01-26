"""
任务管理模块
"""

from .manager import TaskExecutor, submit_evaluation_task
from .deployment import evaluate_deployment

__all__ = ['TaskExecutor', 'submit_evaluation_task', 'evaluate_deployment']
