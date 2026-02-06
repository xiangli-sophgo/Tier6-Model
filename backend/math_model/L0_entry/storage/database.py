"""数据库模块

提供 SQLite 数据库连接和 ORM 模型。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Experiment(Base):
    """实验模型

    Attributes:
        id: 主键
        name: 实验名称
        description: 描述
        created_at: 创建时间
        updated_at: 更新时间
    """

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    tasks = relationship("EvaluationTask", back_populates="experiment", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "taskCount": len(self.tasks) if self.tasks else 0,
        }


class EvaluationTask(Base):
    """评估任务模型

    Attributes:
        id: 主键
        task_id: 任务 UUID
        experiment_id: 外键到 Experiment
        status: 任务状态
        progress: 进度 (0-1)
        config_snapshot: 配置快照 (JSON)
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
        error_message: 错误信息
    """

    __tablename__ = "evaluation_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), unique=True, nullable=False, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    status = Column(String(20), default="pending")
    progress = Column(Float, default=0.0)
    config_snapshot = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="tasks")
    result = relationship("EvaluationResult", back_populates="task", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "taskId": self.task_id,
            "experimentId": self.experiment_id,
            "status": self.status,
            "progress": self.progress,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "errorMessage": self.error_message,
        }


class EvaluationResult(Base):
    """评估结果模型

    Attributes:
        id: 主键
        task_id: 外键到 EvaluationTask
        tps: Tokens per second
        tpot: Time per output token (ms)
        ttft: Time to first token (ms)
        mfu: Model FLOPS utilization
        score: 综合评分
        full_result: 完整结果 (JSON)
        created_at: 创建时间
    """

    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("evaluation_tasks.task_id"), unique=True, nullable=False)

    # Key metrics (indexed for fast query)
    tps = Column(Float, default=0.0)
    tpot = Column(Float, default=0.0)
    ttft = Column(Float, default=0.0)
    mfu = Column(Float, default=0.0)
    score = Column(Float, default=0.0)

    # Full result data
    full_result = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    task = relationship("EvaluationTask", back_populates="result")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "taskId": self.task_id,
            "tps": self.tps,
            "tpot": self.tpot,
            "ttft": self.ttft,
            "mfu": self.mfu,
            "score": self.score,
            "fullResult": self.full_result,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class Database:
    """数据库管理类"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """初始化

        Args:
            db_path: 数据库文件路径 (默认为 tier6.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "tier6.db"
        else:
            db_path = Path(db_path)

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """创建所有表"""
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """获取数据库会话"""
        return self.Session()


# 全局数据库实例
_database: Database | None = None


def get_database() -> Database:
    """获取全局数据库实例

    Returns:
        Database: 数据库实例
    """
    global _database
    if _database is None:
        _database = Database()
        _database.create_tables()
    return _database


def init_database(db_path: str | Path | None = None) -> Database:
    """初始化数据库

    Args:
        db_path: 数据库文件路径

    Returns:
        Database: 数据库实例
    """
    global _database
    _database = Database(db_path)
    _database.create_tables()
    return _database
