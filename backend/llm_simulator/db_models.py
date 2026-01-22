"""
数据库 ORM 模型
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from .database import Base


class TaskStatus(str, enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment(Base):
    """实验元数据表（轻量级容器）"""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 统计信息
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)

    # 关系
    tasks = relationship("EvaluationTask", back_populates="experiment", cascade="all, delete-orphan")


class EvaluationTask(Base):
    """评估任务表（保存完整配置快照）"""
    __tablename__ = "evaluation_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)

    # 任务状态
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    progress = Column(Float, default=0.0)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 完整配置快照（每个任务独立保存所有配置）
    config_snapshot = Column(JSON, nullable=False)  # 包含 model, inference, topology, protocol, network, chip_latency

    # 配置文件引用（追溯配置来源）
    benchmark_name = Column(String(255), nullable=True)  # Benchmark 配置文件名称
    topology_config_name = Column(String(255), nullable=True)  # 拓扑配置文件名称

    # 搜索配置
    search_mode = Column(String(20), nullable=False)  # 'manual' or 'auto'
    manual_parallelism = Column(JSON, nullable=True)  # 手动模式的并行策略
    search_constraints = Column(JSON, nullable=True)  # 自动模式的搜索约束

    # 搜索统计（自动模式）
    search_stats = Column(JSON, nullable=True)  # 搜索统计信息

    # 关系
    experiment = relationship("Experiment", back_populates="tasks")
    results = relationship("EvaluationResult", back_populates="task", cascade="all, delete-orphan")


class EvaluationResult(Base):
    """评估结果表"""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("evaluation_tasks.id"), nullable=False)

    # 并行策略
    dp = Column(Integer, nullable=False)
    tp = Column(Integer, nullable=False)
    pp = Column(Integer, default=1)
    ep = Column(Integer, default=1)
    sp = Column(Integer, default=1)
    moe_tp = Column(Integer, nullable=True)

    # 资源使用
    chips = Column(Integer, nullable=False)

    # 性能指标（对齐 DS_TPU）
    total_elapse_us = Column(Float, nullable=False)  # 总延迟 (微秒)
    total_elapse_ms = Column(Float, nullable=False)  # 总延迟 (毫秒)
    comm_elapse_us = Column(Float, nullable=False)   # 通信延迟 (微秒)
    tps = Column(Float, nullable=False)              # tokens/s (total throughput)
    tps_per_batch = Column(Float, nullable=False)    # tokens/s per batch
    tps_per_chip = Column(Float, nullable=False)     # tokens/s/chip
    mfu = Column(Float, nullable=False)              # Model FLOPs Utilization (0-1)

    # 计算量和内存
    flops = Column(Float, nullable=False)            # 总 FLOPs
    dram_occupy = Column(Float, nullable=False)      # 内存占用 (bytes)

    # 综合得分
    score = Column(Float, nullable=False)
    is_feasible = Column(Integer, default=1)  # 1=可行, 0=不可行
    infeasible_reason = Column(Text, nullable=True)

    # 完整结果数据（JSON）
    full_result = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    task = relationship("EvaluationTask", back_populates="results")
