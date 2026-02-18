"""
数据库配置、会话管理和 ORM 模型

合并自 database.py 和 db_models.py
"""

import os
import enum
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship


# ============================================
# 数据库配置
# ============================================

# 数据库文件路径
DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "llm_evaluations.db"

# 数据库 URL
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 基类
Base = declarative_base()


def get_db() -> Session:
    """获取数据库会话（依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    获取数据库会话的上下文管理器（用于非 FastAPI 依赖注入场景）

    使用示例:
        with get_db_session() as db:
            task = db.query(EvaluationTask).first()
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """初始化数据库（创建所有表）"""
    Base.metadata.create_all(bind=engine)


# ============================================
# ORM 模型
# ============================================

# TaskStatus 已移除（不再需要中间的 Task 层）


class Experiment(Base):
    """实验元数据表"""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系 - 直接关联到结果
    results = relationship("EvaluationResult", back_populates="experiment", cascade="all, delete-orphan")


class EvaluationResult(Base):
    """评估结果表（合并原 Task 和 Result）"""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)

    # 任务标识（UUID，用于前端展示和 WebSocket 通知）
    task_id = Column(String(36), nullable=False, index=True)

    # 配置快照（每个结果独立保存所有配置，确保可复现）
    config_snapshot = Column(JSON, nullable=False)

    # 配置文件引用（追溯配置来源）
    benchmark_name = Column(String(255), nullable=True)
    topology_config_name = Column(String(255), nullable=True)

    # 搜索配置
    search_mode = Column(String(20), nullable=False)  # 'manual' or 'auto'
    manual_parallelism = Column(JSON, nullable=True)
    search_constraints = Column(JSON, nullable=True)
    search_stats = Column(JSON, nullable=True)

    # 并行策略
    dp = Column(Integer, nullable=False)
    tp = Column(Integer, nullable=False)
    pp = Column(Integer, default=1)
    ep = Column(Integer, default=1)
    sp = Column(Integer, default=1)
    moe_tp = Column(Integer, nullable=True)

    # 资源使用
    chips = Column(Integer, nullable=False)

    # 性能指标
    total_elapse_us = Column(Float, nullable=False)
    total_elapse_ms = Column(Float, nullable=False)
    comm_elapse_us = Column(Float, nullable=False)
    tps = Column(Float, nullable=False)
    tps_per_batch = Column(Float, nullable=False)
    tps_per_chip = Column(Float, nullable=False)
    ttft = Column(Float, nullable=True)  # Time To First Token (ms)
    tpot = Column(Float, nullable=True)  # Time Per Output Token (ms)
    mfu = Column(Float, nullable=False)

    # 计算量和内存
    flops = Column(Float, nullable=False)
    dram_occupy = Column(Float, nullable=False)

    # 综合得分
    score = Column(Float, nullable=False)
    is_feasible = Column(Integer, default=1)
    infeasible_reason = Column(Text, nullable=True)

    # 完整结果数据
    full_result = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系
    experiment = relationship("Experiment", back_populates="results")
