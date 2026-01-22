"""
FastAPI 接口模块

提供 LLM 推理模拟的 REST API 接口。
"""

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
from sqlalchemy.orm import Session

from .simulator import (
    run_simulation,
    validate_model_config,
    validate_hardware_config,
    validate_parallelism_config,
    validate_mla_config,
    validate_moe_config,
)
from .evaluators import ARCH_PRESETS
from .types import ProtocolConfig, NetworkInfraConfig
from .database import get_db, init_db
from .websocket_manager import ws_manager
from . import task_manager
from .db_models import Experiment, EvaluationTask, TaskStatus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Pydantic 模型
# ============================================

class SimulationRequest(BaseModel):
    """模拟请求"""
    topology: dict[str, Any]
    model: dict[str, Any]
    inference: dict[str, Any]
    parallelism: dict[str, Any]
    hardware: dict[str, Any]
    config: dict[str, Any] | None = None


class SimulationResponse(BaseModel):
    """模拟响应"""
    ganttChart: dict[str, Any]
    stats: dict[str, Any]
    timestamp: float


class BenchmarkConfig(BaseModel):
    """Benchmark 配置"""
    id: str
    name: str
    model: dict[str, Any]
    inference: dict[str, Any]


class EvaluationRequest(BaseModel):
    """评估请求"""
    experiment_name: str
    description: str = ""
    topology: dict[str, Any]
    model: dict[str, Any]
    hardware: dict[str, Any]
    inference: dict[str, Any]
    search_mode: str  # 'manual' or 'auto'
    manual_parallelism: Optional[dict[str, Any]] = None
    search_constraints: Optional[dict[str, Any]] = None


class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    task_id: str
    message: str


# Benchmark 文件存储目录
BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
BENCHMARKS_DIR.mkdir(exist_ok=True)


# ============================================
# FastAPI 应用
# ============================================

app = FastAPI(
    title="LLM 推理模拟器 API",
    description="基于拓扑的 GPU/加速器侧精细模拟服务",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 启动事件：初始化数据库和 WebSocket 回调
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("初始化数据库...")
    init_db()
    logger.info("设置 WebSocket 广播回调...")
    task_manager.set_ws_broadcast_callback(ws_manager.broadcast_task_update)
    logger.info("应用启动完成")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "LLM 推理模拟器 API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/api/presets/chips")
async def get_chip_presets():
    """
    获取所有芯片预设配置

    返回后端定义的所有硬件架构配置，包括：
    - 芯片名称
    - 算力精度 (BF16/FP16)
    - 峰值算力 (TFLOPS)
    - 核心数量
    - 内存带宽等
    """
    result = []
    for chip_id, arch in ARCH_PRESETS.items():
        result.append({
            "id": chip_id,
            "name": arch.name,
            "flops_dtype": arch.flops_dtype,
            "compute_tflops": round(arch.flops_per_second / 1e12, 2),  # 四舍五入到2位小数
            "num_cores": arch.num_cores,
            "sram_size_mb": round(arch.sram_size_bytes / (1024 * 1024), 2),
            "dram_bandwidth_gbps": round(arch.dram_bandwidth_bytes / 1e9, 2),
            "intra_bw_gbps": round(arch.intra_bw / 1e9, 2),
            "inter_bw_gbps": round(arch.inter_bw / 1e9, 2),
            # 粗粒度延迟（向后兼容）
            "intra_latency_us": round(arch.intra_latency_us, 2),
            "inter_latency_us": round(arch.inter_latency_us, 2),
            # 细粒度通信延迟（新增，单位: us）
            "comm_latency": {
                "chip_to_chip_us": round(arch.comm_latency.chip_to_chip_us, 2),
                "comm_start_overhead_us": round(arch.comm_latency.comm_start_overhead_us, 2),
                "memory_read_latency_us": round(arch.comm_latency.memory_read_latency_us, 2),
                "memory_write_latency_us": round(arch.comm_latency.memory_write_latency_us, 2),
            },
        })
    return {"chips": result}


@app.get("/api/presets/runtime")
async def get_runtime_presets():
    """
    获取运行时配置预设

    返回协议和网络基础设施的默认配置值，供前端显示和配置。
    """
    # 使用默认值创建配置对象
    protocol = ProtocolConfig()
    network = NetworkInfraConfig()

    return {
        "protocol": {
            "rtt_tp_us": protocol.rtt_tp_us,
            "rtt_ep_us": protocol.rtt_ep_us,
            "bandwidth_utilization": protocol.bandwidth_utilization,
            "sync_latency_us": protocol.sync_latency_us,
        },
        "network": {
            "switch_delay_us": network.switch_delay_us,
            "cable_delay_us": network.cable_delay_us,
            "link_delay_us": network.link_delay_us,
        },
    }


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """
    运行 LLM 推理模拟

    Args:
        request: 模拟请求，包含拓扑、模型、推理、并行策略、硬件配置

    Returns:
        模拟结果，包含甘特图数据和统计信息
    """
    try:
        logger.info(f"开始模拟: model={request.model.get('model_name', 'Unknown')}")
        result = run_simulation(
            topology_dict=request.topology,
            model_dict=request.model,
            inference_dict=request.inference,
            parallelism_dict=request.parallelism,
            hardware_dict=request.hardware,
            config_dict=request.config,
        )
        logger.info("模拟完成")
        return SimulationResponse(**result)
    except ValueError as e:
        logger.warning(f"配置验证失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        logger.warning(f"配置缺少必要字段: {e}")
        raise HTTPException(status_code=400, detail=f"配置缺少必要字段: {e}")
    except TypeError as e:
        logger.warning(f"配置类型错误: {e}")
        raise HTTPException(status_code=400, detail=f"配置类型错误: {e}")
    except Exception as e:
        logger.error(f"模拟失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模拟失败: {str(e)}")


@app.post("/api/validate")
async def validate_config(request: SimulationRequest):
    """
    验证配置是否有效

    检查：
    - 模型配置的有效性
    - 硬件配置的合理性
    - 并行策略的正确性
    - MLA/MoE 配置的完整性
    - 拓扑中芯片数量是否满足并行策略需求
    """
    errors = []

    # 验证模型配置
    try:
        validate_model_config(request.model)
    except ValueError as e:
        errors.append(f"模型配置: {e}")

    # 验证硬件配置
    try:
        validate_hardware_config(request.hardware)
    except ValueError as e:
        errors.append(f"硬件配置: {e}")

    # 验证并行策略
    try:
        validate_parallelism_config(request.parallelism, request.model)
    except ValueError as e:
        errors.append(f"并行策略: {e}")

    # 验证 MLA 配置（如果存在）
    mla_dict = request.model.get("mla_config")
    if mla_dict:
        try:
            validate_mla_config(mla_dict)
        except ValueError as e:
            errors.append(f"MLA 配置: {e}")

    # 验证 MoE 配置（如果存在）
    moe_dict = request.model.get("moe_config")
    if moe_dict:
        try:
            validate_moe_config(moe_dict)
        except ValueError as e:
            errors.append(f"MoE 配置: {e}")

    # 验证芯片数量
    topology = request.topology
    parallelism = request.parallelism

    required_chips = (
        parallelism.get("dp", 1) *
        parallelism.get("tp", 1) *
        parallelism.get("pp", 1) *
        parallelism.get("ep", 1)
    )

    available_chips = 0
    for pod in topology.get("pods", []):
        for rack in pod.get("racks", []):
            for board in rack.get("boards", []):
                available_chips += len(board.get("chips", []))

    if available_chips < required_chips:
        errors.append(f"芯片数量不足: 需要 {required_chips} 个，拓扑中只有 {available_chips} 个")

    if errors:
        logger.warning(f"配置验证失败: {errors}")
        return {
            "valid": False,
            "errors": errors,
            "required_chips": required_chips,
            "available_chips": available_chips,
        }

    logger.info("配置验证通过")
    return {
        "valid": True,
        "required_chips": required_chips,
        "available_chips": available_chips,
    }


# ============================================
# Benchmark 管理 API
# ============================================

@app.get("/api/benchmarks")
async def list_benchmarks():
    """
    获取所有自定义 Benchmark 列表

    从 benchmarks 目录读取所有 JSON 文件
    """
    benchmarks = []
    for file_path in BENCHMARKS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                benchmarks.append(data)
        except Exception as e:
            logger.warning(f"读取 benchmark 文件失败 {file_path}: {e}")
    return {"benchmarks": benchmarks}


@app.get("/api/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """
    获取单个 Benchmark 配置
    """
    file_path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark 不存在: {benchmark_id}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmarks")
async def create_benchmark(benchmark: BenchmarkConfig):
    """
    创建新的 Benchmark 配置

    保存为 JSON 文件到 benchmarks 目录
    """
    file_path = BENCHMARKS_DIR / f"{benchmark.id}.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"创建 benchmark: {benchmark.id}")
        return {"success": True, "id": benchmark.id}
    except Exception as e:
        logger.error(f"保存 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/benchmarks/{benchmark_id}")
async def update_benchmark(benchmark_id: str, benchmark: BenchmarkConfig):
    """
    更新 Benchmark 配置
    """
    file_path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark 不存在: {benchmark_id}")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"更新 benchmark: {benchmark_id}")
        return {"success": True, "id": benchmark_id}
    except Exception as e:
        logger.error(f"更新 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/benchmarks/{benchmark_id}")
async def delete_benchmark(benchmark_id: str):
    """
    删除 Benchmark 配置
    """
    file_path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark 不存在: {benchmark_id}")

    try:
        file_path.unlink()
        logger.info(f"删除 benchmark: {benchmark_id}")
        return {"success": True, "id": benchmark_id}
    except Exception as e:
        logger.error(f"删除 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 评估任务管理 API
# ============================================

@app.post("/api/evaluation/submit", response_model=TaskSubmitResponse)
async def submit_evaluation(request: EvaluationRequest):
    """
    提交评估任务到后台执行

    Args:
        request: 评估请求，包含实验名称、拓扑、模型、硬件、推理配置等

    Returns:
        任务 ID 和提示消息
    """
    try:
        task_id = task_manager.create_and_submit_task(
            experiment_name=request.experiment_name,
            description=request.description,
            topology=request.topology,
            model_config=request.model,
            hardware_config=request.hardware,
            inference_config=request.inference,
            search_mode=request.search_mode,
            manual_parallelism=request.manual_parallelism,
            search_constraints=request.search_constraints,
        )
        logger.info(f"评估任务已提交: {task_id}")
        return TaskSubmitResponse(
            task_id=task_id,
            message="评估任务已提交，正在后台运行"
        )
    except Exception as e:
        logger.error(f"提交评估任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"提交评估任务失败: {str(e)}")


@app.get("/api/evaluation/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """获取任务状态"""
    status = task_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return status


@app.get("/api/evaluation/tasks/{task_id}/results")
async def get_task_results_endpoint(task_id: str):
    """获取任务的完整结果"""
    results = task_manager.get_task_results(task_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return results


@app.post("/api/evaluation/tasks/{task_id}/cancel")
async def cancel_task_endpoint(task_id: str):
    """取消任务"""
    success = task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {"success": True, "message": "任务已取消"}


@app.delete("/api/evaluation/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    """删除任务（及其结果）"""
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {"success": True, "message": "任务已删除"}


@app.get("/api/evaluation/tasks")
async def list_tasks(
    status: Optional[str] = None,
    experiment_name: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    获取任务列表

    Args:
        status: 过滤状态 (pending, running, completed, failed, cancelled)
        experiment_name: 过滤实验名称
        limit: 返回数量限制
    """
    query = db.query(EvaluationTask).join(Experiment)

    if status:
        try:
            status_enum = TaskStatus(status)
            query = query.filter(EvaluationTask.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的状态: {status}")

    if experiment_name:
        query = query.filter(Experiment.name == experiment_name)

    tasks = query.order_by(EvaluationTask.created_at.desc()).limit(limit).all()

    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "experiment_name": task.experiment.name,
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            }
            for task in tasks
        ]
    }


@app.get("/api/evaluation/running")
async def get_running_tasks_endpoint():
    """获取所有运行中的任务"""
    return {"tasks": task_manager.get_running_tasks()}


@app.get("/api/evaluation/experiments")
async def list_experiments(db: Session = Depends(get_db)):
    """获取所有实验列表"""
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return {
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "total_tasks": exp.total_tasks,
                "completed_tasks": exp.completed_tasks,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }
            for exp in experiments
        ]
    }


@app.get("/api/evaluation/experiments/{experiment_id}")
async def get_experiment_details(experiment_id: int, db: Session = Depends(get_db)):
    """获取实验详情（包含所有任务）"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    tasks = db.query(EvaluationTask).filter(EvaluationTask.experiment_id == experiment_id).all()

    return {
        "id": experiment.id,
        "name": experiment.name,
        "description": experiment.description,
        "model_config": experiment.model_config,
        "hardware_config": experiment.hardware_config,
        "inference_config": experiment.inference_config,
        "total_tasks": experiment.total_tasks,
        "completed_tasks": experiment.completed_tasks,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "tasks": [
            {
                "id": task.id,
                "task_id": task.task_id,
                "experiment_id": task.experiment_id,
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "error": task.error,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "search_mode": task.search_mode,
                "manual_parallelism": task.manual_parallelism,
                "search_constraints": task.search_constraints,
                "search_stats": task.search_stats,
            }
            for task in tasks
        ]
    }


@app.delete("/api/evaluation/experiments/{experiment_id}")
async def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """删除实验及其所有任务和结果"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    try:
        # 删除实验（级联删除任务和结果）
        db.delete(experiment)
        db.commit()
        return {"message": f"实验 '{experiment.name}' 已删除"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ============================================
# WebSocket 端点
# ============================================

@app.websocket("/ws/tasks")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 端点：实时推送任务状态更新

    客户端连接后会收到所有任务的状态变化推送
    消息格式:
    {
        "type": "task_update",
        "task_id": "...",
        "status": "running",
        "progress": 50.0,
        "message": "...",
        ...
    }
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # 保持连接，等待客户端消息（目前不处理客户端消息）
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import os
    import uvicorn
    from pathlib import Path
    from dotenv import load_dotenv

    # 加载 Tier6+model/.env 共享配置
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    port = int(os.environ.get("VITE_API_PORT", "8001"))
    print(f"Tier6+互联建模平台启动在端口: {port}")
    uvicorn.run("llm_simulator.api:app", host="0.0.0.0", port=port, reload=True)
