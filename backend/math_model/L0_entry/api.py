"""API 接口模块

提供 FastAPI 路由定义，包含完整的预设管理、Benchmark/Topology CRUD、
仿真、任务管理、实验管理和 WebSocket 端点。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Body

from math_model.L0_entry.config_loader import get_config_loader
from math_model.L0_entry.config_schema import (
    BenchmarkCreateRequest,
    BenchmarkUpdateRequest,
    CalculateParamsRequest,
    CalculateParamsResponse,
    ChipPresetUpdateRequest,
    EvaluationRequest,
    ExecutorConfigRequest,
    ExperimentUpdateRequest,
    BatchDeleteRequest,
    ImportExecuteRequest,
    ModelPresetCreateRequest,
    ModelPresetUpdateRequest,
    SimulateRequest,
    TopologyCreateRequest,
    TopologyUpdateRequest,
    ValidateRequest,
    ValidationResponse,
    validate_chip_config,
    validate_evaluation_config,
    validate_model_config,
)
from math_model.L0_entry.tasks import TaskInfo, TaskStatus, get_task_manager
from math_model.L0_entry.websocket import get_ws_manager

# 使用 math_model 自己的数据库存储
from math_model.core.database import (
    get_db_session,
    Experiment as DBExperiment,
    EvaluationTask as DBEvaluationTask,
    EvaluationResult as DBEvaluationResult,
    TaskStatus as DBTaskStatus,
)

logger = logging.getLogger(__name__)

# ============================================
# API 路由
# ============================================

router = APIRouter(prefix="/api", tags=["math_model"])


# ============================================
# 健康检查
# ============================================


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "0.2.0"}


# ============================================
# 预设管理 - 芯片
# ============================================


@router.get("/presets/chips")
async def list_chips():
    """列出所有芯片预设 (含完整配置)"""
    loader = get_config_loader()
    names = loader.list_chips()
    presets = []
    for name in names:
        try:
            config = loader.load_chip(name)
            presets.append({"name": name, "config": config})
        except Exception as e:
            logger.warning(f"Failed to load chip preset {name}: {e}")
    return {"presets": presets}


@router.get("/presets/chips/{name}")
async def get_chip_preset(name: str):
    """获取芯片预设详情"""
    try:
        loader = get_config_loader()
        return loader.load_chip(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chip preset not found: {name}")


@router.post("/presets/chips")
async def save_chip(config: dict[str, Any] = Body(...)):
    """保存芯片预设到 YAML 文件 (Tier6 格式)"""
    name = config.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' field")

    try:
        loader = get_config_loader()
        loader.save_chip(name, config)
        return {"message": f"Chip preset '{name}' saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save chip preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/presets/chips/{name}")
async def update_chip_preset(name: str, request: ChipPresetUpdateRequest):
    """更新芯片预设"""
    loader = get_config_loader()

    # 检查是否存在
    try:
        loader.load_chip(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chip preset not found: {name}")

    # 验证配置
    errors = validate_chip_config(request.config, source=name)
    if errors:
        raise HTTPException(status_code=400, detail=f"Invalid chip config: {'; '.join(errors)}")

    try:
        loader.save_chip(name, request.config)
        return {"message": f"Chip preset '{name}' updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update chip preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/presets/chips/{name}")
async def delete_chip(name: str):
    """删除芯片预设"""
    try:
        loader = get_config_loader()
        loader.delete_chip(name)
        return {"message": f"Chip preset '{name}' deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chip preset not found: {name}")
    except Exception as e:
        logger.error(f"Failed to delete chip preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 预设管理 - 模型
# ============================================


@router.get("/presets/models")
async def list_models():
    """列出所有模型预设 (含完整配置)"""
    loader = get_config_loader()
    names = loader.list_models()
    presets = []
    for name in names:
        try:
            config = loader.load_model(name)
            presets.append({"name": name, "config": config})
        except Exception as e:
            logger.warning(f"Failed to load model preset {name}: {e}")
    return {"presets": presets}


@router.get("/presets/models/{name}")
async def get_model_preset(name: str):
    """获取模型预设详情"""
    try:
        loader = get_config_loader()
        return loader.load_model(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model preset not found: {name}")


@router.post("/presets/models")
async def create_model_preset(request: ModelPresetCreateRequest):
    """创建模型预设"""
    loader = get_config_loader()

    # 验证配置
    errors = validate_model_config(request.config, source=request.name)
    if errors:
        raise HTTPException(status_code=400, detail=f"Invalid model config: {'; '.join(errors)}")

    try:
        loader.save_model(request.name, request.config)
        return {"message": f"Model preset '{request.name}' saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save model preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/presets/models/{name}")
async def update_model_preset(name: str, request: ModelPresetUpdateRequest):
    """更新模型预设"""
    loader = get_config_loader()

    # 检查是否存在
    try:
        loader.load_model(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model preset not found: {name}")

    # 验证配置
    errors = validate_model_config(request.config, source=name)
    if errors:
        raise HTTPException(status_code=400, detail=f"Invalid model config: {'; '.join(errors)}")

    try:
        loader.save_model(name, request.config)
        return {"message": f"Model preset '{name}' updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update model preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/presets/models/{name}")
async def delete_model(name: str):
    """删除模型预设"""
    try:
        loader = get_config_loader()
        loader.delete_model(name)
        return {"message": f"Model preset '{name}' deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model preset not found: {name}")
    except Exception as e:
        logger.error(f"Failed to delete model preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Benchmark 管理
# ============================================


@router.get("/benchmarks")
async def list_benchmarks():
    """列出所有 Benchmark"""
    loader = get_config_loader()
    benchmarks = loader.list_all_benchmarks()
    return {"benchmarks": benchmarks}


@router.get("/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """获取 Benchmark 详情"""
    try:
        loader = get_config_loader()
        return loader.load_benchmark(benchmark_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")


@router.post("/benchmarks")
async def create_benchmark(request: BenchmarkCreateRequest):
    """创建 Benchmark"""
    loader = get_config_loader()

    # 检查是否已存在
    existing = loader.list_all_benchmarks()
    for b in existing:
        if b["id"] == request.id or b["filename"] == request.id:
            raise HTTPException(status_code=409, detail=f"Benchmark already exists: {request.id}")

    # 保存配置
    config = {
        "id": request.id,
        "name": request.name,
        "model": request.model,
        "topology": request.topology,
        "inference": request.inference,
    }
    loader.save_benchmark(request.id, config)

    return {"message": "Benchmark created", "id": request.id}


@router.put("/benchmarks/{benchmark_id}")
async def update_benchmark(benchmark_id: str, request: BenchmarkUpdateRequest):
    """更新 Benchmark"""
    loader = get_config_loader()

    # 加载现有配置
    try:
        config = loader.load_benchmark(benchmark_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")

    # 更新字段
    if request.name is not None:
        config["name"] = request.name
    if request.model is not None:
        config["model"] = request.model
    if request.inference is not None:
        config["inference"] = request.inference
    if request.topology is not None:
        config["topology"] = request.topology
        # 清除旧的拓扑引用，避免 save_benchmark 用旧引用覆盖新值
        config.pop("topology_preset_ref", None)

    # 同样处理模型引用
    if request.model is not None:
        config.pop("model_preset_ref", None)

    # 保存
    loader.save_benchmark(benchmark_id, config)

    return {"message": "Benchmark updated", "id": benchmark_id}


@router.delete("/benchmarks/{benchmark_id}")
async def delete_benchmark(benchmark_id: str):
    """删除 Benchmark"""
    loader = get_config_loader()

    if not loader.delete_benchmark(benchmark_id):
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")

    return {"message": "Benchmark deleted", "id": benchmark_id}


# ============================================
# Topology 管理
# ============================================


@router.get("/topologies")
async def list_topologies():
    """列出所有拓扑配置"""
    loader = get_config_loader()
    names = loader.list_topologies()
    topologies = []
    for name in names:
        try:
            config = loader.load_topology(name)
            chip_count = _count_topology_chips(config)
            topologies.append({
                "name": name,
                "chip_count": chip_count,
                "topology_type": config.get("topology", {}).get("type", ""),
            })
        except Exception as e:
            logger.warning(f"Failed to load topology {name}: {e}")
    return {"topologies": topologies}


@router.get("/topologies/{name}")
async def get_topology(name: str):
    """获取拓扑配置详情"""
    try:
        loader = get_config_loader()
        return loader.load_topology(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Topology not found: {name}")


@router.post("/topologies")
async def create_topology(request: TopologyCreateRequest):
    """创建拓扑配置"""
    loader = get_config_loader()

    # 检查是否已存在
    if request.name in loader.list_topologies():
        raise HTTPException(status_code=409, detail=f"Topology already exists: {request.name}")

    # 保存完整配置（前端 SavedConfig 格式）
    config = request.model_dump(exclude_none=True)
    loader.save_topology(request.name, config)

    return {"message": "Topology created", "name": request.name}


@router.put("/topologies/{name}")
async def update_topology(name: str, request: TopologyUpdateRequest):
    """更新拓扑配置"""
    loader = get_config_loader()

    # 加载现有配置
    try:
        config = loader.load_topology(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Topology not found: {name}")

    # 用新数据更新配置
    update_data = request.model_dump(exclude_none=True)
    config.update(update_data)

    # 处理重命名
    new_name = request.name if request.name else name
    if new_name != name:
        loader.delete_topology(name)

    loader.save_topology(new_name, config)

    return {"message": "Topology updated", "name": new_name}


@router.delete("/topologies/{name}")
async def delete_topology(name: str):
    """删除拓扑配置"""
    loader = get_config_loader()

    if not loader.delete_topology(name):
        raise HTTPException(status_code=404, detail=f"Topology not found: {name}")

    return {"message": "Topology deleted", "name": name}


# ============================================
# 仿真
# ============================================


@router.post("/simulate")
async def simulate(request: SimulateRequest):
    """同步仿真 - 立即返回结果"""
    from math_model.L0_entry.engine import run_evaluation

    # 构建配置
    config = {
        "chip_preset": request.chip_preset,
        "model_preset": request.model_preset,
        "topology_preset": request.topology_preset,
        "chip_config": request.chip_config,
        "model_config": request.model_params,
        "topology_config": request.topology_config,
        "deployment": request.parallelism.model_dump() if request.parallelism else {},
        "inference": request.inference,
    }

    try:
        result = run_evaluation(config)
        return result
    except Exception as e:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_config(request: ValidateRequest):
    """验证配置"""
    config = {
        "chip_config": request.chip_config,
        "model_config": request.model_params,
        "topology_config": request.topology_config,
        "parallelism": request.parallelism,
        "inference": request.inference,
    }

    result = validate_evaluation_config(config)

    return ValidationResponse(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
    )


@router.post("/model/calculate-params", response_model=CalculateParamsResponse)
async def calculate_model_params(request: CalculateParamsRequest):
    """计算模型参数量

    注意: 此端点为前端辅助计算工具，使用典型默认值以便快速估算。
    实际仿真评估时必须提供完整的模型配置（见 run_evaluation）。
    """
    model_config = request.model_params

    # 基本参数（使用典型默认值用于快速估算）
    hidden_size = model_config.get("hidden_size", 4096)
    num_layers = model_config.get("num_layers", 32)
    vocab_size = model_config.get("vocab_size", 32000)
    intermediate_size = model_config.get("intermediate_size", hidden_size * 4)
    num_heads = model_config.get("num_attention_heads", 32)
    dtype = model_config.get("dtype", "bf16")

    # 数据类型字节数
    dtype_bytes = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    bytes_per_param = dtype_bytes.get(dtype, 2)

    # Embedding 参数
    embed_params = vocab_size * hidden_size

    # Attention 参数 (Q, K, V, O projections)
    attn_params_per_layer = 4 * hidden_size * hidden_size

    # MLA 配置
    mla_config = model_config.get("MLA", {})
    if mla_config:
        q_lora_rank = mla_config.get("q_lora_rank", 0)
        kv_lora_rank = mla_config.get("kv_lora_rank", 0)
        if q_lora_rank > 0 and kv_lora_rank > 0:
            # MLA: 使用 LoRA 投影
            attn_params_per_layer = (
                hidden_size * q_lora_rank +  # Q down
                q_lora_rank * hidden_size +  # Q up
                hidden_size * kv_lora_rank +  # KV down
                kv_lora_rank * hidden_size * 2  # K up + V up
            )

    # FFN 参数
    ffn_params_per_layer = 3 * hidden_size * intermediate_size  # gate, up, down

    # MoE 配置
    moe_config = model_config.get("MoE", {})
    num_routed_experts = moe_config.get("num_routed_experts", 0)
    num_shared_experts = moe_config.get("num_shared_experts", 0)
    num_activated_experts = moe_config.get("num_activated_experts", 1)
    moe_intermediate_size = moe_config.get("intermediate_size", intermediate_size // 4)

    num_dense_layers = model_config.get("num_dense_layers", num_layers if num_routed_experts == 0 else 0)
    num_moe_layers = model_config.get("num_moe_layers", 0 if num_routed_experts == 0 else num_layers - num_dense_layers)

    # Dense 层参数
    dense_layer_params = attn_params_per_layer + ffn_params_per_layer
    total_dense_params = dense_layer_params * num_dense_layers

    # MoE 层参数
    moe_expert_params = 3 * hidden_size * moe_intermediate_size  # 每个专家
    moe_layer_params = (
        attn_params_per_layer +
        num_routed_experts * moe_expert_params +
        num_shared_experts * moe_expert_params +
        hidden_size * num_routed_experts  # router
    )
    total_moe_params = moe_layer_params * num_moe_layers

    # 激活参数 (MoE 只激活部分专家)
    active_moe_layer_params = (
        attn_params_per_layer +
        num_activated_experts * moe_expert_params +
        num_shared_experts * moe_expert_params +
        hidden_size * num_routed_experts
    )
    total_active_moe_params = active_moe_layer_params * num_moe_layers

    # LM Head 参数
    lm_head_params = hidden_size * vocab_size

    # 总参数
    total_params = embed_params + total_dense_params + total_moe_params + lm_head_params
    active_params = embed_params + total_dense_params + total_active_moe_params + lm_head_params

    # 权重大小
    weight_size_bytes = int(total_params * bytes_per_param)

    return CalculateParamsResponse(
        total_params=int(total_params),
        total_params_b=total_params / 1e9,
        active_params=int(active_params),
        active_params_b=active_params / 1e9,
        weight_size_bytes=weight_size_bytes,
        weight_size_gb=weight_size_bytes / (1024 ** 3),
        breakdown={
            "embedding": int(embed_params),
            "dense_layers": int(total_dense_params),
            "moe_layers": int(total_moe_params),
            "lm_head": int(lm_head_params),
            "num_dense_layers": num_dense_layers,
            "num_moe_layers": num_moe_layers,
        }
    )


# ============================================
# 任务管理
# ============================================


@router.post("/evaluation/submit")
async def submit_evaluation(request: EvaluationRequest):
    """提交评估任务"""
    from math_model.L0_entry.engine import run_evaluation_from_request

    # 构建配置快照
    # 提取模型和推理配置（兼容前端）
    model_config = request.benchmark_config.get("model", {}) if request.benchmark_config else {}
    inference_config = request.benchmark_config.get("inference", {}) if request.benchmark_config else {}

    config_snapshot = {
        "experiment_name": request.experiment_name,
        "description": request.description,
        "benchmark_name": request.benchmark_name,
        "topology_config_name": request.topology_config_name,
        "benchmark_config": request.benchmark_config,
        "topology_config": request.topology_config,
        # 前端兼容字段（Results 页面期望这些字段）
        "topology": request.topology_config,  # 别名，前端使用 topology
        "model": model_config,                # 前端使用 model
        "inference": inference_config,        # 前端使用 inference
        "search_mode": request.search_mode,
        "manual_parallelism": request.manual_parallelism.model_dump() if request.manual_parallelism else None,
        "search_constraints": request.search_constraints.model_dump() if request.search_constraints else None,
        "max_workers": request.max_workers,
        "enable_tile_search": request.enable_tile_search,
        "enable_partition_search": request.enable_partition_search,
        "max_simulated_tokens": request.max_simulated_tokens,
    }

    # 提交任务到线程池
    task_manager = get_task_manager()
    task_id = task_manager.submit(
        name=request.experiment_name,
        func=run_evaluation_from_request,
        config=config_snapshot,
        config_snapshot=config_snapshot,
    )

    # 创建数据库记录
    db_experiment_id = None
    db_task_id = None
    try:
        with get_db_session() as db:
            # 创建实验
            db_experiment = DBExperiment(
                name=request.experiment_name or f"Tier6 Experiment",
                description=request.description or "",
                total_tasks=1,
                completed_tasks=0,
            )
            db.add(db_experiment)
            db.flush()  # 获取 ID
            db_experiment_id = db_experiment.id

            # 创建任务记录
            db_task = DBEvaluationTask(
                task_id=task_id,
                experiment_id=db_experiment.id,
                status=DBTaskStatus.PENDING,
                progress=0.0,
                config_snapshot=config_snapshot,
                benchmark_name=request.benchmark_name,
                topology_config_name=request.topology_config_name,
                search_mode=request.search_mode or "manual",
                manual_parallelism=request.manual_parallelism.model_dump() if request.manual_parallelism else None,
                search_constraints=request.search_constraints.model_dump() if request.search_constraints else None,
            )
            db.add(db_task)
            db.flush()
            db_task_id = db_task.id

            db.commit()
            logger.info(f"Created experiment {db_experiment_id} with task {task_id}")
    except Exception as e:
        logger.error(f"Failed to create database records: {e}")

    # 添加任务完成回调
    def on_task_complete(task_info: TaskInfo) -> None:
        """任务完成时保存结果到数据库"""
        if task_info.status == TaskStatus.COMPLETED and task_info.result:
            _save_task_result_to_db(task_id, db_task_id, db_experiment_id, task_info)
        elif task_info.status == TaskStatus.FAILED:
            _update_task_status_in_db(task_id, DBTaskStatus.FAILED, task_info.error)

    task_manager.add_callback(task_id, on_task_complete)

    return {
        "task_id": task_id,
        "experiment_id": db_experiment_id,
        "status": TaskStatus.PENDING.value,
        "progress": 0.0,
    }


@router.get("/evaluation/tasks")
async def list_tasks(
    status: str | None = None,
    skip: int = 0,
    limit: int = 100,
):
    """列出任务"""
    task_manager = get_task_manager()

    task_status = TaskStatus(status) if status else None
    all_tasks = task_manager.list_tasks(status=task_status)

    # 获取总数（分页前）
    total = len(all_tasks)

    # 应用分页
    tasks = all_tasks[skip:skip + limit]

    return {
        "tasks": [t.to_dict() for t in tasks],
        "total": total,
    }


@router.get("/evaluation/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


@router.get("/evaluation/tasks/{task_id}/results")
async def get_task_results(task_id: str):
    """获取任务结果"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Status: {task.status.value}",
        )

    return task.result


@router.post("/evaluation/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    task_manager = get_task_manager()

    if not task_manager.cancel(task_id):
        raise HTTPException(status_code=400, detail="Failed to cancel task")

    return {"message": "Task cancelled", "task_id": task_id}


@router.delete("/evaluation/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # 取消运行中的任务
    if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        task_manager.cancel(task_id)

    # 从管理器中移除
    task_manager._tasks.pop(task_id, None)
    task_manager._futures.pop(task_id, None)
    task_manager._callbacks.pop(task_id, None)

    return {"message": "Task deleted", "task_id": task_id}


@router.get("/evaluation/running")
async def get_running_tasks():
    """获取运行中的任务"""
    task_manager = get_task_manager()
    tasks = task_manager.list_tasks(status=TaskStatus.RUNNING)

    return {
        "tasks": [t.to_dict() for t in tasks],
        "count": len(tasks),
    }


@router.get("/evaluation/config")
async def get_executor_config():
    """获取执行器配置"""
    task_manager = get_task_manager()

    return {
        "max_workers": task_manager.max_workers,
        "max_queued": task_manager.max_queued,
    }


@router.put("/evaluation/config")
async def update_executor_config(request: ExecutorConfigRequest):
    """更新执行器配置"""
    task_manager = get_task_manager()

    # 注意: 更新配置需要重建线程池，这里简单更新属性
    # 实际生产环境需要更复杂的处理
    task_manager.max_workers = request.max_workers
    task_manager.max_queued = request.max_queued

    return {
        "message": "Config updated",
        "max_workers": request.max_workers,
        "max_queued": request.max_queued,
    }


# ============================================
# 实验管理
# ============================================

@router.get("/evaluation/experiments")
async def list_experiments(skip: int = 0, limit: int = 20):
    """列出实验"""
    try:
        with get_db_session() as db:
            total = db.query(DBExperiment).count()
            experiments = db.query(DBExperiment).order_by(
                DBExperiment.created_at.desc()
            ).offset(skip).limit(limit).all()

            return {
                "experiments": [
                    {
                        "id": exp.id,
                        "name": exp.name,
                        "description": exp.description,
                        "total_tasks": exp.total_tasks,
                        "completed_tasks": exp.completed_tasks,
                        "created_at": exp.created_at.isoformat() + 'Z' if exp.created_at else None,
                    }
                    for exp in experiments
                ],
                "total": total,
            }
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/experiments/{experiment_id}")
async def get_experiment(experiment_id: int):
    """获取实验详情（含所有评估结果）"""
    try:
        with get_db_session() as db:
            experiment = db.query(DBExperiment).filter(
                DBExperiment.id == experiment_id
            ).first()
            if not experiment:
                raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

            tasks = db.query(DBEvaluationTask).filter(
                DBEvaluationTask.experiment_id == experiment_id
            ).all()

            # 展开每个任务的所有结果为单独条目
            tasks_with_results = []
            for task in tasks:
                results = db.query(DBEvaluationResult).filter(
                    DBEvaluationResult.task_id == task.id
                ).order_by(DBEvaluationResult.score.desc()).all()

                if results:
                    for idx, result in enumerate(results):
                        full_result = result.full_result or {}
                        task_dict = {
                            "id": task.id,
                            "task_id": task.task_id,
                            "result_id": result.id,
                            "result_rank": idx + 1,
                            "experiment_id": task.experiment_id,
                            "status": task.status.value,
                            "progress": task.progress,
                            "message": task.message,
                            "error": task.error,
                            "created_at": result.created_at.isoformat() + 'Z' if result.created_at else None,
                            "started_at": task.started_at.isoformat() + 'Z' if task.started_at else None,
                            "completed_at": task.completed_at.isoformat() + 'Z' if task.completed_at else None,
                            "config_snapshot": task.config_snapshot,
                            "benchmark_name": task.benchmark_name,
                            "topology_config_name": task.topology_config_name,
                            "search_mode": task.search_mode,
                            "manual_parallelism": task.manual_parallelism,
                            "search_constraints": task.search_constraints,
                            "search_stats": task.search_stats,
                            "result": {
                                "tps": result.tps,
                                "tps_per_chip": result.tps_per_chip,
                                "tps_per_batch": result.tps_per_batch,
                                "tpot": result.tpot,
                                "ttft": result.ttft,
                                "mfu": result.mfu,
                                "mbu": full_result.get("mbu", 0),
                                "score": result.score,
                                "chips": result.chips,
                                "dram_occupy": result.dram_occupy,
                                "flops": result.flops,
                                "cost": full_result.get("cost"),
                                "parallelism": {
                                    "dp": result.dp,
                                    "tp": result.tp,
                                    "pp": result.pp,
                                    "ep": result.ep,
                                    "sp": result.sp,
                                    "moe_tp": result.moe_tp,
                                },
                            },
                        }
                        tasks_with_results.append(task_dict)

            return {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "total_tasks": experiment.total_tasks,
                "completed_tasks": experiment.completed_tasks,
                "created_at": experiment.created_at.isoformat() + 'Z' if experiment.created_at else None,
                "updated_at": experiment.updated_at.isoformat() + 'Z' if experiment.updated_at else None,
                "tasks": tasks_with_results,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/evaluation/experiments/{experiment_id}")
async def update_experiment(experiment_id: int, request: ExperimentUpdateRequest):
    """更新实验名称/描述"""
    try:
        with get_db_session() as db:
            experiment = db.query(DBExperiment).filter(
                DBExperiment.id == experiment_id
            ).first()
            if not experiment:
                raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

            if request.name is not None:
                experiment.name = request.name
            if request.description is not None:
                experiment.description = request.description

            db.commit()
            db.refresh(experiment)

            return {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "created_at": experiment.created_at.isoformat() + 'Z' if experiment.created_at else None,
                "updated_at": experiment.updated_at.isoformat() + 'Z' if experiment.updated_at else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/evaluation/experiments/{experiment_id}")
async def delete_experiment(experiment_id: int):
    """删除实验（级联删除任务和结果）"""
    try:
        with get_db_session() as db:
            experiment = db.query(DBExperiment).filter(
                DBExperiment.id == experiment_id
            ).first()
            if not experiment:
                raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

            db.delete(experiment)
            db.commit()
            return {"message": f"Experiment '{experiment.name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/experiments/batch-delete")
async def batch_delete_experiments(request: BatchDeleteRequest):
    """批量删除实验"""
    try:
        with get_db_session() as db:
            experiments = db.query(DBExperiment).filter(
                DBExperiment.id.in_(request.ids)
            ).all()

            deleted_count = len(experiments)
            deleted_ids = [exp.id for exp in experiments]
            for exp in experiments:
                db.delete(exp)

            db.commit()
            return {
                "deleted": deleted_ids,
                "count": deleted_count,
            }
    except Exception as e:
        logger.error(f"Failed to batch delete experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/experiments/{experiment_id}/results/batch-delete")
async def batch_delete_results(experiment_id: int, request: BatchDeleteRequest):
    """批量删除实验结果"""
    try:
        with get_db_session() as db:
            experiment = db.query(DBExperiment).filter(
                DBExperiment.id == experiment_id
            ).first()
            if not experiment:
                raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

            results = db.query(DBEvaluationResult).filter(
                DBEvaluationResult.id.in_(request.ids)
            ).all()

            deleted_count = len(results)
            deleted_ids = [r.id for r in results]
            for r in results:
                db.delete(r)

            db.commit()
            return {
                "deleted": deleted_ids,
                "count": deleted_count,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to batch delete results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation/experiments/export")
async def export_experiments(experiment_ids: str = ""):
    """导出实验"""
    try:
        with get_db_session() as db:
            if experiment_ids:
                ids = [int(id_str) for id_str in experiment_ids.split(",")]
                experiments = db.query(DBExperiment).filter(
                    DBExperiment.id.in_(ids)
                ).all()
            else:
                experiments = db.query(DBExperiment).all()

            export_data = {
                "version": "1.0",
                "export_time": datetime.now().isoformat(),
                "experiments": [],
            }

            for exp in experiments:
                exp_data = {
                    "name": exp.name,
                    "description": exp.description,
                    "created_at": exp.created_at.isoformat() + 'Z' if exp.created_at else None,
                    "tasks": [],
                }
                for task in exp.tasks:
                    task_data = {
                        "config_snapshot": task.config_snapshot,
                        "search_mode": task.search_mode,
                        "results": [],
                    }
                    for result in task.results:
                        task_data["results"].append(result.full_result or {})
                    exp_data["tasks"].append(task_data)
                export_data["experiments"].append(exp_data)

            return export_data
    except Exception as e:
        logger.error(f"Failed to export experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 临时文件存储 (用于导入流程)
_temp_imports: dict[str, dict[str, Any]] = {}


@router.post("/evaluation/experiments/check-import")
async def check_import(file: UploadFile = File(...)):
    """检查导入文件"""
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

    experiments = data.get("experiments", [])
    if not experiments:
        raise HTTPException(status_code=400, detail="No experiments found in file")

    # 检查冲突（从数据库查询）
    conflicts = []
    try:
        with get_db_session() as db:
            existing_names = {exp.name for exp in db.query(DBExperiment).all()}
            for exp in experiments:
                if exp.get("name", "") in existing_names:
                    conflicts.append(exp.get("name", ""))
    except Exception as e:
        logger.error(f"Failed to check conflicts: {e}")

    # 保存临时数据
    temp_id = str(uuid.uuid4())
    _temp_imports[temp_id] = {
        "experiments": experiments,
        "created_at": datetime.now().isoformat(),
    }

    return {
        "temp_file_id": temp_id,
        "experiments": [
            {"name": e.get("name", ""), "task_count": len(e.get("tasks", []))}
            for e in experiments
        ],
        "conflicts": conflicts,
    }


@router.post("/evaluation/experiments/execute-import")
async def execute_import(request: ImportExecuteRequest):
    """执行导入"""
    temp_data = _temp_imports.pop(request.temp_file_id, None)
    if not temp_data:
        raise HTTPException(status_code=404, detail="Import session expired or not found")

    experiments = temp_data["experiments"]
    imported = []
    skipped = []

    try:
        with get_db_session() as db:
            existing_names = {exp.name: exp for exp in db.query(DBExperiment).all()}

            for exp in experiments:
                exp_name = exp.get("name", "")

                if exp_name in existing_names:
                    if request.conflict_strategy == "skip":
                        skipped.append(exp_name)
                        continue
                    elif request.conflict_strategy == "overwrite":
                        db.delete(existing_names[exp_name])
                        db.flush()
                    elif request.conflict_strategy == "rename":
                        exp_name = f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    else:
                        skipped.append(exp_name)
                        continue

                # 创建实验
                db_exp = DBExperiment(
                    name=exp_name,
                    description=exp.get("description", ""),
                    total_tasks=len(exp.get("tasks", [])),
                )
                db.add(db_exp)
                db.flush()

                # 导入任务和结果
                for task_data in exp.get("tasks", []):
                    db_task = DBEvaluationTask(
                        task_id=str(uuid.uuid4()),
                        experiment_id=db_exp.id,
                        status=DBTaskStatus.COMPLETED,
                        progress=1.0,
                        config_snapshot=task_data.get("config_snapshot", {}),
                        search_mode=task_data.get("search_mode", "manual"),
                    )
                    db.add(db_task)
                    db.flush()

                    for result_data in task_data.get("results", []):
                        parallelism = result_data.get("parallelism", {})
                        chips = parallelism.get("tp", 1) * parallelism.get("pp", 1) * parallelism.get("dp", 1) * parallelism.get("ep", 1)
                        db_result = DBEvaluationResult(
                            task_id=db_task.id,
                            dp=parallelism.get("dp", 1),
                            tp=parallelism.get("tp", 1),
                            pp=parallelism.get("pp", 1),
                            ep=parallelism.get("ep", 1),
                            sp=parallelism.get("sp", 1),
                            moe_tp=parallelism.get("moe_tp"),
                            chips=chips,
                            total_elapse_us=float(result_data.get("total_elapse_us", 0)),
                            total_elapse_ms=float(result_data.get("total_elapse_ms", 0)),
                            comm_elapse_us=float(result_data.get("comm_elapse_us", 0)),
                            tps=float(result_data.get("tps", 0)),
                            tps_per_batch=float(result_data.get("tps_per_batch", 0)),
                            tps_per_chip=float(result_data.get("tps_per_chip", 0)),
                            ttft=float(result_data.get("ttft", 0)),
                            tpot=float(result_data.get("tpot", 0)),
                            mfu=float(result_data.get("mfu", 0)),
                            flops=float(result_data.get("flops", 0)),
                            dram_occupy=float(result_data.get("dram_occupy", 0)),
                            score=float(result_data.get("score", 0)),
                            is_feasible=1 if result_data.get("is_feasible", True) else 0,
                            infeasible_reason=result_data.get("infeasible_reason"),
                            full_result=result_data,
                        )
                        db.add(db_result)

                imported.append(exp_name)

            db.commit()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "imported": imported,
        "skipped": skipped,
        "count": len(imported),
    }


# ============================================
# WebSocket
# ============================================


@router.websocket("/ws/tasks")
async def websocket_tasks(websocket: WebSocket):
    """WebSocket 任务状态推送"""
    await websocket.accept()

    ws_manager = get_ws_manager()
    queue = ws_manager.subscribe()

    try:
        while True:
            # 等待消息
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.unsubscribe(queue)


# ============================================
# 辅助函数
# ============================================


def _save_task_result_to_db(
    task_uuid: str,
    db_task_id: int | None,
    db_experiment_id: int | None,
    task_info: TaskInfo,
) -> None:
    """保存任务结果到数据库

    Args:
        task_uuid: 任务 UUID
        db_task_id: 数据库任务 ID
        db_experiment_id: 数据库实验 ID
        task_info: 任务信息
    """
    result = task_info.result
    if not result:
        return

    try:
        with get_db_session() as db:
            # 查找任务记录
            db_task = db.query(DBEvaluationTask).filter(
                DBEvaluationTask.task_id == task_uuid
            ).first()

            if not db_task:
                logger.warning(f"Task not found in database: {task_uuid}")
                return

            # 更新任务状态
            db_task.status = DBTaskStatus.COMPLETED
            db_task.progress = 1.0
            db_task.completed_at = datetime.now()

            # 保存搜索统计
            search_stats = result.get("search_stats", {})
            db_task.search_stats = search_stats

            # 提取并保存结果
            top_k_plans = result.get("top_k_plans", [])
            for plan in top_k_plans:
                parallelism = plan.get("parallelism", {})
                # 性能指标在 aggregates 或直接在 plan 顶层
                aggregates = plan.get("aggregates", {})

                # 获取 TPS、MFU 等指标（优先从顶层获取，其次从 aggregates）
                tps = plan.get("tps", aggregates.get("tps", 0))
                mfu = plan.get("mfu", aggregates.get("mfu", 0))
                ttft = plan.get("ttft", aggregates.get("ttft_ms", aggregates.get("total_time_ms", 0)))
                tpot = plan.get("tpot", aggregates.get("tpot_ms", 0))
                score = plan.get("score", tps)

                # 从 aggregates 获取详细指标
                total_time_ms = aggregates.get("total_time_ms", 0)
                compute_time_ms = aggregates.get("compute_time_ms", 0)
                comm_time_ms = aggregates.get("comm_time_ms", 0)

                # 计算芯片数
                chips = parallelism.get("tp", 1) * parallelism.get("pp", 1) * parallelism.get("dp", 1) * parallelism.get("ep", 1)

                db_result = DBEvaluationResult(
                    task_id=db_task.id,
                    # 并行策略
                    dp=parallelism.get("dp", 1),
                    tp=parallelism.get("tp", 1),
                    pp=parallelism.get("pp", 1),
                    ep=parallelism.get("ep", 1),
                    sp=parallelism.get("sp", 1),
                    moe_tp=parallelism.get("moe_tp"),
                    # 资源
                    chips=chips,
                    # 性能指标
                    total_elapse_us=total_time_ms * 1000,  # ms -> us
                    total_elapse_ms=total_time_ms,
                    comm_elapse_us=comm_time_ms * 1000,  # ms -> us
                    tps=tps,
                    tps_per_batch=aggregates.get("tps_per_batch", tps),
                    tps_per_chip=aggregates.get("tps_per_chip", tps / max(chips, 1)),
                    ttft=ttft,
                    tpot=tpot,
                    mfu=mfu,
                    # 计算量
                    flops=aggregates.get("total_flops", 0),
                    dram_occupy=aggregates.get("dram_occupy_gb", aggregates.get("memory_gb", 0)),
                    # 得分
                    score=score,
                    is_feasible=1 if plan.get("is_feasible", True) else 0,
                    infeasible_reason=plan.get("infeasible_reason"),
                    # 完整结果
                    full_result=plan,
                )
                db.add(db_result)

            # 更新实验统计
            if db_experiment_id:
                db_experiment = db.query(DBExperiment).filter(
                    DBExperiment.id == db_experiment_id
                ).first()
                if db_experiment:
                    db_experiment.completed_tasks += 1

            db.commit()
            logger.info(f"Saved {len(top_k_plans)} results to database for task {task_uuid}")

    except Exception as e:
        logger.error(f"Failed to save result to database: {e}")


def _update_task_status_in_db(task_uuid: str, status: DBTaskStatus, error: str | None = None) -> None:
    """更新数据库中的任务状态

    Args:
        task_uuid: 任务 UUID
        status: 新状态
        error: 错误信息
    """
    try:
        with get_db_session() as db:
            db_task = db.query(DBEvaluationTask).filter(
                DBEvaluationTask.task_id == task_uuid
            ).first()

            if db_task:
                db_task.status = status
                db_task.error = error
                db_task.completed_at = datetime.now()
                db.commit()
                logger.info(f"Updated task {task_uuid} status to {status.value}")
    except Exception as e:
        logger.error(f"Failed to update task status: {e}")


def _count_topology_chips(config: dict[str, Any]) -> int:
    """计算拓扑中的芯片总数"""
    from .topology_format import count_chips
    return count_chips(config)
