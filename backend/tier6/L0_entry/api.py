"""API 接口模块

提供 FastAPI 路由定义。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tier6.L0_entry.config_loader import (
    get_config_loader,
    load_benchmark,
    load_chip_preset,
    load_model_preset,
    load_topology,
)
from tier6.L0_entry.tasks import TaskStatus, get_task_manager


# ============================================
# Pydantic 模型
# ============================================


class ParallelismConfigRequest(BaseModel):
    """并行配置请求"""

    dp: int = Field(1, ge=1)
    tp: int = Field(1, ge=1)
    pp: int = Field(1, ge=1)
    ep: int = Field(1, ge=1)
    sp: int = Field(1, ge=1)


class InferenceConfigRequest(BaseModel):
    """推理配置请求"""

    batch_size: int = Field(1, gt=0)
    input_seq_length: int = Field(1024, gt=0)
    output_seq_length: int = Field(128, gt=0)


class EvaluationRequest(BaseModel):
    """评估请求"""

    chip_preset: str | None = None
    model_preset: str | None = None
    topology_preset: str | None = None
    chip_params: dict[str, Any] | None = Field(None, alias="chip_config")
    model_params: dict[str, Any] | None = Field(None, alias="model_config")
    topology_params: dict[str, Any] | None = Field(None, alias="topology_config")
    parallelism: ParallelismConfigRequest = Field(default_factory=ParallelismConfigRequest)
    inference: InferenceConfigRequest = Field(default_factory=InferenceConfigRequest)


class TaskResponse(BaseModel):
    """任务响应"""

    task_id: str
    status: str
    progress: float
    error: str | None = None


class PresetListResponse(BaseModel):
    """预设列表响应"""

    presets: list[str]


# ============================================
# API 路由
# ============================================

router = APIRouter(prefix="/api", tags=["tier6"])


@router.get("/presets/chips", response_model=PresetListResponse)
async def list_chip_presets():
    """列出所有芯片预设"""
    loader = get_config_loader()
    return PresetListResponse(presets=loader.list_chip_presets())


@router.get("/presets/chips/{name}")
async def get_chip_preset(name: str):
    """获取芯片预设详情"""
    try:
        return load_chip_preset(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chip preset not found: {name}")


@router.get("/presets/models", response_model=PresetListResponse)
async def list_model_presets():
    """列出所有模型预设"""
    loader = get_config_loader()
    return PresetListResponse(presets=loader.list_model_presets())


@router.get("/presets/models/{name}")
async def get_model_preset(name: str):
    """获取模型预设详情"""
    try:
        return load_model_preset(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model preset not found: {name}")


@router.get("/presets/topologies", response_model=PresetListResponse)
async def list_topologies():
    """列出所有拓扑预设"""
    loader = get_config_loader()
    return PresetListResponse(presets=loader.list_topologies())


@router.get("/presets/topologies/{name}")
async def get_topology(name: str):
    """获取拓扑预设详情"""
    try:
        return load_topology(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Topology not found: {name}")


@router.get("/presets/benchmarks", response_model=PresetListResponse)
async def list_benchmarks():
    """列出所有 Benchmark"""
    loader = get_config_loader()
    return PresetListResponse(presets=loader.list_benchmarks())


@router.get("/presets/benchmarks/{name}")
async def get_benchmark(name: str):
    """获取 Benchmark 详情"""
    try:
        return load_benchmark(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {name}")


@router.post("/evaluation/submit", response_model=TaskResponse)
async def submit_evaluation(request: EvaluationRequest):
    """提交评估任务"""
    from tier6.L0_entry.engine import run_evaluation

    # 构建配置
    config = {
        "chip_preset": request.chip_preset,
        "model_preset": request.model_preset,
        "topology_preset": request.topology_preset,
        "chip_config": request.chip_params,
        "model_config": request.model_params,
        "topology_config": request.topology_params,
        "parallelism": request.parallelism.model_dump(),
        "inference": request.inference.model_dump(),
    }

    # 提交任务
    task_manager = get_task_manager()
    task_id = task_manager.submit(
        name="evaluation",
        func=run_evaluation,
        config=config,
        config_snapshot=config,
    )

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING.value,
        progress=0.0,
    )


@router.get("/evaluation/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        error=task.error,
    )


@router.get("/evaluation/tasks/{task_id}/result")
async def get_task_result(task_id: str):
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

    return {"message": "Task cancelled"}


@router.get("/evaluation/tasks")
async def list_tasks(
    status: str | None = None,
    limit: int = 100,
):
    """列出任务"""
    task_manager = get_task_manager()

    task_status = TaskStatus(status) if status else None
    tasks = task_manager.list_tasks(status=task_status, limit=limit)

    return {
        "tasks": [t.to_dict() for t in tasks],
        "total": len(tasks),
    }


# ============================================
# 健康检查
# ============================================


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "0.1.0"}
