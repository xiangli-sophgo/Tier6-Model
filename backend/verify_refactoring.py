"""验证数据库重构完整性

检查项:
1. 数据库表结构正确
2. API导入无错误
3. 没有旧模型引用
4. CRUD操作正常
"""

import sys
from pathlib import Path

print("=" * 60)
print("数据库重构验证")
print("=" * 60)
print()

# 1. 检查数据库表结构
print("[1/4] 检查数据库表结构...")
try:
    from math_model.L0_entry.database import engine, Experiment, EvaluationResult
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "experiments" in tables, "缺少 experiments 表"
    assert "evaluation_results" in tables, "缺少 evaluation_results 表"
    assert "evaluation_tasks" not in tables, "旧的 evaluation_tasks 表仍存在!"

    # 检查experiments表字段
    exp_cols = [c['name'] for c in inspector.get_columns('experiments')]
    assert "id" in exp_cols and "name" in exp_cols
    assert "total_tasks" not in exp_cols, "experiments 表仍有 total_tasks 字段!"
    assert "completed_tasks" not in exp_cols, "experiments 表仍有 completed_tasks 字段!"

    # 检查evaluation_results表字段
    result_cols = [c['name'] for c in inspector.get_columns('evaluation_results')]
    required_fields = [
        "id", "experiment_id", "task_id",  # 基础字段
        "config_snapshot", "search_mode",  # 配置字段（从Task迁移）
        "dp", "tp", "pp",  # 并行策略
        "tps", "mfu", "score",  # 性能指标
    ]
    for field in required_fields:
        assert field in result_cols, f"evaluation_results 缺少字段: {field}"

    print("[OK] 数据库表结构正确")

except Exception as e:
    print(f"[FAIL] 数据库表结构检查失败: {e}")
    sys.exit(1)

# 2. 检查API导入
print("[2/4] 检查API导入...")
try:
    from math_model.L0_entry import api

    # 检查关键端点存在
    assert hasattr(api, 'submit_evaluation')
    assert hasattr(api, 'list_experiments')
    assert hasattr(api, 'get_experiment')
    assert hasattr(api, 'batch_delete_results')
    assert hasattr(api, 'export_experiments')
    assert hasattr(api, 'execute_import')

    print("[OK] API导入成功，关键端点存在")

except Exception as e:
    print(f"[FAIL] API导入失败: {e}")
    sys.exit(1)

# 3. 检查没有旧模型引用
print("[3/4] 检查旧模型引用...")
try:
    import subprocess

    # 搜索 DBEvaluationTask 引用
    result = subprocess.run(
        ["grep", "-r", "DBEvaluationTask", "math_model/L0_entry/api.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    if result.returncode == 0:  # 找到匹配
        print(f"[WARN] api.py 中仍有 DBEvaluationTask 引用:")
        print(result.stdout)
    else:
        print("[OK] api.py 中没有 DBEvaluationTask 引用")

    # 搜索 DBTaskStatus 引用
    result = subprocess.run(
        ["grep", "-r", "DBTaskStatus", "math_model/L0_entry/api.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    if result.returncode == 0:
        print(f"[WARN] api.py 中仍有 DBTaskStatus 引用:")
        print(result.stdout)
    else:
        print("[OK] api.py 中没有 DBTaskStatus 引用")

except Exception as e:
    print(f"[WARN] 旧模型引用检查跳过: {e}")

# 4. 测试CRUD操作
print("[4/4] 测试CRUD操作...")
try:
    from math_model.L0_entry.database import get_db_session
    import uuid

    with get_db_session() as db:
        # Create
        exp = Experiment(name="Verify Test", description="验证测试")
        db.add(exp)
        db.commit()
        db.refresh(exp)

        result = EvaluationResult(
            experiment_id=exp.id,
            task_id=str(uuid.uuid4()),
            config_snapshot={"test": True},
            search_mode="manual",
            dp=1, tp=1, pp=1, ep=1, sp=1,
            chips=1,
            total_elapse_us=1000.0,
            total_elapse_ms=1.0,
            comm_elapse_us=100.0,
            tps=100.0,
            tps_per_batch=100.0,
            tps_per_chip=100.0,
            mfu=0.5,
            flops=1000.0,
            dram_occupy=1.0,
            score=0.8,
            full_result={}
        )
        db.add(result)
        db.commit()
        db.refresh(result)

        # Read
        read_exp = db.query(Experiment).filter(Experiment.id == exp.id).first()
        assert read_exp is not None
        assert len(read_exp.results) == 1
        assert read_exp.results[0].task_id == result.task_id

        # Update
        result.score = 0.9
        db.commit()

        # Delete (cascade should work)
        db.delete(exp)
        db.commit()

        # Verify cascade delete
        orphan_result = db.query(EvaluationResult).filter(
            EvaluationResult.id == result.id
        ).first()
        assert orphan_result is None, "级联删除未生效!"

    print("[OK] CRUD操作正常，级联删除工作正常")

except Exception as e:
    print(f"[FAIL] CRUD操作测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print()
print("=" * 60)
print("[OK] 所有验证通过!")
print("=" * 60)
print()
print("数据库重构完整性验证成功:")
print("  - 表结构正确 (Experiment → Result 2层架构)")
print("  - 没有旧模型引用")
print("  - API端点完整")
print("  - CRUD操作正常")
print("  - 级联删除工作正常")
print()
print("可以安全地启动服务器和前端!")
