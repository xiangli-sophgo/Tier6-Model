"""数据库迁移脚本: 3层结构 -> 2层结构

将旧的 Experiment → Task → Result 结构迁移到新的 Experiment → Result 结构

使用方法:
    python migrate_database.py
"""

import shutil
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 数据库路径
DB_DIR = Path(__file__).parent / "perf_model" / "data"
DB_PATH = DB_DIR / "llm_evaluations.db"
BACKUP_PATH = DB_DIR / f"llm_evaluations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"


def backup_database():
    """备份数据库"""
    if not DB_PATH.exists():
        print(f"[INFO] 数据库文件不存在: {DB_PATH}")
        print("[INFO] 将创建新数据库")
        return False

    print(f"[INFO] 备份数据库: {DB_PATH} -> {BACKUP_PATH}")
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"[OK] 备份完成")
    return True


def migrate_database():
    """执行数据库迁移"""

    # 1. 备份
    has_existing_data = backup_database()

    if not has_existing_data:
        # 如果没有现有数据库，直接初始化新结构
        print("[INFO] 初始化新数据库结构...")
        from perf_model.L0_entry.database import init_db
        init_db()
        print("[OK] 新数据库结构创建完成")
        return

    # 2. 连接数据库
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 3. 检查旧表是否存在
        result = session.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluation_tasks'"
        ))
        has_old_structure = result.fetchone() is not None

        if not has_old_structure:
            print("[INFO] 数据库已经是新结构，无需迁移")
            return

        print("[INFO] 检测到旧的3层结构，开始迁移...")

        # 4. 读取旧数据
        print("[INFO] 读取旧数据...")

        # 读取experiments
        experiments = session.execute(text(
            "SELECT id, name, description, created_at, updated_at FROM experiments"
        )).fetchall()
        print(f"[INFO] 读取 {len(experiments)} 个实验")

        # 读取tasks
        tasks = session.execute(text(
            "SELECT id, task_id, experiment_id, status, progress, message, error, "
            "config_snapshot, benchmark_name, topology_config_name, search_mode, "
            "manual_parallelism, search_constraints, search_stats, "
            "created_at, started_at, completed_at FROM evaluation_tasks"
        )).fetchall()
        print(f"[INFO] 读取 {len(tasks)} 个任务")

        # 读取results
        results = session.execute(text(
            "SELECT id, task_id, dp, tp, pp, ep, sp, moe_tp, chips, "
            "total_elapse_us, total_elapse_ms, comm_elapse_us, "
            "tps, tps_per_batch, tps_per_chip, ttft, tpot, mfu, "
            "flops, dram_occupy, score, is_feasible, infeasible_reason, "
            "full_result, created_at FROM evaluation_results"
        )).fetchall()
        print(f"[INFO] 读取 {len(results)} 个结果")

        # 5. 创建task_id到task数据的映射
        task_map = {task[0]: task for task in tasks}  # task.id -> task_row

        # 6. 删除旧表
        print("[INFO] 删除旧表结构...")
        session.execute(text("DROP TABLE IF EXISTS evaluation_results"))
        session.execute(text("DROP TABLE IF EXISTS evaluation_tasks"))
        session.execute(text("DROP TABLE IF EXISTS experiments"))
        session.commit()
        print("[OK] 旧表已删除")

        # 7. 创建新表结构
        print("[INFO] 创建新表结构...")
        from perf_model.L0_entry.database import Base, engine as new_engine
        Base.metadata.create_all(bind=new_engine)
        print("[OK] 新表结构创建完成")

        # 8. 迁移数据
        print("[INFO] 迁移数据...")

        # 迁移experiments（移除total_tasks/completed_tasks字段）
        for exp in experiments:
            session.execute(text(
                "INSERT INTO experiments (id, name, description, created_at, updated_at) "
                "VALUES (:id, :name, :description, :created_at, :updated_at)"
            ), {
                "id": exp[0],
                "name": exp[1],
                "description": exp[2],
                "created_at": exp[3],
                "updated_at": exp[4],
            })
        print(f"[OK] 迁移 {len(experiments)} 个实验")

        # 迁移results（合并task字段）
        migrated_count = 0
        for result in results:
            task_id = result[1]  # result.task_id (FK to task.id)
            task = task_map.get(task_id)

            if not task:
                print(f"[WARN] 结果 {result[0]} 的任务 {task_id} 不存在，跳过")
                continue

            # task字段索引：
            # 0:id, 1:task_id(UUID), 2:experiment_id, 3:status, 4:progress, 5:message, 6:error,
            # 7:config_snapshot, 8:benchmark_name, 9:topology_config_name, 10:search_mode,
            # 11:manual_parallelism, 12:search_constraints, 13:search_stats,
            # 14:created_at, 15:started_at, 16:completed_at

            session.execute(text(
                """
                INSERT INTO evaluation_results (
                    experiment_id, task_id,
                    config_snapshot, benchmark_name, topology_config_name,
                    search_mode, manual_parallelism, search_constraints, search_stats,
                    dp, tp, pp, ep, sp, moe_tp, chips,
                    total_elapse_us, total_elapse_ms, comm_elapse_us,
                    tps, tps_per_batch, tps_per_chip, ttft, tpot, mfu,
                    flops, dram_occupy, score, is_feasible, infeasible_reason,
                    full_result, created_at
                ) VALUES (
                    :experiment_id, :task_id,
                    :config_snapshot, :benchmark_name, :topology_config_name,
                    :search_mode, :manual_parallelism, :search_constraints, :search_stats,
                    :dp, :tp, :pp, :ep, :sp, :moe_tp, :chips,
                    :total_elapse_us, :total_elapse_ms, :comm_elapse_us,
                    :tps, :tps_per_batch, :tps_per_chip, :ttft, :tpot, :mfu,
                    :flops, :dram_occupy, :score, :is_feasible, :infeasible_reason,
                    :full_result, :created_at
                )
                """
            ), {
                # 从task获取
                "experiment_id": task[2],
                "task_id": task[1],  # UUID字符串
                "config_snapshot": task[7],
                "benchmark_name": task[8],
                "topology_config_name": task[9],
                "search_mode": task[10],
                "manual_parallelism": task[11],
                "search_constraints": task[12],
                "search_stats": task[13],
                # 从result获取
                "dp": result[2],
                "tp": result[3],
                "pp": result[4],
                "ep": result[5],
                "sp": result[6],
                "moe_tp": result[7],
                "chips": result[8],
                "total_elapse_us": result[9],
                "total_elapse_ms": result[10],
                "comm_elapse_us": result[11],
                "tps": result[12],
                "tps_per_batch": result[13],
                "tps_per_chip": result[14],
                "ttft": result[15],
                "tpot": result[16],
                "mfu": result[17],
                "flops": result[18],
                "dram_occupy": result[19],
                "score": result[20],
                "is_feasible": result[21],
                "infeasible_reason": result[22],
                "full_result": result[23],
                "created_at": result[24],
            })
            migrated_count += 1

        session.commit()
        print(f"[OK] 迁移 {migrated_count} 个结果")

        print("\n" + "=" * 60)
        print("[OK] 数据库迁移完成!")
        print(f"[INFO] 备份文件: {BACKUP_PATH}")
        print(f"[INFO] 新数据库: {DB_PATH}")
        print("=" * 60)

    except Exception as e:
        session.rollback()
        print(f"\n[FAIL] 迁移失败: {e}")
        print(f"[INFO] 可以从备份恢复: {BACKUP_PATH}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("数据库迁移脚本: 3层结构 -> 2层结构")
    print("=" * 60)
    print()

    try:
        migrate_database()
    except Exception as e:
        print(f"\n[FAIL] 迁移过程中出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
