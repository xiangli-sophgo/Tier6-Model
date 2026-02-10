# 08 - 系统集成与 TPUPerf 直接接入方案

## 1. 方案对比

有两种实现路径:

| | 方案 A: Python 全实现 | 方案 B: 直接接入 TPUPerf |
|--|---------------------|------------------------|
| **实现方式** | Python 重写仿真引擎 | 将 TPUPerf C++ 编译为 .so, Python 调用 |
| **工作量** | 10-18 周 | 3-5 周 |
| **维护成本** | 高 (两套代码) | 中 (只维护接口层) |
| **灵活性** | 高 (可随意修改) | 低 (受限于 TPUPerf 接口) |
| **性能** | 慢 50-100x | 与 TPUPerf 一致 |
| **指令生成** | 原生支持 | 需要额外适配 |
| **依赖** | 无外部依赖 | 依赖 SystemC + TPUPerf 编译 |
| **跨平台** | 天然跨平台 | 需要为每个平台编译 |

## 2. 方案 B: TPUPerf 直接接入

### 2.1 技术路线

```
Python (Tier6+Model)
    |
    v
pybind11 / ctypes 接口层
    |
    v
C++ 封装层 (新增)
    |
    v
TPUPerf C++ 仿真引擎 (现有)
    |
    v
SystemC 库
```

### 2.2 接口设计

需要在 TPUPerf 中新增一个 Python 绑定层:

```cpp
// tpu_perf_binding.cpp (新增文件)

#include <pybind11/pybind11.h>
#include "tpu_subsys.h"
#include "profiler.h"

namespace py = pybind11;

class TPUPerfWrapper {
public:
    // 初始化: 设置芯片参数
    void init(const std::string& chip_arch, int core_num,
              const std::string& config_json);

    // 加载指令
    void load_commands(int core_id,
                       const std::string& tiu_file,
                       const std::string& dma_file);

    // 运行仿真
    SimResult run();

    // 获取结果
    py::dict get_profiling_data();
    py::list get_gantt_events();
    int get_total_cycles();
};

PYBIND11_MODULE(tpu_perf, m) {
    py::class_<TPUPerfWrapper>(m, "TPUPerf")
        .def(py::init<>())
        .def("init", &TPUPerfWrapper::init)
        .def("load_commands", &TPUPerfWrapper::load_commands)
        .def("run", &TPUPerfWrapper::run)
        .def("get_profiling_data", &TPUPerfWrapper::get_profiling_data)
        .def("get_gantt_events", &TPUPerfWrapper::get_gantt_events)
        .def("get_total_cycles", &TPUPerfWrapper::get_total_cycles);
}
```

### 2.3 封装层需要解决的问题

**问题 1: SystemC 的 sc_main 只能调用一次**

TPUPerf 的入口是 `sc_main()`, SystemC 规范要求只能调用一次。需要:
- 方案: 将仿真逻辑从 `sc_main` 提取为独立函数
- 或: 使用子进程模式, 每次仿真 fork 一个新进程

**问题 2: 全局状态**

TPUPerf 使用大量全局变量 (`Utility` 类的静态成员)。需要:
- 方案 1: 每次仿真前重置全局状态
- 方案 2: 封装为 Singleton, 提供 reset 接口

**问题 3: 输出捕获**

TPUPerf 输出到文件 (CSV/JSON/tiuRegInfo)。需要:
- 将输出重定向到内存 buffer
- 解析 profiler 输出为 Python 字典

**问题 4: SystemC 编译依赖**

用户需要安装 SystemC 才能编译。需要:
- 提供预编译的 .so/.dylib
- 或: 在 CI 中构建, 发布 wheel 包

### 2.4 实现步骤

```
阶段 1 (1 周): 接口设计
  - 定义 Python 调用接口
  - 确定数据交换格式

阶段 2 (1-2 周): C++ 封装层
  - 从 sc_main 提取核心逻辑
  - 解决全局状态问题
  - 实现 pybind11 绑定

阶段 3 (1 周): Python 集成
  - 封装为 Tier6+Model 的后端
  - 结果适配到现有 Gantt/图表系统
  - API 接口对接

阶段 4 (0.5 周): 构建和发布
  - CMake 集成 pybind11
  - 多平台编译测试
```

### 2.5 方案 B 的限制

1. **无法自动生成指令**: TPUPerf 只能读取二进制文件, 不能接受 Python 生成的指令对象
   - 解决: Python 侧生成指令 -> 写入二进制文件 -> TPUPerf 读取
   - 或: 扩展接口, 直接传递指令数组

2. **调试困难**: C++ 仿真内部状态不透明
   - 解决: 增加更多 profiling 输出

3. **修改受限**: 如果需要修改仿真逻辑, 需要改 C++ 代码
   - 解决: 对于常见的参数调整, 通过配置暴露

## 3. 推荐: 混合方案

**结合方案 A 和 B 的优势**:

```
短期 (1-2 月): 方案 B 快速接入
  - pybind11 封装 TPUPerf
  - 获得完整的仿真能力
  - 验证集成流程

中期 (3-6 月): 方案 A 逐步实现
  - Python 事件引擎
  - TIU 延迟计算 (从 tiuImpl.cc 翻译)
  - DMA 引擎 (简化版)
  - 用 TPUPerf 结果作为验证基准

长期: 方案 A 替代方案 B
  - Python 引擎精度达标后, 方案 B 变为可选后端
  - 保留两种模式供用户选择
```

## 4. 与现有 Tier6+Model 的集成

### 4.1 API 层设计

```python
# backend/math_model/L0_entry/api.py 中新增端点

@router.post("/api/instruction-simulate")
async def instruction_simulate(config: InstructionSimConfig):
    """指令级仿真接口

    输入:
    - command_source: "binary" | "auto_generate"
    - binary_files: {tiu: path, dma: path}  (binary 模式)
    - model_config: {...}                     (auto 模式)
    - chip_config: {...}
    - core_num: int
    - backend: "python" | "tpuperf"          (选择仿真后端)

    输出:
    - total_cycles: int
    - gantt_data: [...]
    - profiling: {...}
    - cost: {...}  (复用现有成本模型)
    """
```

### 4.2 结果适配

指令级仿真的输出需要适配到现有的结果系统:

```python
class InstructionSimResultAdapter:
    """将指令级仿真结果转换为 Tier6+Model 的标准格式

    输入 (仿真引擎输出):
    - per_instruction_events: [{cmd_id, type, start_cycle, end_cycle}, ...]
    - per_core_stats: [{tiu_cycles, dma_cycles, idle_cycles}, ...]
    - total_cycles: int

    输出 (Tier6+Model 标准格式):
    - gantt_data: 兼容现有 gantt.py 格式
    - performance_metrics: {tps, tpot, ttft, mfu}
    - bottleneck_analysis: {compute_bound, memory_bound, comm_bound}
    - cost_breakdown: 复用 CostEvaluator
    """
```

### 4.3 前端集成

```
Results 页面新增:
  - 仿真模式切换: [数学建模] [指令级仿真]
  - 指令上传: 拖放 .BD/.GDMA 文件
  - 自动生成选项: 选择模型 + 芯片 -> 自动生成指令
  - Gantt 图增强: 显示 TIU/DMA 指令级事件
  - 新增面板: TIU 利用率、DMA 带宽、Bank Conflict 统计
```

### 4.4 评估任务集成

```
现有评估任务流程:
  配置 -> submit -> 数学建模 -> 结果存储

新增指令级仿真流程:
  配置 -> submit -> 指令生成/加载 -> 指令级仿真 -> 结果适配 -> 结果存储

两种模式共享:
  - 任务队列 (ThreadPoolExecutor)
  - 结果存储 (SQLite)
  - 可视化 (Gantt/图表)
  - 实验管理 (对比/导出)
```

## 5. 文件组织

```
backend/
  math_model/
    instruction_simulator/       # 新增: 指令级仿真引擎
      __init__.py
      core/
        event_scheduler.py
        signal.py
        process.py
      engines/
        tiu_engine.py
        tiu_delay.py
        dma_engine.py
        dma_delay.py
      memory/
        lmem.py
        ddr.py
        cache.py
        address_map.py
      interconnect/
        bus.py
        c2c.py
        cdma.py
      command/
        binary_parser.py
        instruction_gen.py
        types.py
      top/
        single_core.py
        multi_core.py
        config_loader.py
      profiler/
        profiler.py
        gantt_adapter.py
      tpuperf_backend/          # 方案 B: TPUPerf 封装
        __init__.py
        wrapper.py              # Python 调用接口
        build.py                # 编译脚本
```
