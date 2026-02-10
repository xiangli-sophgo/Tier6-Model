# 02 - 事件驱动仿真内核

## 1. 对标: SystemC 调度模型

TPUPerf 使用 SystemC 提供的两种进程模型:

### SC_METHOD
- **特点**: 每次被触发时执行完整函数, 不可挂起
- **触发**: 敏感列表中的信号变化 (如 `clock.pos()`, `tiuSyncId`)
- **TPUPerf 用途**: TIU 状态机 (`Tiu::run`), DMA 命令调度 (`cmd_dispatch_mth`)

### SC_THREAD
- **特点**: 可以调用 `wait()` 挂起, 下次事件触发时从挂起点恢复
- **触发**: 显式 `wait(event)` 或 `wait(time)`
- **TPUPerf 用途**: DMA 数据通路 (32个线程), C2C 通信, Cache 处理

### Delta Cycle
- SystemC 在同一仿真时间内可以有多个 delta cycle
- 信号写入在当前 delta cycle 结束后才可见
- 保证了因果关系的正确性

## 2. Python 实现方案

### 2.1 EventScheduler 核心类

```python
class EventScheduler:
    """
    轻量级事件驱动调度器, 替代 SystemC 内核。

    支持两种进程:
    - Method: 每次触发执行完整函数 (对应 SC_METHOD)
    - Thread: 协程, 可通过 yield 挂起 (对应 SC_THREAD)

    支持两种事件:
    - 时间事件: 在指定仿真时间触发
    - 信号事件: 信号值变化时触发
    """
```

核心数据结构:
```
- current_time: int          # 当前仿真时间 (cycle)
- event_queue: heapq         # (time, priority, callback) 最小堆
- signal_waiters: dict       # signal_id -> [waiter_list]
- clock_methods: list        # 每周期执行的 method 列表
- pending_signals: list      # 本 delta cycle 内的信号更新
- running: bool              # 仿真是否在运行
```

### 2.2 时间推进模型

```
主循环:
  while event_queue 非空:
    1. 取出最早的事件时间 T
    2. current_time = T
    3. 执行所有 time=T 的事件
    4. Delta cycle 处理:
       while pending_signals 非空:
         a. 提交所有 pending 信号更新
         b. 触发信号敏感的 method/thread
         c. 收集新的 pending 信号更新
    5. 如果 T 是时钟边沿:
       a. 执行所有 clock_methods
       b. 再次做 delta cycle 处理
```

### 2.3 进程模型

**Method (对应 SC_METHOD)**:
```python
class Method:
    def __init__(self, func, sensitive_signals):
        self.func = func                    # 普通函数
        self.sensitive = sensitive_signals   # 敏感信号列表

    def trigger(self):
        self.func()  # 直接调用, 不可挂起
```

**Thread (对应 SC_THREAD)**:
```python
class Thread:
    def __init__(self, coroutine_func):
        self.coro = coroutine_func()  # 协程生成器
        self.waiting_for = None       # 当前等待的事件

    def resume(self):
        try:
            self.waiting_for = next(self.coro)  # 执行到下一个 yield
        except StopIteration:
            self.active = False
```

Thread 的使用方式:
```python
def dma_read_thread(self):
    while True:
        cmd = yield self.wait_fifo(self.cmd_fifo)      # 等待命令
        segments = self.segment_cmd(cmd)
        for seg in segments:
            yield self.wait_time(seg.delay)             # 等待延迟
            yield self.send_request(seg.addr, seg.size) # 发送请求
        yield self.signal_write(self.done_signal, cmd.id) # 通知完成
```

### 2.4 信号模型 (对应 sc_signal)

```python
class Signal:
    def __init__(self, name, init_value=0):
        self.name = name
        self.current_value = init_value
        self.next_value = init_value      # delta cycle 延迟更新
        self.waiters = []                 # 等待此信号变化的进程

    def read(self):
        return self.current_value

    def write(self, value):
        # 不立即生效, 放入 pending 队列
        self.next_value = value
        scheduler.add_pending_signal(self)

    def commit(self):
        if self.next_value != self.current_value:
            self.current_value = self.next_value
            # 唤醒所有等待者
            for waiter in self.waiters:
                scheduler.activate(waiter)
```

### 2.5 FIFO 模型 (对应 sc_fifo)

```python
class FIFO:
    def __init__(self, name, depth):
        self.name = name
        self.depth = depth
        self.buffer = deque()
        self.write_event = Event()   # 有数据写入
        self.read_event = Event()    # 有数据被读取 (空间释放)

    def write(self, data):
        # Thread 中: yield fifo.write(data) 可能阻塞
        while len(self.buffer) >= self.depth:
            yield self.wait_event(self.read_event)
        self.buffer.append(data)
        self.write_event.notify()

    def read(self):
        # Thread 中: data = yield fifo.read() 可能阻塞
        while len(self.buffer) == 0:
            yield self.wait_event(self.write_event)
        data = self.buffer.popleft()
        self.read_event.notify()
        return data
```

### 2.6 Semaphore 模型 (对应 sc_semaphore)

```python
class Semaphore:
    """用于 outstanding 控制"""
    def __init__(self, name, count):
        self.count = count
        self.max_count = count
        self.release_event = Event()

    def acquire(self):
        while self.count <= 0:
            yield self.wait_event(self.release_event)
        self.count -= 1

    def release(self):
        self.count += 1
        self.release_event.notify()
```

## 3. 时钟域支持

TPUPerf 有多个时钟域 (TPU clock, TIU clock, Bus clock, DDR clock)。

```python
class Clock:
    def __init__(self, name, period_ns):
        self.name = name
        self.period = period_ns
        self.methods = []       # 绑定到此时钟的 method

    def register_method(self, method):
        self.methods.append(method)
```

调度器在初始化时为每个时钟生成周期性事件:
```python
for t in range(0, max_sim_time, clock.period):
    scheduler.schedule_event(t, clock.tick)
```

## 4. 与 SystemC 的差异和简化

| SystemC 特性 | Python 实现 | 说明 |
|-------------|-------------|------|
| `sc_signal<T>` | `Signal` 类 | 保留 delta cycle 语义 |
| `SC_METHOD` | `Method` 类 | 保留敏感列表触发 |
| `SC_THREAD` | Python 协程 | `yield` 替代 `wait()` |
| `sc_fifo` | `FIFO` 类 | 保留阻塞读写 |
| `sc_semaphore` | `Semaphore` 类 | 保留信用控制 |
| `sc_time` | 整数 cycle | 简化为整数运算 |
| TLM-2.0 nb_transport | 请求-响应模型 | 简化 4 phase 为 2 phase |
| `sc_module` | Python 类 | 无特殊处理 |
| `sc_port/sc_export` | 直接引用 | Python 不需要端口绑定语法 |

**关键简化**: 不实现 SystemC 的模块层次结构和端口绑定机制。Python 直接通过对象引用传递, 比 SystemC 的 port binding 简单得多。
