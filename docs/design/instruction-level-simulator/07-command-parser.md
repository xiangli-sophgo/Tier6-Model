# 07 - 指令解析与生成

## 1. 指令来源

两种指令来源:
- **路径 A**: 解析 TPUPerf 编译器产出的二进制文件 (.BD/.GDMA)
- **路径 B**: 从 LLM 模型结构自动生成虚拟指令序列

## 2. 路径 A: 二进制指令解析

### 2.1 对标: TPUPerf 指令格式

**TIU 指令 (.BD 文件)**:

对标 `c_model/include/tpu/tpu_cmd.h` 中的 `parse_tpu_cmdbuf()`:

```
文件结构: 连续的变长指令序列

每条指令:
  byte[0] bit[0]: des_cmd_short (0=长格式, 1=短格式)
  byte[5] bit[1:4]: des_tsk_typ (指令类型)
  byte[5:6] bit[5:9]: des_tsk_eu_typ (执行单元类型)

指令类型映射:
  tsk_typ=0:  CONV (卷积)
  tsk_typ=1:  PorD (池化)
  tsk_typ=2:  MM/MM2 (矩阵乘, 由 eu_typ 区分)
    eu_typ=1: MM (v1)
    eu_typ=4: MM2.nn
    eu_typ=5: MM2.nt
    eu_typ=6: MM2.tt
  tsk_typ=3:  AR (算术)
  tsk_typ=4:  CMP (比较)
  tsk_typ=5:  SG (Scatter/Gather)
  tsk_typ=6:  SFU (特殊函数)
  tsk_typ=7:  LIN (线性)
  tsk_typ=8:  VC (向量乘)
  tsk_typ=9:  RQ (量化)
  tsk_typ=15: SYS (系统指令)
```

每种指令类型有自己的字段布局 (寄存器结构), 定义在:
- `c_model/include/sg2260/spec/include/bd_reg_def.h`
- `c_model/include/sg2260/spec/include/bd_reg_value.h`

**关键字段** (以 CONV 为例):
```
des_cmd_id:          指令自身 ID
des_cmd_id_dep:      依赖的 DMA 指令 ID
des_res0_addr:       结果地址 (LMEM)
des_opd0_addr:       操作数 0 地址 (LMEM)
des_opd1_addr:       操作数 1 地址 (LMEM)
des_res0_n/c/h/w:    结果 tensor shape
des_opd0_n/c/h/w:    操作数 tensor shape
des_opd0_prec:       操作数精度 (INT8/BF16/FP32)
```

**GDMA 指令 (.GDMA 文件)**:

对标 `parse_gdma_cmdbuf()`:

```
每条指令: 默认 96 字节, 短格式按类型有不同大小

关键字段:
  cmd_type:        DMA 命令类型 (tensor/matrix/cw_trans/...)
  cmd_id:          指令自身 ID
  cmd_id_dep:      依赖的 TIU 指令 ID
  src_start_addr:  源地址
  dst_start_addr:  目标地址
  src_nsize/csize/hsize/wsize:  源 tensor shape
  src_nstride/cstride/hstride/wstride:  源 tensor stride
  dst_*:           目标 tensor shape/stride
  data_format:     数据格式
  direction:       搬运方向 (LMEM->DDR, DDR->LMEM, LMEM->LMEM, DDR->DDR)
```

### 2.2 Python 二进制解析器

```python
class BinaryCommandParser:
    """二进制指令文件解析器

    支持:
    - .BD 文件 (TIU 指令)
    - .GDMA 文件 (DMA 指令)
    - 多核文件命名: basename.ext.0, basename.ext.1, ...

    解析流程:
    1. 读取文件到 byte buffer
    2. 逐条解析:
       a. 读取指令头 (判断类型和长度)
       b. 根据类型选择 decoder
       c. 解析字段到 TIUCommand / DMACommand 对象
    3. 返回命令列表
    """
```

**TIU 指令数据类**:
```python
class TIUCommand:
    cmd_id: int
    cmd_id_dep: int        # 依赖的 DMA cmd_id
    tsk_typ: int           # 指令大类
    eu_typ: int            # 执行单元子类
    # 结果/操作数地址和 shape
    res0_addr: int
    res0_n: int; res0_c: int; res0_h: int; res0_w: int
    opd0_addr: int
    opd0_n: int; opd0_c: int; opd0_h: int; opd0_w: int
    opd1_addr: int
    precision: int         # 数据精度
    # ... 其他字段根据指令类型不同
```

**DMA 指令数据类**:
```python
class DMACommand:
    cmd_id: int
    cmd_id_dep: int        # 依赖的 TIU cmd_id
    cmd_type: int          # DMA 类型 (tensor/matrix/...)
    src_addr: int
    dst_addr: int
    src_shape: tuple       # (n, c, h, w)
    src_stride: tuple      # (n_stride, c_stride, h_stride, w_stride)
    dst_shape: tuple
    dst_stride: tuple
    data_format: int
    direction: int         # 搬运方向
```

### 2.3 寄存器结构定义

需要从 TPUPerf 的头文件翻译寄存器位域定义:

```
来源文件:
  sg2260/spec/include/bd_reg_def.h      -> TIU 寄存器定义
  sg2260/spec/include/gdma_reg_def.h    -> GDMA 寄存器定义
  sg2260/spec/include/cdma_reg_def.h    -> CDMA 寄存器定义

翻译方式:
  C 位域结构 -> Python struct.unpack + 位操作
  或使用 ctypes.Structure 保持接近 C 的定义
```

## 3. 路径 B: 从 LLM 模型生成指令

### 3.1 设计思路

从 LLM 模型的计算图自动生成 TIU/DMA 指令序列, 类似一个**简化编译器**。

这是 Tier6+Model 的独特价值: TPUPerf 需要真实编译器产出, 而本项目可以从模型定义直接生成。

### 3.2 LLM 推理的计算图

以 Transformer 一层为例:

```
Prefill 阶段 (一层):

  1. QKV Projection: X(B,S,H) @ W_qkv(H,3H) -> QKV(B,S,3H)
     -> TIU: MM2.nn 指令
     -> DMA: 加载 W_qkv 到 LMEM, 加载 X 到 LMEM, 存储 QKV

  2. Attention Score: Q(B,S,H) @ K^T(B,H,S) -> Score(B,S,S)
     -> TIU: MM2.nt 指令
     -> DMA: reshape/transpose 数据

  3. Softmax: softmax(Score)
     -> TIU: SFU (exp) + AR (sum) + AR (div)

  4. Attention Output: Score(B,S,S) @ V(B,S,H) -> Attn(B,S,H)
     -> TIU: MM2.nn 指令

  5. Output Projection: Attn(B,S,H) @ W_o(H,H) -> Out(B,S,H)
     -> TIU: MM2.nn 指令

  6. FFN: Out @ W1(H,4H) -> GELU -> @ W2(4H,H)
     -> TIU: MM2.nn + SFU (gelu) + MM2.nn
     -> DMA: 加载 W1, W2
```

### 3.3 指令生成器

```python
class InstructionGenerator:
    """从 LLM 模型配置生成 TIU/DMA 指令序列

    输入:
    - model_config: 模型配置 (hidden_size, num_layers, num_heads, ...)
    - chip_config: 芯片配置 (lmem_size, lane_num, ...)
    - inference_config: 推理配置 (batch_size, seq_len, ...)
    - parallelism: 并行策略 (TP/PP/DP)

    输出:
    - tiu_commands: list[TIUCommand]  (每核)
    - dma_commands: list[DMACommand]  (每核)

    生成流程:
    1. 构建计算图 (层级: Model -> Layer -> Operation)
    2. Tiling: 将大矩阵切分为适合 LMEM 的块
    3. 调度: 确定 TIU/DMA 执行顺序和依赖关系
    4. 生成: 为每个操作生成具体的 TIU/DMA 指令
    """
```

### 3.4 Tiling 策略

大矩阵无法一次放入 LMEM, 需要分块:

```
MatMul: C(M,N) = A(M,K) @ B(K,N)

LMEM 容量限制:
  A_tile + B_tile + C_tile <= LMEM_SIZE (per lane)

Tiling 维度:
  M 方向: tile_m = min(M, max_m)  受 lane_num 限制
  K 方向: tile_k = min(K, max_k)  受 LMEM 限制
  N 方向: tile_n = min(N, max_n)  受 eu_num 限制

循环嵌套:
  for m_tile in range(0, M, tile_m):
    for n_tile in range(0, N, tile_n):
      DMA: load C_tile (partial sum, if k > 0)
      for k_tile in range(0, K, tile_k):
        DMA: load A_tile(m_tile, k_tile)
        DMA: load B_tile(k_tile, n_tile)
        TIU: MM2.nn(A_tile, B_tile, C_tile)
      DMA: store C_tile
```

### 3.5 cmd_id 依赖生成

```
规则:
- TIU 指令的 cmd_id_dep = 它依赖的最近一条 DMA 指令的 cmd_id
  (即: "我需要的数据由这条 DMA 加载")

- DMA 指令的 cmd_id_dep = 它依赖的最近一条 TIU 指令的 cmd_id
  (即: "我需要存储的数据由这条 TIU 计算")

- 不相关的 TIU/DMA 指令可以并行执行 (dep 指向更早的指令)

示例:
  DMA_0: load A_0           cmd_id=0, dep=0
  DMA_1: load B_0           cmd_id=1, dep=0
  TIU_0: matmul(A_0, B_0)   cmd_id=0, dep=1  (等 DMA_1 完成)
  DMA_2: load A_1           cmd_id=2, dep=0  (与 TIU_0 并行)
  DMA_3: store C_0          cmd_id=3, dep=0  (等 TIU_0... 需要更精确的 dep)
  TIU_1: matmul(A_1, B_0)   cmd_id=1, dep=2  (等 DMA_2 完成)
```

### 3.6 MoE/MLA 特殊处理

**MoE (Mixture of Experts)**:
```
额外生成:
1. Router: TIU AR 指令 (小矩阵乘 + topk)
2. Token Dispatch: DMA scatter/gather 指令 (按 expert 分发 token)
3. Expert FFN: 每个激活的 expert 生成独立的 MM2 指令
4. Token Combine: DMA gather 指令 (合并 expert 输出)
5. 如果 EP > 1: CDMA send/recv 指令 (跨核传输 token)
```

**MLA (Multi-head Latent Attention)**:
```
额外生成:
1. KV 压缩: MM2 指令 (hidden -> kv_lora_rank)
2. Q LoRA: MM2 指令 (hidden -> q_lora_rank -> head_dim)
3. 减少的 KV cache 加载: DMA 指令 (只加载压缩后的 KV)
```

## 4. 指令文件 I/O

### 4.1 读取

```python
# 读取二进制文件
tiu_cmds = BinaryCommandParser.parse_tiu("model.BD")
dma_cmds = BinaryCommandParser.parse_dma("model.GDMA")

# 多核: 按 .0, .1, ... 后缀
for core_id in range(core_num):
    tiu_cmds[core_id] = parser.parse_tiu(f"model.BD.{core_id}")
    dma_cmds[core_id] = parser.parse_dma(f"model.GDMA.{core_id}")
```

### 4.2 自动生成

```python
# 从模型配置生成
generator = InstructionGenerator(model_config, chip_config)
tiu_cmds, dma_cmds = generator.generate(
    batch_size=1,
    seq_len=4096,
    phase="prefill",     # prefill 或 decode
    parallelism={"tp": 8, "pp": 1}
)
```

### 4.3 导出 (可选)

```python
# 将生成的指令导出为二进制格式 (可供 TPUPerf 验证)
BinaryCommandWriter.write_tiu(tiu_cmds, "generated.BD")
BinaryCommandWriter.write_dma(dma_cmds, "generated.GDMA")
```
