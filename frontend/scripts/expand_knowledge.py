import pandas as pd
import json
import re
from collections import Counter

# 读取原始Excel
df = pd.read_excel(r'C:\Users\xiang\Downloads\名词定义.xlsx')
df.columns = ['name', 'definition', 'category_raw', 'notes', 'status', 'updated_at']

# 扩展的知识点
extended_nodes = [
    # 网络拓扑
    {"id": "torus", "name": "Torus", "definition": "环面拓扑，一种将节点排列成多维网格并在每个维度首尾相连形成环的网络拓扑结构。常用于超级计算机互联，如3D Torus。", "category": "topology"},
    {"id": "fat_tree", "name": "Fat-Tree", "definition": "胖树拓扑，一种多层交换机组成的网络结构，上层交换机带宽逐层增加以避免过载。广泛用于数据中心网络。", "category": "topology"},
    {"id": "dragonfly", "name": "Dragonfly", "definition": "蜻蜓拓扑，一种高效的网络拓扑，将节点分组成路由器组，组内全连接，组间通过全局链路连接。", "category": "topology"},
    {"id": "mesh", "name": "Mesh", "definition": "网格拓扑，节点排列成规则的多维网格，每个节点与相邻节点直接连接。是NoC中最常见的拓扑之一。", "category": "topology"},
    {"id": "ring", "name": "Ring", "definition": "环形拓扑，所有节点连成一个闭环，数据沿环单向或双向传输。结构简单但延迟随节点数线性增长。", "category": "topology"},
    {"id": "crossbar", "name": "Crossbar", "definition": "交叉开关，一种全连接的交换结构，任意输入端口可同时连接到任意输出端口，无阻塞但成本高。", "category": "topology"},

    # 互联技术
    {"id": "nvlink", "name": "NVLink", "definition": "NVIDIA的高速GPU互联技术，提供比PCIe更高的带宽（900GB/s），用于GPU间直接通信。", "category": "interconnect"},
    {"id": "nvswitch", "name": "NVSwitch", "definition": "NVIDIA的交换芯片，实现多GPU间的全连接NVLink网络，支持8-16个GPU全速互联。", "category": "interconnect"},
    {"id": "infinity_fabric", "name": "Infinity Fabric", "definition": "AMD的可扩展互联架构，连接CPU核心、GPU和I/O设备，支持一致性缓存访问。", "category": "interconnect"},
    {"id": "upi", "name": "UPI", "definition": "Ultra Path Interconnect，Intel的CPU间互联技术，取代QPI，用于多路服务器中CPU间通信。", "category": "interconnect"},
    {"id": "cxl", "name": "CXL", "definition": "Compute Express Link，基于PCIe物理层的开放互联标准，支持CPU与加速器、内存扩展设备间的缓存一致性访问。", "category": "interconnect"},
    {"id": "pcie", "name": "PCIe", "definition": "Peripheral Component Interconnect Express，高速串行计算机扩展总线标准，PCIe 5.0单通道带宽32GT/s。", "category": "interconnect"},
    {"id": "infiniband", "name": "InfiniBand", "definition": "高性能计算互联标准，提供低延迟、高带宽的RDMA通信，HDR速率达200Gb/s。", "category": "interconnect"},
    {"id": "roce", "name": "RoCE", "definition": "RDMA over Converged Ethernet，在以太网上实现RDMA的技术，RoCEv2使用UDP封装支持路由。", "category": "interconnect"},
    {"id": "ethernet", "name": "Ethernet", "definition": "以太网，最广泛使用的局域网技术，数据中心常用100GbE/400GbE/800GbE。", "category": "interconnect"},

    # 通信原语
    {"id": "allreduce", "name": "AllReduce", "definition": "集合通信原语，所有进程贡献数据进行规约运算（如求和），结果分发给所有进程。分布式训练中用于梯度同步。", "category": "collective"},
    {"id": "allgather", "name": "AllGather", "definition": "集合通信原语，每个进程发送数据给所有其他进程，最终每个进程都拥有所有数据的完整副本。", "category": "collective"},
    {"id": "alltoall", "name": "AllToAll", "definition": "集合通信原语，每个进程向每个其他进程发送不同的数据块，实现全排列交换。EP并行中用于专家路由。", "category": "collective"},
    {"id": "reduce_scatter", "name": "ReduceScatter", "definition": "集合通信原语，先对所有进程的数据进行规约，再将结果分散到各进程。是AllReduce的分解操作之一。", "category": "collective"},
    {"id": "broadcast", "name": "Broadcast", "definition": "集合通信原语，一个进程将数据发送给所有其他进程。常用于模型参数初始化分发。", "category": "collective"},
    {"id": "reduce", "name": "Reduce", "definition": "集合通信原语，所有进程的数据规约到一个根进程。与AllReduce区别是结果只在根进程。", "category": "collective"},
    {"id": "scatter", "name": "Scatter", "definition": "集合通信原语，根进程将数据分块发送给所有进程。与Broadcast区别是每个进程收到不同数据。", "category": "collective"},
    {"id": "gather", "name": "Gather", "definition": "集合通信原语，所有进程将数据发送到根进程汇总。是Scatter的逆操作。", "category": "collective"},

    # 大模型架构
    {"id": "transformer", "name": "Transformer", "definition": "一种基于自注意力机制的神经网络架构，由编码器和解码器组成，是现代大语言模型的基础。", "category": "model_arch"},
    {"id": "attention", "name": "Attention", "definition": "注意力机制，通过计算Query和Key的相似度来加权Value，使模型能关注输入的不同部分。", "category": "model_arch"},
    {"id": "mha", "name": "MHA", "definition": "Multi-Head Attention多头注意力，将注意力计算分成多个头并行执行，每个头学习不同的注意力模式。", "category": "model_arch"},
    {"id": "mqa", "name": "MQA", "definition": "Multi-Query Attention，所有注意力头共享同一组Key和Value，大幅减少KV Cache内存占用。", "category": "model_arch"},
    {"id": "gqa", "name": "GQA", "definition": "Grouped Query Attention，将Query头分组，每组共享一组Key/Value，是MHA和MQA的折中方案。", "category": "model_arch"},
    {"id": "mla", "name": "MLA", "definition": "Multi-head Latent Attention，DeepSeek-V2提出的注意力变体，通过低秩投影压缩KV Cache。", "category": "model_arch"},
    {"id": "moe", "name": "MoE", "definition": "Mixture of Experts混合专家，使用门控网络动态选择部分专家子网络激活，实现条件计算。", "category": "model_arch"},
    {"id": "ffn", "name": "FFN", "definition": "Feed-Forward Network前馈网络，Transformer中的两层全连接网络，占模型参数的2/3。", "category": "model_arch"},
    {"id": "kv_cache", "name": "KV Cache", "definition": "键值缓存，存储已生成token的Key和Value向量，避免重复计算，是推理加速的关键技术。", "category": "model_arch"},
    {"id": "rope", "name": "RoPE", "definition": "Rotary Position Embedding旋转位置编码，通过旋转矩阵编码相对位置信息，支持长度外推。", "category": "model_arch"},

    # 训练/推理优化
    {"id": "flash_attention", "name": "FlashAttention", "definition": "一种IO感知的精确注意力算法，通过分块计算和融合操作减少HBM访问，显著提升训练和推理效率。", "category": "optimization"},
    {"id": "paged_attention", "name": "PagedAttention", "definition": "vLLM提出的KV Cache管理技术，将KV Cache分页存储，实现动态内存分配和共享。", "category": "optimization"},
    {"id": "speculative_decoding", "name": "Speculative Decoding", "definition": "推测解码，用小模型快速生成多个候选token，大模型并行验证，加速自回归生成。", "category": "optimization"},
    {"id": "continuous_batching", "name": "Continuous Batching", "definition": "连续批处理，动态调整批次中的请求，已完成的请求立即退出，新请求随时加入。", "category": "optimization"},
    {"id": "quantization", "name": "Quantization", "definition": "量化，将模型权重和激活从高精度（FP32/FP16）转为低精度（INT8/INT4），减少内存和计算量。", "category": "optimization"},
    {"id": "gradient_checkpointing", "name": "Gradient Checkpointing", "definition": "梯度检查点，只保存部分层的激活值，反向传播时重新计算，用内存换计算。", "category": "optimization"},
    {"id": "zero", "name": "ZeRO", "definition": "Zero Redundancy Optimizer，DeepSpeed的优化器状态分片技术，消除数据并行中的冗余存储。", "category": "optimization"},

    # 硬件相关
    {"id": "hbm", "name": "HBM", "definition": "High Bandwidth Memory高带宽内存，采用3D堆叠技术，提供数TB/s的带宽，用于GPU和AI加速器。", "category": "hardware"},
    {"id": "npu", "name": "NPU", "definition": "Neural Processing Unit神经网络处理单元，专门优化神经网络计算的AI加速芯片。", "category": "hardware"},
    {"id": "gpu", "name": "GPU", "definition": "Graphics Processing Unit图形处理单元，大规模并行处理器，是深度学习训练和推理的主力硬件。", "category": "hardware"},
    {"id": "tpu", "name": "TPU", "definition": "Tensor Processing Unit张量处理单元，Google专为机器学习设计的ASIC，采用脉动阵列架构。", "category": "hardware"},
    {"id": "systolic_array", "name": "Systolic Array", "definition": "脉动阵列，一种高效的矩阵乘法硬件架构，数据在处理单元间有规律地流动。", "category": "hardware"},
    {"id": "roofline", "name": "Roofline Model", "definition": "屋顶线模型，用于分析程序性能瓶颈的可视化模型，横轴为计算强度，纵轴为性能。", "category": "hardware"},

    # 分布式系统
    {"id": "nccl", "name": "NCCL", "definition": "NVIDIA Collective Communications Library，NVIDIA的集合通信库，优化多GPU通信。", "category": "distributed"},
    {"id": "gloo", "name": "Gloo", "definition": "Facebook的集合通信库，支持CPU和GPU，是PyTorch分布式训练的后端之一。", "category": "distributed"},
    {"id": "mpi", "name": "MPI", "definition": "Message Passing Interface消息传递接口，分布式计算的标准通信接口，定义了点对点和集合通信原语。", "category": "distributed"},
    {"id": "rdma", "name": "RDMA", "definition": "Remote Direct Memory Access远程直接内存访问，绕过CPU直接访问远程内存，实现超低延迟通信。", "category": "distributed"},
    {"id": "gpudirect", "name": "GPUDirect", "definition": "NVIDIA的GPU直接通信技术，包括P2P（GPU间直传）和RDMA（GPU直接网络访问）。", "category": "distributed"},
]

# 分类映射
def map_category(name, definition):
    name_lower = name.lower() if isinstance(name, str) else ''

    if name_lower in ['tp', 'sp', 'ep', 'dp', 'pp']:
        return 'parallel_strategy'
    elif name_lower in ['scale up', 'scale out']:
        return 'scaling'
    elif name_lower in ['pod', 'rack', 'server'] or '1u' in name_lower or '2u' in name_lower:
        return 'hardware'
    elif name_lower in ['cbfc', 'pfc', 'fec', 'ecc', 'crc', 'icrc', 'fecn'] or 'go back' in name_lower:
        return 'flow_control'
    elif name_lower in ['gva', 'spa', 'npa', 'mmu']:
        return 'address_space'
    elif name_lower in ['prefill', 'decode']:
        return 'inference'
    elif name_lower in ['psn', 'urpc', 'rtp', 'utp', 'ctp', 'pcs']:
        return 'protocol'
    elif name_lower in ['pam4', 'nrz']:
        return 'encoding'
    else:
        return 'general'

# 从Excel生成节点
nodes = []
name_to_id = {}

for idx, row in df.iterrows():
    name = str(row['name']).strip() if pd.notna(row['name']) else ''
    definition = str(row['definition']) if pd.notna(row['definition']) else ''

    if not name or not definition or definition == 'nan':
        continue

    node_id = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', name.lower()).strip('_')
    if not node_id:
        node_id = f"node_{idx}"

    category = map_category(name, definition)

    node = {
        'id': node_id,
        'name': name,
        'definition': definition,
        'category': category
    }

    nodes.append(node)
    name_to_id[name.lower()] = node_id
    if len(name) <= 5:
        name_to_id[name] = node_id

# 添加扩展节点
for ext_node in extended_nodes:
    if ext_node['id'] not in [n['id'] for n in nodes]:
        nodes.append(ext_node)
        name_to_id[ext_node['name'].lower()] = ext_node['id']
        name_to_id[ext_node['id']] = ext_node['id']

print(f"总节点数: {len(nodes)}")

# 定义关系
manual_relations = [
    # 并行策略关系
    ("tp", "sp", "related_to"),
    ("tp", "dp", "contrasts_with"),
    ("tp", "pp", "related_to"),
    ("ep", "moe", "depends_on"),
    ("dp", "allreduce", "depends_on"),
    ("tp", "allreduce", "depends_on"),

    # 扩展方式
    ("scale_up", "scale_out", "contrasts_with"),
    ("scale_up", "nvlink", "related_to"),
    ("scale_out", "ethernet", "related_to"),
    ("scale_out", "infiniband", "related_to"),

    # 互联技术关系
    ("nvlink", "nvswitch", "related_to"),
    ("nvlink", "gpu", "related_to"),
    ("pcie", "cxl", "related_to"),
    ("infiniband", "rdma", "depends_on"),
    ("roce", "rdma", "depends_on"),
    ("roce", "ethernet", "depends_on"),
    ("ethernet", "pfc", "related_to"),

    # 拓扑关系
    ("torus", "mesh", "related_to"),
    ("fat_tree", "pod", "related_to"),
    ("ring", "mesh", "related_to"),
    ("dragonfly", "fat_tree", "contrasts_with"),

    # 通信原语关系
    ("allreduce", "reduce_scatter", "related_to"),
    ("allreduce", "allgather", "related_to"),
    ("alltoall", "ep", "related_to"),
    ("broadcast", "scatter", "contrasts_with"),
    ("gather", "scatter", "contrasts_with"),
    ("reduce", "allreduce", "related_to"),

    # 模型架构关系
    ("transformer", "attention", "depends_on"),
    ("transformer", "ffn", "depends_on"),
    ("mha", "attention", "related_to"),
    ("mqa", "mha", "related_to"),
    ("gqa", "mha", "related_to"),
    ("gqa", "mqa", "related_to"),
    ("mla", "mha", "related_to"),
    ("moe", "ffn", "related_to"),
    ("kv_cache", "attention", "related_to"),
    ("kv_cache", "prefill", "related_to"),
    ("kv_cache", "decode", "related_to"),
    ("rope", "attention", "related_to"),

    # 优化技术关系
    ("flash_attention", "attention", "related_to"),
    ("paged_attention", "kv_cache", "related_to"),
    ("speculative_decoding", "decode", "related_to"),
    ("continuous_batching", "decode", "related_to"),
    ("zero", "dp", "related_to"),

    # 硬件关系
    ("hbm", "gpu", "related_to"),
    ("hbm", "npu", "related_to"),
    ("gpu", "npu", "contrasts_with"),
    ("tpu", "systolic_array", "depends_on"),
    ("roofline", "hbm", "related_to"),

    # 分布式系统关系
    ("nccl", "allreduce", "related_to"),
    ("nccl", "nvlink", "related_to"),
    ("gloo", "mpi", "related_to"),
    ("rdma", "gpudirect", "related_to"),
    ("gpudirect", "nvlink", "related_to"),

    # 流控纠错关系
    ("ecc", "crc", "related_to"),
    ("fec", "crc", "related_to"),
    ("pfc", "cbfc", "related_to"),
    ("fecn", "pfc", "related_to"),

    # 推理阶段关系
    ("prefill", "decode", "related_to"),
    ("prefill", "tp", "related_to"),
    ("decode", "kv_cache", "depends_on"),

    # 地址空间关系
    ("gva", "spa", "related_to"),
    ("spa", "npa", "related_to"),
    ("mmu", "gva", "related_to"),

    # 硬件架构关系
    ("pod", "rack", "related_to"),
    ("rack", "server", "related_to"),
]

# 构建关系
relations = []
relation_set = set()
node_ids = set(n['id'] for n in nodes)

for source, target, rel_type in manual_relations:
    source_id = name_to_id.get(source.lower(), source)
    target_id = name_to_id.get(target.lower(), target)

    if source_id in node_ids and target_id in node_ids:
        rel_key = tuple(sorted([source_id, target_id]))
        if rel_key not in relation_set:
            relations.append({
                'source': source_id,
                'target': target_id,
                'type': rel_type
            })
            relation_set.add(rel_key)

# 自动从定义中提取关系
for node in nodes:
    definition = node['definition'].lower()
    for other_name, other_id in name_to_id.items():
        if other_id == node['id']:
            continue
        if len(other_name) < 2:
            continue
        if other_name.lower() in definition:
            rel_key = tuple(sorted([node['id'], other_id]))
            if rel_key not in relation_set:
                relations.append({
                    'source': node['id'],
                    'target': other_id,
                    'type': 'related_to'
                })
                relation_set.add(rel_key)

print(f"总关系数: {len(relations)}")

# 保存
knowledge_graph = {
    'nodes': nodes,
    'relations': relations,
    'metadata': {
        'version': '2.0.0',
        'nodeCount': len(nodes),
        'relationCount': len(relations)
    }
}

output_path = r'C:\Users\xiang\Documents\code\CrossRing\Tier6+model\frontend\src\data\knowledge-graph.json'

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)

print(f"\n已保存到: {output_path}")

# 统计分类
cat_count = Counter(n['category'] for n in nodes)
print("\n分类统计:")
for cat, count in cat_count.most_common():
    print(f"  {cat}: {count}")
