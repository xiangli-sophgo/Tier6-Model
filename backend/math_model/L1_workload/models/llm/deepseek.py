"""DeepSeek 模型实现

定义 DeepSeek V3 系列模型。
"""

from __future__ import annotations

from typing import Any

from math_model.L0_entry.types import DataType
from math_model.L1_workload.layers.embedding import EmbeddingLayer
from math_model.L1_workload.layers.ffn import FFNLayer
from math_model.L1_workload.layers.lmhead import LMHeadLayer
from math_model.L1_workload.layers.mla_absorb import MLAAbsorbLayer
from math_model.L1_workload.layers.mla import MLALayer
from math_model.L1_workload.layers.moe import MoELayer
from math_model.L1_workload.metadata import ModelMetadata
from math_model.L1_workload.models import model_registry
from math_model.L1_workload.models.base import ModelBase


@model_registry.register("deepseek_v3")
class DeepSeekV3Model(ModelBase):
    """DeepSeek V3 模型

    支持 DeepSeek V3/V3.2 配置，包含 MLA + MoE 架构。

    模型结构:
        - Embedding
        - N_dense x (MLA + FFN)      # 前几层为 Dense
        - N_moe x (MLA + MoE)        # 后续层为 MoE
        - RMSNorm
        - LM Head

    Config 参数:
        基础配置:
            - hidden_size: 隐藏层大小 (7168)
            - num_layers: 总层数 (61)
            - num_dense_layers: Dense 层数 (3)
            - num_moe_layers: MoE 层数 (58)
            - num_heads: 注意力头数 (128)
            - vocab_size: 词表大小 (129280)
            - seq_len: 序列长度
            - batch: 批次大小

        MLA 配置:
            - q_lora_rank: Q 低秩维度 (1536)
            - kv_lora_rank: KV 低秩维度 (512)
            - qk_nope_head_dim: 非 RoPE 的 head 维度 (128)
            - qk_rope_head_dim: RoPE 的 head 维度 (64)
            - v_head_dim: V 的 head 维度 (128)

        MoE 配置:
            - n_routed_experts: 专家总数 (256)
            - n_shared_experts: 共享专家数 (1)
            - n_activated_experts: 激活专家数 (8)
            - moe_intermediate_size: MoE FFN 中间层 (2048)

        FFN Dense 配置:
            - intermediate_size: Dense FFN 中间层 (18432)

    Example:
        >>> model = DeepSeekV3Model({
        ...     "hidden_size": 7168,
        ...     "num_layers": 61,
        ...     "num_dense_layers": 3,
        ...     "num_moe_layers": 58,
        ...     "num_heads": 128,
        ...     "vocab_size": 129280,
        ...     "q_lora_rank": 1536,
        ...     "kv_lora_rank": 512,
        ...     "n_routed_experts": 256,
        ...     "n_activated_experts": 8,
        ...     "moe_intermediate_size": 2048,
        ...     "intermediate_size": 18432,
        ...     "seq_len": 4096,
        ...     "batch": 1,
        ... })
        >>> ir = model.to_ir()
        >>> print(ir.get_ops_breakdown())
    """

    @classmethod
    def from_model_config(cls, mc: "ModelConfig") -> DeepSeekV3Model:
        """从 EvalConfig.ModelConfig 构建模型

        将结构化 ModelConfig 转为 layer 层期望的 flat dict，
        这样 EmbeddingLayer / MLALayer 等无需修改。

        Args:
            mc: 类型化模型配置 (from math_model.L0_entry.eval_config)

        Returns:
            DeepSeekV3Model 实例
        """
        from math_model.L0_entry.eval_config import ModelConfig  # noqa: F811
        config = {
            "hidden_size": mc.hidden_size,
            "num_layers": mc.num_layers,
            "num_dense_layers": mc.num_dense_layers,
            "num_moe_layers": mc.num_moe_layers,
            "num_heads": mc.num_attention_heads,
            "vocab_size": mc.vocab_size,
            "intermediate_size": mc.intermediate_size,
            # MLA (从嵌套提取为 flat)
            "q_lora_rank": mc.mla.q_lora_rank,
            "kv_lora_rank": mc.mla.kv_lora_rank,
            "qk_nope_head_dim": mc.mla.qk_nope_head_dim,
            "qk_rope_head_dim": mc.mla.qk_rope_head_dim,
            "v_head_dim": mc.mla.v_head_dim,
            # MoE (key rename 对齐 layer 层期望)
            "n_routed_experts": mc.moe.num_routed_experts,
            "n_shared_experts": mc.moe.num_shared_experts,
            "n_activated_experts": mc.moe.num_activated_experts,
            "moe_intermediate_size": mc.moe.intermediate_size,
            # 运行时参数
            "weight_dtype": mc.weight_dtype,
            "activation_dtype": mc.activation_dtype,
            "seq_len": mc.seq_len,
            "kv_seq_len": mc.kv_seq_len,
            "q_seq_len": mc.q_seq_len,
            "batch": mc.batch,
            "is_prefill": mc.is_prefill,
        }
        return cls(config)

    @property
    def name(self) -> str:
        """模型名称"""
        return "deepseek_v3"

    def build(self) -> None:
        """构建模型层结构

        DeepSeek V3 结构:
            1. Embedding
            2. Dense Layers (MLA + FFN) x num_dense_layers
            3. MoE Layers (MLA + MoE) x num_moe_layers
            4. LM Head
        """
        # 从 config 获取层数配置（必需）
        if "num_layers" not in self._config:
            raise ValueError("Missing required field 'num_layers' in DeepSeek model config")
        total_layers = self._config["num_layers"]

        # dense/moe 层数拆分：优先使用显式指定，否则从 total_layers 推导
        if "num_dense_layers" in self._config and "num_moe_layers" in self._config:
            num_dense_layers = self._config["num_dense_layers"]
            num_moe_layers = self._config["num_moe_layers"]
            # 校验一致性
            if num_dense_layers + num_moe_layers != total_layers:
                raise ValueError(
                    f"num_dense_layers ({num_dense_layers}) + num_moe_layers ({num_moe_layers}) "
                    f"!= num_layers ({total_layers})"
                )
        elif "num_dense_layers" in self._config:
            num_dense_layers = self._config["num_dense_layers"]
            num_moe_layers = total_layers - num_dense_layers
        elif "num_moe_layers" in self._config:
            num_moe_layers = self._config["num_moe_layers"]
            num_dense_layers = total_layers - num_moe_layers
        else:
            raise ValueError(
                "Must specify either 'num_dense_layers', 'num_moe_layers', or both in DeepSeek model config"
            )
        is_prefill_cfg = self._config.get("is_prefill")
        if is_prefill_cfg is None:
            # 兼容历史行为：未显式指定时使用 MLALayer。
            mla_cls = MLALayer
        else:
            is_prefill = bool(is_prefill_cfg)
            mla_cls = MLALayer if is_prefill else MLAAbsorbLayer

        # 1. Embedding 层
        self._layers.append(EmbeddingLayer("embedding", self._config))

        # 2. Dense 层 (前 num_dense_layers 层)
        for i in range(num_dense_layers):
            # MLA 层
            self._layers.append(mla_cls(f"layers.{i}.mla", self._config))
            # FFN 层 (Dense)
            self._layers.append(FFNLayer(f"layers.{i}.ffn", self._config))

        # 3. MoE 层 (后 num_moe_layers 层)
        for i in range(num_dense_layers, num_dense_layers + num_moe_layers):
            # MLA 层
            self._layers.append(mla_cls(f"layers.{i}.mla", self._config))
            # MoE 层
            self._layers.append(MoELayer(f"layers.{i}.moe", self._config))

        # 4. LM Head
        self._layers.append(LMHeadLayer("lm_head", self._config))

    def _build_metadata(self) -> ModelMetadata:
        """构建模型元数据"""
        # 必需字段检查（与 ModelMetadata.from_dict 对齐）
        required_fields = ["hidden_size", "num_layers", "num_heads"]
        missing = [f for f in required_fields if f not in self._config]
        if missing:
            raise ValueError(f"Missing required fields in DeepSeek model config: {missing}")

        return ModelMetadata(
            name=self.name,
            dtype=DataType.from_string(self._config.get("dtype", "bf16")),  # dtype 可选
            hidden_size=self._config["hidden_size"],
            num_layers=self._config["num_layers"],
            num_heads=self._config["num_heads"],
            vocab_size=self._config.get("vocab_size"),  # 可选
            seq_len=self._config.get("seq_len"),  # 可选
            batch=self._config.get("batch"),  # 可选
        )

    def get_info(self) -> dict[str, Any]:
        """获取模型汇总信息（扩展版）"""
        base_info = super().get_info()
        # 添加 DeepSeek 特有信息（仅包含已配置的字段）
        if "num_dense_layers" in self._config:
            base_info["num_dense_layers"] = self._config["num_dense_layers"]
        if "num_moe_layers" in self._config:
            base_info["num_moe_layers"] = self._config["num_moe_layers"]
        if "n_routed_experts" in self._config:
            base_info["n_routed_experts"] = self._config["n_routed_experts"]
        if "n_activated_experts" in self._config:
            base_info["n_activated_experts"] = self._config["n_activated_experts"]
        if "q_lora_rank" in self._config:
            base_info["q_lora_rank"] = self._config["q_lora_rank"]
        if "kv_lora_rank" in self._config:
            base_info["kv_lora_rank"] = self._config["kv_lora_rank"]
        return base_info


@model_registry.register("deepseek")
class DeepSeekModel(DeepSeekV3Model):
    """DeepSeek 模型别名

    默认指向 DeepSeek V3。
    """

    @property
    def name(self) -> str:
        return "deepseek"
