/**
 * 配置适配器
 *
 * 在 ModelPreset (新格式, types/math_model.ts) 和 LLMModelConfig (旧格式, types.ts) 之间双向转换。
 * 用于渐进式迁移：新代码使用 ModelPreset，旧代码通过适配器继续使用 LLMModelConfig。
 */

import type { ModelPreset } from '../../types/math_model';
import type { LLMModelConfig, AttentionType } from './types';

/**
 * ModelPreset -> LLMModelConfig
 *
 * 将后端 YAML 格式的模型预设转换为前端旧格式。
 * 注意: ModelPreset 没有 dtype 信息，精度由 InferenceConfig 管理。
 */
export function modelPresetToLLMConfig(preset: ModelPreset): LLMModelConfig {
  const isMoE = !!preset.MoE;

  // 推断 attention 类型
  let attention_type: AttentionType = 'mha';
  if (preset.MLA) {
    attention_type = 'mla';
  } else if (
    preset.num_key_value_heads !== undefined &&
    preset.num_key_value_heads !== preset.num_attention_heads
  ) {
    attention_type = preset.num_key_value_heads === 1 ? 'mqa' : 'gqa';
  }

  const config: LLMModelConfig = {
    model_name: preset.name,
    model_type: isMoE ? 'moe' : 'dense',
    hidden_size: preset.hidden_size,
    num_layers: preset.num_layers,
    num_attention_heads: preset.num_attention_heads,
    num_kv_heads: preset.num_key_value_heads ?? preset.num_attention_heads,
    intermediate_size: preset.intermediate_size,
    vocab_size: preset.vocab_size,
    max_seq_length: preset.max_seq_len ?? 4096,
    norm_type: (preset.norm_type as 'rmsnorm' | 'layernorm') || 'rmsnorm',
    attention_type,
  };

  if (preset.MoE) {
    config.moe_config = {
      num_experts: preset.MoE.num_routed_experts,
      num_experts_per_tok: preset.MoE.num_activated_experts,
      num_shared_experts: preset.MoE.num_shared_experts,
      expert_intermediate_size: preset.MoE.intermediate_size,
      first_k_dense_replace: preset.num_dense_layers,
    };
  }

  if (preset.MLA) {
    config.mla_config = {
      kv_lora_rank: preset.MLA.kv_lora_rank,
      q_lora_rank: preset.MLA.q_lora_rank,
      qk_nope_head_dim: preset.MLA.qk_nope_head_dim,
      qk_rope_head_dim: preset.MLA.qk_rope_head_dim,
      v_head_dim: preset.MLA.v_head_dim,
    };
  }

  return config;
}

/**
 * LLMModelConfig -> ModelPreset
 *
 * 将前端旧格式转换为后端 YAML 格式。
 * 注意: 会丢失 dtype、capacity_factor 等旧格式专有字段。
 */
export function llmConfigToModelPreset(config: LLMModelConfig): ModelPreset {
  const preset: ModelPreset = {
    name: config.model_name,
    vocab_size: config.vocab_size,
    hidden_size: config.hidden_size,
    intermediate_size: config.intermediate_size,
    num_layers: config.num_layers,
    num_attention_heads: config.num_attention_heads,
    num_key_value_heads: config.num_kv_heads,
    max_seq_len: config.max_seq_length,
    norm_type: config.norm_type,
  };

  if (config.moe_config) {
    preset.MoE = {
      num_routed_experts: config.moe_config.num_experts,
      num_activated_experts: config.moe_config.num_experts_per_tok,
      intermediate_size: config.moe_config.expert_intermediate_size ?? config.intermediate_size,
      num_shared_experts: config.moe_config.num_shared_experts,
    };
    if (config.moe_config.first_k_dense_replace) {
      preset.num_dense_layers = config.moe_config.first_k_dense_replace;
      preset.num_moe_layers = config.num_layers - config.moe_config.first_k_dense_replace;
    }
  }

  if (config.mla_config) {
    preset.MLA = {
      q_lora_rank: config.mla_config.q_lora_rank,
      kv_lora_rank: config.mla_config.kv_lora_rank,
      qk_nope_head_dim: config.mla_config.qk_nope_head_dim,
      qk_rope_head_dim: config.mla_config.qk_rope_head_dim,
      v_head_dim: config.mla_config.v_head_dim,
    };
  }

  return preset;
}
