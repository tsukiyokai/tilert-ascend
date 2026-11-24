"""GLM-5.2 geometry from the retained 2026-10-20 configuration snapshot."""


def model_config():
    """Return the published layer pattern without changing its indexer frequency."""
    full = [max(index - 3 + 1, 0) % 4 == 0 for index in range(78)]
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "ep_size": 1,
        "first_k_dense_replace": 3,
        "head_dim": 192,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_share_for_mtp_iteration": True,
        "index_skip_topk_offset": 3,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_topk_pattern": None,
        "indexer_rope_interleave": True,
        "indexer_types": ["full" if item else "shared" for item in full],
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "kv_lora_rank": 512,
        "max_position_embeddings": 1048576,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "model_type": "glm_moe_dsa",
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 78,
        "num_key_value_heads": 64,
        "num_nextn_predict_layers": 1,
        "pretraining_tp": 1,
        "q_lora_rank": 2048,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-5,
        "rope_interleave": True,
        "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "transformers_version": "5.12.0",
        "use_cache": True,
        "v_head_dim": 256,
        "vocab_size": 154880,
    }


def selection(layer_index):
    config = model_config()
    if not 0 <= layer_index < config["num_hidden_layers"]:
        raise ValueError("Layer index is outside the retained GLM-5.2 backbone")
    return {
        "layer_index": layer_index,
        "ffn": config["mlp_layer_types"][layer_index],
        "indexer": config["indexer_types"][layer_index],
        "first_k_dense_replace": config["first_k_dense_replace"],
        "index_topk_freq": config["index_topk_freq"],
        "index_skip_topk_offset": config["index_skip_topk_offset"],
        "index_topk_pattern": config["index_topk_pattern"],
    }
