from dataclasses import asdict, dataclass
from pathlib import Path

WORLD_SIZE = 8
EP_SIZE = 8
LOCAL_EXPERTS = 32
PAGE_SIZE = 128
TOPK = 2048
MAIN_LAYERS = 78
DRAFT_LAYER = 78
TOTAL_LAYERS = 79
VOCAB_SIZE = 154880
HIDDEN_SIZE = 6144
FULL_INDEX_LAYERS = (0, 1, 2, *range(6, 79, 4))
INT8_INDEX_LAYERS = frozenset(range(6, 67, 4))
BF16_INDEX_LAYERS = (0, 1, 2, 70, 74, 78)
K_NORM_EPS = 1e-6
RMS_NORM_EPS = 1e-5
INV_TEMP = 0.95
DEFAULT_CAPACITY_TOKENS = 9280
PREFIX_LEN = 8192
MTP_PREFIX_LEN = 8191
DECODE_TOKENS = 1023
FINAL_POS = 9215
WIDTHS = (1, 2, 4, 6, 8, 12, 16, 24, 32)


@dataclass(frozen=True)
class DecodeConfig:
    vocab_size: int = 154880
    hidden_size: int = 6144
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    first_k_dense_replace: int = 3
    routed_scaling_factor: float = 2.5
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8000000.0

    @classmethod
    def smoke(cls) -> "DecodeConfig":
        return cls(
            vocab_size=256,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            moe_intermediate_size=128,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=64,
            index_n_heads=4,
            index_head_dim=128,
            index_topk=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
        )

    def has_indexer(self, layer: int) -> bool:
        return layer < 3 or (layer - 2) % 4 == 0

    def to_hf_config(self):
        from transformers import GlmMoeDsaConfig

        values = asdict(self)
        values.pop("rope_theta")
        config = GlmMoeDsaConfig(
            **values,
            num_key_value_heads=self.num_attention_heads,
            mlp_layer_types=[
                "dense" if layer < self.first_k_dense_replace else "sparse"
                for layer in range(self.num_hidden_layers)
            ],
            indexer_types=[
                "full" if self.has_indexer(layer) else "shared"
                for layer in range(self.num_hidden_layers)
            ],
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            max_position_embeddings=16384,
            rope_parameters={"rope_type": "default", "rope_theta": self.rope_theta},
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        return config


@dataclass(frozen=True)
class RuntimeConfig:
    model_dir: Path
    cal_l78_path: Path
    whole_layer_json: Path
    rank: int
    rot_path: Path | None = None
    world_size: int = WORLD_SIZE
    ep_size: int = EP_SIZE
    local_experts: int = LOCAL_EXPERTS
    capacity_tokens: int = DEFAULT_CAPACITY_TOKENS
    device: str = "npu:0"
    tree_widths: tuple[int, ...] = (12, 16, 24)
    inv_temp: float = INV_TEMP
    max_rope_positions: int = 16384
