# GLM-5.2 Ascend (`tilert.models.glm_5_2_ascend`)

Standalone Ascend A3 TP8/EP8 MTP speculative-decode runtime for GLM-5.2 (`DecodeEngine` / `Glm52AscendShowHands`) with zero `vllm` dependencies.

## Architecture (`end2end.py` + `dspine/`)

- `config.py`: Model geometry (`VOCAB_SIZE=154880`, `HIDDEN_SIZE=6144`, `K_NORM_EPS=1e-6`, `RMS_NORM_EPS=1e-5`, `WIDTHS=(1, 2, 4, 6, 8, 12, 16, 24, 32)`), `DecodeConfig`, and `RuntimeConfig`.
- `weights.py` & `abi.py`: `build_model_arena`, layer/draft weight packing (`pack_value_weights`, `pack_output_weights`, `pack_shared_weights`, `pack_indexer_weights`), `Libraries`, `Window`, `OneSidedReduce`, `FrozenPrefixArena`, and `ModelArena`.
- `spec.py`: `MtpState`, `MtpRoundExecutor`, `accepted_path`, and `accepted_tokens`.
- `dspine/`: Calibrated L78 drafter (`calibrated_l78.py`), `aclmdlRI`/NPU-Graph TP drafter (`tp_drafter.py`), and custom C++/AscendC dspine ops (`build_dspine_ops.py`).
- `end2end.py`: Standalone `Glm52AscendShowHands` (`DecodeEngine`) orchestrating single-graph `G_round(W)` capture and `in_flight=2` host overlap (`ShowHandsLoop`) across target layers `L0-77` (`ChainKernel`) and L78 speculative drafting (`CalibratedL78V3Runner`).
- `ops/`: Megakernel, indexer (`cann_arch22_scoring_contract`), rmsnorm, rope, sfa, mla, moe, and ep8 collectives.
- `reference.py`: Numerical reference utilities.

## Usage

```python
from pathlib import Path
from tilert.models.glm_5_2_ascend import Glm52AscendShowHands
from tilert.models.glm_5_2_ascend.config import RuntimeConfig

cfg = RuntimeConfig(
    model_dir=Path("/path/to/model"),
    cal_l78_path=Path("/path/to/cal_l78_v3_hybrid_best.pt"),
    whole_layer_json=Path("/path/to/build/whole_layer.json"),
    rank=0,
    world_size=8,
)
engine = Glm52AscendShowHands(
    cfg=cfg,
    case_dir=Path("/path/to/case-01-coding"),
    mtp_prefix_path=Path("/path/to/mtp_prefix.pt"),
    build_dir=Path("/path/to/build"),
)
engine.load()
result = engine.warmup_and_run(warmups=2, formals=5, max_steps=1023)
engine.close()
```
