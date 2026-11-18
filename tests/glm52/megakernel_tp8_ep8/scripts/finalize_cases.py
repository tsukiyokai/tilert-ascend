from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def main() -> None:
    model_dir = sys.argv[1]
    cases_dir = Path(sys.argv[2])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    slugs = ("case-01-coding", "case-02-review", "case-03-refactor", "case-04-math")
    overview = []

    for idx, slug in enumerate(slugs, start=1):
        cdir = cases_dir / slug
        pmeta = json.loads((cdir / "prompt.json").read_text(encoding="utf-8"))
        bmeta = json.loads((cdir / "bench_result.json").read_text(encoding="utf-8"))
        out_ids = bmeta["output_token_ids"]
        out_text = tok.decode(out_ids, skip_special_tokens=False)

        manifest = {
            "status": "capture_complete",
            "request_id": pmeta["prompt_sha256"][:16],
            "case_id": slug,
            "scenario": pmeta["scenario"],
            "capture_index": idx,
            "seq_len": 8192,
            "generation": 1,
            "chat_template": pmeta["chat_template"],
            "enable_thinking": pmeta["enable_thinking"],
            "prompt_token_ids": pmeta["prompt_token_ids"],
            "prompt_sha256": pmeta["prompt_sha256"],
            "first_token_id": bmeta["first_token_id"],
            "delivered_tokens": [bmeta["first_token_id"]],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "topology": "TP8+EP8",
            "reset_complete": True,
        }
        (cdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        decoded = {
            "case_id": slug,
            "scenario": pmeta["scenario"],
            "meta": {
                "status": "capture_complete",
                "capture_index": idx,
                "seq_len": 8192,
                "prompt_sha256": pmeta["prompt_sha256"],
                "first_token_id": bmeta["first_token_id"],
                "output_sha256": bmeta["output_sha256"],
                "eos_position_in_1024": bmeta["eos_position_in_1024"],
                "unique_4gram_ratio": bmeta["unique_4gram_ratio"],
                "lookup_min2": bmeta["lookup_min2"],
                "recency_lookup": bmeta["recency_lookup"],
                **(
                    {"layer78_mtp_w4": bmeta["layer78_mtp_w4"]}
                    if "layer78_mtp_w4" in bmeta
                    else {}
                ),
                **(
                    {"layer78_mtp_w8": bmeta["layer78_mtp_w8"]}
                    if "layer78_mtp_w8" in bmeta
                    else {}
                ),
                **(
                    {"hybrid_lookup3_mtp8": bmeta["hybrid_lookup3_mtp8"]}
                    if "hybrid_lookup3_mtp8" in bmeta
                    else {}
                ),
            },
            "prompt_len": 8192,
            "prompt_text": pmeta["rendered_prompt_text"],
            "output_len": len(out_ids),
            "output_text": out_text,
        }
        (cdir / "decoded.json").write_text(
            json.dumps(decoded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        overview.append({
            "case_id": slug,
            "scenario": pmeta["scenario"],
            "seq_len": 8192,
            "output_len": len(out_ids),
            "first_token_id": bmeta["first_token_id"],
            "eos_position_in_1024": bmeta["eos_position_in_1024"],
            "unique_4gram_ratio": bmeta["unique_4gram_ratio"],
            "lookup_min2": bmeta["lookup_min2"],
            "recency_lookup": bmeta["recency_lookup"],
            **(
                {"layer78_mtp_w4": bmeta["layer78_mtp_w4"]}
                if "layer78_mtp_w4" in bmeta
                else {}
            ),
            **(
                {"layer78_mtp_w8": bmeta["layer78_mtp_w8"]}
                if "layer78_mtp_w8" in bmeta
                else {}
            ),
            **(
                {"hybrid_lookup3_mtp8": bmeta["hybrid_lookup3_mtp8"]}
                if "hybrid_lookup3_mtp8" in bmeta
                else {}
            ),
            "output_preview_head": out_text[:320],
            "output_preview_tail": out_text[-240:],
        })
        print(f"[Decoded] {slug:18s} | out_chars={len(out_text)} | head={out_text[:80]!r}")

    (cases_dir / "bench-summary.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
