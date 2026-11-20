import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


SLUGS = ("case-01-coding", "case-02-review", "case-03-refactor", "case-04-math")
TOPOLOGY_BY_TRACK = {
    "a3-tp8-ep8": "TP8+EP8",
    "a3-tp16-ep16": "TP16+EP16",
    "a5-tp8-ep8": "TP8+EP8",
}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("cases_dir", nargs="?", type=Path, default=root)
    parser.add_argument("--track", default="a3-tp8-ep8")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cases_dir = Path(args.cases_dir)
    track = str(args.track)
    topology = TOPOLOGY_BY_TRACK.get(track, track)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    overview = []

    for idx, slug in enumerate(SLUGS, start=1):
        cdir = cases_dir / slug
        gdir = cdir / "golden" / track
        result_path = gdir / "result.json"
        if not result_path.is_file():
            continue
        pmeta = json.loads((cdir / "prompt.json").read_text(encoding="utf-8"))
        bmeta = json.loads(result_path.read_text(encoding="utf-8"))
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
            "track": track,
            "topology": topology,
            "reset_complete": True,
        }
        (gdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        extra_keys = (
            "eos_position_in_1024",
            "unique_4gram_ratio",
            "lookup_min2",
            "recency_lookup",
            "layer78_mtp_w4",
            "layer78_mtp_w8",
            "hybrid_lookup3_mtp8",
        )
        extras = {key: bmeta[key] for key in extra_keys if key in bmeta}
        decoded = {
            "case_id": slug,
            "scenario": pmeta["scenario"],
            "track": track,
            "meta": {
                "status": "capture_complete",
                "capture_index": idx,
                "seq_len": 8192,
                "prompt_sha256": pmeta["prompt_sha256"],
                "first_token_id": bmeta["first_token_id"],
                "output_sha256": bmeta["output_sha256"],
                **extras,
            },
            "prompt_len": 8192,
            "prompt_text": pmeta["rendered_prompt_text"],
            "output_len": len(out_ids),
            "output_text": out_text,
        }
        (gdir / "decoded.json").write_text(
            json.dumps(decoded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        overview.append(
            {
                "case_id": slug,
                "scenario": pmeta["scenario"],
                "track": track,
                "seq_len": 8192,
                "output_len": len(out_ids),
                "first_token_id": bmeta["first_token_id"],
                "output_sha256": bmeta["output_sha256"],
                **extras,
                "output_preview_head": out_text[:320],
                "output_preview_tail": out_text[-240:],
            }
        )
        print(
            f"[Decoded:{track}] {slug:18s} | out_chars={len(out_text)} | head={out_text[:80]!r}"
        )

    summary_path = cases_dir / "golden" / track / "bench-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
