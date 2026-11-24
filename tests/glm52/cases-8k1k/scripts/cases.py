import hashlib
import json
from pathlib import Path
import re
import sys


TARGET_LEN = 8192

CASES = (
    ("case-01-coding", "case01_coding.txt", "coding", False),
    ("case-02-review", "case02_review.txt", "code_review", True),
    ("case-03-refactor", "case03_refactor.txt", "refactoring", True),
    ("case-04-math", "case04_math.txt", "math_reasoning", False),
)

PROSE_FILLERS = (
    "Ensure deterministic behavior across edge conditions.",
    "Maintain strict consistency across all subsystem boundaries.",
    "Verify exact invariants at every state transition.",
    "Preserve clean separation of concerns throughout.",
)

WORD_PAD = (
    "strictly", "cleanly", "always", "consistently", "deterministically",
    "rigorously", "explicitly", "uniformly", "precisely", "completely",
    "accurately", "reliably", "seamlessly", "directly", "exact", "valid",
)


def encode_chat(tok, content: str) -> list[int]:
    res = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(res["input_ids"] if hasattr(res, "keys") else res)


def prompt_sha256(tokens: list[int]) -> str:
    raw = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def calibrate_prompt(tok, raw_template: str, is_code_trim: bool) -> tuple[str, list[int]]:
    cleaned = raw_template.replace("\n<<<BODY_END>>>\n", "\n\n").replace(
        "\n<<<FINAL_DIRECTIVE>>>\n", "\n\n"
    )
    m = re.search(r"<<<TRIM_START>>>\n(.*?)\n<<<TRIM_END>>>", cleaned, flags=re.S)
    if not m:
        raise ValueError("Missing <<<TRIM_START>>> ... <<<TRIM_END>>>")
    prefix = cleaned[: m.start()]
    trim_lines = m.group(1).splitlines()
    suffix = cleaned[m.end() :]

    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    suffix_ids = tok.encode(suffix, add_special_tokens=False)
    fixed_overhead = 6 + len(prefix_ids) + len(suffix_ids)

    def fast_len(mid_str: str) -> int:
        return fixed_overhead + len(tok.encode(mid_str, add_special_tokens=False))

    lo, hi = 0, len(trim_lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fast_len("\n".join(trim_lines[:mid])) <= TARGET_LEN - 10:
            lo = mid
        else:
            hi = mid - 1
    kept: list[str] = list(trim_lines[:lo])

    if is_code_trim:
        import ast

        while kept:
            try:
                ast.parse("\n".join(kept))
                break
            except SyntaxError:
                kept.pop()

    base_trim = "\n".join(kept).rstrip()
    base_trim_ids = tok.encode(base_trim, add_special_tokens=False) if base_trim else []
    note_hdr = "# Note:" if is_code_trim else "Note:"
    pad_prefix = (
        (f"\n\n{note_hdr}" if is_code_trim else f"\n{note_hdr}") if base_trim else note_hdr
    )

    def build_mid(wlist: list[str]) -> str:
        return base_trim if not wlist else f"{base_trim}{pad_prefix} {' '.join(wlist)}."

    def build_text(wlist: list[str]) -> str:
        return prefix + build_mid(wlist) + suffix

    pool = (
        "verify runtime invariants across boundary transitions and edge states "
        "under concurrent execution with deterministic resource accounting "
        "and strict state consistency across all active components"
    ).split() * 40
    w_lo, w_hi = 0, len(pool)
    while w_lo < w_hi:
        w_mid = (w_lo + w_hi + 1) // 2
        pad_str = f"{pad_prefix} {' '.join(pool[:w_mid])}."
        pad_len = len(tok.encode(pad_str, add_special_tokens=False))
        if fixed_overhead + len(base_trim_ids) + pad_len <= TARGET_LEN:
            w_lo = w_mid
        else:
            w_hi = w_mid - 1
    words = list(pool[:w_lo])
    cur_ids = encode_chat(tok, build_text(words))
    for _ in range(12):
        diff = TARGET_LEN - len(cur_ids)
        if diff == 0:
            return build_text(words), cur_ids
        if diff < 0 and words:
            words.pop()
        else:
            words.append("all" if diff == 1 else "strict")
        cur_ids = encode_chat(tok, build_text(words))
    while len(cur_ids) > TARGET_LEN and words:
        words.pop()
        cur_ids = encode_chat(tok, build_text(words))
    for w1 in list(WORD_PAD) + ["valid", "clean", "exact", "pure", "full", "safe", "core"]:
        cand_text = build_text(words + [w1])
        mid_len = len(tok.encode(build_mid(words + [w1]), add_special_tokens=False))
        if abs(fixed_overhead + mid_len - TARGET_LEN) <= 3:
            ids = encode_chat(tok, cand_text)
            if len(ids) == TARGET_LEN:
                return cand_text, ids
    raise RuntimeError(f"Failed to hit exact 8192 tokens (got {len(cur_ids)})")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_dir = sys.argv[1]
    src_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "sources"
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else root
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    all_prompts: list[list[int]] = []
    summary = []

    for slug, fname, scenario, is_code in CASES:
        raw = (src_dir / fname).read_text(encoding="utf-8")
        user_text, token_ids = calibrate_prompt(tok, raw, is_code)
        assert len(token_ids) == TARGET_LEN, f"{slug}: got {len(token_ids)}"
        assert token_ids[:3] == [154822, 154824, 154827], f"{slug}: bad prefix {token_ids[:3]}"
        assert token_ids[-3:] == [154828, 154841, 154842], f"{slug}: bad suffix {token_ids[-3:]}"
        sha = prompt_sha256(token_ids)
        case_dir = out_dir / slug
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": slug,
            "scenario": scenario,
            "seq_len": TARGET_LEN,
            "chat_template": "[gMASK]<sop><|user|>...<|assistant|><think></think>",
            "enable_thinking": False,
            "prompt_sha256": sha,
            "user_text": user_text,
            "rendered_prompt_text": tok.decode(token_ids, skip_special_tokens=False),
            "prompt_token_ids": token_ids,
        }
        (case_dir / "prompt.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        all_prompts.append(token_ids)
        summary.append(
            {
                "case_id": slug,
                "scenario": scenario,
                "seq_len": len(token_ids),
                "prompt_sha256": sha,
            }
        )
        print(
            f"[OK] {slug:18s} | scenario={scenario:15s} | "
            f"seq_len={len(token_ids)} | sha256={sha[:16]}..."
        )

    (out_dir / "prompts-8192.json").write_text(
        json.dumps(all_prompts, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (out_dir / "cases-index.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
