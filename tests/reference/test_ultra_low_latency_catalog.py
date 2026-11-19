"""The catalog checker reads the shipped reference file, not a side ledger."""

import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs" / "reference" / "ultra-low-latency-2025.md"
README = ROOT / "docs" / "reference" / "README.md"
FIELDS = ("url", "date", "mechanism", "regime", "number", "quote", "relevance")
START = date(2025, 1, 1)


def records(text):
    chunks = re.split(r"\n(?=## r\d{3}\. )", text)
    found = []
    for chunk in chunks:
        if not chunk.startswith("## r"):
            continue
        if chunk.startswith("## Gaps"):
            continue
        head = re.match(r"## (r\d{3})\. (.+)", chunk)
        row = {"rid": head.group(1), "title": head.group(2).strip()}
        for line in chunk.splitlines():
            if line.startswith("- ") and ": " in line:
                key, value = line[2:].split(": ", 1)
                row[key] = value.strip()
        found.append(row)
    return found


def test_catalog_meets_the_acceptance_count():
    text = CATALOG.read_text()
    rows = records(text)
    assert len(rows) == 100
    urls = []
    arxiv_ids = []
    hosts = {"arxiv": 0, "x": 0}
    for row in rows:
        for field in FIELDS:
            assert row.get(field), field
        assert row.get("authors") or row.get("handle")
        url = row["url"]
        urls.append(url)
        stamped = date.fromisoformat(row["date"])
        assert stamped >= START
        if url.startswith("https://arxiv.org/abs/"):
            hosts["arxiv"] += 1
            arxiv_ids.append(url.removeprefix("https://arxiv.org/abs/"))
            assert re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_ids[-1])
        else:
            assert re.fullmatch(r"https://x\.com/[^/]+/status/\d+", url)
            hosts["x"] += 1
        assert len(row["quote"]) >= 20
        assert "11.6198" not in row["relevance"]
        note = row["relevance"]
        assert "10.8528" in note
        other = ("3008", "13.58", "7.1", "5.35")
        names_baseline = all(token in note for token in other)
        says_why_not = re.search(
            r"does not move|does not cut|does not transfer|does not replace|"
            r"not measured|not remeasured|not an A3|not this A3|"
            r"was not run|not comparable|not a measurement",
            note,
            re.I,
        )
        assert names_baseline or says_why_not, row["url"]
    assert len(urls) == len(set(urls))
    assert len(arxiv_ids) == len(set(arxiv_ids))
    assert hosts["arxiv"] >= 20
    assert hosts["x"] >= 20
    assert "## Gaps" in text
    assert "10.8528" in text
    assert "11.6198" not in text
    readme = README.read_text()
    assert "ultra-low-latency-2025.md" in readme
    assert "10.8528" in readme


DRAFT_BOILER = "a longer accepted draft banks more tokens"
KV_BOILER = "this moves KV or prefill/decode placement"
LAUNCH_BOILER = "this cuts kernel launches on the decode step"
QUANT_BOILER = "a foreign quant kernel's speedup does not transfer"
SPECULATIVE = re.compile(r"specul|draft|mtp|eagle|acceptance|medusa", re.I)
KV_TECHNIQUE = re.compile(r"\bkv\b|offload|disaggreg|handoff|transfer", re.I)
LAUNCH_TECHNIQUE = re.compile(
    r"megakernel|launch|cuda graph|persistent|fused kernel", re.I
)
QUANT_TECHNIQUE = re.compile(r"quant|int2|int4|fp4|fp8|bit|nibble", re.I)
NOT_DRAFT_OR_KV_BOILER = {
    "r043",
    "r044",
    "r053",
    "r057",
    "r059",
    "r060",
    "r065",
    "r066",
    "r067",
    "r068",
    "r069",
}


def test_relevance_boilerplate_matches_the_technique():
    rows = records(CATALOG.read_text())
    for row in rows:
        blob = row["title"] + " " + row["mechanism"]
        note = row["relevance"]
        if DRAFT_BOILER in note:
            assert SPECULATIVE.search(blob), row["rid"]
        if KV_BOILER in note:
            assert KV_TECHNIQUE.search(blob), row["rid"]
        if LAUNCH_BOILER in note:
            assert LAUNCH_TECHNIQUE.search(blob), row["rid"]
        if QUANT_BOILER in note:
            assert QUANT_TECHNIQUE.search(blob), row["rid"]
        if row["rid"] in NOT_DRAFT_OR_KV_BOILER:
            assert DRAFT_BOILER not in note, row["rid"]
            assert KV_BOILER not in note, row["rid"]
            assert LAUNCH_BOILER not in note, row["rid"]
            assert QUANT_BOILER not in note, row["rid"]
