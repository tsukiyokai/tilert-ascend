"""Measure TP16 PD decode from exact token IDs and producer callbacks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
from itertools import pairwise
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm.decode_timing import (  # noqa: E402
    atomic_json, load_timing, timing_path, token_metrics,
)
from tilert.pd_vllm.wire import derive_rid  # noqa: E402
from tilert.models.glm_5_2_ascend.mtp_state import WIDTHS  # noqa: E402


PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 1024
VOCAB_SIZE = 154880


class DecodeStream:
    """Timestamp complete NDJSON messages at their last byte's arrival."""

    def __init__(self):
        self.pending = b""
        self.token_ids = []
        self.timestamps_ns = []
        self.messages = []
        self.reads = []
        self.done = None
        self.done_ns = None

    def feed(self, data, received_ns):
        self.reads.append({"received_ns": received_ns, "bytes": len(data)})
        self.pending += data
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if not line.strip():
                continue
            if self.done is not None:
                raise ValueError("Decode sent data after its terminal message")
            message = json.loads(line)
            if "t" in message and "done" in message:
                raise ValueError("Token and terminal messages must differ")
            if "t" in message:
                ids = message["t"]
                if (not isinstance(ids, list) or not ids
                        or any(type(token) is not int
                               or not 0 <= token < VOCAB_SIZE
                               for token in ids)):
                    raise ValueError("Decode sent invalid token IDs")
                self.token_ids.extend(ids)
                self.timestamps_ns.extend([received_ns] * len(ids))
                self.messages.append({"received_ns": received_ns,
                                      "token_ids": ids})
            elif message.get("done") is True:
                self.done = message
                self.done_ns = received_ns
            else:
                raise ValueError(f"Decode stream error: {message}")

    def finish(self, first_token, prompt_tokens=PROMPT_TOKENS,
               output_tokens=OUTPUT_TOKENS):
        if self.pending.strip() or self.done is None:
            raise ValueError("Decode stream ended without a complete done")
        if (type(self.done.get("n")) is not int
                or self.done["n"] != len(self.token_ids)
                or len(self.token_ids) != output_tokens):
            raise ValueError("Decode terminal count differs from token IDs")
        if (type(self.done.get("seq_len")) is not int
                or self.done["seq_len"] != prompt_tokens):
            raise ValueError("Decode prefix is not the requested length")
        if not self.token_ids or self.token_ids[0] != first_token:
            raise ValueError("Decode first token differs from prefill")
        if self.done.get("finish_reason") != "length":
            raise ValueError("Decode did not complete the fixed output length")
        return token_metrics(self.token_ids, self.timestamps_ns)


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def aggregate(requests, source):
    metrics = [request[source] for request in requests]
    tpots = [item["tpot_ms"] for item in metrics]
    intervals = [value for item in metrics for value in item["intervals_ns"]]
    span = sum(item["span_ns"] for item in metrics)
    count = sum(item["output_tokens"] - 1 for item in metrics)
    return {
        "requests": len(requests),
        "request_p50_ms": percentile(tpots, 0.50),
        "request_p95_ms": percentile(tpots, 0.95),
        "request_mean_ms": sum(tpots) / len(tpots),
        "token_weighted_ms": span / count / 1e6,
        "token_weighted_tok_s": count * 1e9 / span if span else None,
        "interval_p50_ms": percentile(intervals, 0.50) / 1e6,
        "interval_p95_ms": percentile(intervals, 0.95) / 1e6,
        "zero_intervals": sum(value == 0 for value in intervals),
        "max_interval_ms": max(intervals) / 1e6,
    }


def prompt_digest(prompt):
    encoded = json.dumps(prompt, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_diversity_policy(path):
    if path is None:
        return None
    path = Path(path).resolve()
    raw = path.read_bytes()
    policy = json.loads(raw)
    fields = ("min_distinct_tokens", "min_distinct_bigrams")
    if (not isinstance(policy, dict)
            or any(type(policy.get(name)) is not int or policy[name] <= 1
                   for name in fields)):
        raise ValueError(
            "Diversity minima must be explicit integers greater than one",
        )
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            **{name: policy[name] for name in fields}}


def freeze_diversity_policy(policy, directory):
    if policy is None:
        return None
    raw = Path(policy["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy["sha256"]:
        raise ValueError("Diversity policy changed before its snapshot")
    name = "diversity-policy.json"
    with (Path(directory) / name).open("xb") as stream:
        stream.write(raw)
    return {**policy, "file": name,
            "source": {"file": policy["file"], "sha256": policy["sha256"]}}


def output_diversity(tokens):
    if (not isinstance(tokens, list) or len(tokens) != OUTPUT_TOKENS
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE
                   for token in tokens)):
        raise ValueError("Diversity requires all 1024 actual output token IDs")
    # N - longest proper border gives the minimum period, including a final
    # partial repetition; divisibility by the period is not required.
    borders = [0] * len(tokens)
    for index in range(1, len(tokens)):
        length = borders[index - 1]
        while length and tokens[index] != tokens[length]:
            length = borders[length - 1]
        if tokens[index] == tokens[length]:
            length += 1
        borders[index] = length
    return {
        "output_tokens": len(tokens),
        "distinct_output_tokens": len(set(tokens)),
        "distinct_output_bigrams": len(set(pairwise(tokens))),
        "minimum_period": len(tokens) - borders[-1],
        "output_sha256": prompt_digest(tokens),
    }


def validate_prompts(prompts, precision):
    expected = precision.get("prompt_sha256s")
    if (not isinstance(expected, list) or len(expected) != 8
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in expected)
            or len(set(expected)) != 8):
        raise ValueError("Precision requires eight distinct prompt hashes")
    actual = [prompt_digest(prompt) for prompt in prompts]
    if any(digest not in expected for digest in actual):
        raise ValueError("Benchmark prompt has no matching precision case")
    return actual


def passes_target(summary, measured, directory):
    """Recheck saved outputs; diversity alone does not establish model quality."""
    try:
        count = len(measured)
        names = summary["requests"]
        coverage = dict(Counter(item["prompt_sha256"] for item in measured))
        prompts = summary["precision"]["prompt_sha256s"]
        if (len(summary["warmup_requests"]) < 3 or count < 32
                or len(names) != count or len(set(names)) != count
                or len(prompts) != 8 or len(set(prompts)) != 8
                or coverage != summary["measured_prompt_counts"]
                or set(coverage) != set(prompts)):
            return False
        for source in ("producer", "client"):
            actual = aggregate(measured, source)
            if (summary[source] != actual or not all(
                    actual[stat] <= 2.0
                    for stat in ("request_p50_ms", "request_p95_ms"))):
                return False
        policy = summary.get("diversity_policy")
        if policy is None or policy["file"] != "diversity-policy.json":
            return False
        frozen = load_diversity_policy(Path(directory) / policy["file"])
        if (policy != {**frozen, "file": policy["file"],
                       "source": policy["source"]}
                or policy["source"]["sha256"] != frozen["sha256"]):
            return False
        for name, request in zip(names, measured):
            if Path(name).name != name:
                return False
            raw = (Path(directory) / name).read_bytes()
            if (hashlib.sha256(raw).hexdigest()
                    != summary["request_sha256"][name]
                    or json.loads(raw) != request):
                return False
            tokens = request["client_token_ids"]
            stats = output_diversity(tokens)
            producer = request["producer_report"]
            if (request.get("diversity") != stats
                    or producer["token_ids"] != tokens
                    or producer["returned_token_ids"] != tokens
                    or any(request[source]["output_tokens"] != len(tokens)
                           for source in ("producer", "client"))
                    or stats["minimum_period"] * 2 <= len(tokens)
                    or stats["distinct_output_tokens"]
                    < policy["min_distinct_tokens"]
                    or stats["distinct_output_bigrams"]
                    < policy["min_distinct_bigrams"]):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def validate_runtime(producer, precision):
    runtime = producer.get("runtime", {})
    if runtime.get("rank") != 0 or runtime.get("world_size") != 16:
        raise ValueError("Timing must describe the TP16 rank-zero server")
    for field in ("checkpoint_sha256", "mtp_width", "expert_backend"):
        if runtime.get(field) != precision[field]:
            raise ValueError(f"Runtime and precision differ: {field}")
    for package, version in precision["packages"].items():
        if runtime.get(package) != version:
            raise ValueError(f"Runtime package differs: {package}")
    sources = runtime.get("source_sha256", {})
    for name, digest in precision["source_sha256"].items():
        if sources.get(name) != digest:
            raise ValueError(f"Runtime source differs from precision: {name}")
    name = "tilert/pd_vllm/decode_timing.py"
    current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if (sources.get(name) != current
            or producer.get("timing_source_sha256") != current):
        raise ValueError("Timing source differs between client and server")


def command_rounds(operation):
    count = (1 if operation["operation"] == "advance_mtp"
             else operation["rounds"])
    if type(count) is not int or count < 1:
        raise ValueError("MTP command must complete at least one round")
    return count


def validate_linear_acceptance(span, tokens, accepted):
    """Recompute fixed-length, ignore-EOS acceptance from actual round rows."""
    width = span["width"]
    candidates, predictions = span.get("candidates"), span.get("predictions")
    if (span.get("parents") is not None
            or not all(isinstance(values, list) and len(values) == width
                       for values in (candidates, predictions))
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE
                   for values in (candidates, predictions) for token in values)
            or candidates[0] != tokens[accepted]):
        raise ValueError("MTP round lacks linear candidate/prediction rows")
    matched = 0
    for proposed, predicted in zip(candidates[1:], predictions):
        if proposed != predicted:
            break
        matched += 1
    natural = matched + 1
    committed = min(natural, OUTPUT_TOKENS - 1 - accepted)
    if (any(type(span[field]) is not int for field in (
            "matched_drafts", "natural_accepted", "accepted"))
            or span["matched_drafts"] != matched
            or span["natural_accepted"] != natural
            or span["accepted"] != committed
            or tokens[1 + accepted:1 + accepted + committed]
            != predictions[:committed]):
        raise ValueError("MTP acceptance differs from candidate/prediction rows")


def validate_round_commands(operations, stats, mtp_width, tokens):
    spans = stats["accepted_spans"]
    if (not operations or not spans
            or stats["round_calls"] != len(spans)
            or stats["completed_rounds"] != len(spans)):
        raise ValueError("MTP commands lack completed round spans")
    offset, position, accepted = 0, PROMPT_TOKENS, 0
    for operation in operations:
        count = command_rounds(operation)
        group = spans[offset:offset + count]
        count_tokens = sum(item["accepted"] for item in group)
        if (len(group) != count
                or operation["accepted"] != count_tokens
                or operation["matched_drafts"]
                != sum(item["matched_drafts"] for item in group)
                or operation["natural_accepted"]
                != sum(item["natural_accepted"] for item in group)):
            raise ValueError("MTP command and round spans disagree")
        if operation.get("accepted_tokens") != tokens[
            1 + accepted:1 + accepted + count_tokens
        ]:
            raise ValueError("MTP command tokens differ from delivery")
        for item in group:
            if (type(item["width"]) is not int
                    or item["width"] not in WIDTHS or item["width"] > mtp_width
                    or not 1 <= item["accepted"] <= item["width"]
                    or item["output_tokens"] != item["accepted"]
                    or item["main_start"] != position
                    or item["main_retained"] != position + item["accepted"]):
                raise ValueError("MTP round width, delivery or cache differs")
            validate_linear_acceptance(item, tokens, accepted)
            position = item["main_retained"]
            accepted += item["accepted"]
        offset += count
    if (offset != len(spans) or accepted != OUTPUT_TOKENS - 1
            or stats["accepted"] != accepted
            or stats["initial_position"] != PROMPT_TOKENS
            or stats["committed_position"] != position):
        raise ValueError("MTP rounds and delivered output count differ")


def round_summary(operations):
    durations = [(item["end_ns"] - item["start_ns"]) / 1e6
                 for item in operations]
    count = sum(command_rounds(item) for item in operations)
    accepted = sum(item["accepted"] for item in operations)
    return {
        "count": count, "command_count": len(operations), "accepted": accepted,
        "mean_accepted": accepted / count if count else None,
        "timing_scope": "Host command; a command can execute several rounds",
        "p50_command_ms": percentile(durations, 0.5),
        "p95_command_ms": percentile(durations, 0.95),
    }


def validate_producer(producer, received, first_token, mtp_width):
    if (producer.get("status") != "complete"
            or producer.get("reset_complete") is not True):
        raise ValueError("Producer request did not complete and reset")
    if (producer.get("seq_len") != PROMPT_TOKENS
            or producer.get("first_token_id") != first_token
            or producer.get("max_tokens") != OUTPUT_TOKENS):
        raise ValueError("Producer request shape differs from the benchmark")
    if (producer.get("token_ids") != received.token_ids
            or producer.get("returned_token_ids") != received.token_ids):
        raise ValueError("Producer and client token sequences differ")
    stats = producer["engine_stats"]
    if (stats.get("finish_reason") != "length"
            or stats.get("output_tokens") != OUTPUT_TOKENS
            or stats.get("finish_error")):
        raise ValueError("Producer statistics describe incomplete delivery")
    operations = producer["operations"]
    if any("error" in operation for operation in operations):
        raise ValueError("A timed TP command failed")
    if any(operation["end_ns"] < operation["start_ns"]
           for operation in operations):
        raise ValueError("A TP command clock ran backwards")
    rounds = [item for item in operations
              if item["operation"] in ("advance_mtp", "advance_mtp_n")]
    if producer.get("runtime", {}).get("mtp_width") != mtp_width:
        raise ValueError("Actual server MTP width differs from the benchmark")
    if mtp_width == 1:
        calls = sum(item["operation"] == "step" for item in operations)
        if rounds or calls != OUTPUT_TOKENS - 1:
            raise ValueError("Serial producer has the wrong step count")
    else:
        validate_round_commands(rounds, stats, mtp_width, producer["token_ids"])
    metrics = token_metrics(producer["token_ids"], producer["callback_ns"])
    return metrics, rounds


def post(url, path, payload, timeout):
    address = urlsplit(url)
    classes = {"http": http.client.HTTPConnection,
               "https": http.client.HTTPSConnection}
    if address.scheme not in classes or address.hostname is None:
        raise ValueError("Use an http:// or https:// service URL")
    connection = classes[address.scheme](address.hostname, address.port,
                                        timeout=timeout)
    body = json.dumps(payload).encode()
    try:
        connection.request("POST", address.path.rstrip("/") + path,
                           body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {response.read(4096)!r}"
            )
        return connection, response
    except BaseException:
        connection.close()
        raise


def prefill(args, prompt):
    payload = {
        "model": args.model, "prompt": prompt, "max_tokens": 1,
        "add_special_tokens": False, "temperature": 0, "top_k": 1,
        "ignore_eos": True, "logprobs": 1, "stream": False,
        "kv_transfer_params": {"tilert_host": args.decode_host,
                               "tilert_ctrl_port": args.control_port},
    }
    connection, response = post(args.prefill_url, "/v1/completions", payload,
                                args.timeout)
    try:
        result = json.loads(response.read())
    finally:
        connection.close()
    return result


def prefill_token(result, prompt_tokens=PROMPT_TOKENS):
    usage = result.get("usage", {})
    if (type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or usage.get("prompt_tokens") != prompt_tokens
            or usage.get("completion_tokens") != 1):
        raise ValueError(
            f"Prefill usage must be exactly {prompt_tokens} input + 1 output"
        )
    tokens = result["choices"][0]["logprobs"]["tokens"]
    if (len(tokens) != 1 or not isinstance(tokens[0], str)
            or not tokens[0].startswith("token_id:")):
        raise ValueError("Prefill requires --return-tokens-as-token-ids")
    token = int(tokens[0].split(":", 1)[1])
    if not 0 <= token < VOCAB_SIZE:
        raise ValueError("Prefill first token is outside the vocabulary")
    return token


def decode(args, rid, first, received=None):
    payload = {"rid": rid, "first_token_id": first,
               "max_tokens": OUTPUT_TOKENS, "stream": True,
               "timeout_s": args.timeout,
               "sampling": {"temperature": 0, "top_k": 1,
                            "ignore_eos": True}}
    if received is None:
        received = DecodeStream()
    connection, response = post(args.decode_url, "/pd/decode", payload,
                                args.timeout)
    try:
        while True:
            data = response.read1(65536)
            timestamp = perf_counter_ns()
            if not data:
                break
            received.feed(data, timestamp)
    finally:
        connection.close()
    received.finish(first)
    return received


def run_request(args, prompt, precision):
    start = perf_counter_ns()
    response = prefill(args, prompt)
    rid = derive_rid(response["id"])
    received = DecodeStream()
    try:
        first = prefill_token(response)
        decode(args, rid, first, received)
        producer = load_timing(args.timing_dir, rid)
        validate_runtime(producer, precision)
        metrics, rounds = validate_producer(producer, received, first,
                                            args.mtp_width)
    except BaseException as error:
        failure = {
            "rid": rid, "status": "error", "prefill_response": response,
            "error": f"{type(error).__name__}: {error}",
            "client_token_ids": received.token_ids,
            "client_token_ns": received.timestamps_ns,
            "client_messages": received.messages,
            "client_reads": received.reads, "client_done": received.done,
        }
        atomic_json(timing_path(args.output / "failures", rid), failure)
        try:
            connection, reply = post(args.decode_url, "/pd/cancel",
                                     {"rid": rid}, args.timeout)
            reply.read()
            connection.close()
        except Exception:
            pass
        raise
    client = received.finish(first)
    return {
        "rid": rid, "request_start_ns": start,
        "prefill_response": response, "first_token_id": first,
        "prompt_sha256": prompt_digest(prompt),
        "diversity": output_diversity(received.token_ids),
        "client": client, "client_token_ns": received.timestamps_ns,
        "client_token_ids": received.token_ids,
        "client_messages": received.messages, "client_reads": received.reads,
        "client_ttft_ms": (received.timestamps_ns[0] - start) / 1e6,
        "client_done_ns": received.done_ns,
        "client_burst_sizes": [len(item["token_ids"])
                               for item in received.messages],
        "client_done": received.done, "producer": metrics,
        "rounds": rounds, "producer_report": producer,
        "producer_report_path": str(timing_path(args.timing_dir, rid)),
    }


def load_prompts(path):
    prompts = json.loads(Path(path).read_text())
    if isinstance(prompts, list) and prompts and type(prompts[0]) is int:
        prompts = [prompts]
    if (not isinstance(prompts, list) or not prompts
            or any(not isinstance(prompt, list) or len(prompt) != PROMPT_TOKENS
                   or any(type(token) is not int
                          or not 0 <= token < VOCAB_SIZE for token in prompt)
                   for prompt in prompts)):
        raise ValueError("Supply JSON lists containing exactly 8192 token IDs")
    return prompts


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", default="http://127.0.0.1:18080")
    parser.add_argument("--decode-host", required=True)
    parser.add_argument("--control-port", type=int, default=5556)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--timing-dir", type=Path, required=True)
    parser.add_argument("--precision-dir", type=Path, required=True)
    parser.add_argument("--diversity-policy", type=Path,
                        help="Explicit token/bigram minima; without a policy, "
                             "passes_2ms remains false")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mtp-width", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--expert-bank", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.requests < 1 or args.timeout <= 0:
        parser.error("warmup >= 0, requests >= 1 and timeout > 0 required")
    return args


def main():
    args = parse_arguments()
    from precision_gate import validate_workload_precision

    prompts = load_prompts(args.prompt_ids)
    diversity_policy = load_diversity_policy(args.diversity_policy)
    precision = validate_workload_precision(
        args.precision_dir, args.checkpoint, mtp_width=args.mtp_width,
        expert_backend="bank" if args.expert_bank else "per_expert",
        source_root=ROOT,
    )
    prompt_hashes = validate_prompts(prompts, precision)
    atomic_json(args.output / "prompts.json", prompts)
    diversity_policy = freeze_diversity_policy(diversity_policy, args.output)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "A3 TP16 GLM-5.2-W4A8C8 batch1 PD decode; 8192 -> 1024",
        "python": platform.python_version(), "precision": precision,
        "diversity_policy": diversity_policy,
        "submitted_prompt_sha256s": prompt_hashes,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "percentile": "nearest rank", "clock": "perf_counter_ns",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest() for path in (
                Path(__file__), Path(__file__).with_name("precision_gate.py"),
                ROOT / "tilert/pd_vllm/decode_timing.py",
                ROOT / "tilert/pd_vllm/wire.py",
            )
        },
        "requests": [], "warmup_requests": [], "request_sha256": {},
        "status": "running",
    }
    seen = set()
    measured = []
    try:
        for number in range(args.warmup + args.requests):
            warmup = number < args.warmup
            result = run_request(
                args, prompts[number % len(prompts)], precision,
            )
            if result["rid"] in seen:
                raise ValueError("Prefill reused a benchmark request ID")
            seen.add(result["rid"])
            name = f"request-{number:04d}.json"
            atomic_json(args.output / name, result)
            report["request_sha256"][name] = hashlib.sha256(
                (args.output / name).read_bytes(),
            ).hexdigest()
            report["warmup_requests" if warmup else "requests"].append(name)
            if not warmup:
                measured.append(result)
        for source in ("producer", "client"):
            report[source] = aggregate(measured, source)
        rounds = [item for request in measured for item in request["rounds"]]
        report["rounds"] = round_summary(rounds)
        report["client_ttft_ms"] = {
            "p50": percentile([item["client_ttft_ms"] for item in measured],
                              0.50),
            "p95": percentile([item["client_ttft_ms"] for item in measured],
                              0.95),
        }
        report["status"] = "measured"
        report["measured_prompt_counts"] = dict(Counter(
            item["prompt_sha256"] for item in measured
        ))
        report["passes_2ms"] = passes_target(report, measured, args.output)
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "summary.json", report)
    print(json.dumps({"summary": str(args.output / "summary.json"),
                      "passes_2ms": report["passes_2ms"]}))


if __name__ == "__main__":
    main()
