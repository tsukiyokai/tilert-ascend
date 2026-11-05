"""Measure serial 8192-to-1024-token vLLM completions from streamed IDs."""

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import platform
from statistics import pstdev
import sys
from time import perf_counter_ns
from urllib.parse import urlsplit


PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


class Client:
    def __init__(self, url, timeout):
        address = urlsplit(url)
        if address.scheme != "http" or address.path not in ("", "/"):
            raise ValueError("Use the local http://host:port service URL")
        self.host = address.hostname
        self.port = address.port
        self.timeout = timeout

    def connect(self):
        return http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout)

    def get(self, path):
        connection = self.connect()
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {data[:4096]!r}")
            return data
        finally:
            connection.close()

    def completion(self, model, prompt):
        payload = {
            "model": model, "prompt": prompt, "max_tokens": OUTPUT_TOKENS,
            "n": 1, "temperature": 0, "seed": 0,
            "echo": False, "add_special_tokens": False,
            "ignore_eos": True, "stop": [], "stop_token_ids": [],
            "stream": True, "return_token_ids": True, "stream_interval": 1,
            "stream_options": {"include_usage": True},
        }
        connection = self.connect()
        started = perf_counter_ns()
        try:
            connection.request(
                "POST", "/v1/completions", body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {response.read(4096)!r}"
                )
            result = read_completion(response, prompt)
        finally:
            connection.close()
        first = result["events"][0]["received_ns"]
        last = result["events"][-1]["received_ns"]
        result.update(ttft_ms=(first - started) / 1e6,
                      tpot_ms=(last - first) / (OUTPUT_TOKENS - 1) / 1e6,
                      decode_span_ns=last - first)
        return result


def read_completion(response, prompt):
    events, tokens, usage = [], [], None
    text = []
    finish_reason, prompt_ids, request_id = None, None, None
    done = False
    while line := response.readline():
        received = perf_counter_ns()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
            break
        message = json.loads(data)
        if "error" in message:
            raise RuntimeError(f"SSE error: {message['error']}")
        if request_id is None:
            request_id = message["id"]
        if message["id"] != request_id:
            raise ValueError("SSE request ID changed")
        if message.get("usage") is not None:
            usage = message["usage"]
        for choice in message["choices"]:
            if choice["index"] != 0:
                raise ValueError("The baseline requires one sequence")
            if choice.get("prompt_token_ids") is not None:
                prompt_ids = choice["prompt_token_ids"]
            ids = choice.get("token_ids") or []
            if any(type(token) is not int or not 0 <= token < 154880
                   for token in ids):
                raise ValueError("The service returned invalid token IDs")
            if choice.get("text") and not ids:
                raise ValueError("Streamed text has no token IDs")
            text.append(choice.get("text", ""))
            if ids:
                tokens.extend(ids)
                events.append({"received_ns": received, "token_ids": ids,
                               "cumulative_tokens": len(tokens)})
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    expected_usage = {"prompt_tokens": PROMPT_TOKENS,
                      "completion_tokens": OUTPUT_TOKENS,
                      "total_tokens": PROMPT_TOKENS + OUTPUT_TOKENS}
    if not done or finish_reason != "length":
        raise ValueError("The stream did not finish at the requested length")
    if prompt_ids != prompt:
        raise ValueError("Returned prompt IDs differ from the 8192 input IDs")
    if len(tokens) != OUTPUT_TOKENS or usage is None or any(
        usage.get(name) != value for name, value in expected_usage.items()
    ):
        raise ValueError("Token IDs and usage must describe exactly 8K to 1K")
    return {"request_id": request_id, "usage": usage,
            "finish_reason": finish_reason, "token_ids": tokens,
            "text": "".join(text), "events": events}


def summarize(requests):
    values = sorted(item["tpot_ms"] for item in requests)

    def percentile(fraction):
        return values[math.ceil(len(values) * fraction) - 1]

    mean = sum(values) / len(values)
    return {
        "requests": len(values), "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95), "mean_ms": mean,
        "p99_ms": percentile(0.99), "stddev_ms": pstdev(values),
        "token_weighted_tok_s": 1000 / mean,
        "unique_outputs": len({tuple(item["token_ids"]) for item in requests}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.warmup < 1 or args.requests < 1:
        parser.error("Warmup and measured request counts must be positive")
    prompt = json.loads(args.prompt.read_text())
    if (not isinstance(prompt, list) or len(prompt) != PROMPT_TOKENS
            or any(type(token) is not int or not 0 <= token < 154880
                   for token in prompt)):
        raise ValueError("Use exactly 8192 GLM-5.2 token IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "status": "running",
        "scope": "Client delivery TPOT; vLLM prefill and decode, batch one",
        "formula": "(last token arrival - first token arrival) / 1023",
        "source_sha256": digest(Path(__file__)),
        "prompt_path": str(args.prompt), "prompt_sha256": digest(args.prompt),
        "runtime": json.loads(args.runtime.read_text()),
        "command": [sys.executable, *sys.argv],
        "warmup_requests": [], "requests": [],
    }
    client = Client(args.url, args.timeout)
    try:
        client.get("/health")
        report["models"] = json.loads(client.get("/v1/models"))
        for index in range(args.warmup + args.requests):
            result = client.completion(args.model, prompt)
            field = "warmup_requests" if index < args.warmup else "requests"
            report[field].append(result)
            save(args.output, report)
            print(json.dumps({"phase": field, "index": len(report[field]),
                              "tpot_ms": result["tpot_ms"]}), flush=True)
        client.get("/health")
        report.update(status="complete", summary=summarize(report["requests"]))
        print(json.dumps(report["summary"]))
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
    save(args.output, report)
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
