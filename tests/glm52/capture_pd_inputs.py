"""Drive batch-one prefill and retain its HIXL prefix and first-token IDs."""

import argparse
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from bench_pd_decode import post, prefill_token


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402
from tilert.pd_vllm.prefix_capture import digest, prompt_digest  # noqa: E402
from tilert.pd_vllm.wire import derive_rid  # noqa: E402


def validate_capture_service(url, timeout):
    address = urlsplit(url)
    classes = {"http": http.client.HTTPConnection,
               "https": http.client.HTTPSConnection}
    if address.scheme not in classes or address.hostname is None:
        raise ValueError("Capture service requires an http:// or https:// URL")
    connection = classes[address.scheme](address.hostname, address.port,
                                        timeout=timeout)
    try:
        connection.request("GET", address.path.rstrip("/") + "/capabilities")
        response = connection.getresponse()
        payload = json.loads(response.read())
        if (response.status != 200
                or payload.get("engine") != "PrefixCaptureEngine"):
            raise ValueError("The destination is not a prefix capture service")
    finally:
        connection.close()


def load_prompts(path):
    prompts = json.loads(Path(path).read_text())
    if isinstance(prompts, list) and prompts and type(prompts[0]) is int:
        prompts = [prompts]
    if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 8
            or any(not isinstance(prompt, list)
                   or not 1 <= len(prompt) <= 8192
                   or any(type(token) is not int or not 0 <= token < 154880
                          for token in prompt) for prompt in prompts)):
        raise ValueError("Supply 1..8 prompts of 1..8192 vocabulary token IDs")
    if len({prompt_digest(prompt) for prompt in prompts}) != len(prompts):
        raise ValueError("Capture prompts must be different")
    return prompts


def request_json(url, path, payload, timeout):
    connection, response = post(url, path, payload, timeout)
    try:
        return json.loads(response.read())
    finally:
        connection.close()


def capture_one(args, prompt):
    rid = None
    try:
        response = request_json(args.prefill_url, "/v1/completions", {
            "model": args.model, "prompt": prompt, "max_tokens": 1,
            "add_special_tokens": False, "temperature": 0, "top_k": 1,
            "ignore_eos": True, "logprobs": 1, "stream": False,
            "kv_transfer_params": {"tilert_host": args.capture_host,
                                   "tilert_ctrl_port": args.control_port},
        }, args.timeout)
        rid = derive_rid(response["id"])
        first = prefill_token(response, prompt_tokens=len(prompt))
        exported = request_json(args.capture_url, "/pd/decode", {
            "rid": rid, "first_token_id": first, "max_tokens": 1,
            "stream": False, "timeout_s": args.timeout,
            "sampling": {"temperature": 0, "top_k": 1, "ignore_eos": True},
        }, args.timeout)
        stats = exported.get("timing_ms", {})
        index = stats.get("capture_index")
        if (exported.get("rid") != rid
                or type(exported.get("seq_len")) is not int
                or exported.get("seq_len") != len(prompt)
                or not isinstance(exported.get("token_ids"), list)
                or any(type(token) is not int
                       for token in exported["token_ids"])
                or exported.get("token_ids") != [first]
                or stats.get("finish_reason") != "length"
                or stats.get("prompt_sha256") != prompt_digest(prompt)
                or type(index) is not int or index < 0):
            raise ValueError("The captured request differs from prefill")
        return {"request_id": rid, "capture_index": index,
                "prompt_token_ids": prompt,
                "prompt_sha256": prompt_digest(prompt),
                "seq_len": len(prompt), "first_token_id": first,
                "prefill_response": response, "capture_response": exported}
    except BaseException as error:
        if rid is not None:
            try:
                request_json(args.capture_url, "/pd/cancel", {"rid": rid},
                             args.timeout)
            except Exception as cancel_error:
                error.add_note(
                    f"Cancelling prefix capture failed: {cancel_error}"
                )
        raise


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--capture-url", default="http://127.0.0.1:18081")
    parser.add_argument("--capture-host", required=True)
    parser.add_argument("--control-port", type=int, default=5557)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    prompts = load_prompts(args.prompt_ids)
    validate_capture_service(args.capture_url, args.timeout)
    sources = (Path(__file__), Path(__file__).with_name("bench_pd_decode.py"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "vLLM/HIXL prefix capture; no decode inference or TPOT",
        "status": "running", "cases": [],
        "command": [sys.executable, *sys.argv],
        "prompt_file_sha256": digest(args.prompt_ids),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                          for path in sources},
    }
    try:
        seen_requests, seen_cases = set(), set()
        for prompt in prompts:
            result = capture_one(args, prompt)
            rid, index = result["request_id"], result["capture_index"]
            if rid in seen_requests or index in seen_cases:
                raise ValueError("Capture service reused a request or case")
            seen_requests.add(rid)
            seen_cases.add(index)
            directory = args.output / f"case-{index:02d}"
            atomic_json(directory / "prefill-response.json", result)
            report["cases"].append({key: result[key] for key in (
                "request_id", "capture_index", "prompt_sha256", "seq_len",
                "first_token_id",
            )})
        report["status"] = "captured"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "inputs.json", report)
    print(json.dumps({
        "status": report["status"], "cases": len(report["cases"]),
        "report": str(args.output / "inputs.json"),
    }))


if __name__ == "__main__":
    main()
