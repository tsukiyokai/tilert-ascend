"""Check the capture CLI against HTTP fixtures; no HIXL or tensor execution."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "tests/glm52/capture_pd_inputs.py"
FIXTURES = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_hash(tokens):
    return hashlib.sha256(json.dumps(
        tokens, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


class HttpFixture:
    """Return explicit protocol payloads without allocating receive caches."""

    def __init__(self, *, engine="PrefixCaptureEngine", first=23,
                 prefill_change=None, decode_change=None, failure=None,
                 repeat_rid=False):
        self.engine = engine
        self.first = first
        self.prefill_change = prefill_change
        self.decode_change = decode_change
        self.failure = failure
        self.repeat_rid = repeat_rid
        self.requests = []
        self.prefill_responses = []
        self.decode_responses = []

    def respond(self, method, path, body):
        self.requests.append((method, path, body))
        if (method, path) == ("GET", "/capture/capabilities"):
            return 200, {"engine": self.engine}
        if (method, path) == ("POST", "/prefill/v1/completions"):
            index = len(self.prefill_responses)
            rid_index = 0 if self.repeat_rid else index
            response = {
                "id": f"cmpl-request{rid_index}q-0",
                "usage": {"prompt_tokens": len(body["prompt"]),
                          "completion_tokens": 1},
                "choices": [{"logprobs": {
                    "tokens": [f"token_id:{self.first + index}"],
                }}],
            }
            if self.prefill_change is not None:
                self.prefill_change(response)
            self.prefill_responses.append(response)
            return 200, response
        if (method, path) == ("POST", "/capture/pd/decode"):
            if self.failure == "decode_http":
                return 500, {"error": "capture fixture failure"}
            if self.failure == "decode_json":
                return 200, b"{unfinished"
            prompt = next(value["prompt"] for _, endpoint, value in
                          reversed(self.requests)
                          if endpoint == "/prefill/v1/completions")
            index = len(self.decode_responses)
            response = {
                "rid": body["rid"], "seq_len": len(prompt),
                "token_ids": [body["first_token_id"]],
                "timing_ms": {"capture_index": index,
                              "prompt_sha256": prompt_hash(prompt),
                              "finish_reason": "length"},
            }
            if self.decode_change is not None:
                self.decode_change(response)
            self.decode_responses.append(response)
            return 200, response
        if (method, path) == ("POST", "/capture/pd/cancel"):
            return 200, {"cancelled": body["rid"]}
        return 404, {"error": "unexpected fixture endpoint"}

    def bodies(self, path):
        return [body for _, endpoint, body in self.requests
                if endpoint == path]


@contextmanager
def serve(fixture):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def handle_request(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else None
            status, response = fixture.respond(self.command, self.path, body)
            data = response if isinstance(response, bytes) else json.dumps(
                response,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()


class CaptureHttpChecks(unittest.TestCase):
    @contextmanager
    def run_client(self, prompts, fixture):
        with tempfile.TemporaryDirectory(prefix="capture-http-") as temp:
            directory = Path(temp)
            prompt_file = directory / "prompts.json"
            prompt_file.write_text(json.dumps(prompts) + "\n")
            output = directory / "output"
            with serve(fixture) as url:
                result = subprocess.run([
                    sys.executable, str(CLIENT),
                    "--prefill-url", url + "/prefill",
                    "--capture-url", url + "/capture",
                    "--capture-host", "127.0.0.1", "--control-port", "5557",
                    "--prompt-ids", str(prompt_file), "--output", str(output),
                    "--timeout", "3",
                ], capture_output=True, text=True, timeout=15)
            FIXTURES.append({
                "test": self.id(), "returncode": result.returncode,
                "requests": [[method, path] for method, path, _ in
                             fixture.requests],
                "stderr": result.stderr,
            })
            yield result, output, prompt_file

    def assert_failed_capture(self, result, output, fixture):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        report = json.loads((output / "inputs.json").read_text())
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["cases"], [])
        self.assertEqual(list(output.glob("case-*/prefill-response.json")), [])
        self.assertEqual(fixture.bodies("/capture/pd/cancel"),
                         [{"rid": "request0q"}])

    def test_short_and_8k_prompts_preserve_p_response_and_hashes(self):
        prompts = [[3], [7, 11, 17],
                   [(index * 13 + 5) % 154880 for index in range(8192)]]
        fixture = HttpFixture()
        with self.run_client(prompts, fixture) as (result, output, path):
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "inputs.json").read_text())
            self.assertEqual(report["status"], "captured")
            self.assertEqual(report["prompt_file_sha256"], digest(path))
            self.assertEqual(len(report["cases"]), 3)
            for index, prompt in enumerate(prompts):
                saved = json.loads((output / f"case-{index:02d}"
                                    / "prefill-response.json").read_text())
                expected = {
                    "request_id": f"request{index}q", "capture_index": index,
                    "prompt_sha256": prompt_hash(prompt),
                    "seq_len": len(prompt), "first_token_id": 23 + index,
                }
                self.assertEqual(report["cases"][index], expected)
                self.assertEqual({key: saved[key] for key in expected},
                                 expected)
                self.assertEqual(saved["prompt_token_ids"], prompt)
                self.assertEqual(saved["prefill_response"],
                                 fixture.prefill_responses[index])
                self.assertEqual(saved["capture_response"],
                                 fixture.decode_responses[index])
                sent = fixture.bodies("/prefill/v1/completions")[index]
                self.assertEqual(sent["prompt"], prompt)
                self.assertEqual(sent["max_tokens"], 1)
                self.assertFalse(sent["add_special_tokens"])
                self.assertFalse(sent["stream"])
                self.assertTrue(sent["ignore_eos"])
                self.assertEqual(sent["kv_transfer_params"], {
                    "tilert_host": "127.0.0.1", "tilert_ctrl_port": 5557,
                })
            self.assertEqual(fixture.bodies("/capture/pd/cancel"), [])

    def test_flat_prompt_input_is_one_capture(self):
        fixture = HttpFixture()
        with self.run_client([3, 10, 17], fixture) as (result, output, _):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((output / "inputs.json").read_text())[
                "cases"][0]["seq_len"], 3)

    def test_wrong_engine_is_refused_before_first_prefill(self):
        fixture = HttpFixture(engine="AscendServingEngine")
        with self.run_client([3], fixture) as (result, output, _):
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(fixture.requests, [
                ("GET", "/capture/capabilities", None),
            ])
            self.assertFalse(output.exists())

    def test_invalid_prompt_file_is_refused_before_http(self):
        cases = ([], [[]], [True], [-1], [154880], [[3], [3]],
                 [0] * 8193, [[index] for index in range(9)])
        for prompts in cases:
            with self.subTest(length=len(prompts)):
                fixture = HttpFixture()
                with self.run_client(prompts, fixture) as (result, output, _):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(fixture.requests, [])
                    self.assertFalse(output.exists())

    def test_wrong_prefill_usage_or_token_cancels_without_decode(self):
        changes = (
            lambda r: r["usage"].update(prompt_tokens=2),
            lambda r: r["usage"].update(completion_tokens=2),
            lambda r: r["usage"].update(prompt_tokens=True),
            lambda r: r["usage"].update(completion_tokens=True),
            lambda r: r["choices"][0]["logprobs"].update(tokens=[]),
            lambda r: r["choices"][0]["logprobs"].update(tokens=["word"]),
            lambda r: r["choices"][0]["logprobs"].update(
                tokens=["token_id:-1"],
            ),
            lambda r: r["choices"][0]["logprobs"].update(
                tokens=["token_id:154880"],
            ),
        )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                fixture = HttpFixture(prefill_change=change)
                with self.run_client([3], fixture) as (result, output, _):
                    self.assert_failed_capture(result, output, fixture)
                    self.assertEqual(fixture.bodies("/capture/pd/decode"), [])

    def test_wrong_decode_identity_is_rejected_and_cancelled(self):
        changes = (
            lambda r: r.update(rid="another-request"),
            lambda r: r.update(seq_len=2),
            lambda r: r.update(seq_len=True),
            lambda r: r.update(token_ids=[2]),
            lambda r: r.update(token_ids=[True]),
            lambda r: r["timing_ms"].update(prompt_sha256="0" * 64),
            lambda r: r["timing_ms"].update(capture_index=-1),
            lambda r: r["timing_ms"].update(capture_index=True),
            lambda r: r["timing_ms"].pop("capture_index"),
            lambda r: r["timing_ms"].update(finish_reason="cancelled"),
        )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                fixture = HttpFixture(first=1, decode_change=change)
                with self.run_client([3], fixture) as (result, output, _):
                    self.assert_failed_capture(result, output, fixture)

    def test_decode_http_and_json_failures_cancel_the_known_request(self):
        for failure in ("decode_http", "decode_json"):
            with self.subTest(failure=failure):
                fixture = HttpFixture(failure=failure)
                with self.run_client([3], fixture) as (result, output, _):
                    self.assert_failed_capture(result, output, fixture)

    def test_reused_request_or_capture_index_cannot_record_second_case(self):
        fixtures = (
            HttpFixture(repeat_rid=True),
            HttpFixture(decode_change=lambda r: r["timing_ms"].update(
                capture_index=0,
            )),
        )
        for index, fixture in enumerate(fixtures):
            with self.subTest(case=index):
                with self.run_client([[3], [4]], fixture) as values:
                    result, output, _ = values
                    self.assertNotEqual(result.returncode, 0)
                    report = json.loads((output / "inputs.json").read_text())
                    self.assertEqual(report["status"], "error")
                    self.assertEqual(len(report["cases"]), 1)
                    saved = json.loads((output / "case-00"
                                        / "prefill-response.json").read_text())
                    self.assertEqual(saved["prompt_token_ids"], [3])
                    self.assertEqual(len(list(output.glob(
                        "case-*/prefill-response.json",
                    ))), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = (Path(__file__), CLIENT, CLIENT.with_name("bench_pd_decode.py"),
             ROOT / "tilert/pd_vllm/prefix_capture.py",
             ROOT / "tilert/pd_vllm/decode_timing.py",
             ROOT / "tilert/pd_vllm/wire.py")
    before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CaptureHttpChecks)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    passed = result.wasSuccessful() and before == after
    report = {
        "status": "pass" if passed else "fail", "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv], "fixtures": FIXTURES,
        "fixture_count": len(FIXTURES), "test_output": stream.getvalue(),
        "source_stable": before == after, "source_sha256": after,
        "torch_imported": "torch" in sys.modules,
        "scope": "Real capture CLI and loopback HTTP protocol fixtures",
        "not_executed": ["vLLM prefill", "HIXL", "TP16 collectives",
                         "NPU execution", "tensor export", "numerical parity"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), end="")
    print(json.dumps({"status": report["status"], "tests": result.testsRun,
                      "fixtures": len(FIXTURES), "report": str(args.output)}))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
