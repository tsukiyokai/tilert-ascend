"""Check the exact tail seam and adversarial arrivals at its retained join."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare import (BODY, EMPTY_TAIL, END_PHASE, FINAL_JOINS, KERNEL,
                     PROFILE_GUARD, TRACE, bind_body, bind_kernel, prepare, rename)


def tail_schedules(require_join):
    """Explore one CUBE and two VECTOR workers with independently delayed stores."""
    initial = (0, 0, 0)
    pending = [initial]
    visited = {initial}
    unsafe = []
    while pending:
        state = pending.pop()
        for worker, step in enumerate(state):
            if step == 4:
                continue
            # 0: work pending; 1: local publish complete; 2: body returned;
            # 3: outer join reached; 4: pipe reset complete.
            if step == 3 and require_join and min(state) < 3:
                continue
            after = tuple(value + (index == worker)
                          for index, value in enumerate(state))
            if step == 3 and min(state) < 3:
                unsafe.append((state, worker))
            if after not in visited:
                visited.add(after)
                pending.append(after)
    return visited, unsafe


class TailJoinContract(unittest.TestCase):
    parent = Path(os.environ.get("FFN_TOKEN_JOIN_PARENT", str(
        Path(__file__).resolve().parents[4]
        / "docs/evidence/ascend-decode/megakernel-device-20261021"
        / "a3-194/ffn-token-output-build-v2")))

    def test_only_redundant_joins_change_executable_source(self):
        original = (self.parent / BODY).read_text()
        candidate = bind_body(original)
        restored = candidate.replace(PROFILE_GUARD, "").replace(
            EMPTY_TAIL.replace("    timing.Barrier(5);\n", ""), EMPTY_TAIL)
        self.assertEqual(restored, original)
        self.assertEqual(original.count("timing.Barrier(5);"), 1)
        self.assertNotIn("timing.Barrier(5);", candidate)
        kernel = (self.parent / KERNEL).read_text()
        reduced = bind_kernel(kernel)
        self.assertEqual(reduced.replace(FINAL_JOINS.replace(
            "    ffn_bridge::device::EndPhase(pipe);\n", "", 1), FINAL_JOINS), kernel)
        self.assertEqual(reduced.count("ffn_bridge::device::EndPhase(pipe);"), 1)

    def test_remaining_tail_is_empty_and_publication_join_keeps_reset_order(self):
        trace = (self.parent / TRACE).read_text()
        for declaration in (
            "__aicore__ inline void Begin(int) {}",
            "__aicore__ inline void FinishFinal(int) {}",
            "__aicore__ inline void Store(__gm__ int64_t*, int64_t, int64_t) {}",
            "#else\nusing Timing = NoTrace;\n#endif",
        ):
            self.assertIn(declaration, trace)
        end_phase = (self.parent / END_PHASE).read_text()
        self.assertIn(
            "__aicore__ inline void EndPhase(TPipe& pipe)\n{\n"
            "    PipeBarrier<PIPE_ALL>();\n"
            "    DataSyncBarrier<MemDsbT::DDR>();\n"
            "    SyncAll<false>();\n"
            "    pipe.Reset();\n}", end_phase)
        kernel = bind_kernel((self.parent / KERNEL).read_text())
        call_end = kernel.index("ffn_bridge::kCombined]);")
        tail = kernel[call_end + len("ffn_bridge::kCombined]);"):]
        tail = tail.split("}\n}  // namespace", 1)[0]
        statements = [line.strip() for line in tail.splitlines()
                      if line.strip() and not line.strip().startswith("//")]
        self.assertEqual(statements, ["ffn_bridge::device::EndPhase(pipe);"])

    def test_profile_compile_guard_precedes_device_includes(self):
        source = bind_body((self.parent / BODY).read_text())
        prefix = source[:source.index("/**")]
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "guard.h"
            header.write_text(prefix + "#endif\n")
            command = ["g++", "-E", "-x", "c++", str(header)]
            subprocess.run(command, check=True, capture_output=True)
            rejected = subprocess.run(command + ["-DROUTED_ENABLE_PROFILE=1"],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("uninstrumented NoTrace", rejected.stderr)

    def test_arbitrary_worker_delays_cannot_reset_before_all_returns(self):
        states, unsafe = tail_schedules(require_join=True)
        self.assertIn((4, 4, 4), states)
        self.assertEqual(unsafe, [])
        _, no_join = tail_schedules(require_join=False)
        self.assertTrue(any(min(state) == 0 for state, _ in no_join))

    def test_all_numeric_helpers_validators_and_public_abi_stay_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            report = prepare(self.parent, output)
            for name in (
                "source/contract.h", "source/binding.cpp", "source/output_native.h",
                "source/output_contract.h", "body/source/routed_int4/host_validate.h",
                "bridge/source/ffn_bridge/host_validate.h", TRACE, END_PHASE,
                "body/source/routed_int4/routed_int4_front.h",
                "body/source/routed_int4/routed_int4_dispatch.h",
                "body/source/routed_int4/routed_int4_pipeline_native.h",
            ):
                self.assertEqual((output / name).read_text(),
                                 rename((self.parent / name).read_text()))
            self.assertEqual(report["public_scratch_count"], 32)
            self.assertEqual(report["additional_gm_bytes"], 0)
            self.assertEqual(report["additional_ub_bytes"], 0)
            self.assertEqual(report["body_barriers"], 5)
            self.assertEqual(report["outer_end_phases"], 1)
            self.assertEqual((output / "check.py").read_text(),
                             (self.parent / "check.py").read_text().replace(
                                 "torch.ops.tilert_ffn_token_output",
                                 "torch.ops.tilert_ffn_token_join"))
            binary = Path(directory) / "ownership"
            subprocess.run([
                "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-include", str(output / "source/contract.h"),
                str(output / "source/ownership_probe.cpp"),
                "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
