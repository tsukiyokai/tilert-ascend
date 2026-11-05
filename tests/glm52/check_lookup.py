"""CPU checks for lookup proposals, cache ownership and measurement records."""

import copy
from itertools import product
import random
from types import SimpleNamespace
import unittest

from bench_cann_mtp import MeasuredRank, validate_local
from check_mtp_round import (
    Hidden, ScalarDraft, ScalarTarget, autoregressive, next_token,
)
from ngram_acceptance import lookup, simulate
from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter


class Target(ScalarTarget):
    def reset(self):
        self.position = 0


class Draft(ScalarDraft):
    def reset(self):
        self.position = 0


def run_request(width, lookup, shortest=2):
    prompt = tuple((index * 7 + 3) % 251 for index in range(8192))
    target = Target(prompt, 8256)
    parent = Hidden("target", target.rows[-1][1], len(prompt))
    first = next_token(parent.context_hash, len(prompt))
    draft = None if lookup else Draft(target, 8191, first, "mixed")
    engine = MeasuredRank(target, draft, None, prompt if lookup else None,
                          lookup_min_ngram=shortest)
    engine.parent_hidden = [parent]
    engine.first_token_id = first
    engine._prompt_length = 8192
    engine.current = {"injected": [8192, None if lookup else 8191],
                      "first_token": first,
                      "rounds": []}
    engine.requests.append(engine.current)
    engine.start_mtp(first, 12, width, ())
    if lookup:
        assert engine._mtp.drafter.shortest == shortest
    tokens = [first]
    while engine._mtp.state.finish_reason is None:
        result = engine.advance_mtp()
        if lookup:
            assert result.commit.draft is None
            assert engine._mtp.state.draft_length is None
        tokens.extend(result.commit.accepted_tokens)
    expected, rows = autoregressive(prompt, first, 12, target.capacity)
    assert tokens == expected
    assert [token for token, _ in target.rows[:target.position]] == rows
    if lookup:
        assert engine.draft is None
        assert engine._mtp.drafter.context == list(prompt) + tokens
    engine.finish_mtp(engine.position)
    engine.reset()
    report = {
        "rank": 1, "rank_requests": engine.requests, "closed": True,
        "final_positions": [target.position,
                            None if draft is None else draft.position],
        "command_serial": 5 + len(engine.requests[0]["rounds"]),
    }
    args = SimpleNamespace(
        warmup=0, requests=1, width=width, max_tokens=12, lookup=lookup,
        target_graph=False, draft_graph=False, conditioned_graph=False,
        automaton=False, tree=False,
    )
    return report, args


class LookupChecks(unittest.TestCase):
    def test_lookup_preserves_serial_tokens_and_actual_cache_lengths(self):
        for width in (2, 8, 32):
            with self.subTest(width=width):
                report, args = run_request(width, lookup=True)
                validate_local(report, args)
                self.assertEqual(report["rank_requests"][0]["finished"],
                                 [8203, None])

    def test_mtp_still_requires_both_caches_to_advance(self):
        report, args = run_request(8, lookup=False)
        validate_local(report, args)
        self.assertEqual(report["rank_requests"][0]["finished"],
                         [8203, 8203])

    def test_unigram_lookup_preserves_serial_tokens_and_cache_lengths(self):
        report, args = run_request(32, lookup=True, shortest=1)
        validate_local(report, args)
        self.assertEqual(report["rank_requests"][0]["finished"],
                         [8203, None])

    def test_unigram_supplies_continuation_when_longer_suffix_is_absent(self):
        prompt = [1, 9, 2, 1]
        self.assertEqual(LookupDrafter(prompt).propose(3), [])
        self.assertEqual(LookupDrafter(prompt, shortest=1).propose(3),
                         [9, 2, 1])

    def test_validator_rejects_fictitious_lookup_cache_and_bad_target(self):
        report, args = run_request(8, lookup=True)
        for field in ("draft_retained", "retained"):
            damaged = copy.deepcopy(report)
            damaged["rank_requests"][0]["rounds"][0][field] = 9999
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_local(damaged, args)

    def test_lookup_prefers_longest_then_most_recent_match(self):
        drafter = LookupDrafter([1, 2, 3, 9, 1, 2, 7, 1, 2])
        self.assertEqual(drafter.propose(3), [7, 1, 2])
        drafter.commit([3, 9, 1, 2])
        self.assertEqual(drafter.propose(3), [7, 1, 2])

    def test_distinct_tails_do_not_hide_the_longer_match(self):
        # These distinct tails collide under the former 61-bit polynomial.
        left = [30000, 30712, 30000, 32048, 30503]
        right = [30306, 30000, 30004, 30000, 30000]
        context = left + [111] + right + [222] + left[-4:] + [333] + left
        drafter = LookupDrafter(context, longest=5)
        self.assertEqual(lookup(context, 1, 5, 2), [[111]])
        self.assertEqual(drafter.propose(1), [111])

    def test_match_can_end_inside_a_compressed_edge(self):
        context = [1, 2, 3, 4, 7, 8, 9, 2, 3, 4]
        self.assertEqual(LookupDrafter(context, longest=4).propose(3),
                         [7, 8, 9])

    def test_full_length_tail_uses_the_most_recent_continuation(self):
        tail = list(range(32))
        context = tail + [100] + tail + [101] + tail
        self.assertEqual(LookupDrafter(context).propose(4), [101, 0, 1, 2])

    def test_all_small_binary_contexts_match_independent_reverse_scan(self):
        for length in range(9):
            for context in product(range(2), repeat=length):
                for shortest, longest in ((1, 1), (1, 8), (2, 8), (4, 32)):
                    drafter = LookupDrafter(context[:-2], longest, shortest)
                    drafter.commit(())
                    drafter.commit(iter(context[-2:]))
                    for count in (0, 1, 7):
                        branches = lookup(list(context), count,
                                          longest, shortest)
                        self.assertEqual(drafter.propose(count),
                                         branches[0] if branches else [])

    def test_replay_counts_padding_matches(self):
        rounds = simulate([7, 7, 7], 3, 8, 2, prompt=[1, 2])
        self.assertEqual(rounds, [{"matched_drafts": 2, "accepted": 2}])

    def test_replay_clips_final_bonus_to_output_budget(self):
        rounds = simulate([4, 5], 1, 8, 2, prompt=[3, 4, 5, 3])
        self.assertEqual(rounds, [{"matched_drafts": 1, "accepted": 1}])

    def test_prompt_can_displace_a_better_shorter_match(self):
        context = [1, 2, 3, 9, 4, 1, 2]
        self.assertEqual(lookup(context, 1, 8, 2), [[3]])
        self.assertEqual(lookup([4, 1, 2, 8] + context, 1, 8, 2), [[8]])

    def test_incremental_index_matches_reverse_scan_after_each_commit(self):
        rng = random.Random(42)
        for shortest, longest in ((1, 1), (1, 8), (2, 8), (4, 12),
                                  (1, 32), (2, 32), (32, 32)):
            context = [rng.randrange(4) for _ in range(40)]
            drafter = LookupDrafter(context, longest, shortest)
            for _ in range(80):
                for count in (0, 1, 7, 31):
                    branches = lookup(context, count, longest, shortest)
                    expected = branches[0] if branches else []
                    self.assertEqual(drafter.propose(count), expected)
                added = [rng.randrange(4) for _ in range(rng.randrange(9))]
                context.extend(added)
                drafter.commit(added)


if __name__ == "__main__":
    unittest.main()
