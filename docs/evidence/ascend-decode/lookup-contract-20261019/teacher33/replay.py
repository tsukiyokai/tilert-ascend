"""CPU lookup replay bounded to the trusted first output plus 32 successors."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from lookup_draft import LookupDrafter
from ngram_acceptance import lookup, matched_prefix, simulate

PROMPT = Path('/workspace/experiments/shared-fia-20261020/prompt.json')
TEACHER = PROMPT.with_name('teacher-32.json')
MANIFEST = Path('/workspace/offline/prefix8k-allocation-20261017/capture/case-00/manifest.json')
EXPECTED = {PROMPT: 'd70449c0bb76f23c8641a4ea1037a3586dcd87e22e395d0961b2e0a11712ba0c',
            TEACHER: '82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655',
            MANIFEST: 'b01cceb2d2e30564807ce9efc441ad535999b56b73d0c333e93bb18694ccda74'}
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert all(digest(path) == sha for path, sha in EXPECTED.items())
prompt, teacher, manifest = [json.loads(p.read_text()) for p in (PROMPT, TEACHER, MANIFEST)]
outputs = [teacher['first_token_id']] + teacher['expected_next_tokens']
assert len(prompt) == 8192 and prompt == manifest['prompt_token_ids']
assert teacher['prompt_sha256'] == digest(PROMPT) and teacher['prompt_tokens'] == 8192
assert len(outputs) == 33 and outputs[:-1] == teacher['teacher_tokens']
assert outputs[0] == manifest['first_token_id'] and all(0 <= x < 154880 for x in outputs)

def scalar_proposal(context, count):
    for size in range(8, 0, -1):
        for end in range(len(context) - 1, size - 1, -1):
            if context[end-size:end] == context[-size:]:
                return context[end:end+count], size, end
    return [], 0, None

results = []
for width in (8, 16, 32):
    drafter = LookupDrafter(prompt, longest=8, shortest=1)
    drafter.commit(outputs[:1])
    scalar_context, cursor, scalar_cursor, rounds = prompt + outputs[:1], 1, 1, []
    while cursor < len(outputs):
        assert cursor == scalar_cursor and drafter.context == scalar_context
        # Both proposals and their source positions are fixed before label access.
        proposal = drafter.propose(width - 1)
        reference, size, source = scalar_proposal(scalar_context, width - 1)
        assert proposal == reference == (lookup(scalar_context, width-1, 8, 1) or [[]])[0]
        pending = drafter.context[-1]
        candidates = [pending] + proposal + [pending] * (width - 1 - len(proposal))
        scalar_candidates = [scalar_context[-1]] + reference
        scalar_candidates += [pending] * (width - len(scalar_candidates))
        assert candidates == scalar_candidates and len(candidates) == width
        labels = outputs[cursor:]  # Future labels become visible only after proposal construction.
        matched = matched_prefix(candidates[1:], labels)
        accepted = min(matched + 1, len(labels))
        scalar_matched = 0
        for index in range(1, width):
            if scalar_cursor + index - 1 >= len(outputs):
                break
            if scalar_candidates[index] != outputs[scalar_cursor + index - 1]:
                break
            scalar_matched += 1
        scalar_accepted = min(scalar_matched + 1, len(outputs) - scalar_cursor)
        assert (matched, accepted) == (scalar_matched, scalar_accepted)
        committed = labels[:accepted]
        rounds.append({'round': len(rounds)+1, 'next_output_offset_before': cursor,
            'next_output_offset_after': cursor+accepted, 'pending_token': pending,
            'query_position_start': 8192+cursor-1, 'width': width,
            'matched_suffix_length': size, 'source_context_offset': source,
            'source_positions': [] if source is None else list(range(source, source+len(proposal))),
            'proposal': proposal, 'candidates': candidates,
            'padding_rows': width-1-len(proposal), 'matched_drafts_within_window': matched,
            'natural_match_beyond_window_unknown': matched == len(labels) and len(labels) < width-1,
            'accepted_output_tokens': committed, 'accepted_outputs': accepted,
            'final_bonus_budget_clipped': accepted < matched+1})
        drafter.commit(committed)
        scalar_context.extend(outputs[scalar_cursor:scalar_cursor+scalar_accepted])
        cursor += accepted
        scalar_cursor += scalar_accepted
    existing = simulate(outputs, width-1, 8, 1, prompt=prompt)
    assert [(r['matched_drafts'], r['accepted']) for r in existing] == [
        (r['matched_drafts_within_window'], r['accepted_outputs']) for r in rounds]
    assert cursor == 33 and drafter.context == prompt + outputs
    results.append({'width': width, 'rounds': len(rounds), 'planned_verify_rows': width*len(rounds),
                    'new_outputs_after_prefill': sum(r['accepted_outputs'] for r in rounds),
                    'output_tokens_including_prefill': 33, 'raw_rounds': rounds})
bonus_witness = simulate([4, 5], 1, 8, 2, prompt=[3, 4, 5, 3])
assert bonus_witness == [{'matched_drafts': 1, 'accepted': 1}]
report = {'scope': 'One trusted 8192-prompt/33-output window; CPU replay, no NPU or TPOT/1024 claim',
    'shortest': 1, 'longest': 8, 'first_output_already_emitted_by_prefill': True,
    'oracle': 'Only frozen teacher32 linked successors; no old1024 file is opened. Unverified wrong-prefix tail predictions are not claimed.',
    'source_sha256': {str(p): digest(p) for p in [Path(__file__), *sorted((ROOT/'source').glob('*.py'))]},
    'input_sha256': {str(p): digest(p) for p in EXPECTED}, 'outputs': outputs, 'results': results,
    'separate_existing_bonus_clip_witness': bonus_witness,
    'independent_scalar_proposal_acceptance_cursor': 'pass', 'existing_ngram_simulate': 'pass',
    'torch_imported': 'torch' in sys.modules, 'torch_npu_imported': 'torch_npu' in sys.modules}
(ROOT/'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k != 'raw_rounds'} for r in results], indent=2))
