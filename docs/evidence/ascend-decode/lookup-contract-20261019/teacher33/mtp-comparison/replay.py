"""Replay measured MTP4 windows and causal lookup8 on one trusted teacher33."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from lookup_draft import LookupDrafter

PROMPT = Path('/workspace/experiments/shared-fia-20261020/prompt.json')
TEACHER = PROMPT.with_name('teacher-32.json')
AUDIT = ROOT / 'mtp-teacher-proposal-audit.json'
LOOKUP = Path('/workspace/experiments/lookup-teacher33-20261020/result.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = {PROMPT: 'd70449c0bb76f23c8641a4ea1037a3586dcd87e22e395d0961b2e0a11712ba0c',
            TEACHER: '82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655',
            AUDIT: 'f9b298c69aca0a0c5e42d9a5e3041802cf3f9fe4892a1db7bacfd444f858e185',
            LOOKUP: '546632f0e5cb0d45b70612aac109fe1b4920911caa4ee9fd00c1d78007dab396'}
assert all(digest(path) == sha for path, sha in expected.items())
prompt, teacher, audit, previous = [json.loads(p.read_text()) for p in expected]
outputs = [teacher['first_token_id']] + teacher['expected_next_tokens']
assert len(prompt) == teacher['prompt_tokens'] == 8192 and len(outputs) == 33
assert teacher['teacher_tokens'] == outputs[:-1] and outputs == previous['outputs']
assert teacher['prompt_sha256'] == digest(PROMPT)
assert audit['status'] == 'pass' and audit['cross_rank_candidates_exact'] is True
assert audit['references']['teacher']['sha256'] == digest(TEACHER)
assert audit['references']['prompt']['sha256'] == digest(PROMPT)
assert all(x['different_bytes'] == 0 and x['all_finite'] for x in audit['exact_controls'].values())
windows = {w['i']: w for w in audit['windows']}
assert len(windows) == 30 and set(windows) == set(range(30))
for i, window in windows.items():
    assert window['p'] == 8191 + i and window['labels'] == outputs[i:i + 4]
    for config in ('original', 'sharded'):
        candidate = window[config]['candidate']
        assert len(candidate) == 4 and candidate[0] == outputs[i]
        hits = [a == b for a, b in zip(candidate[1:], outputs[i + 1:i + 4])]
        matched = next((j for j, hit in enumerate(hits) if not hit), 3)
        assert hits == window[config]['hits'] and matched + 1 == window[config]['prefix_coverage']


def replay(kind, budget):
    width = 8 if kind == 'lookup8' else 4
    drafter = LookupDrafter(prompt, longest=8, shortest=1)
    drafter.commit(outputs[:1])
    offset, rounds = 0, []
    while offset < budget:
        if kind != 'lookup8' and offset not in windows:
            break
        if kind == 'lookup8':
            assert drafter.context == prompt + outputs[:offset + 1]
            proposal = drafter.propose(width - 1)
            candidate = [outputs[offset], *proposal]
            candidate += [outputs[offset]] * (width - len(candidate))
            padding = width - 1 - len(proposal)
        else:
            candidate = list(windows[offset][kind]['candidate'])
            padding = 0
        # Candidate IDs are fixed before accessing successor labels.
        labels = outputs[offset + 1:budget + 1]
        matched = 0
        for proposed, label in zip(candidate[1:], labels):
            if proposed != label:
                break
            matched += 1
        accepted = min(matched + 1, budget - offset)
        committed = labels[:accepted]
        rounds.append({'round': len(rounds) + 1, 'pending_output_offset': offset,
                       'pending_token': outputs[offset], 'target_position': 8192 + offset,
                       'mtp_seed_p': 8191 + offset if kind != 'lookup8' else None,
                       'candidates': candidate, 'padding_rows': padding,
                       'matched_within_budget': matched,
                       'accepted': accepted, 'accepted_tokens': committed,
                       'terminal': ('budget_clip' if accepted <= matched else
                                    'bonus' if matched == width - 1 else 'correction'),
                       'next_pending_offset': offset + accepted})
        drafter.commit(committed)
        offset += accepted
    remaining = budget - offset
    lower, upper = len(rounds) + math.ceil(remaining / width), len(rounds) + remaining
    return {'kind': kind, 'new_output_budget': budget, 'known_new_outputs': offset,
            'known_rounds': len(rounds), 'known_target_verify_rows': len(rounds) * width,
            'mean_known_outputs_per_round': offset / len(rounds),
            'missing_pending_window': offset if remaining else None,
            'full_budget_round_bounds': [lower, upper],
            'full_budget_verify_row_bounds': [lower * width, upper * width], 'rounds': rounds}


full = {kind: replay(kind, 32) for kind in ('lookup8', 'original', 'sharded')}
old = next(r for r in previous['results'] if r['width'] == 8)
assert full['lookup8']['known_rounds'] == old['rounds'] == 28
assert [(r['candidates'], r['accepted']) for r in full['lookup8']['rounds']] == [
    (r['candidates'], r['accepted_outputs']) for r in old['raw_rounds']]
assert full['original']['rounds'] == full['sharded']['rounds']
common_budget = min(full[k]['known_new_outputs'] for k in full)
common = {kind: replay(kind, common_budget) for kind in full}
assert all(r['known_new_outputs'] == common_budget for r in common.values())
assert all(r['rounds'] == full[k]['rounds'][:len(r['rounds'])] for k, r in common.items())
assert all(len(row['candidates']) == (8 if kind == 'lookup8' else 4)
           for kind, result in full.items() for row in result['rounds'])
assert all(row['candidates'][0] == outputs[row['pending_output_offset']]
           for result in full.values() for row in result['rounds'])
result = {'scope': 'Teacher-oracle replay of measured candidate IDs, not executed target acceptance or latency',
          'first_output_is_prefill': True, 'shortest_lookup': 1, 'longest_lookup': 8,
          'eos_ignored': True, 'capacity': 9344, 'outputs': outputs,
          'input_sha256': {str(p): digest(p) for p in expected},
          'source_sha256': {str(p): digest(p) for p in [Path(__file__), *sorted((ROOT / 'source').glob('*.py'))]},
          'common_new_output_budget': common_budget, 'common': common, 'full_32_new': full,
          'break_even_Tmtp_round_over_Tlookup_round': {
              'common_strict_less_than': common['lookup8']['known_rounds'] / common['original']['known_rounds'],
              'full_guaranteed_strict_less_than': 28 / full['original']['full_budget_round_bounds'][1],
              'full_possible_strict_less_than': 28 / full['original']['full_budget_round_bounds'][0]},
          'limits': ['MTP windows 30 and 31 are absent; no proposal tokens are synthesized.',
                     'Wrong-prefix target predictions are absent and are not filled with teacher labels.',
                     'Measured M1-conditioned MTP candidates need not equal a live M4-conditioned session.',
                     'MTP round cost includes seed/chain/conditioned passes; target verify rows alone are not cost.',
                     'One fixed trusted trajectory does not establish prompt-set acceptance or a runtime policy.']}
(ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: [{q: v for q, v in x.items() if q != 'rounds'} for x in result[k].values()]
                  for k in ('common', 'full_32_new')}, indent=2))
