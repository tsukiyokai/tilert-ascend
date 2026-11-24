"""Decode one complete width32 frame against an admitted HF reference.

The caller admits the reference and execution identities separately.
This receiver checks layout and typed identities, then returns owned CPU
row snapshots. Numerical comparison keeps the caller's original thresholds.
"""

import hashlib
import json

import torch


WIDTH = 32
COUNTS = {'hidden_states': 79, 'layer_logits': 78,
          'indices': 78, 'routes': 78}
CACHE_FIELDS = ('latent', 'rope', 'index', 'index_scale')
FRAME_KEYS = {
    'tag', 'replay_id', 'input_sha256', 'row_ids', 'lengths', 'tensor',
}
HEADER_KEYS = {
    'origin', 'replay_id', 'width', 'layer_count', 'fp32_head',
    'mla_preprocess',
    'inputs', 'input_sha256', 'source_sha256', 'checkpoint_sha256',
    'trace_sha256', 'weight_arithmetic_sha256', 'loaded_reference_sources',
    'installed_hf_source_sha256',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            typed_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            typed_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def field_tags():
    tags = {'logits', 'last_hidden', 'token_ids'}
    tags.update(f'{field}.{layer}' for field, count in COUNTS.items()
                for layer in range(count))
    tags.update(f'cache_rows.{layer}.{field}' for layer in range(78)
                for field in CACHE_FIELDS)
    return tags


def reference_value(trace, tag):
    if tag == 'last_hidden':
        return trace['hidden_states'][-1]
    if tag == 'token_ids':
        return trace['logits'].argmax(-1)
    if tag == 'logits':
        return trace['logits']
    parts = tag.split('.')
    value = trace[parts[0]][int(parts[1])]
    return value[parts[2]] if len(parts) == 3 else value


def validate_header(header, expected_header, expected):
    require(len(expected) == WIDTH, 'Expected exactly 32 HF reference rows')
    require(type(header) is dict and set(header) == HEADER_KEYS,
            'Complete frame header is required')
    require(type(header['width']) is int and header['width'] == WIDTH
            and type(header['layer_count']) is int
            and header['layer_count'] == 78
            and type(header['fp32_head']) is bool,
            'Header width, layers and head mode must have exact types')
    require(typed_equal(header, expected_header),
            'Typed frame identity differs')
    inputs = header['inputs']
    require(type(inputs) is dict and set(inputs) == {
        'tokens', 'positions', 'committed_positions',
    }, 'Complete input identity is required')
    for name, values in inputs.items():
        require(type(values) is list and len(values) == WIDTH
                and all(type(value) is int for value in values),
                'Input identity must contain 32 integer rows: ' + name)
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':'))
    require(hashlib.sha256(encoded.encode()).hexdigest()
            == header['input_sha256'], 'Input digest differs')
    for row, trace in enumerate(expected):
        for key, field in (('token', 'tokens'), ('position', 'positions'),
                           ('committed_position', 'committed_positions')):
            require(type(trace[key]) is int
                    and trace[key] == inputs[field][row],
                    'HF reference row identity differs')
        for field, count in (*COUNTS.items(), ('cache_rows', 78)):
            require(len(trace[field]) == count, 'Incomplete HF reference')


def decode_frame(package, expected_header, expected):
    """Reject incomplete references and return exactly 32 owned rows."""
    require(type(package) is dict and set(package) == {'header', 'frames'},
            'Frame package fields differ')
    header = package['header']
    validate_header(header, expected_header, expected)
    require(type(package['frames']) is list and len(package['frames']) == 628,
            'Expected exactly 628 fields')
    frames = {}
    for frame in package['frames']:
        require(type(frame) is dict and set(frame) == FRAME_KEYS,
                'Frame field schema differs')
        tag = frame['tag']
        require(type(tag) is str and tag not in frames,
                'Duplicate or invalid field tag')
        require(typed_equal(frame['row_ids'], list(range(WIDTH)))
                and typed_equal(frame['replay_id'], header['replay_id'])
                and typed_equal(frame['input_sha256'], header['input_sha256']),
                'Typed row or replay identity differs')
        lengths = frame['lengths']
        require(lengths is None or (
            type(lengths) is list and len(lengths) == WIDTH
            and all(type(length) is int and length > 0 for length in lengths)
        ), 'Index lengths must be positive integers')
        frames[tag] = frame
    require(set(frames) == field_tags(), 'Complete field coverage is required')
    decoded = []
    for row, reference in enumerate(expected):
        inputs = header['inputs']
        trace = {'token': inputs['tokens'][row],
                 'position': inputs['positions'][row],
                 'committed_position': inputs['committed_positions'][row]}
        trace.update({field: [None] * count
                      for field, count in COUNTS.items()})
        trace['cache_rows'] = [{} for _ in range(78)]
        for tag, frame in frames.items():
            target = reference_value(reference, tag)
            tensor = frame['tensor']
            if tensor is None:
                require(target is None and frame['lengths'] is None,
                        'Unexpected None field: ' + tag)
                value = None
            else:
                require(target is not None
                        and isinstance(tensor, torch.Tensor)
                        and tensor.device.type == 'cpu' and tensor.ndim > 0
                        and tensor.shape[0] == WIDTH,
                        'Expected a complete CPU tensor: ' + tag)
                if tag.startswith('indices.'):
                    lengths = [reference_value(item, tag).numel()
                               for item in expected]
                    require(frame['lengths'] == lengths
                            and tensor.shape == (WIDTH, max(lengths)),
                            'Index lengths or shape differ')
                    length = lengths[row]
                    require(bool((tensor[row, length:] == -1).all()),
                            'Index padding differs')
                    value = tensor[row, :length].reshape(target.shape)
                else:
                    require(frame['lengths'] is None,
                            'Unexpected ragged field')
                    value = tensor[row]
                    if not tag.startswith(('cache_rows.', 'routes.')):
                        value = value.unsqueeze(0)
                    require(value.shape == target.shape,
                            'Tensor row shape differs: ' + tag)
                value = value.detach().clone()
            if tag in ('logits', 'last_hidden', 'token_ids'):
                trace[tag] = value
            else:
                parts = tag.split('.')
                if len(parts) == 3:
                    trace[parts[0]][int(parts[1])][parts[2]] = value
                else:
                    trace[parts[0]][int(parts[1])] = value
        decoded.append(trace)
    return decoded
