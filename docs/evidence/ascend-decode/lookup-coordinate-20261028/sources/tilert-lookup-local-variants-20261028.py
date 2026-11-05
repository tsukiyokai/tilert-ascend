"""Two local-node ablations of the current exact radix lookup."""

import importlib.util
from pathlib import Path


SOURCE = Path("/Users/shanshan/repo/tile-ai/TileRT-Ascend/"
              "tilert/models/glm_5_2_ascend/lookup_draft.py")
SPEC = importlib.util.spec_from_file_location("radix_baseline", SOURCE)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


class CoordinateNode:
    __slots__ = ("hi", "length", "children", "latest")

    def __init__(self, hi, length, latest):
        self.hi = hi
        self.length = length
        self.children = {}
        self.latest = latest


class LocalCoordinateLookup(BASELINE.LookupDrafter):
    """Retain node-local transitions; labels refer to append-only context."""

    def __init__(self, prompt, longest=32, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= longest")
        self.context = []
        self.longest = longest
        self.shortest = shortest
        self.hits = 0
        self.rounds = 0
        self._root = CoordinateNode(0, 0, -1)
        self.commit(prompt)

    def _shared(self, child, query_hi, remaining):
        length = min(child.length, remaining)
        hi, context = child.hi, self.context
        if (context[hi - length + 1:hi + 1]
                == context[query_hi - length + 1:query_hi + 1]):
            return length
        for offset in range(length):
            if context[hi - offset] != context[query_hi - offset]:
                return offset
        raise AssertionError("Unequal edge intervals have no differing token")

    def _insert(self, end):
        window_length = min(self.longest, end)
        node, depth, context = self._root, 0, self.context
        while depth < window_length:
            query_hi = end - 1 - depth
            token = context[query_hi]
            child = node.children.get(token)
            if child is None:
                node.children[token] = CoordinateNode(
                    query_hi, window_length - depth, end,
                )
                return
            shared = self._shared(child, query_hi, window_length - depth)
            if shared == child.length:
                child.latest = end
                depth += shared
                node = child
                continue
            replacement = CoordinateNode(child.hi, shared, end)
            child.hi -= shared
            child.length -= shared
            replacement.children[context[child.hi]] = child
            node.children[token] = replacement
            depth += shared
            if depth < window_length:
                query_hi = end - 1 - depth
                replacement.children[context[query_hi]] = CoordinateNode(
                    query_hi, window_length - depth, end,
                )
            return

    def commit(self, tokens):
        previous = len(self.context)
        self.context.extend(int(token) for token in tokens)
        for end in range(max(previous, self.shortest), len(self.context)):
            self._insert(end)

    def propose(self, count):
        self.rounds += 1
        maximum = min(self.longest, len(self.context) - 1)
        if maximum < self.shortest or count <= 0:
            return []
        node, matched, end = self._root, 0, -1
        context = self.context
        while matched < maximum:
            query_hi = len(context) - 1 - matched
            child = node.children.get(context[query_hi])
            if child is None:
                break
            shared = self._shared(child, query_hi, maximum - matched)
            matched += shared
            end = child.latest
            if shared < child.length:
                break
            node = child
        if matched < self.shortest:
            return []
        proposed = context[end:end + count]
        self.hits += bool(proposed)
        return proposed


class LazyNode:
    __slots__ = ("label", "children", "latest")

    def __init__(self, label, latest):
        self.label = label
        self.children = None
        self.latest = latest


class LazyChildrenLookup(BASELINE.LookupDrafter):
    """Retain tuple labels; allocate a child table only for branch nodes."""

    def _insert(self, word, end):
        node, offset = self._root, 0
        while offset < len(word):
            if node.children is None:
                node.children = {}
            child = node.children.get(word[offset])
            if child is None:
                node.children[word[offset]] = LazyNode(word[offset:], end)
                return
            shared = BASELINE._common_prefix(child.label, word, offset)
            if shared == len(child.label):
                child.latest = end
                offset += shared
                node = child
                continue
            replacement = LazyNode(child.label[:shared], end)
            child.label = child.label[shared:]
            replacement.children = {child.label[0]: child}
            node.children[word[offset]] = replacement
            offset += shared
            if offset < len(word):
                replacement.children[word[offset]] = LazyNode(word[offset:], end)
            return

    def propose(self, count):
        self.rounds += 1
        maximum = min(self.longest, len(self.context) - 1)
        if maximum < self.shortest or count <= 0:
            return []
        word = tuple(reversed(self.context[-maximum:]))
        node, matched, end = self._root, 0, -1
        while matched < maximum and node.children is not None:
            child = node.children.get(word[matched])
            if child is None:
                break
            shared = BASELINE._common_prefix(child.label, word, matched)
            matched += shared
            end = child.latest
            if shared < len(child.label):
                break
            node = child
        if matched < self.shortest:
            return []
        proposed = self.context[end:end + count]
        self.hits += bool(proposed)
        return proposed
