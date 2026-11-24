"""Exhaustively check finite interleavings of the communication protocol.

This verifies ordering and two-plane reuse in a discrete transition model.
It does not execute Ascend instructions or establish device visibility.
"""

from collections import deque
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Progress:
    epoch: int = 1
    phase: int = 0  # 0 payload, 1 ready, 2 read, 3 local completion barrier
    done: int = 0


class ProtocolViolation(RuntimeError):
    pass


def stored_epoch(progress, sender, receiver, plane, ranks, planes, ready):
    """Latest logical epoch in one sender-owned slot, including prior rounds."""
    epoch = progress.epoch - 1
    while epoch % planes != plane:
        epoch -= 1
    if progress.epoch % planes != plane:
        return epoch
    if ready:
        offset = (receiver - sender) % ranks
        published = (progress.phase > 1
                     or (progress.phase == 1 and progress.done > offset))
    else:
        published = (progress.phase > 0
                     or bool(progress.done & (1 << receiver)))
    return progress.epoch if published else epoch


def consumed(progress, sender, epoch):
    if epoch <= 0 or progress.epoch > epoch:
        return True
    return (progress.epoch == epoch
            and (progress.phase == 3
                 or (progress.phase == 2 and progress.done > sender)))


def successors(state, ranks, rounds, payload_planes, ready_planes):
    for rank, progress in enumerate(state):
        epoch, phase, done = progress.epoch, progress.phase, progress.done
        if epoch > rounds:
            continue
        candidates = []
        if phase == 0:
            # AIV payload writers may finish their peers in any order.
            for peer in range(ranks):
                if done & (1 << peer):
                    continue
                old = stored_epoch(progress, rank, peer, epoch % payload_planes,
                                   ranks, payload_planes, ready=False)
                if not consumed(state[peer], rank, old):
                    raise ProtocolViolation("payload overwritten before its reader finished")
                mask = done | (1 << peer)
                candidates.append(Progress(epoch, 1, 0) if mask == (1 << ranks) - 1
                                  else Progress(epoch, 0, mask))
        elif phase == 1:
            # One AIV publishes all ready words after every local writer is quiet.
            peer = (rank + done) % ranks
            old = stored_epoch(progress, rank, peer, epoch % ready_planes,
                               ranks, ready_planes, ready=True)
            if not consumed(state[peer], rank, old):
                raise ProtocolViolation("ready overwritten before its reader observed it")
            candidates.append(Progress(epoch, 2, 0) if done + 1 == ranks
                              else Progress(epoch, 1, done + 1))
        elif phase == 2:
            sender = done
            ready = stored_epoch(state[sender], sender, rank, epoch % ready_planes,
                                 ranks, ready_planes, ready=True)
            if ready != epoch:
                continue
            payload = stored_epoch(state[sender], sender, rank,
                                   epoch % payload_planes, ranks, payload_planes,
                                   ready=False)
            if payload != epoch:
                raise ProtocolViolation("ready exposed another epoch's payload")
            candidates.append(Progress(epoch, 3, 0) if done + 1 == ranks
                              else Progress(epoch, 2, done + 1))
        else:
            candidates.append(Progress(epoch + 1, 0, 0))
        for candidate in candidates:
            following = list(state)
            following[rank] = candidate
            yield tuple(following)


def explore(ranks, rounds, payload_planes=2, ready_planes=2):
    initial = tuple(Progress() for _ in range(ranks))
    seen = {initial}
    queue = deque([initial])
    maximum_skew = 0
    while queue:
        state = queue.popleft()
        epochs = [progress.epoch for progress in state]
        maximum_skew = max(maximum_skew, max(epochs) - min(epochs))
        following = list(successors(state, ranks, rounds,
                                    payload_planes, ready_planes))
        if not following and any(progress.epoch <= rounds for progress in state):
            raise ProtocolViolation("deadlock before all collective rounds complete")
        for candidate in following:
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    return {"ranks": ranks, "rounds": rounds, "states": len(seen),
            "maximum_epoch_skew": maximum_skew}


def main():
    results = [explore(2, 6), explore(3, 4)]
    negative_controls = []
    for payload_planes, ready_planes in ((1, 2), (2, 1)):
        try:
            explore(2, 4, payload_planes, ready_planes)
        except ProtocolViolation as error:
            negative_controls.append(str(error))
        else:
            raise AssertionError("single-plane negative control was not rejected")
    print(json.dumps({"scope": "finite protocol interleavings only",
                      "passed": results,
                      "rejected_single_plane_controls": negative_controls}, indent=2))


if __name__ == "__main__":
    main()
