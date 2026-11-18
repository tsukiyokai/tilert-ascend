from time import perf_counter_ns


class RoundPhases:

    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = {}
        self.counts = {}
        self.rounds = 0
        self._last = None

    def begin(self):
        self.rounds += 1
        self._last = perf_counter_ns()

    def mark(self, step):
        now = perf_counter_ns()
        if self._last is not None:
            self.totals[step] = self.totals.get(step, 0) + now - self._last
            self.counts[step] = self.counts.get(step, 0) + 1
        self._last = now

    def report(self):
        if not self.rounds:
            return {}
        held = {step: nanoseconds / self.rounds / 1e6
                for step, nanoseconds in self.totals.items()}
        held["marks_a_round"] = {step: count / self.rounds
                                 for step, count in self.counts.items()}
        held["rounds"] = self.rounds
        return held
