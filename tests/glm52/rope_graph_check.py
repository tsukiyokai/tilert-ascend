"""The rope refresh's control flow, against stand-ins, on the host.

`RopeGraph` was measured on a die -- 22.54x and bit-exact -- but the branch in
`refresh_hoisted_rope` that decides whether to capture, reuse or fall back has
never run at all. These checks exercise that branch with a stand-in graph, so a
wrong condition shows here rather than on sixteen ranks.
"""
import sys
import types

ROOT = "/home/shanshan/repo/tile-ai/TileRT-Ascend"
sys.path.insert(0, ROOT)


class Refresher:
    """The engine's rope branch, lifted onto a stand-in.

    Copied rather than imported because importing `cann_model` pulls in torch
    and vllm; the branch is five lines and what is under test is its logic.
    """

    def __init__(self, rope_graph, hoisted, build, eager):
        self.rope_graph = rope_graph
        self._hoisted = hoisted
        self._rope_graphs = {}
        self._build = build
        self._eager = eager

    def refresh_hoisted_rope(self, width, positions):
        hoisted = self._hoisted.get(width)
        if hoisted is None:
            return
        if not self.rope_graph:
            self._eager(positions, width, hoisted[2])
            return
        captured = self._rope_graphs.get(width)
        if captured is None:
            captured = self._build(positions, width, hoisted[2])
            self._rope_graphs[width] = captured
        captured.refresh()


class Stand:
    def __init__(self, *_):
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1


def build_refresher(rope_graph=True, widths=(12,)):
    built = []
    eager = []

    def build(positions, width, destinations):
        built.append(width)
        return Stand()

    hoisted = {width: (None, None, [("cos", "sin")]) for width in widths}
    refresher = Refresher(rope_graph, hoisted, build,
                          lambda *arguments: eager.append(arguments))
    return refresher, built, eager


def check_a_width_is_captured_once():
    refresher, built, _ = build_refresher()
    for _ in range(5):
        refresher.refresh_hoisted_rope(12, "positions")
    assert built == [12], built
    assert refresher._rope_graphs[12].refreshes == 5


def check_each_width_gets_its_own_graph():
    refresher, built, _ = build_refresher(widths=(8, 12))
    refresher.refresh_hoisted_rope(8, "p")
    refresher.refresh_hoisted_rope(12, "p")
    refresher.refresh_hoisted_rope(8, "p")
    assert built == [8, 12], built
    assert refresher._rope_graphs[8].refreshes == 2


def check_the_flag_off_runs_eager():
    refresher, built, eager = build_refresher(rope_graph=False)
    refresher.refresh_hoisted_rope(12, "positions")
    assert built == []
    assert len(eager) == 1, eager


def check_a_width_without_hoisted_metadata_does_nothing():
    refresher, built, eager = build_refresher(widths=(12,))
    refresher.refresh_hoisted_rope(24, "positions")
    assert built == [] and eager == []




def check_every_flag_is_registered_before_the_parse():
    """A flag added below `parse_args` never registers, and argparse is silent.

    Four were, in this session: --rope-graph, --round-phases,
    --ai-cpu-collectives and --narrow-bank, all rejected as unrecognized at
    run time while present in the file. The line numbers answer it without
    running anything.
    """
    import ast

    source = open(f"{ROOT}/tests/glm52/bench_cann_mtp.py").read()
    tree = ast.parse(source)
    parse_line = min(
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse_args")
    late = [
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.lineno > parse_line and node.args
        and isinstance(node.args[0], ast.Constant)]
    assert not late, late

def main():
    checks = [value for name, value in sorted(globals().items())
              if name.startswith("check_")]
    for check in checks:
        check()
        print(f"  ok  {check.__name__}")
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
