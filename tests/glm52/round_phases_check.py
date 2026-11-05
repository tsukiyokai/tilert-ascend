"""Checks for RoundPhases, on the host."""
import sys
import time

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")
from tilert.models.glm_5_2_ascend.round_phases import RoundPhases


def check_a_step_is_charged_once():
    phases = RoundPhases()
    phases.begin()
    phases.mark("one")
    assert phases.counts["one"] == 1


def check_steps_average_over_rounds():
    phases = RoundPhases()
    for _ in range(4):
        phases.begin()
        phases.mark("one")
    report = phases.report()
    assert report["rounds"] == 4
    assert report["one"] >= 0


def check_an_unstarted_report_is_empty():
    assert RoundPhases().report() == {}


def check_time_lands_on_the_step_that_spent_it():
    phases = RoundPhases()
    phases.begin()
    time.sleep(0.01)
    phases.mark("slow")
    phases.mark("fast")
    report = phases.report()
    assert report["slow"] > 5.0, report
    assert report["fast"] < 1.0, report

def main():
    checks = [value for name, value in sorted(globals().items())
              if name.startswith("check_")]
    for check in checks:
        check()
        print(f"  ok  {check.__name__}")
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
