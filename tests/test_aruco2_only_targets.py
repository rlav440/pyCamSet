"""
That ``ARUCO2_ONLY_TARGETS`` names exactly the targets that need aruco2.

Five tests parametrize over every registered target, and each has to skip
the ones that cannot be built without the optional aruco2 backend.  That
was five copies of a hand-written tuple of names.  Adding CIco2 updated
two of them; the other three went red on CI -- where aruco2 is not
installed -- and stayed green on every developer machine that had it.

So the list is in one place now, and this checks it against what the
constructors actually do rather than against what someone remembered.  It
runs in a subprocess with ``aruco2`` blocked at import, which is the state
CI is in, so it gives the same answer whether or not aruco2 is installed
here.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import ARUCO2_ONLY_TARGETS, REPO_ROOT
from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES


# Runs in the child: block aruco2, then try to build every registered
# target from its own declared defaults and report which ones refuse.
_PROBE = r'''
import json, sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "aruco2" or name.startswith("aruco2."):
            raise ImportError("No module named 'aruco2'")
        return None

sys.meta_path.insert(0, _Block())

from pyCamSet.calibration_targets.core.target_registry import (
    TARGET_NAMES, target_class, build_target)

needs, failed = [], {}
for name in TARGET_NAMES:
    try:
        spec = {"type": name, **target_class(name).construction_parameters().defaults()}
        build_target(spec)
    except ImportError:
        needs.append(name)
    except Exception as exc:                       # a real fault, not a missing backend
        failed[name] = f"{type(exc).__name__}: {exc}"

print("RESULT " + json.dumps({"needs": needs, "failed": failed}))
'''


@pytest.fixture(scope="module")
def _built_without_aruco2() -> dict:
    """Build every registered target in a child that cannot import aruco2."""
    done = subprocess.run([sys.executable, "-c", _PROBE],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        pytest.fail("the probe did not finish:\n" + done.stdout + done.stderr)
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        pytest.fail("the probe printed no result:\n" + done.stdout + done.stderr)
    return json.loads(line[-1][len("RESULT "):])


def test_the_list_names_exactly_the_targets_that_need_aruco2(_built_without_aruco2):
    """
    The point of the file.

    A target added on the wrong side of this list fails here, on any
    machine, instead of only on CI.
    """
    observed = set(_built_without_aruco2["needs"])
    assert observed == set(ARUCO2_ONLY_TARGETS), (
        "ARUCO2_ONLY_TARGETS disagrees with the constructors: "
        f"only listed {sorted(set(ARUCO2_ONLY_TARGETS) - observed)}, "
        f"only observed {sorted(observed - set(ARUCO2_ONLY_TARGETS))}")


def test_every_other_target_builds_without_aruco2(_built_without_aruco2):
    """
    Nothing else quietly depends on aruco2 being present.

    Kept apart from the test above so a target that breaks for some reason
    of its own does not read as a backend problem.
    """
    assert not _built_without_aruco2["failed"], (
        "these targets failed to build for reasons other than a missing "
        f"aruco2: {_built_without_aruco2['failed']}")


def test_the_list_only_names_registered_targets():
    """A rename that misses the list would otherwise skip nothing, silently."""
    assert set(ARUCO2_ONLY_TARGETS) <= set(TARGET_NAMES), (
        f"not registered targets: {sorted(set(ARUCO2_ONLY_TARGETS) - set(TARGET_NAMES))}")
