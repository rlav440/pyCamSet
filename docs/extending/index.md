# What is extensible

This library was written to enable exploration of calibration targets and
calibration methodologies. A calibration rests on the interaction of three
things — the target, the parameter handler, and the bundle adjustment loss —
and each is meant to be replaced.

Derivatives of these classes substitute into the framework directly, which is
what allows the calibration target, the calibration method, and the type of
optimisation to be customised independently of each other. [Architecture](../architecture.md)
describes how they fit together; these pages are how to replace one.

---

| | |
|---|---|
| [Calibration targets](targets.md) | Where the features are, and how to find them in an image |
| [Parameter handlers](parameters.md) | What the optimisation is free to move, and what it means |
| [Bundle adjustment](bundle-adjustment.md) | What the loss actually measures |

---

## The registry

A target class is reached by name, through
`pyCamSet.calibration_targets.core.target_registry`:

```python exec="true" result="text"
from pyCamSet.calibration_targets.core.target_registry import TARGET_CLASSES

for name, (module, cls) in TARGET_CLASSES.items():
    print(f"{name:<18} {module}.{cls}")
```

Entries are held as module and class *names* rather than as classes, so that
importing the registry costs nothing and so that a target with an optional
dependency is only imported when one is actually asked for. A new target is one
line in that table and nothing else.
