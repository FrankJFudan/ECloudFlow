# Task 13 Verification Report

## RED

Before the sampling package was added, the focused command from the brief could
not collect sampling tests because `ecloudflow.sampling` and its solver/prior
modules did not exist (`ModuleNotFoundError`). This was the expected failing
baseline for the task.

## GREEN

Command:

```text
conda run -n 3dmolecule python -m pytest tests/unit/sampling tests/integration/test_fragment_invariance.py -q
```

Result: `4 passed`.

Coverage includes cavity support and simplex normalization, Euler/Heun order
and NFE accounting, deterministic caller-generator score correction, finite
diagnostics, and bitwise fixed-fragment equality in every saved trajectory
frame.

## Round 1 Review Fixes

### RED

Independent review identified three missing regressions: an impossible cavity
could return unsupported points after rejection exhaustion; score-corrector
hooks accepted only three-argument callbacks and silently skipped valid
two-argument callbacks; and ``steps`` overrides accepted negative, boolean, or
non-integer values.

### GREEN

The focused suite now reports `7 passed`, including explicit bounded-rejection
failure, 3/2/1-argument hook dispatch, and strict step-override validation.

## Round 2 Review Fixes

### RED

Re-review found that the corrector recorded an unconstrained input state as
``frames[0]`` and that a zero score caused an unbounded inverse-norm Langevin
step, risking non-finite coordinates.

### GREEN

The focused suite now reports `9 passed`. Initial corrector states pass through
the canonical hook chain before recording, zero/near-zero score updates remain
finite, and all prior hook/step regressions remain covered.
