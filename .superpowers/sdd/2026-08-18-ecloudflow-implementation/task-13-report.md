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

