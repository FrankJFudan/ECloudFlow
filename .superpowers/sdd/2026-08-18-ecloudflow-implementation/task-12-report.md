# Task 12 TDD and verification report

## RED evidence

- Command: `D:\Anaconda3\envs\3dmolecule\python.exe -m pytest tests/unit/training/test_stages.py tests/integration/test_checkpoint_resume.py -v`
- Result: expected collection failure, exit code 2, 0 tests collected.
- Exact missing production boundaries: `ModuleNotFoundError: No module named 'ecloudflow.training.stages'` and `ModuleNotFoundError: No module named 'ecloudflow.training.callbacks'`.
- Interpretation: the tests reached the intended imports and failed because Task 12 APIs did not exist, before any production implementation was added.

## GREEN and final verification

Implemented and committed as `b01b74d` (`feat: add staged resumable distributed training`).

The implementation provides explicit four-stage parameter/loss policies,
strict `TrainerConfig`, local and production Hydra experiment presets,
rank-local resumable WebDataset cursors, complete Python/NumPy/CPU/CUDA RNG
capture and restoration, semantic configuration compatibility checks, manifest
and preprocessing identity validation, Lightning EMA/loss-scaler checkpoint
contracts, bounded nonfinite diagnostics, and collective rank-zero atomic JSON
publication. New APIs have detailed English Sphinx docstrings. The experiment
presets use Hydra override defaults so they compose cleanly over the root model
and data groups.

Focused verification:

```text
conda run -n 3dmolecule python -m pytest \
  tests/unit/training/test_stages.py tests/integration/test_checkpoint_resume.py -v
11 passed

conda run -n 3dmolecule python -m pytest \
  tests/unit/training tests/integration/test_checkpoint_resume.py -q
61 passed
```

Full verification:

```text
conda run -n 3dmolecule python -m pytest -q
350 passed, 2 skipped

conda run -n 3dmolecule ruff check <Task-12 source/test scope>
All checks passed!

conda run -n 3dmolecule ruff format --check <Task-12 source/test scope>
7 files already formatted

conda run -n 3dmolecule python tools/check_python_docs.py src/ecloudflow
exit 0

conda run -n 3dmolecule python -m mypy --no-site-packages \
  --ignore-missing-imports --follow-imports=normal \
  src/ecloudflow/training src/ecloudflow/config/schema.py
Success: no issues found in 9 source files

git diff --check
exit 0 (only Git LF-to-CRLF checkout warnings)
```

The prescribed Windows `torch.distributed.run --standalone --nproc_per_node=2`
smoke launch was attempted. This environment's PyTorch Windows build lacks
libuv; the launcher requests libuv for its TCP rendezvous and fails before test
workers start (`DistStoreError: use_libuv was requested but PyTorch was built
without libuv support`). The implementation itself retains Gloo-safe fixed
collective ordering and the distributed smoke test remains available for a
libuv-capable/Linux or server environment. The NCCL/H100 test is intentionally
marked server-only.

## Independent review and fix round 1

The first independent review identified three important correctness gaps:

1. CPU-only checkpoint capture could call CUDA RNG APIs unconditionally.
2. A non-finite diagnostic raised after the checkpoint callback could leave the
   consumed-batch cursor advanced.
3. Injected Git revisions were checked only by length, not hexadecimal syntax.

Fix round 1 adds an explicit CUDA availability guard, skips cursor advancement
for non-finite callback outputs, validates `[0-9a-fA-F]{40}` revisions, and adds
regressions for all three cases. Focused verification after the fixes:

```text
python -m pytest tests/integration/test_checkpoint_resume.py \
  tests/unit/training/test_stages.py -q
14 passed, 19 warnings
ruff check src/ecloudflow/training/checkpoint.py \
  tests/integration/test_checkpoint_resume.py
All checks passed!
ruff format --check src/ecloudflow/training/checkpoint.py \
  tests/integration/test_checkpoint_resume.py
All files formatted
```

The review itself could not execute tests in its isolated environment because
PyTorch was unavailable; controller verification above is the authoritative
GREEN evidence for this fix round.
