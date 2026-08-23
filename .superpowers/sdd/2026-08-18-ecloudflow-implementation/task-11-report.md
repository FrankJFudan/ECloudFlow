# Task 11 Report: Composite Scientific Losses and Lightning Training

## Status

Implemented and verified on branch `feature/ecloudflow-implementation` from base
`ee7a2c0`. The implementation provides frozen nested loss configuration, typed
training targets and diagnostics, six-component composite losses, DDP-synchronized
running RMS normalization, checkpointable EMA, and a Lightning 2.5.2 training
module with stable `field_tokenizer`, `field_decoder`, and `joint_backbone` groups.

## TDD evidence

The initial RED run was:

```text
conda run -n 3dmolecule python -m pytest tests/unit/training tests/integration/test_training_step.py -q
3 collection errors: ecloudflow.training and LossConfig did not exist.
```

After adding only importable RED placeholders, substantive tests independently
reached the intended boundaries:

```text
compute_ecloudflow_loss -> NotImplementedError: composite loss is not implemented
ExponentialMovingAverage -> NotImplementedError: EMA behavior is not implemented
ECloudFlowTrainingModule -> NotImplementedError: Lightning training is not implemented
3 failed
```

Further focused RED-GREEN cycles caught:

- missing first-order endpoint/affinity-variance prediction fields;
- a running-scaler advanced-index mutation bug (`4.0` observed versus the
  hand-derived decayed mean square `10.0`);
- Lightning 2.5.2 default transfer rejecting frozen nested dataclasses;
- BF16 `index_add_` source/accumulator dtype disagreement;
- malformed decoder reconstruction dtype accepted by `ModelPrediction`;
- implicit QM density target broadcasting;
- zero-weight scientific terms incorrectly requiring unused context;
- Hydra loss keys requiring `+` because no composed loss group existed.

Each was reproduced by a focused failing behavioral test before its production fix.

## Typed contracts and equations

`LossBreakdown.raw`, `normalized`, and `weighted` each contain exactly:
`flow`, `score`, `discrete`, `ecloud`, `chem`, and `interaction`. Fixed fragment
subterms are represented by typed `LossDiagnostics.flow_fixed` and
`score_fixed`, both exact zero contributions, rather than a seventh mapping key.

For editable scalar entries `M`, the continuous objectives are:

```text
L_flow  = w_pos mean_M ||v_pos - v*_pos||^2
        + w_ele mean_M ||v_ele - v*_ele||^2
L_score = w_pos mean_M ||s_pos - s*_pos||^2
        + w_ele mean_M ||s_ele - s*_ele||^2
```

Positions and velocities use angstroms and angstroms per path-time unit; position
scores use inverse angstroms. Packed electron channels preserve the Task 8/10
irrep layout. Atom, charge, sparse canonical bond-halfedge, and count losses are
masked endpoint cross-entropies with independently validated class ranges.

Genuine-QM ecloud terms use only `qm_mask` rows and a real differentiable
`ElectronReconstruction`:

```text
L_ecloud = w_rho MSE(rho, rho*) + w_grad MSE(grad rho, grad rho*)
         + w_N MSE(N_e, N_e*) + w_mu MSE(mu_e, mu_e*)
         + w_cycle MSE(z_round_trip, z*)
```

Density is electrons/angstrom^3, its gradient electrons/angstrom^4, electron
count is electrons, and dipole is electron-angstrom. Approximate/non-QM rows are
never presented as QM. An active term missing its required prediction or label
context raises explicitly; missing availability masks and disabled zero-weight
subterms give differentiable finite zero.

The chemistry component contains:

- expected bond-order valence overflow against typed per-element/charge-derived
  `valence_limits`;
- element/bond-conditioned Gaussian length residuals supplied as sparse
  halfedge `bond_length_mean/std` values in angstroms;
- sparse ligand nonbonded and per-complex ligand-protein clash hinges;
- sparse ring-triplet standardized angle strain;
- differentiable minimum expected-degree connectivity;
- per-example heteroscedastic affinity NLL
  `0.5*(exp(-clamp(log_var))*residual^2 + clamp(log_var))`.

Affinity log variance is clamped to configured typed bounds before exponentiation.
Interaction supervision uses masked binary focal loss
`(1-p_t)^gamma * BCEWithLogits`; `gamma` is typed and bounded.

`ModelPrediction.endpoint_positions` and `endpoint_electron_latent` are explicit
first-order auxiliary estimators `x_t + (1-t)v_t`. They are exact only for a
straight deterministic path. For curved or stochastic paths they are empirical
estimators and are never documented or interpreted as guaranteed clean endpoints.
The optional field reconstruction is populated only through the compatible real
decoder boundary described by `ElectronDecoderContext`.

## Normalization, precision, DDP, and EMA decisions

`RunningLossScaler` stores persistent float32 `mean_square` and boolean
`initialized` buffers. Active rank-level sufficient statistics are detached,
squared, and all-reduced with Gloo/NCCL-compatible tensor collectives when a
process group exists. A globally missing component is not observed and does not
decay. Every rank therefore applies the same update:

```text
q_t = decay*q_(t-1) + (1-decay)*mean_ranks(L_rank^2)
L_normalized = L / sqrt(detach(q_t) + epsilon)
```

The positive scalar division cannot change a component's within-component
gradient direction. A two-process Windows Gloo test starts from local flow
losses 1 and 3 and verifies both ranks store `(1^2+3^2)/2 = 5` exactly.

FP16/BF16 reductions and chemistry accumulations use float32. Decoder outputs
are allowed to remain float32 under reduced-precision model execution. No code
calls `.cuda()`, selects ranks/devices, or creates dense `[N,N,C]` bonds.
Lightning logs all raw/normalized/weighted/total values with `sync_dist=True`.

EMA shadows and `num_updates` are persistent registered buffers. `update`,
`update_after_step`, `store`, `copy_to`, and consuming `restore` have explicit
semantics. The Lightning optimizer hook delegates precision/AMP handling to
Lightning and updates EMA only after the step returns with at least one finite
gradient. Raised, skipped, missing-gradient, and overflow/non-finite paths do not
mutate EMA. State dictionaries and module dtype/device transfers retain shadows.
The EMA parameter-name layout includes frozen groups so later Task 12 stage
freezing does not invalidate resume state.

## Verification

Final focused suite:

```text
conda run -n 3dmolecule python -m pytest tests/unit/training tests/integration/test_training_step.py -v
28 passed, 17 warnings in 10.83s
```

This includes a real CPU Lightning optimizer/EMA update, two-rank local Gloo,
and the available NVIDIA RTX 4060 CUDA BF16 Lightning smoke (not skipped).

Full suite:

```text
conda run -n 3dmolecule python -m pytest -q
308 passed, 1 skipped, 17 warnings in 23.31s
```

Static and documentation gates:

```text
conda run -n 3dmolecule ruff check <Task-11 source/test/docs scope>
All checks passed!

conda run -n 3dmolecule ruff format --check <Task-11 source/test/docs scope>
15 files already formatted

conda run -n 3dmolecule python tools/check_python_docs.py src/ecloudflow
exit 0

conda run -n 3dmolecule python -m mypy --no-site-packages --ignore-missing-imports \
  --follow-imports=normal src/ecloudflow/training src/ecloudflow/config/schema.py \
  src/ecloudflow/models/ecloudflow.py
Success: no issues found in 7 source files

git diff --check
exit 0 (only Git LF-to-CRLF checkout warnings)
```

`--no-site-packages --ignore-missing-imports` was necessary because the installed
third-party `rdkit-stubs` contains a syntax error (`Non-default argument follows
default argument`). `--follow-imports=normal` still checks all local ECloudFlow
imports in scope rather than skipping them.

## Files

- `configs/config.yaml`, `configs/loss/default.yaml`
- `src/ecloudflow/config/__init__.py`, `src/ecloudflow/config/schema.py`
- `src/ecloudflow/models/ecloudflow.py`
- `src/ecloudflow/training/{__init__,types,losses,ema,module}.py`
- `tests/unit/training/test_losses.py`, `tests/unit/training/test_ema.py`
- `tests/integration/test_training_step.py`
- `tests/unit/models/test_ecloudflow.py`, `tests/unit/test_source_docs.py`
- `tools/check_python_docs.py`
- this report

## Verified invariants versus empirical hypotheses

Verified invariants are the algebra, exact mask exclusion, finite empty/missing
behavior, sparse canonical topology checks, no target broadcasting, class-range
validation, DDP scaler equality, scaler detachment and gradient-direction
preservation, persistent state round trips, variance clamping, fail-fast behavior,
EMA store/copy/restore/skip/update semantics, and real Lightning CPU/CUDA updates.

The default component/subterm weights, focal gamma, warm-up choices, clash
thresholds, minimum-degree target, log-variance bounds, chemistry surrogate
usefulness, and any claim of improved binding quality are empirical hypotheses.
They require real-data ablations and calibration; these unit/integration tests do
not establish chemical validity or binding-quality benefit.

## Self-review and concerns

- No fabricated density, geometry, chemistry, or affinity-variance tensor is used.
- Sparse halfedges remain flattened and canonical; protein clash uses per-complex
  pair distances but never a dense ligand bond tensor.
- Frozen dataclass transfer is functional and strategy-device driven for Lightning
  2.5.2; it contains no manual accelerator/rank policy.
- The repository has seven pre-existing Ruff-format differences in Task 1–3 files
  outside this task. They were not modified. The Task 11 scope is fully formatted.
- The one full-suite skip is the existing external xTB integration test. Warnings
  are third-party pyparsing deprecations plus Lightning environment/worker hints.
