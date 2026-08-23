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

## Fix round 1/5 — active selection, collective fail-fast, and decoder geometry

Review base: `b6faa96`.

### RED evidence

Each behavior below was exercised independently rather than being hidden by a
missing-module collection failure:

```text
conda run -n 3dmolecule python -m pytest \
  tests/unit/training/test_losses.py::test_excluded_nonfinite_and_class_sentinel_rows_are_never_evaluated \
  tests/unit/training/test_losses.py::test_all_false_masks_ignore_nonfinite_placeholders_with_exact_zero_gradient -q
2 failed
```

The first failure was the pre-mask global class-range check; the second reached a
non-finite total because an empty reduction used `values.sum() * 0`. An accidental
initial invocation with the base interpreter failed to import Torch and is not
claimed as scientific RED evidence; every reported RED/GREEN run used the pinned
`3dmolecule` environment.

```text
conda run -n 3dmolecule python -m pytest \
  tests/unit/training/test_losses.py::test_nonfinite_on_one_gloo_rank_raises_consistently_without_deadlock -q
1 failed in 15.23s
rank 0: Gloo recv timed out after 5000 ms while rank 1 had raised locally
```

The bounded process-group timeout demonstrated the actual stranded-rank failure
without leaving Windows spawn children hung.

```text
conda run -n 3dmolecule python -m pytest \
  tests/integration/test_training_step.py::test_decoder_is_not_required_without_enabled_genuine_qm_supervision \
  tests/integration/test_training_step.py::test_real_decoder_uses_predicted_centers_and_all_qm_terms_are_differentiable \
  tests/integration/test_training_step.py::test_decoder_mapping_rejects_duplicates -q
3 failed: ElectronDecoderContext still required external centers
```

```text
conda run -n 3dmolecule python -m pytest \
  tests/unit/training/test_losses.py::test_sparse_scientific_topology_is_validated_without_dense_allocation \
  tests/unit/training/test_losses.py::test_active_bond_prior_requires_positive_finite_stddev \
  tests/unit/training/test_losses.py::test_diagnostics_count_actual_enabled_observations_per_subterm \
  tests/unit/training/test_losses.py::test_disabled_component_does_not_initialize_or_decay_running_scale -q
7 failed
```

Those failures independently exposed bonded/nonbonded overlap, duplicate pairs,
repeated ring nodes, invalid active ring/bond standard deviations, missing exact
counts, and broad-mask scaler presence. Further focused RED tests failed because
`latent_cycle_mask` did not exist and because a zero-weight ecloud component still
required a reconstruction prediction. A final IEEE-zeroing RED showed that an
inactive non-finite raw component multiplied by numerical zero still contaminated
the total; weighted components now branch to a differentiable zero when their
typed weight/warm-up factor is zero. An explicit-context RED also confirmed that
active QM `forward` previously deferred a missing context; it now raises at the
module boundary, while genuinely inactive paths still require no context.

### GREEN behavior and contracts

Masking now expands only boolean prefix metadata, selects active tensor entries,
and performs CE/MSE/NLL/focal/geometry arithmetic only on the selection. Empty
selection uses an empty view sum, yielding finite differentiable zero even when
excluded backing storage is NaN/Inf. Masked class sentinels are not range checked;
selected classes are. Fixed, missing, non-QM, and padded cycle entries have exact
zero loss and gradient. `TrainingTargets.latent_cycle_mask [B,Nmax]` makes missing
cycle-token supervision explicit.

Each finite stage first creates fixed `[6,2]` detached presence/nonfinite
sufficient statistics and performs the same all-reduce on every initialized
rank. All ranks then raise the same ordered component diagnostic, or all proceed
to scaler square/count reductions in fixed order. Locally absent but globally
present supervision contributes no zero observation. Component presence is the
sum of enabled observed subterms and additionally respects component weight and
explicit-step warm-up; disabled/empty terms neither initialize nor decay RMS.

`ElectronDecoderContext` now contains only `query_grid`, `atom_mask`, and
`flat_index`. Both centers and electron latents are gathered from
`ModelPrediction.endpoint_positions` and `endpoint_electron_latent` with the same
mapping. Physical mappings must be complete, unique, in range, and match
`TrainingTargets.node_batch`; duplicate, cross-complex, and incomplete mappings
raise. Only genuine-QM rows needed by enabled, warmed-up terms enter the real
decoder, and results scatter back as differentiable zeros for inactive rows.
Thus density, gradient, count, dipole, and cycle losses reach predicted centers,
while an inactive row may safely carry non-finite placeholders.

Sparse validation remains `O(N+E+P+R)`: canonical pairs are encoded as integer
keys for uniqueness/disjointness, without a dense adjacency. Nonbonded pairs are
same-complex and disjoint from bonds; ring triplets use three distinct in-range
same-complex nodes and positive finite active priors. Optional scientific values
use exact shape/dtype/device checks and active-only finiteness/range checks.
Diagnostics now report actual enabled node, edge, pair, triplet, field-point,
cycle-token, and labeled-example counts.

The real Lightning tests use `ElectronFieldDecoder`, not a mock of scientific
behavior. They exercise all five reconstruction terms, a mixed-QM batch with
NaN/Inf inactive labels/query points, nonzero predicted-center gradient on the
QM complex, exact zero gradient on the non-QM complex, and an optimizer update.
On the available RTX 4060, Lightning's real CUDA FP16 precision plugin/GradScaler
skips a deliberately overflowing step; the parameter and EMA update counter stay
unchanged. EMA/scaler buffers remain normal persistent `state_dict` entries; the
Task 12-owned interrupted/resumed Trainer scenario was intentionally not copied.

These are verified algebraic, mask, topology, collective-order, precision-plugin,
and state-mutation invariants. We still make no empirical conclusion about loss
weights, focal gamma, warm-ups, surrogate chemical utility, or binding quality;
those remain real-data ablation hypotheses.

### Fix-round verification

```text
conda run -n 3dmolecule python -m pytest \
  tests/unit/training tests/integration/test_training_step.py tests/unit/models/test_ecloudflow.py -q
88 passed, 20 warnings in 24.33s

conda run -n 3dmolecule python -m pytest -q
331 passed, 1 skipped, 20 warnings in 34.87s

conda run -n 3dmolecule ruff check <fix-round source/test scope>
All checks passed!

conda run -n 3dmolecule ruff format --check <fix-round source/test scope>
all files formatted

conda run -n 3dmolecule python tools/check_python_docs.py src/ecloudflow
exit 0

conda run -n 3dmolecule python -m mypy --no-site-packages \
  --ignore-missing-imports --follow-imports=normal \
  src/ecloudflow/training src/ecloudflow/config/schema.py \
  src/ecloudflow/models/ecloudflow.py
Success: no issues found in 7 source files

git diff --check
exit 0 (only Git LF-to-CRLF checkout warnings)
```

Warnings are the prior third-party pyparsing deprecations and Lightning GPU/
dataloader-worker hints. The single skip remains the external xTB integration.
