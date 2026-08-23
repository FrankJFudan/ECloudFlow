# Task 10 Report: SE(3)-Equivariant Pocket Encoder and Joint Ligand Backbone

## Outcome

Implemented the first public joint pocket-conditioned ligand model used by
Tasks 11--13. `ECloudFlowModel` is a device-agnostic `torch.nn.Module` with no
implicit transfers or rank-local global state. It consumes the canonical
flattened `MolecularState`/`GenerationCondition` contracts and returns typed
`ModelPrediction` tensors for position and packed-electron flow/score,
categorical atom/charge/bond endpoints, atom count, affinity, and interaction.

## Architecture and contract decisions

- `ModelConfig.vector_dim` remains the pocket/backbone vector multiplicity.
  The packed Task 8 electron multiplicity is a separate, validated and
  overrideable `electron_vector_dim=8` constructor/from-config default.
  Together with `electron_latent_dim=48,lmax=2`, the model validates and uses
  the exact `19x0e + 8x1o + 1x2e` layout.
- Packed electron blocks are never flattened into invariant channels.
  Invariant conditioning uses scalar blocks and irrep norms. Output heads apply
  shared scalar gates across every component of an irrep; only the `1o` block
  receives Cartesian relative-vector injection.
- Pocket, ligand, and pocket-to-ligand neighborhoods use PyG radius operations
  when the optional compiled `torch-cluster` extra is installed. A deterministic
  sparse native fallback is used when PyG raises its documented missing-extra
  `ImportError`. Neither path allocates dense `[N,N,C]` features.
- Coordinate outputs are learned scalar weights multiplying ligand/pocket
  relative unit vectors, encoded pocket vectors, and packed `1o` contractions.
  Proper rotations are equivariant and translations cancel. Reflections are
  intentionally outside the contract.
- Pocket input features are reduced through channel-independent invariant
  summaries before learned projection. The actual positive width is recorded
  and cache-validated, so Task 6 width 50 works without becoming a hidden
  fixture assumption; width 7 is independently tested.
- Pocket cache keys cover pocket position/feature/batch bytes, dtype, device,
  actual feature layout, null mode, encoder architecture, process-local encoder
  ownership, and parameter version counters. A changed pocket/frame/device,
  changed encoder parameters, or another model instance is rejected before
  stale conditioning is used. The caller explicitly owns the cache object.
- Classifier-free null conditioning uses a trainable null scalar embedding,
  zero pocket vectors, retained geometry/batch metadata, and removal of task
  and property signals. Stable SHA-256-derived task features make arbitrary
  non-empty fragment task IDs real conditioning without a process-global
  vocabulary.
- Canonical halfedges remain one unordered `src < dst` row per pair. The bond
  head uses commutative `source+target` and `abs(source-target)` features and
  produces `O(E)` logits.
- Fragment coordinate velocities/scores are zeroed before output. Fixed
  atom/charge/bond categorical fields are functionally restored from the input
  state before output, remaining compatible with downstream exact clamping.
  Count logits are masked before normalization, giving exact zero probability
  below each complex's own fixed-atom count.
- Empty ligand states derive batch size from pocket/time and still return count
  and auxiliary values. Zero-valued RMS contractions clamp the mean-square
  before square root so the backward derivative stays finite.
- `PocketEncoding`, `PocketEncoder.encode`, `AtomCountPredictor.forward`,
  `ECloudFlowModel.encode_pocket`, `ECloudFlowModel.forward`, and
  `ModelPrediction` validation are in the semantic documentation registry with
  shape/dtype/device/frame/units/mask/cache/equivariance/gradient/distributed
  requirements as applicable.

## TDD RED evidence

Initial required command:

`conda run -n 3dmolecule python -m pytest tests/unit/models tests/integration/test_model_equivariance.py -v`

collected no tests and stopped with four expected collection errors:

- `ImportError: cannot import name 'ECloudFlowModel' from 'ecloudflow.models'`
- `ModuleNotFoundError: No module named 'ecloudflow.models.heads'`
- `ImportError: cannot import name 'AtomCountPredictor' from 'ecloudflow.models'`
- the integration test repeated the missing `ECloudFlowModel` public export.

The first implementation run exposed an invalid hand-built test halfedge that
crossed complexes. The canonical `MolecularState` correctly rejected it; the
fixture was repaired to use two independently derived within-complex pairs.

Additional behavior cycles were also observed RED before their minimal fixes:

- The Task 10 documentation test failed because all six required designated
  APIs were absent from `DESIGNATED_APIS`; after registry addition, it reported
  two missing shape topics, which were documented before GREEN.
- With the pocket translated 100 angstroms outside the cross cutoff, moving a
  ligand atom left atom logits exactly equal. This caught the missing ligand
  radius-message branch. Sparse ligand scalar/vector messages made it GREEN.
- All-zero atom, charge, bond, electron, and pocket features produced NaN input
  gradients from unguarded RMS square roots. Clamping mean-square values at the
  dtype epsilon made every asserted gradient finite.
- A pocket cache made by a parameter-diverged second model was accepted because
  the key covered architecture but not learned-feature ownership. Encoder
  ownership and parameter versions now make it fail clearly.
- BF16 pocket encoding raised `TypeError: Got unsupported ScalarType BFloat16`
  while hashing through NumPy. Hashing the raw `uint8` view preserves exact
  bytes/dtype and passes.

Every test names the mutation it catches and asserts real outputs with literal
or independently transformed expected values. No dependency mocks are used.

## Files changed

- `src/ecloudflow/models/layers.py`
- `src/ecloudflow/models/pocket_encoder.py`
- `src/ecloudflow/models/count_predictor.py`
- `src/ecloudflow/models/heads.py`
- `src/ecloudflow/models/backbone.py`
- `src/ecloudflow/models/ecloudflow.py`
- `src/ecloudflow/models/__init__.py`
- `tests/unit/models/test_layers.py`
- `tests/unit/models/test_pocket_encoder.py`
- `tests/unit/models/test_ecloudflow.py`
- `tests/integration/test_model_equivariance.py`
- `tests/unit/test_source_docs.py`
- `tools/check_python_docs.py`
- this report.

## Final commands and results

- `conda run -n 3dmolecule python -m pytest tests/unit/models
  tests/integration/test_model_equivariance.py -v`: **22 passed**, including
  deterministic CPU and available CUDA coverage.
- `conda run -n 3dmolecule python -m pytest -q`: **248 passed, 1 skipped**;
  the skip is the existing external xTB smoke test. The 14 warnings are existing
  third-party Matplotlib/PyParsing deprecations.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: passed.
- `conda run -n 3dmolecule python -m ruff format --check
  src/ecloudflow/models tests/unit/models
  tests/integration/test_model_equivariance.py tests/unit/test_source_docs.py
  tools/check_python_docs.py`: 13 files already formatted.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/models tools/check_python_docs.py`: passed with no output.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/models`: success,
  no issues in 7 source files.
- `git diff --check`: passed; Git emitted only Windows LF-to-CRLF checkout
  notices for three already tracked files.

## Self-review and concerns

The test matrix covers proper rotation plus translation for Cartesian outputs,
the complete e3nn packed electron representation, every scalar head, cache
reuse/rejection, endpoint symmetry, mask/count semantics, null/task behavior,
empty and multi-complex inputs, malformed time/layout/cache values, all-zero and
ordinary finite gradients, BF16 cache bytes, and CUDA placement.

The only remaining operational concern is scale when `torch-cluster` is absent:
the native fallback creates per-complex pair index candidates with quadratic
index memory before radius filtering. It is correct and sparse in feature
channels for the tested installation, but production large-pocket training
should install the PyG-compatible `torch-cluster` wheel so compiled radius
queries are used. No correctness blocker remains within Task 10.

## Commit

All Task 10 source, tests, documentation-gate, and report changes are included
in the planned commit message `feat: add joint SE3 ECloudFlow backbone`; the
final commit identifier is reported in the task handoff because a commit cannot
contain its own final hash.

## Fix Round 1

### Review findings addressed

- Atom, charge, and bond heads now have true independently learned output
  channels. `atom_classes=6`, `charge_classes=4`, and `bond_classes=5` are
  explicit validated constructor and `from_config` overrides; forward rejects
  canonical states whose channel widths disagree. The unordered bond head
  remains exactly symmetric but returns `[E,K]`, not one broadcast scalar.
- Pocket-field values are encoded as invariant masked moments and a
  translation-free polar first moment. Interaction targets accept validated
  `[B]` or `[B,I]` tensors and retain invariant mean/RMS information. Named
  properties use the first eight SHA-256 bytes as a stable device/dtype-local
  identity plus an independent value aggregate; identity remains present when
  the value is exactly zero. Each condition stream has its own learned
  projection and finite gradients.
- The classifier-free null branch now skips pocket-to-ligand edges entirely and
  excludes pocket features/geometry, pocket field, interaction targets,
  property identity/value, and fragment task ID. It retains only per-complex
  batch cardinality, learned null features, ligand state, and time.
- Ligand hidden geometry now remains `[N,V,3]` through every block. Cross and
  ligand messages generate separate weights per vector channel; vector gates,
  coordinate heads, and packed-electron injection consume all channels.
  `ModelConfig.vector_dim` therefore controls the joint ligand backbone as well
  as the pocket encoder, while `electron_vector_dim` remains the separate Task
  8 packed `1o` multiplicity.
- Every block applies invariant time FiLM scale/shift before messaging. Pocket
  cross-attention uses a learned invariant logit and exact per-destination
  segment softmax, so scalar and vector aggregation are normalized even when
  ligand atoms have unequal pocket-neighbor counts.
- Cross products between polar vector channels and their scalar triple product
  add a chiral pseudoscalar update. This is invariant under proper rotations,
  covariant with translation-free vectors, and sign-sensitive under reflection;
  the architecture is SE(3)-equivariant without being forced to O(3)-invariant.
- PyG radius operations remain the preferred path. Both same-set and
  bipartite native fallbacks now generate deterministic source-row chunks with
  a default bound of roughly 65,536 resident candidate pairs rather than one
  per-complex Cartesian grid. Exact directed edge and batch semantics remain
  unchanged.
- The Task 10 semantic registry now enforces parity, `[N,V,3]` vector channels,
  time FiLM, normalized attention, pocket field, interaction-target, and stable
  property-identity documentation in addition to the original tensor/cache/
  equivariance/distributed requirements.

### RED evidence

All fix tests were written before their corresponding production changes.

1. The first grouped focused command stopped during collection because
   `_candidate_pair_chunks` did not exist:

   `conda run -n 3dmolecule python -m pytest tests/unit/models
   tests/integration/test_model_equivariance.py -v`

   Output: 31 items discovered plus one collection error,
   `ImportError: cannot import name '_candidate_pair_chunks'`.

2. After isolating that missing-helper blocker, the model/integration RED run
   collected 24 tests and failed all 24. New model tests reported
   `TypeError: ECloudFlowModel.from_config() got an unexpected keyword argument
   'atom_classes'`; the existing proper-SE(3) test separately reached the
   changed backbone and reported the missing `vector_dim` constructor argument.
   This proved the public vocabulary and full-vector APIs were absent rather
   than hidden behind the initial collection error. The reviewed implementation
   also had the exact broadcast formulas
   `state.atom_logits + scalar[:,None]`, equivalent charge/bond formulas, no
   condition-field/interaction use, cross messages in the null branch, and one
   `[N,3]` ligand direction, which the new behavioral expectations target.

3. Extending the designated documentation contract before editing the
   docstring produced seven independent errors at `ECloudFlowModel.forward`:
   missing parity, vector-channels, time-FiLM, normalized-attention,
   pocket-field, interaction-target, and property-identity topics.

4. The zero-property adversarial test was run after the main condition encoder
   was green and failed with identical affinity values
   `tensor([1.2691, 1.2050])` for names `"affinity"` and `"logp"` at value zero.
   Separating name identity from numeric value made it GREEN.

The tests assert real model probabilities, per-class gradient rows, individual
condition ablations, target gradients, exact null equality under simultaneous
condition changes, reflected-versus-properly-rotated behavior, retained vector
channel shape/use, FiLM gradients in every block, hand-derived attention
weights, exact chunked edges, and candidate chunk bounds. No mocks or source
text assertions are used.

### GREEN evidence and exact commands

- `conda run -n 3dmolecule python -m pytest tests/unit/models
  tests/integration/test_model_equivariance.py -v`: **40 passed in 4.26s**,
  including deterministic CPU tests and the available CUDA regression.
- `conda run -n 3dmolecule python -m pytest -q`: **266 passed, 1 skipped,
  14 warnings in 16.61s**. The skip is the existing external xTB smoke test;
  warnings are unchanged third-party Matplotlib/PyParsing deprecations.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: passed.
- `conda run -n 3dmolecule python -m ruff format --check
  src/ecloudflow/models tests/unit/models
  tests/integration/test_model_equivariance.py tests/unit/test_source_docs.py
  tools/check_python_docs.py`: 13 files already formatted.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/models`: success,
  no issues in 7 source files.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/models tools/check_python_docs.py`: exit code 0 with no output.
- `git diff --check`: passed; only ordinary Windows LF-to-CRLF checkout
  notices were emitted.

### Files changed

- `src/ecloudflow/models/heads.py`
- `src/ecloudflow/models/layers.py`
- `src/ecloudflow/models/backbone.py`
- `src/ecloudflow/models/ecloudflow.py`
- `tests/unit/models/test_layers.py`
- `tests/unit/models/test_ecloudflow.py`
- `tests/integration/test_model_equivariance.py`
- `tools/check_python_docs.py`
- this report.

### Self-review and concerns

Existing empty-ligand, multi-complex, fragment mask/count lower-bound, cache
reuse/rejection, BF16 hashing, zero-feature and ordinary finite-gradient, packed
`l=2`, proper rotation/translation, malformed time/layout, symmetry, feature
width, and CUDA tests all remain green. New tests additionally prove that each
categorical class has a trainable row, every condition independently changes a
prediction and receives gradient, null predictions are bitwise independent of
all removed condition content, all `V>1` channels carry signal, both FiLM blocks
receive gradient, attention sums to one per destination, mirrors are
distinguishable, and fallback chunks obey their bound.

The fallback bounds candidate workspace, but a genuinely dense radius graph can
still have quadratic output edge count; no implementation can store those exact
edges in subquadratic output memory. Production should continue to install the
matching `torch-cluster` wheel for speed. Segment softmax is a deterministic
portable loop in the fallback-compatible implementation and may later merit a
compiled scatter-softmax optimization. Dataset/checkpoint callers whose
vocabularies differ from the explicit defaults must pass their actual atom,
charge, and bond class counts. No correctness blocker remains for this review.

All fix-round source, tests, registry, and report changes are included in the
follow-up commit reported in the task handoff.

## Fix Round 2

### Review findings addressed

- Stable SHA-256 property identity now occupies eight invariant feature slots
  independently of the numeric target value in the ninth slot. No finite value,
  including the prior alias at exactly `-1`, can algebraically erase a name.
  Tensor values remain differentiable and retain the existing dtype, device,
  scalar/`[B]`, batch, and classifier-free-null contracts.
- The public `ECloudFlowModel` constructor now rejects `vector_dim < 3` because
  the chosen parity-sensitive scalar triple product requires three vector
  channels. `from_config` reaches the same validation, including for an
  externally constructed invalid configuration; ordinary `ModelConfig`
  presets already require at least four channels. Constructor and factory
  documentation state the mathematical lower bound.
- Both complete-model equivariance regressions now instantiate the exact
  minimum supported width of three. They prove that this boundary retains
  proper-rotation/translation equivariance while allowing mirrored chiral
  inputs to produce different invariant predictions.

### RED evidence

The new behavior tests were added before production changes. This command was
run against the `707845b` implementation:

`conda run -n 3dmolecule python -m pytest tests/unit/models/test_ecloudflow.py
tests/integration/test_model_equivariance.py -k "property_identity or
vector_width or model_is_se3 or chiral_mirror" -v`

Result: **5 failed, 4 passed, 24 deselected in 3.62s**. At value `-1`, names
`"affinity"` and `"logp"` produced the identical affinity tensor
`tensor([1.3219, 1.2601])`. Direct construction and `from_config` each failed
to raise for both vector widths 1 and 2. The value 0 and 2.5 cases, including
their independent value-gradient checks, and both minimum-width-3 proper-SE(3)
and mirror tests passed; thus the failures isolated the two reviewed defects.

The name regression catches reintroducing any numeric multiplier that can zero
the identity. The boundary regressions catch deleting either direct or factory
validation. Expectations use public predictions and real gradients, with no
mocks or implementation-derived expected values.

### GREEN evidence and exact commands

- The RED command above after the minimal fixes: **9 passed, 24 deselected in
  3.36s**.
- `conda run -n 3dmolecule python -m pytest tests/unit/models
  tests/integration/test_model_equivariance.py -v`: **46 passed in 4.22s**,
  including available CUDA coverage.
- `conda run -n 3dmolecule python -m pytest -q`: **272 passed, 1 skipped, 14
  warnings in 16.80s**. The skip and third-party Matplotlib/PyParsing warnings
  are unchanged.
- `conda run -n 3dmolecule python -m ruff format src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: 13 files left
  unchanged.
- The equivalent `ruff format --check` command: 13 files already formatted.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: all checks passed.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/models`: success, no
  issues in 7 source files.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/models tools/check_python_docs.py`: exit code 0 with no output.
- `git diff --check`: passed; only ordinary Windows LF-to-CRLF checkout
  notices were emitted.

An initial concurrent gate launch exposed a Windows `conda run` temporary-file
activation race before Python started for format-check, Mypy, and pytest. Each
affected command was rerun serially with the successful results above; this was
an environment orchestration failure, not a source or test failure.

### Files changed

- `src/ecloudflow/models/ecloudflow.py`
- `tests/unit/models/test_ecloudflow.py`
- `tests/integration/test_model_equivariance.py`
- this report.

The semantic documentation registry already enforced property identity and
parity topics, so no registry change was required in this round.

### Self-review and concerns

Property name bytes are now unconditionally present when conditioning is
enabled, while value gradients are finite and nonzero at `-1`, 0, and 2.5.
The null path still excludes the entire property feature vector. The minimum
width check occurs before allocating submodules and is shared naturally by the
factory. Width three exercises all operands of the existing triple product and
passes the full packed-electron SE(3) transform test.

The unchanged operational concerns from Fix Round 1 remain: exact dense radius
graphs can have quadratic output size, and the deterministic portable segment
softmax may later benefit from a compiled implementation. No correctness
blocker remains from this review.

## Fix Round 3

### Review finding addressed

Property conditioning now preserves each name-value association before
permutation-invariant aggregation. Every property contributes three pieces to
a 17-channel invariant representation: eight stable SHA-256 identity values,
one differentiable numeric value, and eight identity-times-value joint values.
The independent identity component cannot be erased by any finite numeric
value, while the joint component distinguishes swapped assignments such as
`{"affinity": 1, "logp": 2}` and `{"affinity": 2, "logp": 1}`. Properties
are still processed in sorted-name order and summed, so mapping insertion order
does not affect floating-point evaluation. The complete vector remains absent
from the classifier-free null branch.

### RED evidence

All new behavioral regressions were written before the 17-channel production
change. The focused RED command on `1807d29` was:

`conda run -n 3dmolecule python -m pytest
tests/unit/models/test_ecloudflow.py -k "property or null_branch" -v`

Result: **1 failed, 11 passed, 25 deselected in 3.59s**. The swapped-pair test
failed because the two mappings produced the exact same affinity tensor
`tensor([1.1246, 1.1257])`. This held property names and the per-complex sum of
values constant, isolating the missing association rather than merely detecting
another name or value change.

The 11 companion cases were independently GREEN on the reviewed code: reversed
mapping insertion order was identical; single versus multiple names stayed
distinct at negative, zero, and positive values; explicit zero differed from
omission; each tensor value had a finite nonzero gradient; repeated seeded
model construction was deterministic; and multi-property changes remained
fully removed by the null branch. Together, these cases constrain the minimal
fix without depending on source text, private structure, or mocks.

### GREEN evidence and exact commands

- The RED command after the joint encoder change: **12 passed, 25 deselected in
  3.59s**.
- `conda run -n 3dmolecule python -m pytest tests/unit/models
  tests/integration/test_model_equivariance.py -q`: **52 passed in 4.40s**,
  including the available CUDA test.
- `conda run -n 3dmolecule python -m pytest -q`: **278 passed, 1 skipped, 14
  warnings in 16.78s**. The skip and Matplotlib/PyParsing warnings are unchanged.
- `conda run -n 3dmolecule python -m ruff format src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: one file
  reformatted, 12 unchanged.
- The equivalent `ruff format --check` command: 13 files already formatted.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: all checks passed.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/models tools/check_python_docs.py`: exit code 0 with no output.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/models`: success, no
  issues in 7 source files.
- `git diff --check`: passed; only ordinary Windows LF-to-CRLF checkout
  notices were emitted.

### Files changed

- `src/ecloudflow/models/backbone.py`
- `src/ecloudflow/models/ecloudflow.py`
- `tests/unit/models/test_ecloudflow.py`
- this report.

The existing semantic documentation registry already requires stable property
identity; the designated forward doc now additionally explains independent and
joint name-value slots, so no registry edit was needed.

### Self-review and concerns

The association term is formed per property before summation, and sorted-name
iteration makes the aggregate deterministic for equivalent mappings regardless
of insertion order. Separate identity slots preserve explicit zero as a real
condition and prevent the earlier `-1` erasure. Separate leaves for both values
receive finite nonzero gradients. Existing dtype/device/batch validation occurs
before every contribution, and the null backbone path still skips the complete
property projection.

The fixed 17-channel representation is intentionally small and collision risk
is limited to the documented eight-byte SHA-256 identity prefix; no new global
vocabulary or distributed state was introduced. The unchanged radius-output
and portable segment-softmax performance concerns remain. No correctness
blocker remains from this review.
