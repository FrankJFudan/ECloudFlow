# ECloudFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-grade, pocket-conditioned 3D molecular generator that jointly models complete ligand graphs, binding-pose coordinates, and equivariant electron-cloud latents, with de novo and fragment-conditioned modes, distributed training, constrained sampling, docking-based ranking, comprehensive evaluation, and publication-quality visualization.

**Architecture:** A cached SE(3)-equivariant pocket/electron-field encoder conditions a joint ligand backbone trained as a stochastic interpolant. Continuous coordinates and electron latents use flow matching plus an optional terminal score corrector; categorical atoms, charges, and bonds use simplex probability paths. Fragment invariants and chemistry constraints are enforced throughout sampling, followed by exact bond decoding and transparent raw/relaxed evaluation.

**Tech Stack:** Python 3.10–3.12, PyTorch 2.7+, PyTorch Geometric, e3nn, Lightning 2.5+, Hydra/OmegaConf, Pydantic 2, Typer, RDKit, Biopython, WebDataset, NumPy/SciPy, xTB, OR-Tools, AutoDock Vina, PoseBusters, pandas/pyarrow/openpyxl, Py3Dmol, Plotly, Matplotlib/Seaborn, pytest/Hypothesis/Ruff/Mypy.

**Spec:** `docs/superpowers/specs/2026-08-18-ecloudflow-design.md`

## Global Constraints

- The first delivery is a complete research implementation and tiny end-to-end smoke workflow, not a claim of a converged SOTA checkpoint.
- Local smoke tests must run in the `3dmolecule` Conda environment on an RTX 4060 8GB; production configs target 4× NVIDIA H100 80GB.
- All Python comments and docstrings are English. Designated core APIs use detailed Sphinx/reStructuredText docstrings with `:param:`, tensor shape/dtype/device/units, `:return:`, `:rtype:`, `:raises:`, invariants, side effects, and numerical/distributed notes.
- The model is SE(3)-equivariant for proper rotations and translations and does not force reflection invariance.
- Fixed fragment atom types, charges, internal bonds, and coordinates are exactly restored after every solver, corrector, and projection step.
- Paths, credentials, and machine-specific values are configuration inputs and are never committed.
- Raw generated poses and relaxed poses are stored and evaluated separately.
- `num_molecules` means valid unique molecules; `max_attempts` is bounded and defaults to five times the target.
- Ranked identifiers use `<POCKET_ID>-<RANK:06d>` with no `ECLF`, ordered by Vina ascending, QED descending, SA ascending, then canonical isomeric SMILES.
- Reference repositories remain read-only. Adapted upstream code is attributed in `THIRD_PARTY_NOTICES.md`.
- TDD is mandatory: every task starts with a failing focused test and ends with the relevant suite passing.
- Internal ligand batches use flattened PyG-style tensors: `N` is the total nodes across the batch, `E` is the total unordered ligand halfedges, `halfedge_index[0] < halfedge_index[1]`, positions are `[N,3]`, atom/charge/electron values are `[N,C]`, bond values are `[E,Cb]`, and node/halfedge batch indices map entries to complexes. Dense symmetric bond matrices are materialized only for final decoding and reporting.
- Original ECloudFlow source files use the MIT license; adapted files retain upstream notices and applicable license terms.

## Execution Skill Routing

- Tasks 1–21: use `superpowers:test-driven-development`; use `superpowers:systematic-debugging` for any unexpected failure and `superpowers:verification-before-completion` before final handoff.
- Tasks 3, 6, 14–17: use the RDKit/medicinal-chemistry skill for chemical semantics.
- Tasks 8–13 and 20: use the PyTorch Lightning skill for training, checkpointing, and distributed execution.
- Tasks 11, 17, and 20: use the scientific-critical-thinking skill to separate tested invariants from empirical hypotheses.
- Task 18: use scientific-visualization and academic-plotting skills.
- Task 21: use scientific-writing for README and theory documentation.

## File Responsibility Map

```text
pyproject.toml                       Package metadata, dependency groups, lint/test entry points
environment.yml                     Reproducible CUDA 12.8 Conda environment
configs/                            Typed composable model/data/train/sample/experiment presets
src/ecloudflow/config/              Schema, Hydra composition, resolved-config serialization
src/ecloudflow/core/                Tensor data contracts, coordinate frames, fragment masks
src/ecloudflow/chemistry/           Vocabulary, standardization, valence projection, exact decoding
src/ecloudflow/ecloud/              Density fields, spherical basis, xTB adapter, field tokenizer
src/ecloudflow/data/                Parsers, leakage splits, fragment tasks, shards, Lightning DataModule
src/ecloudflow/models/              Equivariant blocks, pocket encoder, joint backbone and heads
src/ecloudflow/process/             Continuous/categorical interpolants and schedules
src/ecloudflow/training/            Loss composition, EMA, LightningModule, callbacks
src/ecloudflow/sampling/            Cavity prior, ODE solvers, score corrector, constrained pipeline
src/ecloudflow/docking/             Docking interfaces and Vina adapter
src/ecloudflow/evaluation/          Metric registry, ranking, tables, aggregate statistics
src/ecloudflow/visualization/       Interactive 3D, electron fields, plots, HTML reports
src/ecloudflow/cli/                 Typer commands delegating to package services
tests/                              Unit, integration, distributed, CLI, and scientific invariant tests
docs/                               Theory, operations, distributed execution, and reproducibility
```

## Test Fixture Contract

- Every helper name shown in a test snippet (for example, `molecular_state_fixture`, `record`, or `tiny_model_and_batch`) is implemented as a deterministic local factory at the top of that test module or in `tests/conftest.py` before the assertion-focused test is added.
- Fixture factories use fixed seeds, CPU tensors by default, the canonical flattened halfedge contract, and the smallest chemically meaningful graph that exercises the behavior.
- Test-only factories are not promoted into the production API. Shared fixture data belongs under `tests/fixtures/`; reusable pytest fixtures belong in `tests/conftest.py`.
- Tests that need xTB, Vina, NCCL, or a production dataset use explicit markers or injected deterministic backends. The unmarked local suite never silently substitutes fabricated scientific values.

---

### Task 1: Package Skeleton, Typed Configuration, and English Documentation Gate

**Files:**
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `src/ecloudflow/__init__.py`
- Create: `src/ecloudflow/config/__init__.py`
- Create: `src/ecloudflow/config/schema.py`
- Create: `src/ecloudflow/config/loader.py`
- Create: `configs/config.yaml`
- Create: `configs/model/tiny.yaml`
- Create: `configs/model/base.yaml`
- Create: `configs/model/large.yaml`
- Create: `configs/sample/fast.yaml`
- Create: `configs/sample/balanced.yaml`
- Create: `configs/sample/quality.yaml`
- Create: `tools/check_python_docs.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_source_docs.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: the approved design specification.
- Produces: `AppConfig`, `ModelConfig`, `SampleConfig`, `load_config(overrides: Sequence[str]) -> AppConfig`, and `check_paths(paths: Sequence[Path]) -> list[str]`.

- [ ] **Step 1: Write failing configuration and documentation-policy tests**

```python
# tests/unit/test_config.py
import pytest
from pydantic import ValidationError

from ecloudflow.config.loader import load_config


def test_balanced_config_resolves_bounded_attempts():
    config = load_config([
        "model=tiny",
        "sample=balanced",
        "sample.num_molecules=12",
    ])
    assert config.model.name == "tiny"
    assert config.sample.num_molecules == 12
    assert config.sample.resolved_max_attempts == 60
    assert config.sample.solver == "heun"


def test_unknown_config_key_is_rejected():
    with pytest.raises((ValidationError, KeyError)):
        load_config(["sample.unknown_switch=true"])
```

```python
# tests/unit/test_source_docs.py
from pathlib import Path

from tools.check_python_docs import check_paths


def test_checker_rejects_cjk_comment_and_incomplete_core_docstring(tmp_path: Path):
    source = tmp_path / "bad.py"
    source.write_text(
        '# 中文注释\n'
        'def core_step(x):\n'
        '    """Return x."""\n'
        '    return x\n',
        encoding="utf-8",
    )
    errors = check_paths([source], designated={"bad.core_step"})
    assert any("English-only" in error for error in errors)
    assert any(":param x:" in error for error in errors)
```

- [ ] **Step 2: Run the focused tests and verify import/config failures**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/test_config.py tests/unit/test_source_docs.py -v`
Expected: FAIL because `ecloudflow.config` and `tools.check_python_docs` do not exist.

- [ ] **Step 3: Add packaging, strict Pydantic schemas, Hydra composition, and source checker**

```python
# src/ecloudflow/config/schema.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    name: Literal["tiny", "base", "large"] = "tiny"
    scalar_dim: int = Field(default=64, ge=16)
    vector_dim: int = Field(default=16, ge=4)
    num_blocks: int = Field(default=3, ge=1)
    lmax: int = Field(default=2, ge=0, le=4)


class SampleConfig(StrictModel):
    profile: Literal["fast", "balanced", "quality"] = "balanced"
    num_molecules: int = Field(default=100, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    solver: Literal["euler", "heun"] = "heun"
    num_steps: int = Field(default=40, ge=1)
    corrector_steps: int = Field(default=2, ge=0)

    @computed_field
    @property
    def resolved_max_attempts(self) -> int:
        """Return the explicit attempt bound or five times the target count.

        :return: Positive maximum number of generation attempts.
        :rtype: int
        """
        return self.max_attempts or 5 * self.num_molecules


class AppConfig(StrictModel):
    seed: int = 2026
    model: ModelConfig = ModelConfig()
    sample: SampleConfig = SampleConfig()
```

```python
# tools/check_python_docs.py
import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Iterable

CJK = re.compile(r"[\u3400-\u9fff]")
REQUIRED_FIELDS = (":param ", ":return:", ":rtype:")


def check_paths(
    paths: Iterable[Path], designated: set[str] | None = None
) -> list[str]:
    """Validate English-only comments and designated API docstrings.

    :param paths: Python source files to inspect.
    :param designated: Fully qualified ``module.function`` names requiring
        detailed Sphinx fields.
    :return: Human-readable policy violations.
    :rtype: list[str]
    """
    designated = designated or set()
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and CJK.search(token.string):
                errors.append(f"{path}:{token.start[0]} English-only comments required")
        tree = ast.parse(text)
        module = path.stem
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{module}.{node.name}"
                doc = ast.get_docstring(node) or ""
                if CJK.search(doc):
                    errors.append(f"{path}:{node.lineno} English-only docstrings required")
                if name in designated:
                    for field in REQUIRED_FIELDS:
                        if field not in doc:
                            errors.append(f"{path}:{node.lineno} missing {field}")
    return errors
```

Create Hydra defaults that compose `model: tiny` and `sample: balanced`; implement `load_config` with `hydra.initialize_config_dir`, `hydra.compose`, `OmegaConf.to_container(resolve=True)`, and `AppConfig.model_validate`.

```toml
# pyproject.toml core sections
[project]
name = "ecloudflow"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
  "torch>=2.7",
  "torch-geometric>=2.6",
  "e3nn>=0.5.6",
  "lightning>=2.5",
  "hydra-core>=1.3",
  "pydantic>=2.8",
  "typer>=0.12",
  "rdkit>=2024.9",
  "biopython>=1.83",
  "webdataset>=0.2.100",
  "numpy>=1.26,<3",
  "scipy>=1.12",
]

[project.scripts]
ecloudflow = "ecloudflow.cli.main:app"

[project.optional-dependencies]
train = ["wandb>=0.17"]
eval = ["ortools>=9.10", "posebusters>=0.3", "pandas>=2.2", "pyarrow>=17", "openpyxl>=3.1"]
viz = ["py3Dmol>=2.4", "plotly>=5.24", "matplotlib>=3.9", "seaborn>=0.13", "jinja2>=3.1"]
dev = ["pytest>=8.2", "pytest-cov>=5", "hypothesis>=6.110", "ruff>=0.6", "mypy>=1.11"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["external: requires xTB/Vina", "server: requires multi-GPU NCCL"]
```

Add an `argparse` entry point to `tools/check_python_docs.py` that recursively expands source directories, prints one violation per line, and exits with status one when `check_paths` returns errors.

- [ ] **Step 4: Install the editable package and pass focused tests**

Run: `conda run -n 3dmolecule python -m pip install -e ".[dev,train,eval,viz]"`
Run: `conda run -n 3dmolecule python -m pytest tests/unit/test_config.py tests/unit/test_source_docs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit the foundation**

```bash
git add pyproject.toml environment.yml configs src/ecloudflow tools tests
git commit -m "build: initialize typed ECloudFlow package"
```

---

### Task 2: Tensor Contracts, Coordinate Frames, and Fragment Masks

**Files:**
- Create: `src/ecloudflow/core/__init__.py`
- Create: `src/ecloudflow/core/types.py`
- Create: `src/ecloudflow/core/frames.py`
- Create: `src/ecloudflow/core/masks.py`
- Create: `src/ecloudflow/exceptions.py`
- Create: `tests/unit/core/test_types.py`
- Create: `tests/unit/core/test_frames.py`
- Create: `tests/unit/core/test_masks.py`

**Interfaces:**
- Consumes: PyTorch and configuration constants from Task 1.
- Produces: `PocketGraph`, `LigandGraph`, `ElectronField`, `MolecularState`, `FragmentCondition`, `GenerationCondition`, `ComplexSample`, `CoordinateFrame`, and typed package exceptions.

- [ ] **Step 1: Write failing round-trip and exact-fragment-clamping tests**

```python
import torch

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.masks import clamp_fragment
from ecloudflow.core.types import FragmentCondition, MolecularState


def test_coordinate_frame_round_trip():
    points = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    frame = CoordinateFrame.from_pocket(points)
    assert torch.allclose(frame.to_global(frame.to_local(points)), points)
    assert torch.allclose(frame.to_local(points).mean(dim=0), torch.zeros(3))


def test_clamp_fragment_restores_all_fixed_fields():
    reference = molecular_state_fixture(num_atoms=3)
    noisy = reference.replace(
        positions=reference.positions + 7.0,
        atom_logits=reference.atom_logits.roll(1, dims=-1),
        bond_logits=reference.bond_logits.roll(1, dims=-1),
    )
    condition = FragmentCondition.from_atom_mask(
        torch.tensor([True, False, True]), reference
    )
    clamped = clamp_fragment(noisy, condition)
    assert torch.equal(clamped.positions[[0, 2]], reference.positions[[0, 2]])
    assert torch.equal(clamped.atom_logits[[0, 2]], reference.atom_logits[[0, 2]])
    edge_0_2 = find_halfedge(reference.halfedge_index, 0, 2)
    assert torch.equal(clamped.bond_logits[edge_0_2], reference.bond_logits[edge_0_2])
```

- [ ] **Step 2: Run the tests and verify missing core contracts**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/core -v`
Expected: FAIL because the core types do not exist.

- [ ] **Step 3: Implement immutable tensor contracts and frame transforms**

```python
@dataclass(frozen=True)
class MolecularState:
    """Represent one batched ligand state along a generative trajectory.

    :param positions: Flattened coordinates with shape ``[N, 3]`` in angstroms
        in centered pocket frames.
    :param atom_logits: Flattened atom-type values with shape ``[N, A]``.
    :param charge_logits: Flattened formal-charge values with shape ``[N, Q]``.
    :param halfedge_index: Canonical unordered pairs with shape ``[2, E]`` and
        row-zero indices strictly smaller than row-one indices.
    :param bond_logits: One bond-class vector per halfedge, shape ``[E, B]``.
    :param electron_latent: Equivariant field tokens with shape ``[N, C]``.
    :param node_batch: Complex index per node with shape ``[N]``.
    :param halfedge_batch: Complex index per halfedge with shape ``[E]``.
    :return: Immutable molecular state used by training and sampling.
    :rtype: MolecularState
    :raises ValueError: If node counts, ranks, or bond symmetry disagree.
    """
    positions: torch.Tensor
    atom_logits: torch.Tensor
    charge_logits: torch.Tensor
    halfedge_index: torch.Tensor
    bond_logits: torch.Tensor
    electron_latent: torch.Tensor
    node_batch: torch.Tensor
    halfedge_batch: torch.Tensor

    def replace(self, **changes: torch.Tensor) -> "MolecularState":
        return dataclasses.replace(self, **changes)
```

Validate ranks and shared node counts in `__post_init__`. Implement masks with `torch.where`, including symmetric fixed-bond masks and diagonal exclusion. `ComplexSample` must retain source ID, pocket/ligand graphs, fields, properties, frame, and provenance.

- [ ] **Step 4: Run core tests and documentation checks**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/core -v`
Run: `conda run -n 3dmolecule python tools/check_python_docs.py src/ecloudflow/core`
Expected: PASS.

- [ ] **Step 5: Commit core contracts**

```bash
git add src/ecloudflow/core src/ecloudflow/exceptions.py tests/unit/core
git commit -m "feat: add molecular tensor contracts and frames"
```

---

### Task 3: Chemical Vocabulary, Standardization, and Trajectory Valence Projection

**Files:**
- Create: `src/ecloudflow/chemistry/__init__.py`
- Create: `src/ecloudflow/chemistry/vocabulary.py`
- Create: `src/ecloudflow/chemistry/standardize.py`
- Create: `src/ecloudflow/chemistry/valence.py`
- Create: `src/ecloudflow/chemistry/projector.py`
- Create: `tests/unit/chemistry/test_vocabulary.py`
- Create: `tests/unit/chemistry/test_standardize.py`
- Create: `tests/unit/chemistry/test_projector.py`

**Interfaces:**
- Consumes: `MolecularState` from Task 2 and RDKit molecules.
- Produces: `ChemicalVocabulary`, `ValenceTable`, `standardize_molecule`, `ChemicalProjector.project`, and `ProjectedState` with projected logits, expected valence, and allowed-new-bond masks.

- [ ] **Step 1: Write failing valence, symmetry, and unsupported-element tests**

```python
import pytest
import torch

from ecloudflow.chemistry.projector import ChemicalProjector
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary


def test_projector_masks_self_bonds_and_saturated_carbon_additions():
    vocab = ChemicalVocabulary.default_ligand()
    state = make_carbon_with_four_single_bonds(vocab)
    projected = ChemicalProjector(vocab).project(state)
    assert (state.halfedge_index[0] < state.halfedge_index[1]).all()
    assert not (state.halfedge_index[0] == state.halfedge_index[1]).any()
    assert projected.allowed_new_bonds[edges_touching_atom(state, atom=0)].sum() == 0


def test_vocabulary_rejects_unsupported_ligand_metal():
    vocab = ChemicalVocabulary.default_ligand()
    with pytest.raises(ValueError, match="unsupported ligand element"):
        vocab.atom_index("Fe")
```

- [ ] **Step 2: Run chemistry tests and verify missing projector**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/chemistry -v`
Expected: FAIL on missing chemistry modules.

- [ ] **Step 3: Implement explicit vocabularies and differentiable feasibility projection**

Use ligand atoms `C,N,O,S,P,F,Cl,Br,I,B,Si,Se`, charges `-2,-1,0,1,2`, and bond classes `none,single,double,triple`. Pocket elements use a separate expandable vocabulary including common metals. Preserve stereochemistry and formal charges in standardization; kekulize aromatic bonds for model targets and retain isomeric SMILES.

```python
class ChemicalProjector:
    def project(
        self,
        state: MolecularState,
        fixed: FragmentCondition | None = None,
    ) -> ProjectedState:
        """Mask bond updates that cannot satisfy configured valence rules.

        :param state: Current probabilistic molecular state.
        :param fixed: Optional fragment condition whose fields are immutable.
        :return: State with symmetric bond logits, masked self-edges, an
            allowed-new-bond matrix, and expected-valence diagnostics.
        :rtype: ProjectedState
        :raises ValueError: If atom, charge, or bond channels do not match the
            chemical vocabulary.

        The projection is differentiable with respect to unmasked logits. It
        does not discretize the graph and therefore cannot guarantee final
        sanitization; exact feasibility is handled by the final decoder.
        """
```

Build `ValenceTable` from vetted defaults plus optional dataset counts adapted from CoCoGraph semantics. Keep source attribution in module docs.

- [ ] **Step 4: Pass chemistry and source-documentation tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/chemistry -v`
Run: `conda run -n 3dmolecule python tools/check_python_docs.py src/ecloudflow/chemistry`
Expected: PASS.

- [ ] **Step 5: Commit chemical foundations**

```bash
git add src/ecloudflow/chemistry tests/unit/chemistry
git commit -m "feat: add constrained chemical vocabulary"
```

---

### Task 4: Equivariant Electron Basis and Density Projection

**Files:**
- Create: `src/ecloudflow/ecloud/__init__.py`
- Create: `src/ecloudflow/ecloud/basis.py`
- Create: `src/ecloudflow/ecloud/field.py`
- Create: `tests/unit/ecloud/test_basis.py`
- Create: `tests/unit/ecloud/test_field.py`

**Interfaces:**
- Consumes: `ElectronField` and coordinate frames from Task 2; e3nn spherical harmonics.
- Produces: `SphericalFieldBasis`, `project_density_to_atoms`, `reconstruct_density`, `integrated_electron_count`, and `multipole_moments`.

- [ ] **Step 1: Write failing density round-trip, count, and rotation tests**

```python
import torch

from ecloudflow.ecloud.basis import SphericalFieldBasis
from ecloudflow.ecloud.field import (
    integrated_electron_count,
    project_density_to_atoms,
    reconstruct_density,
)


def test_gaussian_density_preserves_electron_count_and_rotation():
    grid, density, centers, weights = gaussian_field_fixture()
    basis = SphericalFieldBasis(n_radial=6, lmax=2, cutoff=4.0)
    coeff = project_density_to_atoms(density, grid, centers, weights, basis)
    reconstructed = reconstruct_density(coeff, grid, centers, basis)
    assert abs(integrated_electron_count(reconstructed, weights) - 6.0) < 0.15

    rotation = random_proper_rotation(seed=17)
    rotated_coeff = project_density_to_atoms(
        density, grid @ rotation.T, centers @ rotation.T, weights, basis
    )
    rotated_reconstruction = reconstruct_density(
        rotated_coeff, grid @ rotation.T, centers @ rotation.T, basis
    )
    assert torch.allclose(reconstructed, rotated_reconstruction, atol=2e-3, rtol=2e-3)
```

- [ ] **Step 2: Run field tests and verify missing e3nn implementation**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/ecloud/test_basis.py tests/unit/ecloud/test_field.py -v`
Expected: FAIL because electron basis modules do not exist.

- [ ] **Step 3: Implement normalized radial bases and spherical projection**

```python
def project_density_to_atoms(
    density: torch.Tensor,
    grid: torch.Tensor,
    centers: torch.Tensor,
    integration_weights: torch.Tensor,
    basis: SphericalFieldBasis,
) -> torch.Tensor:
    """Project a sampled density into atom-centered equivariant coefficients.

    :param density: Non-negative density values with shape ``[G]`` in
        electrons per cubic angstrom.
    :param grid: Grid coordinates with shape ``[G, 3]`` in angstroms.
    :param centers: Atom centers with shape ``[N, 3]`` in the same frame.
    :param integration_weights: Positive quadrature volumes with shape ``[G]``.
    :param basis: Radial and real-spherical-harmonic basis definition.
    :return: Coefficients with shape ``[N, R, (lmax + 1) ** 2]``.
    :rtype: torch.Tensor
    :raises ValueError: If density is negative, shapes disagree, or tensors
        use different devices.

    Coefficients with angular order ``l`` transform under the matching e3nn
    irreducible representation. Accumulation runs in float32 even under BF16
    autocast to avoid electron-count drift.
    """
```

Use `e3nn.o3.spherical_harmonics`, compact-support Gaussian/Bessel radial functions, quadrature weights, and chunked grid evaluation to cap memory. Normalize reconstruction so the `l=0` channel preserves the integrated count within tolerance.

- [ ] **Step 4: Pass field and equivariance tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/ecloud -v`
Expected: PASS on CPU and CUDA when available.

- [ ] **Step 5: Commit electron basis**

```bash
git add src/ecloudflow/ecloud tests/unit/ecloud
git commit -m "feat: add equivariant electron field basis"
```

---

### Task 5: xTB Adapter, Physical Pocket Fields, and Provenance

**Files:**
- Create: `src/ecloudflow/ecloud/xtb.py`
- Create: `src/ecloudflow/ecloud/pocket.py`
- Create: `src/ecloudflow/ecloud/provenance.py`
- Create: `tests/unit/ecloud/test_xtb.py`
- Create: `tests/unit/ecloud/test_pocket.py`
- Create: `tests/fixtures/xtb/success.cube`
- Create: `tests/fixtures/xtb/failure.stderr`

**Interfaces:**
- Consumes: electron-field basis from Task 4, RDKit/Biopython structures, and `CoordinateFrame`.
- Produces: `XTBRunner.calculate_ligand`, `PocketFieldBuilder.build`, `QMResult`, `ToolProvenance`, and `FieldBuilderBundle`.

- [ ] **Step 1: Write failing adapter and pocket-field tests**

```python
from pathlib import Path

from ecloudflow.ecloud.pocket import PocketFieldBuilder
from ecloudflow.ecloud.xtb import QMStatus, XTBRunner


def test_xtb_runner_records_failed_qm_without_fake_density(tmp_path: Path):
    runner = XTBRunner(executable="missing-xtb", work_root=tmp_path)
    result = runner.calculate_ligand(methane_molecule(), charge=0, multiplicity=1)
    assert result.status is QMStatus.TOOL_MISSING
    assert result.density is None
    assert result.qm_mask is False
    assert "missing-xtb" in result.provenance.command


def test_pocket_field_is_deterministic_and_centered():
    pocket = toy_pocket_with_donor_acceptor_and_metal()
    first = PocketFieldBuilder.default().build(pocket)
    second = PocketFieldBuilder.default().build(pocket)
    assert first.channel_names == (
        "density", "partial_charge", "donor", "acceptor", "hydrophobic", "aromatic"
    )
    assert first.frame == second.frame
    assert first.values.equal(second.values)
```

- [ ] **Step 2: Run tests and verify missing tool adapters**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/ecloud/test_xtb.py tests/unit/ecloud/test_pocket.py -v`
Expected: FAIL on missing adapters.

- [ ] **Step 3: Implement safe subprocess execution and physical pocket channels**

```python
class XTBRunner:
    def calculate_ligand(
        self,
        molecule: Chem.Mol,
        charge: int,
        multiplicity: int,
    ) -> QMResult:
        """Calculate a ligand electron-density cube with an isolated xTB run.

        :param molecule: Sanitized RDKit molecule with one three-dimensional
            conformer and explicit hydrogens.
        :param charge: Integer molecular charge passed to xTB.
        :param multiplicity: Positive spin multiplicity.
        :return: Density, grid metadata, status, mask, and complete provenance.
        :rtype: QMResult
        :raises ValueError: If coordinates, charge, or multiplicity are invalid.

        The command runs in a unique temporary directory. A non-zero exit,
        timeout, malformed cube, or missing executable returns a typed failed
        result with ``qm_mask=False`` and never returns approximate density.
        """
```

Use `subprocess.run` with an argument list, timeout, captured output, and a unique `TemporaryDirectory`. Adapt the cube parser/interpolator from ECloudGen with attribution. Build pocket channels from element Gaussian widths, Gasteiger/available partial charge, RDKit donor/acceptor/aromatic flags, and residue hydrophobicity. Store tool/version/command/source hashes without credentials.

- [ ] **Step 4: Pass adapter tests and one opt-in real xTB smoke**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/ecloud/test_xtb.py tests/unit/ecloud/test_pocket.py -v`
Run when xTB is installed: `conda run -n 3dmolecule python -m pytest tests/integration/test_xtb_real.py -m external -v`
Expected: unit tests PASS; external test PASS or is explicitly skipped when xTB is unavailable.

- [ ] **Step 5: Commit field builders**

```bash
git add src/ecloudflow/ecloud tests/unit/ecloud tests/fixtures/xtb
git commit -m "feat: add traceable electron field builders"
```

---

### Task 6: Protein/Ligand Parsing and Fragment Task Construction

**Files:**
- Create: `src/ecloudflow/data/__init__.py`
- Create: `src/ecloudflow/data/parsers.py`
- Create: `src/ecloudflow/data/features.py`
- Create: `src/ecloudflow/data/fragments.py`
- Create: `tests/unit/data/test_parsers.py`
- Create: `tests/unit/data/test_fragments.py`
- Create: `tests/fixtures/complex/toy_pocket.pdb`
- Create: `tests/fixtures/complex/toy_ligand.sdf`

**Interfaces:**
- Consumes: Task 2 contracts, Task 3 chemical standardization, and Task 5 field builders.
- Produces: `parse_pocket_pdb`, `parse_ligand_sdf`, `build_complex_sample`, and `FragmentTaskSampler.sample`.

- [ ] **Step 1: Write failing parser, inverse-frame, and fragment-mode tests**

```python
from pathlib import Path
import torch

from ecloudflow.data.fragments import FragmentMode, FragmentTaskSampler
from ecloudflow.data.parsers import build_complex_sample


def test_complex_parser_centers_and_restores_ligand_pose(fixture_dir: Path):
    sample = build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="TOY",
    )
    restored = sample.frame.to_global(sample.ligand.positions)
    assert torch.allclose(restored, sample.provenance.original_ligand_positions, atol=1e-5)
    assert sample.pocket.features.shape[0] == sample.pocket.positions.shape[0]


def test_fragment_sampler_builds_all_four_optimization_modes():
    ligand = medicinal_ligand_fixture()
    sampler = FragmentTaskSampler(seed=23)
    modes = {sampler.sample(ligand, forced_mode=mode).mode for mode in FragmentMode}
    assert modes == {
        FragmentMode.GROW,
        FragmentMode.LINK,
        FragmentMode.REPLACE,
        FragmentMode.MERGE,
    }
```

- [ ] **Step 2: Run data parser tests and verify missing modules**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/data/test_parsers.py tests/unit/data/test_fragments.py -v`
Expected: FAIL on missing parsers and fragment sampler.

- [ ] **Step 3: Implement explicit parsing and ring-aware fragment masks**

```python
def build_complex_sample(
    pocket_path: Path,
    ligand_path: Path,
    sample_id: str,
    field_builders: FieldBuilderBundle | None = None,
) -> ComplexSample:
    """Parse one cocrystal pair into the canonical model data contract.

    :param pocket_path: Protein-pocket PDB path in the original coordinate frame.
    :param ligand_path: SDF path containing exactly one bonded ligand conformer.
    :param sample_id: Stable source identifier stored in every artifact.
    :param field_builders: Optional pocket/ligand field builder bundle. The
        default builds a physical pocket field and attempts xTB ligand density.
    :return: Centered graph, fields, inverse frame, properties, and provenance.
    :rtype: ComplexSample
    :raises DataValidationError: If files are unreadable, the ligand is
        unsanitizable, coordinates are absent, or no pocket atoms are present.

    Parsing never falls back to a different sample. All skipped inputs receive
    a manifest record from the calling preprocessing pipeline.
    """
```

Use Biopython for protein records and RDKit for ligand chemistry. Implement BRICS, Murcko, ring-bond, and linker cuts with seeded choice. Create exact atom/bond/coordinate/attachment masks. Require pocket-frame fragment coordinates for the fixed-pose path.

- [ ] **Step 4: Pass parsing, fragment, and source-policy tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/data -v`
Expected: PASS.

- [ ] **Step 5: Commit parsers and fragment tasks**

```bash
git add src/ecloudflow/data tests/unit/data tests/fixtures/complex
git commit -m "feat: parse complexes and build fragment tasks"
```

---

### Task 7: Leakage-Controlled Splits, WebDataset Shards, and DataModule

**Files:**
- Create: `src/ecloudflow/data/splits.py`
- Create: `src/ecloudflow/data/manifest.py`
- Create: `src/ecloudflow/data/shards.py`
- Create: `src/ecloudflow/data/datamodule.py`
- Create: `src/ecloudflow/data/diffgui_lmdb.py`
- Modify: `src/ecloudflow/config/schema.py`
- Create: `configs/data/pdbbind.yaml`
- Create: `configs/data/crossdocked.yaml`
- Create: `configs/data/ligand_pretrain.yaml`
- Create: `tests/unit/data/test_splits.py`
- Create: `tests/unit/data/test_shards.py`
- Create: `tests/integration/test_datamodule.py`

**Interfaces:**
- Consumes: `ComplexSample` from Task 6.
- Produces: `DataConfig`, `build_grouped_split`, `ShardWriter`, `stream_samples`, `DiffGuiLMDBImporter`, and `ECloudDataModule`.

- [ ] **Step 1: Write failing leakage and rank/worker coverage tests**

```python
def test_grouped_split_keeps_homologs_and_similar_ligands_together():
    records = split_fixture_with_homologs_and_scaffold_duplicates()
    split = build_grouped_split(records, sequence_identity=0.4, ligand_tanimoto=0.8, seed=7)
    assert split.partition_of("protein-A") == split.partition_of("protein-A-homolog")
    assert split.partition_of("ligand-X") == split.partition_of("ligand-X-analog")
    assert split.hash.startswith("sha256:")


def test_webdataset_rank_worker_partition_has_exact_coverage(tmp_path):
    paths = write_ten_sample_shards(tmp_path)
    seen = []
    for rank in range(2):
        for worker in range(2):
            seen.extend(sample_ids_for_partition(paths, rank, 2, worker, 2))
    assert sorted(seen) == [f"sample-{index}" for index in range(10)]
    assert len(seen) == len(set(seen))
```

- [ ] **Step 2: Run split/shard tests and verify missing data backend**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/data/test_splits.py tests/unit/data/test_shards.py tests/integration/test_datamodule.py -v`
Expected: FAIL because split, shard, and DataModule implementations are absent.

- [ ] **Step 3: Implement manifests, atomic shards, and distributed streaming**

```python
class ShardWriter:
    def write(self, samples: Iterable[ComplexSample], output_dir: Path) -> DatasetManifest:
        """Serialize validated samples into atomic WebDataset tar shards.

        :param samples: Stream of canonical complex samples.
        :param output_dir: Destination containing shards and manifest JSON.
        :return: Manifest with sample IDs, source hashes, shard hashes, skips,
            preprocessing version, and split metadata.
        :rtype: DatasetManifest
        :raises ShardWriteError: If serialization fails before atomic rename.

        Shards are first written with a ``.partial`` suffix, hashed, fsynced,
        and renamed. Failed samples are recorded and never replaced by another
        sample. Rank partitioning occurs before worker partitioning.
        """
```

Use sequence-cluster identifiers when supplied and Morgan-scaffold similarity grouping for ligand leakage. Implement 0.5–2GB target shard sizing, deterministic shuffle buffer, rank/worker split, and node-count bucketed batching. The DiffGui importer reads the existing LMDB without changing it and converts records into the canonical schema.

- [ ] **Step 4: Pass data integration tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/data tests/integration/test_datamodule.py -v`
Expected: PASS, including exact sample coverage.

- [ ] **Step 5: Commit the scalable data layer**

```bash
git add src/ecloudflow/data src/ecloudflow/config/schema.py configs/data tests/unit/data tests/integration/test_datamodule.py
git commit -m "feat: add leakage-safe sharded data pipeline"
```

---

### Task 8: Electron-Field Tokenizer and Graph–Field Decoder

**Files:**
- Create: `src/ecloudflow/ecloud/tokenizer.py`
- Create: `src/ecloudflow/ecloud/decoder.py`
- Create: `src/ecloudflow/models/__init__.py`
- Create: `tests/unit/ecloud/test_tokenizer.py`
- Create: `tests/integration/test_ecloud_autoencoder.py`

**Interfaces:**
- Consumes: spherical coefficients from Task 4 and graph atom features from Task 6.
- Produces: `EquivariantFieldTokenizer`, `ElectronFieldDecoder`, and `ElectronReconstruction`.

- [ ] **Step 1: Write failing shape, equivariance, and backward tests**

```python
def test_field_tokenizer_round_trip_is_differentiable_and_equivariant():
    batch = electron_coefficient_batch(requires_grad=True)
    tokenizer = EquivariantFieldTokenizer(
        n_radial=6, lmax=2, scalar_dim=32, vector_dim=8, latent_dim=48
    )
    latent = tokenizer.encode(batch.coefficients, batch.atom_features, batch.mask)
    reconstruction = tokenizer.decode(latent, batch.centers, batch.query_grid, batch.mask)
    assert latent.shape == (2, 5, 48)
    reconstruction.density.sum().backward()
    assert batch.coefficients.grad is not None
    assert_rotation_equivariant(tokenizer, batch, atol=3e-4)
```

- [ ] **Step 2: Run tokenizer tests and verify missing neural modules**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/ecloud/test_tokenizer.py tests/integration/test_ecloud_autoencoder.py -v`
Expected: FAIL because tokenizer and decoder do not exist.

- [ ] **Step 3: Implement irreducible field encoding and multiresolution decoding**

Use `e3nn.o3.Linear` within each irrep block, gated scalar/non-scalar activations, atom-feature conditioning, masked pooling for global multipoles, and chunked query decoding. Return density, gradient, electron count, dipole, and latent round-trip terms. Accumulate density and moments in float32 under mixed precision.

```python
class EquivariantFieldTokenizer(nn.Module):
    def forward(
        self,
        coefficients: torch.Tensor,
        atom_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atom-centered density coefficients into equivariant tokens.

        :param coefficients: Tensor ``[B, N, R, H]`` where ``H`` stores real
            spherical-harmonic components through the configured ``lmax``.
        :param atom_features: Invariant features with shape ``[B, N, F]``.
        :param mask: Boolean physical-atom mask with shape ``[B, N]``.
        :return: Padded per-atom latent tokens with shape ``[B, N, C]`` and
            a documented packed irrep layout.
        :rtype: torch.Tensor
        :raises ValueError: If the harmonic layout or masks are inconsistent.
        """
```

- [ ] **Step 4: Pass tokenizer and autoencoder smoke tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/ecloud tests/integration/test_ecloud_autoencoder.py -v`
Expected: PASS with finite gradients.

- [ ] **Step 5: Commit electron tokenizer**

```bash
git add src/ecloudflow/ecloud src/ecloudflow/models tests/unit/ecloud tests/integration/test_ecloud_autoencoder.py
git commit -m "feat: tokenize equivariant electron fields"
```

---

### Task 9: Continuous Stochastic Interpolant and Categorical Probability Paths

**Files:**
- Create: `src/ecloudflow/process/__init__.py`
- Create: `src/ecloudflow/process/schedules.py`
- Create: `src/ecloudflow/process/continuous.py`
- Create: `src/ecloudflow/process/categorical.py`
- Create: `tests/unit/process/test_schedules.py`
- Create: `tests/unit/process/test_continuous.py`
- Create: `tests/unit/process/test_categorical.py`

**Interfaces:**
- Consumes: `MolecularState`, edit masks, and category vocabularies.
- Produces: `InterpolantSchedule`, `ContinuousPath.sample`, `ContinuousPath.targets`, `CategoricalPath.sample`, and `CategoricalPath.endpoint_loss`.

- [ ] **Step 1: Write failing endpoint, velocity, score, simplex, and mask tests**

```python
def test_continuous_path_endpoints_and_targets():
    path = ContinuousPath(LinearBridge(interior_noise=0.2))
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)
    assert torch.allclose(path.mean(x0, x1, torch.tensor(0.0)), x0)
    assert torch.allclose(path.mean(x0, x1, torch.tensor(1.0)), x1)
    sample = path.sample(x0, x1, torch.tensor(0.4), generator=seeded_generator(4))
    velocity, score = path.targets(x0, x1, sample)
    assert velocity.shape == score.shape == x0.shape
    assert torch.isfinite(velocity).all() and torch.isfinite(score).all()


def test_categorical_path_stays_on_simplex_and_ignores_fixed_nodes():
    path = CategoricalPath(num_classes=5, prior=torch.ones(5) / 5)
    target = torch.tensor([1, 3, 2])
    fixed = torch.tensor([True, False, False])
    sample = path.sample(target, torch.tensor(0.6), fixed_mask=fixed)
    assert torch.allclose(sample.probabilities.sum(-1), torch.ones(3))
    assert sample.classes[0] == target[0]
```

- [ ] **Step 2: Run process tests and verify missing interpolants**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/process -v`
Expected: FAIL because process modules do not exist.

- [ ] **Step 3: Implement numerically stable paths with explicit target conventions**

```python
class ContinuousPath:
    def targets(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        sample: ContinuousSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conditional velocity and denoising-score targets.

        :param x0: Prior endpoint with arbitrary leading dimensions.
        :param x1: Data endpoint with the same shape, dtype, and device.
        :param sample: Interior path sample containing ``t`` and sampled noise.
        :return: ``(velocity, score)`` tensors matching ``x0``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If endpoints disagree or ``t`` is outside the open
            score-training interval.

        Score targets use ``-epsilon / gamma(t)`` and clamp only the schedule
        denominator at the configured numerical epsilon. Endpoint times are
        excluded from score training rather than assigned an artificial score.
        """
```

Implement linear and cosine bridge options, schedule derivatives, antithetic time sampling, Dirichlet/simplex interpolation, class-prior validation, endpoint cross-entropy, and edit-mask reduction. Add distribution normalization tests in float32 and BF16-compatible code paths.

- [ ] **Step 4: Pass process and gradient tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/process -v`
Expected: PASS.

- [ ] **Step 5: Commit stochastic paths**

```bash
git add src/ecloudflow/process tests/unit/process
git commit -m "feat: add hybrid stochastic interpolant paths"
```

---

### Task 10: SE(3)-Equivariant Pocket Encoder and Joint Ligand Backbone

**Files:**
- Create: `src/ecloudflow/models/layers.py`
- Create: `src/ecloudflow/models/pocket_encoder.py`
- Create: `src/ecloudflow/models/count_predictor.py`
- Create: `src/ecloudflow/models/heads.py`
- Create: `src/ecloudflow/models/backbone.py`
- Create: `src/ecloudflow/models/ecloudflow.py`
- Create: `tests/unit/models/test_layers.py`
- Create: `tests/unit/models/test_pocket_encoder.py`
- Create: `tests/unit/models/test_ecloudflow.py`
- Create: `tests/integration/test_model_equivariance.py`

**Interfaces:**
- Consumes: pocket/ligand contracts, electron tokens, configs, and time/condition embeddings.
- Produces: `PocketEncoding`, `PocketEncoder.encode`, `AtomCountPredictor`, `ECloudFlowModel.forward`, and `ModelPrediction`.

- [ ] **Step 1: Write failing SE(3), cache, output-shape, and gradient tests**

```python
def test_model_is_se3_equivariant_and_scalar_outputs_are_invariant():
    model = ECloudFlowModel.from_config(tiny_model_config())
    batch = tiny_model_batch()
    prediction = model(batch.state, batch.time, batch.condition)
    rotation = random_proper_rotation(seed=31)
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed = batch.transform(rotation, translation)
    transformed_prediction = model(
        transformed.state, transformed.time, transformed.condition
    )
    assert torch.allclose(
        transformed_prediction.position_velocity,
        prediction.position_velocity @ rotation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    assert torch.allclose(
        transformed_prediction.atom_logits, prediction.atom_logits, atol=5e-4
    )
    assert torch.allclose(
        transformed_prediction.bond_logits, prediction.bond_logits, atol=5e-4
    )


def test_cached_pocket_encoding_is_reused_across_steps():
    model, batch = tiny_model_and_batch()
    encoded = model.encode_pocket(batch.condition.pocket)
    first = model(batch.state, torch.tensor([0.2]), batch.condition, encoded)
    second = model(batch.state, torch.tensor([0.8]), batch.condition, encoded)
    assert encoded.cache_key == first.pocket_cache_key == second.pocket_cache_key
```

- [ ] **Step 2: Run model tests and verify missing backbone**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/models tests/integration/test_model_equivariance.py -v`
Expected: FAIL because model components do not exist.

- [ ] **Step 3: Implement relative-vector equivariant blocks and all prediction heads**

```python
class ECloudFlowModel(nn.Module):
    def forward(
        self,
        state: MolecularState,
        time: torch.Tensor,
        condition: GenerationCondition,
        pocket_encoding: PocketEncoding | None = None,
    ) -> ModelPrediction:
        """Predict joint flow, score, categorical endpoints, and auxiliaries.

        :param state: Batched noisy ligand graph, coordinates, and electron tokens.
        :param time: Per-complex path times with shape ``[B]`` in ``[0, 1]``.
        :param condition: Pocket, electron, fragment, property, interaction, and
            task conditions with explicit node/edge batch indices.
        :param pocket_encoding: Optional cached pocket representation. When
            omitted it is computed once by ``encode_pocket``.
        :return: Coordinate/electron velocities and scores; atom, charge, and
            symmetric bond logits; count, affinity, and interaction outputs.
        :rtype: ModelPrediction
        :raises ValueError: If batch indices, channels, devices, or cache keys
            are inconsistent.

        Coordinate updates are sums of learned scalar weights times relative
        displacement vectors, so translation cancels and proper rotations act
        on outputs exactly. Scalar logits depend only on invariant contractions.
        """
```

Build radius/kNN ligand, pocket, and cross edges with PyG. Use invariant radial features, scalar/vector channels, gated equivariant updates, time FiLM, pocket cross-attention, electron-token messages, and symmetric pair heads. Apply masks before output. Include classifier-free null embeddings and fragment task embeddings. The atom-count predictor returns a categorical distribution constrained to at least the fixed-fragment size.

- [ ] **Step 4: Pass CPU/CUDA model and equivariance tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/models tests/integration/test_model_equivariance.py -v`
Expected: PASS, with finite gradients and no dependence on absolute translation.

- [ ] **Step 5: Commit the model backbone**

```bash
git add src/ecloudflow/models tests/unit/models tests/integration/test_model_equivariance.py
git commit -m "feat: add joint SE3 ECloudFlow backbone"
```

---

### Task 11: Composite Scientific Losses and Lightning TrainingModule

**Files:**
- Create: `src/ecloudflow/training/__init__.py`
- Create: `src/ecloudflow/training/losses.py`
- Create: `src/ecloudflow/training/ema.py`
- Create: `src/ecloudflow/training/module.py`
- Create: `src/ecloudflow/training/types.py`
- Modify: `src/ecloudflow/config/schema.py`
- Create: `tests/unit/training/test_losses.py`
- Create: `tests/unit/training/test_ema.py`
- Create: `tests/integration/test_training_step.py`

**Interfaces:**
- Consumes: Task 9 paths and Task 10 model predictions.
- Produces: `LossConfig`, `TrainingTargets`, `LossBreakdown`, `RunningLossScaler`, `compute_ecloudflow_loss`, `ExponentialMovingAverage`, and `ECloudFlowTrainingModule`.

Extend `AppConfig` with a frozen `LossConfig` containing every component weight, warm-up boundary, and normalization option used below; unknown loss keys remain forbidden.

- [ ] **Step 1: Write failing component-loss, fixed-mask, and one-step tests**

```python
def test_composite_loss_reports_raw_and_weighted_finite_terms():
    fixture = loss_fixture_with_qm_and_fragment_masks()
    result = compute_ecloudflow_loss(fixture.prediction, fixture.targets, fixture.config)
    expected = {"flow", "score", "discrete", "ecloud", "chem", "interaction"}
    assert set(result.raw) == expected
    assert set(result.weighted) == expected
    assert torch.isfinite(result.total)
    assert result.raw["flow_fixed"] == 0.0


def test_tiny_lightning_training_step_updates_parameters(tmp_path):
    module, batch = tiny_training_module_and_batch()
    before = clone_trainable_parameters(module)
    trainer = lightning.Trainer(
        accelerator="cpu", devices=1, max_steps=1, logger=False,
        enable_checkpointing=False, default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=[batch])
    assert any_parameter_changed(before, module)
```

- [ ] **Step 2: Run training tests and verify missing loss/module code**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/training tests/integration/test_training_step.py -v`
Expected: FAIL because training modules do not exist.

- [ ] **Step 3: Implement normalized losses, auxiliary chemistry, and EMA**

```python
def compute_ecloudflow_loss(
    prediction: ModelPrediction,
    targets: TrainingTargets,
    config: LossConfig,
    scaler: RunningLossScaler | None = None,
) -> LossBreakdown:
    """Compute all normalized ECloudFlow training objectives.

    :param prediction: Joint outputs from ``ECloudFlowModel``.
    :param targets: Velocity, score, endpoints, density, geometry, chemistry,
        interaction, QM, and editable-mask targets.
    :param config: Component weights and warm-up schedules at the current step.
    :param scaler: Optional detached running RMS normalizer per component.
    :return: Total loss plus raw, normalized, weighted, and diagnostic terms.
    :rtype: LossBreakdown
    :raises FloatingPointError: If any active component is NaN or infinite.

    QM-density terms are multiplied by ``qm_mask``. Fixed fragment fields are
    excluded from generative losses. Running scales are updated from detached
    values and do not alter gradient directions within a component.
    """
```

Implement flow/score MSE, categorical endpoint CE, density/gradient/count/dipole/cycle terms, expected-valence overflow, bond-length prior, clash loss, connectivity surrogate, interaction focal loss, and heteroscedastic affinity NLL. Log raw/normalized/weighted terms. EMA updates after successful optimizer steps and stores/restores separately.

- [ ] **Step 4: Pass loss, EMA, and training-step tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/training tests/integration/test_training_step.py -v`
Expected: PASS.

- [ ] **Step 5: Commit training objectives**

```bash
git add src/ecloudflow/training src/ecloudflow/config/schema.py tests/unit/training tests/integration/test_training_step.py
git commit -m "feat: train joint flow score and electron objectives"
```

---

### Task 12: Four-Stage Experiments, DDP Checkpoints, and Resumability

**Files:**
- Create: `src/ecloudflow/training/callbacks.py`
- Create: `src/ecloudflow/training/checkpoint.py`
- Create: `src/ecloudflow/training/stages.py`
- Create: `configs/train/stage1.yaml`
- Create: `configs/train/stage2.yaml`
- Create: `configs/train/stage3.yaml`
- Create: `configs/train/stage4.yaml`
- Create: `configs/experiment/smoke.yaml`
- Create: `configs/experiment/pdbbind_large.yaml`
- Modify: `src/ecloudflow/config/schema.py`
- Create: `tests/unit/training/test_stages.py`
- Create: `tests/integration/test_checkpoint_resume.py`
- Create: `tests/distributed/test_ddp_smoke.py`

**Interfaces:**
- Consumes: `ECloudDataModule` and `ECloudFlowTrainingModule`.
- Produces: `TrainerConfig`, `TrainingStage`, `configure_stage`, `ReproducibleCheckpoint`, and DDP-ready experiment presets. `TrainerConfig` owns accelerator, strategy, precision, device count, accumulation, clipping, checkpoint, and resume settings.

- [ ] **Step 1: Write failing stage-freeze and interrupted/resumed equivalence tests**

```python
def test_stage_parameter_policies_are_explicit():
    module = tiny_training_module()
    configure_stage(module, TrainingStage.ELECTRON_TOKENIZER)
    assert trainable_groups(module) == {"field_tokenizer", "field_decoder"}
    configure_stage(module, TrainingStage.POCKET_MULTITASK)
    assert "joint_backbone" in trainable_groups(module)
    assert module.loss_config.interaction.weight > 0


def test_resumed_two_step_run_matches_uninterrupted(tmp_path):
    uninterrupted = run_deterministic_steps(steps=2, root=tmp_path / "full")
    checkpoint = run_deterministic_steps(steps=1, root=tmp_path / "partial").checkpoint
    resumed = run_deterministic_steps(
        steps=2, root=tmp_path / "resumed", resume_from=checkpoint
    )
    assert_state_dict_close(uninterrupted.model, resumed.model, atol=1e-6, rtol=1e-6)
    assert uninterrupted.dataset_epoch == resumed.dataset_epoch
```

- [ ] **Step 2: Run stage/checkpoint/DDP tests and verify failures**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/training/test_stages.py tests/integration/test_checkpoint_resume.py -v`
Expected: FAIL because staged configuration and reproducible checkpoints do not exist.

- [ ] **Step 3: Implement stage policies and complete checkpoint state**

Store model, optimizer, scheduler, EMA, global step, epoch, per-rank CPU/CUDA RNG, WebDataset epoch/position, resolved config, dataset manifest hash, preprocessing version, and Git revision. Validate manifest/config compatibility before resume. Add a bounded NaN diagnostic callback and rank-zero atomic artifact writer.

```python
class ReproducibleCheckpoint(Callback):
    def on_save_checkpoint(
        self,
        trainer: lightning.Trainer,
        pl_module: ECloudFlowTrainingModule,
        checkpoint: dict[str, object],
    ) -> None:
        """Attach reproducibility state to a Lightning checkpoint.

        :param trainer: Active trainer providing rank, epoch, and data state.
        :param pl_module: ECloudFlow module containing EMA and loss scalers.
        :param checkpoint: Mutable checkpoint dictionary written by Lightning.
        :return: None. The input dictionary is updated in place.
        :rtype: None
        :raises CheckpointStateError: If manifest or RNG state cannot be read.

        Every rank contributes RNG state through a distributed object gather;
        only rank zero writes the atomic checkpoint file.
        """
```

- [ ] **Step 4: Pass local resume and distributed smoke tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/training/test_stages.py tests/integration/test_checkpoint_resume.py -v`
Run: `conda run -n 3dmolecule python -m torch.distributed.run --standalone --nproc_per_node=2 -m pytest tests/distributed/test_ddp_smoke.py -q`
Expected: PASS with two Gloo ranks locally; NCCL marker remains server-only.

- [ ] **Step 5: Commit staged distributed training**

```bash
git add src/ecloudflow/training src/ecloudflow/config/schema.py configs/train configs/experiment tests/unit/training tests/integration/test_checkpoint_resume.py tests/distributed
git commit -m "feat: add staged resumable distributed training"
```

---

### Task 13: Cavity-Aware Priors, ODE Solvers, Score Corrector, and Fragment Clamping

**Files:**
- Create: `src/ecloudflow/sampling/__init__.py`
- Create: `src/ecloudflow/sampling/profiles.py`
- Create: `src/ecloudflow/sampling/priors.py`
- Create: `src/ecloudflow/sampling/solver.py`
- Create: `src/ecloudflow/sampling/corrector.py`
- Create: `tests/unit/sampling/test_priors.py`
- Create: `tests/unit/sampling/test_solver.py`
- Create: `tests/unit/sampling/test_corrector.py`
- Create: `tests/integration/test_fragment_invariance.py`

**Interfaces:**
- Consumes: trained model, pocket encoding, interpolant paths, projector, and fragment conditions.
- Produces: `CavityAwarePrior`, `EulerSolver`, `HeunSolver`, `ScoreCorrector`, `SamplingTrajectory`, `VectorFieldCallable`, `StateHook`, and profile factories.

- [ ] **Step 1: Write failing prior-support, solver-order, and per-step fragment tests**

```python
def test_cavity_prior_places_free_atoms_inside_supported_volume():
    prior = CavityAwarePrior(seed=11)
    state = prior.sample(pocket_condition_fixture(), num_atoms=24)
    support = pocket_condition_fixture().cavity.contains(state.positions)
    assert support.float().mean() > 0.95


def test_heun_is_more_accurate_than_euler_on_linear_field():
    initial = single_atom_state(x=1.0)
    field = molecular_state_linear_field()
    generator = torch.Generator().manual_seed(5)
    euler = EulerSolver(num_steps=8).integrate(initial, field, (), generator)
    generator = torch.Generator().manual_seed(5)
    heun = HeunSolver(num_steps=8).integrate(initial, field, (), generator)
    exact = torch.exp(torch.tensor(1.0))
    euler_error = abs(euler.final.positions[0, 0] - exact)
    heun_error = abs(heun.final.positions[0, 0] - exact)
    assert heun_error < euler_error


def test_fixed_fragment_is_bitwise_equal_in_every_saved_frame():
    trajectory = sample_tiny_fragment_trajectory(save_every_step=True)
    reference = trajectory.condition.fragment.reference
    for frame in trajectory.frames:
        assert_fragment_equal(frame, reference, bitwise=True)
```

- [ ] **Step 2: Run sampling tests and verify missing solvers**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/sampling tests/integration/test_fragment_invariance.py -v`
Expected: FAIL because sampling components do not exist.

- [ ] **Step 3: Implement cached conditional integration with projection hooks**

```python
class HeunSolver:
    def integrate(
        self,
        state: MolecularState,
        vector_field: VectorFieldCallable,
        hooks: Sequence[StateHook],
        generator: torch.Generator,
    ) -> SamplingTrajectory:
        """Integrate the joint probability-flow ODE with Heun updates.

        :param state: Initial prior state in the centered pocket frame.
        :param vector_field: Callable returning continuous velocities and
            categorical probability derivatives at a requested time.
        :param hooks: Ordered post-substep hooks for fragment clamping,
            chemistry projection, cavity support, and trajectory recording.
        :param generator: Device-matched generator controlling stochastic hooks.
        :return: Final state, network evaluation count, timings, diagnostics,
            and optionally retained trajectory frames.
        :rtype: SamplingTrajectory
        :raises SamplingNumericsError: If a substep becomes non-finite.

        Hooks run after both predictor and corrector evaluations. This ordering
        makes fixed-fragment equality an invariant instead of an endpoint fix.
        """
```

Cache pocket encodings outside time loops. Implement `fast=20 Euler/K0`, `balanced=40 Heun/K2`, `quality=100 Heun/K8` defaults. The score corrector uses a configured Langevin signal-to-noise ratio, edit masks, the same hooks, and a device generator. Record NFE and wall time.

- [ ] **Step 4: Pass solver, corrector, and fragment-invariant tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/sampling tests/integration/test_fragment_invariance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit numerical sampling core**

```bash
git add src/ecloudflow/sampling tests/unit/sampling tests/integration/test_fragment_invariance.py
git commit -m "feat: add constrained flow and score samplers"
```

---

### Task 14: CP-SAT Bond Decoder, RDKit Reconstruction, and Raw/Relaxed Separation

**Files:**
- Create: `src/ecloudflow/chemistry/decoder.py`
- Create: `src/ecloudflow/chemistry/reconstruct.py`
- Create: `src/ecloudflow/chemistry/relax.py`
- Create: `tests/unit/chemistry/test_decoder.py`
- Create: `tests/unit/chemistry/test_reconstruct.py`
- Create: `tests/integration/test_raw_relaxed_outputs.py`

**Interfaces:**
- Consumes: final probabilistic state, vocabulary, valence table, and optional fixed fragment.
- Produces: `ExactBondDecoder.decode`, `GreedyBondDecoder.decode`, `reconstruct_rdkit_molecule`, and `relax_molecule`.

- [ ] **Step 1: Write failing exact-feasibility, timeout, aromaticity, and pose-separation tests**

```python
def test_exact_decoder_maximizes_feasible_graph_and_preserves_fixed_bond():
    problem = carbonyl_fragment_decode_fixture()
    decoded = ExactBondDecoder(timeout_seconds=2.0).decode(problem)
    assert decoded.status == DecodeStatus.OPTIMAL
    assert decoded.bond_orders[problem.fixed_edge] == problem.fixed_bond_order
    assert decoded.connected
    assert decoded.valence_valid


def test_raw_pose_is_not_overwritten_by_relaxation(tmp_path):
    molecule = strained_but_valid_pose_fixture()
    raw_path, relaxed_path = write_raw_and_relaxed(molecule, tmp_path)
    raw = read_coordinates(raw_path)
    relaxed = read_coordinates(relaxed_path)
    assert not torch.equal(raw, relaxed)
    assert read_coordinates(raw_path).equal(raw)
```

- [ ] **Step 2: Run decoder/reconstruction tests and verify missing exact solver**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/chemistry/test_decoder.py tests/unit/chemistry/test_reconstruct.py tests/integration/test_raw_relaxed_outputs.py -v`
Expected: FAIL because exact decoding and artifact separation do not exist.

- [ ] **Step 3: Implement integer constraints, connectivity flow, and transparent fallback**

```python
class ExactBondDecoder:
    def decode(self, problem: BondDecodeProblem) -> BondDecodeResult:
        """Solve the highest-probability chemically feasible bond graph.

        :param problem: Atom/charge assignments, pair bond log probabilities,
            allowed valences, immutable bonds, and decoder limits.
        :return: Bond orders, objective, solver status, feasibility diagnostics,
            and whether the solution is optimal or merely feasible.
        :rtype: BondDecodeResult
        :raises ValueError: If fixed bonds already violate configured valence.

        CP-SAT variables select exactly one bond class per unordered pair.
        Linear valence constraints apply per atom. A single-commodity flow from
        atom zero enforces connectivity over selected non-zero bonds. Timeout
        returns the best feasible solution; no feasible solution is explicit.
        """
```

Convert feasible Kekulé bonds into an RDKit molecule, add conformer coordinates, sanitize, assign stereochemistry from 3D, and let RDKit perceive aromaticity. Implement constrained MMFF/UFF relaxation with fixed fragment atoms. The greedy fallback uses the same masks and returns `FALLBACK_FEASIBLE`, never `OPTIMAL`.

- [ ] **Step 4: Pass exact decoder and artifact tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/chemistry tests/integration/test_raw_relaxed_outputs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit exact chemical decoding**

```bash
git add src/ecloudflow/chemistry tests/unit/chemistry tests/integration/test_raw_relaxed_outputs.py
git commit -m "feat: decode exact connected molecular graphs"
```

---

### Task 15: High-Level Generation Pipeline and Bounded Unique Yield

**Files:**
- Create: `src/ecloudflow/pipeline.py`
- Create: `src/ecloudflow/sampling/pipeline.py`
- Create: `src/ecloudflow/sampling/results.py`
- Create: `tests/unit/sampling/test_results.py`
- Create: `tests/integration/test_generation_pipeline.py`
- Create: `tests/integration/test_all_generation_modes.py`

**Interfaces:**
- Consumes: model/checkpoint, parsers, priors, solvers, projector, decoder, and optional relaxer.
- Produces: `ECloudFlowPipeline.from_pretrained`, `ECloudFlowPipeline.generate`, `GenerationMode`, `GenerationResult`, `GenerationRecord`, `GenerationAttempt`, and mode-specific requests. `GenerationResult` exposes `to_excel(path)` through the shared output writer.

- [ ] **Step 1: Write failing bounded-yield, deduplication, and all-mode tests**

```python
def test_pipeline_returns_target_unique_valid_count_with_bounded_attempts(tmp_path):
    pipeline = deterministic_stub_pipeline(sequence=["CCO", "CCO", "CCN", "invalid"])
    result = pipeline.generate(
        pocket=toy_pocket_path(),
        num_molecules=2,
        max_attempts=4,
        output_dir=tmp_path,
    )
    assert [record.canonical_smiles for record in result.valid] == ["CCO", "CCN"]
    assert result.attempts == 3
    assert result.duplicate_count == 1


def test_one_checkpoint_supports_all_generation_modes():
    pipeline = tiny_real_pipeline()
    for request in de_novo_grow_link_replace_merge_requests():
        result = pipeline.generate_request(request)
        assert result.model_checkpoint_hash == pipeline.checkpoint_hash
        if request.fragment is not None:
            assert_exact_fragment_preservation(result, request.fragment)
```

- [ ] **Step 2: Run generation integration tests and verify missing service**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/sampling/test_results.py tests/integration/test_generation_pipeline.py tests/integration/test_all_generation_modes.py -v`
Expected: FAIL because high-level generation services do not exist.

- [ ] **Step 3: Implement one typed pipeline with explicit attempt statuses**

```python
class ECloudFlowPipeline:
    def generate(
        self,
        pocket: str | Path,
        num_molecules: int,
        fragment: str | Path | None = None,
        mode: GenerationMode = GenerationMode.DE_NOVO,
        profile: str = "balanced",
        max_attempts: int | None = None,
        output_dir: str | Path | None = None,
        seed: int = 2026,
    ) -> GenerationResult:
        """Generate valid unique ligands directly in a protein pocket.

        :param pocket: Pocket PDB path in the desired output coordinate frame.
        :param num_molecules: Target count of valid unique canonical isomeric
            SMILES after exact graph decoding and RDKit sanitization.
        :param fragment: Optional positioned SDF for grow/link/replace/merge.
        :param mode: De novo or fragment-conditioned generation task.
        :param profile: Named fast, balanced, or quality numerical preset.
        :param max_attempts: Bounded attempt count; defaults to five times the
            target count.
        :param output_dir: Optional run directory for atomic artifacts.
        :param seed: Master seed used to derive per-attempt device generators.
        :return: Valid, rejected, and failed attempts with timing/provenance.
        :rtype: GenerationResult
        :raises GenerationShortfallError: Only when strict-count mode is enabled
            and the bounded attempt budget cannot meet the requested count.
        """
```

Deduplicate by canonical isomeric SMILES before counting. Keep all attempts and structured reasons. Use stable temporary attempt IDs until docking/ranking. Save resolved config, pocket encoding cache key, checkpoint hash, and raw/relaxed SDFs atomically.

- [ ] **Step 4: Pass pipeline and generation-mode tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/sampling tests/integration/test_generation_pipeline.py tests/integration/test_all_generation_modes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit the public generation service**

```bash
git add src/ecloudflow/pipeline.py src/ecloudflow/sampling tests/unit/sampling tests/integration/test_generation_pipeline.py tests/integration/test_all_generation_modes.py
git commit -m "feat: expose bounded multi-mode generation pipeline"
```

---

### Task 16: Docking Adapter, Deterministic Ranking, and Tabular/SDF Outputs

**Files:**
- Create: `src/ecloudflow/docking/__init__.py`
- Create: `src/ecloudflow/docking/base.py`
- Create: `src/ecloudflow/docking/vina.py`
- Create: `src/ecloudflow/evaluation/__init__.py`
- Create: `src/ecloudflow/evaluation/ranking.py`
- Create: `src/ecloudflow/evaluation/outputs.py`
- Create: `tests/unit/docking/test_vina.py`
- Create: `tests/unit/evaluation/test_ranking.py`
- Create: `tests/integration/test_ranked_outputs.py`
- Modify: `src/ecloudflow/pipeline.py`

**Interfaces:**
- Consumes: valid generation records and original pocket coordinates.
- Produces: `DockingBackend`, `VinaBackend.score`, `rank_molecules`, `assign_rank_ids`, and `write_ranked_outputs`.

- [ ] **Step 1: Write failing score-order, tie-break, ID, failed-dock, and workbook tests**

```python
def test_ranking_uses_vina_qed_sa_smiles_and_no_eclf():
    records = [
        record("B", vina=-9.0, qed=0.7, sa=2.0),
        record("A", vina=-10.0, qed=0.6, sa=3.0),
        record("C", vina=-9.0, qed=0.8, sa=4.0),
        record("D", vina=None, qed=0.9, sa=1.0, status="dock_failed"),
    ]
    ranked, unranked = rank_molecules("3ZTX", records)
    assert [item.canonical_smiles for item in ranked] == ["A", "C", "B"]
    assert [item.molecule_id for item in ranked] == [
        "3ZTX-000001", "3ZTX-000002", "3ZTX-000003"
    ]
    assert all("ECLF" not in item.molecule_id for item in ranked)
    assert unranked[0].temporary_id == records[-1].temporary_id


def test_output_bundle_has_required_tables_and_sdf_order(tmp_path):
    bundle = write_ranked_outputs(ranked_fixture(), failed_fixture(), tmp_path)
    assert {path.name for path in bundle.paths} >= {
        "samples.csv", "samples.parquet", "summary.xlsx",
        "ranked.sdf", "summary.json",
    }
    assert excel_sheet_names(tmp_path / "summary.xlsx") == {
        "ranked", "failed", "aggregate"
    }
    assert sdf_ids(tmp_path / "ranked.sdf") == ["3ZTX-000001", "3ZTX-000002"]
```

- [ ] **Step 2: Run docking/ranking tests and verify missing modules**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/docking tests/unit/evaluation/test_ranking.py tests/integration/test_ranked_outputs.py -v`
Expected: FAIL because docking, ranking, and outputs are absent.

- [ ] **Step 3: Implement safe Vina execution and exact output schema**

```python
def rank_molecules(
    pocket_id: str,
    records: Sequence[GenerationRecord],
) -> tuple[list[RankedMolecule], list[GenerationRecord]]:
    """Rank successfully docked unique molecules and assign formal IDs.

    :param pocket_id: Sanitized pocket identifier used as the ID prefix.
    :param records: Valid unique generation records with SA, QED, docking status,
        and canonical isomeric SMILES.
    :return: Ranked molecules and unranked docking failures.
    :rtype: tuple[list[RankedMolecule], list[GenerationRecord]]
    :raises ValueError: If pocket ID is empty or a score is non-finite.

    Sorting is ascending Vina score, descending QED, ascending conventional
    SA score, then lexicographic canonical isomeric SMILES. Failed or disabled
    docking retains temporary IDs and never receives a formal rank ID.
    """
```

The Vina adapter uses argument lists, recorded box center/size, timeout, version, and raw output. Write typed Parquet, convenient CSV, Excel sheets, JSON aggregates, and rank-ordered SDF properties. Compute count/mean/std/median/min/max/Q1/Q3 for SA, QED, and Vina.

- [ ] **Step 4: Pass docking mock and output tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/docking tests/unit/evaluation/test_ranking.py tests/integration/test_ranked_outputs.py -v`
Expected: PASS without requiring a local Vina binary; real Vina remains an external marker.

- [ ] **Step 5: Commit docking ranking and outputs**

```bash
git add src/ecloudflow/pipeline.py src/ecloudflow/docking src/ecloudflow/evaluation tests/unit/docking tests/unit/evaluation tests/integration/test_ranked_outputs.py
git commit -m "feat: rank generated ligands by docking score"
```

---

### Task 17: Metric Registry and Seven-Domain Evaluation Protocol

**Files:**
- Create: `src/ecloudflow/evaluation/types.py`
- Create: `src/ecloudflow/evaluation/registry.py`
- Create: `src/ecloudflow/evaluation/chemistry.py`
- Create: `src/ecloudflow/evaluation/distribution.py`
- Create: `src/ecloudflow/evaluation/geometry.py`
- Create: `src/ecloudflow/evaluation/binding.py`
- Create: `src/ecloudflow/evaluation/ecloud.py`
- Create: `src/ecloudflow/evaluation/conditional.py`
- Create: `src/ecloudflow/evaluation/efficiency.py`
- Create: `src/ecloudflow/evaluation/aggregate.py`
- Modify: `src/ecloudflow/config/schema.py`
- Create: `tests/unit/evaluation/test_registry.py`
- Create: `tests/unit/evaluation/test_metrics.py`
- Create: `tests/unit/evaluation/test_aggregate.py`
- Create: `tests/integration/test_evaluation_pipeline.py`

**Interfaces:**
- Consumes: ranked/unranked molecules, pockets, references, fields, timing, and training-set indexes.
- Produces: `EvaluationConfig`, `Metric`, `MetricRegistry`, `EvaluationContext`, seven metric groups, `EvaluationResult`, `bootstrap_macro_summary`, and `evaluate_run`. `EvaluationConfig` selects metric groups, optional backends, bootstrap seed/count, reference data, and raw/relaxed policy.

- [ ] **Step 1: Write failing registry, macro-average, confidence, and missing-tool tests**

```python
def test_registry_exposes_all_seven_metric_domains():
    registry = MetricRegistry.default()
    assert set(registry.groups) == {
        "chemistry", "distribution", "geometry", "binding",
        "ecloud", "conditional", "efficiency",
    }


def test_macro_average_weights_pockets_equally():
    rows = metric_rows({"small-pocket": [1.0], "large-pocket": [0.0] * 20})
    summary = bootstrap_macro_summary(rows, value="valid", seed=19, resamples=500)
    assert summary.mean == 0.5
    assert summary.ci_low <= summary.mean <= summary.ci_high


def test_optional_vina_metric_reports_unavailable_without_fake_value():
    metric = VinaScoreMetric(backend=missing_vina_backend())
    result = metric.compute(single_record_fixture())
    assert result.status == MetricStatus.UNAVAILABLE
    assert result.value is None
```

- [ ] **Step 2: Run evaluation tests and verify missing registry**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/evaluation tests/integration/test_evaluation_pipeline.py -v`
Expected: FAIL because evaluation registry and metric groups are incomplete.

- [ ] **Step 3: Implement typed metrics with per-molecule and per-pocket outputs**

```python
class Metric(Protocol):
    name: str
    group: str

    def compute(self, context: EvaluationContext) -> MetricResult:
        """Compute one metric without mutating generated molecules.

        :param context: Read-only generated molecule, pocket, reference,
            electron-field, timing, and provenance inputs.
        :return: Typed scalar/vector value, status, units, and diagnostics.
        :rtype: MetricResult
        :raises MetricInputError: If required in-memory fields are malformed.

        Missing optional external software returns ``UNAVAILABLE``. It never
        returns zero, NaN-as-success, or a value from another molecule.
        """
```

Implement RDKit validity, PoseBusters adapter, valence/connectivity, uniqueness/novelty/diversity, descriptor JSD/FCD adapter, bond/angle/dihedral JSD, ring/clash/strain/raw-relaxed RMSD, Vina/PLIF/shape contacts, density/count/dipole/cycle metrics, property/fragment success, NFE/time/memory/yield, and 1/2/4 GPU scaling fields. Aggregate by pocket with seeded bootstrap 95% CIs and retain per-molecule records.

- [ ] **Step 4: Pass evaluation and no-mutation tests**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/evaluation tests/integration/test_evaluation_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit evaluation protocol**

```bash
git add src/ecloudflow/evaluation src/ecloudflow/config/schema.py tests/unit/evaluation tests/integration/test_evaluation_pipeline.py
git commit -m "feat: evaluate molecular quality across seven domains"
```

---

### Task 18: Interactive Molecular/Electron Viewer and Publication Reports

**Files:**
- Create: `src/ecloudflow/visualization/__init__.py`
- Create: `src/ecloudflow/visualization/molecule.py`
- Create: `src/ecloudflow/visualization/fields.py`
- Create: `src/ecloudflow/visualization/plots.py`
- Create: `src/ecloudflow/visualization/report.py`
- Create: `src/ecloudflow/visualization/styles/ecloudflow.mplstyle`
- Create: `src/ecloudflow/visualization/templates/report.html.j2`
- Modify: `src/ecloudflow/config/schema.py`
- Create: `tests/unit/visualization/test_viewer.py`
- Create: `tests/unit/visualization/test_plots.py`
- Create: `tests/integration/test_html_report.py`

**Interfaces:**
- Consumes: evaluation results, ranked artifacts, pocket structures, electron fields, and trajectories.
- Produces: `VisualizationConfig`, `render_complex_html`, `render_electron_field_html`, plot functions, `ReportBundle`, and `build_report`. `VisualizationConfig` owns themes, field isovalues, top-N limits, dimensions, formats, DPI, and deterministic seed.

- [ ] **Step 1: Write failing self-contained HTML and deterministic figure tests**

```python
def test_complex_viewer_contains_ranked_molecule_fragment_and_density(tmp_path):
    path = render_complex_html(viewer_fixture(), tmp_path / "complex.html")
    html = path.read_text(encoding="utf-8")
    assert "3ZTX-000001" in html
    assert "fixed-fragment" in html
    assert "ligand-density-isosurface" in html
    assert "raw-pose" in html and "relaxed-pose" in html


def test_publication_plot_is_deterministic_and_exports_vector(tmp_path):
    first = plot_quality_speed_pareto(plot_fixture(), tmp_path / "first.svg", seed=3)
    second = plot_quality_speed_pareto(plot_fixture(), tmp_path / "second.svg", seed=3)
    assert normalized_svg_hash(first) == normalized_svg_hash(second)
```

- [ ] **Step 2: Run visualization tests and verify missing renderers**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/visualization tests/integration/test_html_report.py -v`
Expected: FAIL because viewers and reports do not exist.

- [ ] **Step 3: Implement colorblind-safe interactive and static outputs**

```python
def build_report(
    evaluation: EvaluationResult,
    output_dir: Path,
    top_n: int = 20,
) -> ReportBundle:
    """Build a self-contained scientific HTML and publication figure bundle.

    :param evaluation: Per-molecule, per-pocket, aggregate, confidence, and
        provenance results from the metric registry.
    :param output_dir: Destination for HTML, SVG, PDF, PNG, and data tables.
    :param top_n: Number of ranked molecules shown with 2D/3D panels.
    :return: Paths and content hashes for every generated report artifact.
    :rtype: ReportBundle
    :raises ReportError: If required ranked artifacts are missing or corrupt.

    Plot functions use fixed seeds, explicit dimensions, colorblind-safe
    palettes, embedded fonts where supported, and never recompute metrics.
    """
```

Use Py3Dmol for pocket/ligand/interaction views and Plotly for field isosurfaces/trajectory controls. Provide ECDF/violin with CIs, paired pocket plots, geometry heatmaps, property distributions, top-molecule grids, and quality-speed Pareto figures. Export SVG/PDF/300-DPI PNG. Keep metric data downloadable in the HTML.

- [ ] **Step 4: Pass visual artifact tests and inspect generated fixtures**

Run: `conda run -n 3dmolecule python -m pytest tests/unit/visualization tests/integration/test_html_report.py -v`
Expected: PASS and produce a readable fixture report under pytest temporary output.

- [ ] **Step 5: Commit visualization and reports**

```bash
git add src/ecloudflow/visualization src/ecloudflow/config/schema.py tests/unit/visualization tests/integration/test_html_report.py
git commit -m "feat: visualize pockets electron fields and metrics"
```

---

### Task 19: Unified Typer CLI, Doctor Command, and End-to-End Smoke Workflow

**Files:**
- Create: `src/ecloudflow/cli/__init__.py`
- Create: `src/ecloudflow/cli/main.py`
- Create: `src/ecloudflow/cli/doctor.py`
- Create: `src/ecloudflow/cli/data.py`
- Create: `src/ecloudflow/cli/train.py`
- Create: `src/ecloudflow/cli/sample.py`
- Create: `src/ecloudflow/cli/evaluate.py`
- Create: `src/ecloudflow/cli/report.py`
- Create: `src/ecloudflow/cli/config.py`
- Create: `tests/cli/test_doctor.py`
- Create: `tests/cli/test_commands.py`
- Create: `tests/integration/test_cli_end_to_end.py`

**Interfaces:**
- Consumes: all package services from Tasks 1–18.
- Produces: `ecloudflow` console entry point with `doctor`, `config`, `data`, `train`, `sample`, `evaluate`, `visualize`, and `report` commands.

- [ ] **Step 1: Write failing help, config, sample-count, and smoke tests**

```python
from typer.testing import CliRunner

from ecloudflow.cli.main import app

runner = CliRunner()


def test_sample_help_exposes_simple_count_and_fragment_options():
    result = runner.invoke(app, ["sample", "--help"])
    assert result.exit_code == 0
    assert "--num-molecules" in result.stdout
    assert "--fragment" in result.stdout
    assert "--profile" in result.stdout


def test_cli_tiny_end_to_end(tmp_path):
    result = runner.invoke(app, [
        "sample", str(toy_pocket_path()), "-n", "2",
        "--checkpoint", str(tiny_checkpoint_path()),
        "--output-dir", str(tmp_path), "--profile", "fast",
    ])
    assert result.exit_code == 0, result.stdout
    evaluate = runner.invoke(app, ["evaluate", str(tmp_path), "--profile", "smoke"])
    report = runner.invoke(app, ["report", str(tmp_path)])
    assert evaluate.exit_code == report.exit_code == 0
    assert (tmp_path / "samples.csv").exists()
    assert (tmp_path / "report.html").exists()
```

- [ ] **Step 2: Run CLI tests and verify missing application entry point**

Run: `conda run -n 3dmolecule python -m pytest tests/cli tests/integration/test_cli_end_to_end.py -v`
Expected: FAIL because CLI modules are absent.

- [ ] **Step 3: Implement thin commands and comprehensive doctor checks**

```python
@app.command()
def sample(
    pocket: Path,
    num_molecules: Annotated[int, typer.Option("--num-molecules", "-n", min=1)] = 100,
    fragment: Annotated[Path | None, typer.Option()] = None,
    profile: Annotated[str, typer.Option()] = "balanced",
    checkpoint: Annotated[Path, typer.Option(exists=True)] = Path("checkpoints/ecloudflow.ckpt"),
    output_dir: Annotated[Path, typer.Option()] = Path("runs/sample"),
) -> None:
    """Generate, dock, rank, and summarize ligands for one pocket.

    :param pocket: Input pocket PDB in the desired output coordinate frame.
    :param num_molecules: Target valid unique molecule count.
    :param fragment: Optional positioned fragment SDF.
    :param profile: Fast, balanced, or quality sampling profile.
    :param checkpoint: Trained ECloudFlow checkpoint.
    :param output_dir: Atomic run artifact directory.
    :return: None. A concise summary and artifact paths are printed.
    :rtype: None
    """
```

Commands delegate to package services and share config composition. `doctor` checks Python/dependencies, CUDA/GPU, xTB, Vina, OpenBabel when configured, dataset paths, checkpoint schema, and writable output. `config show` prints resolved YAML; `config explain` reads Pydantic field descriptions. Keep advanced Hydra overrides as trailing `key=value` arguments.

The end-to-end CLI test injects the deterministic docking backend through the smoke configuration, so it verifies ranking and `samples.csv` creation without requiring a local Vina executable. Real Vina execution remains covered by the `external` marker.

- [ ] **Step 4: Pass CLI and complete tiny workflow tests**

Run: `conda run -n 3dmolecule python -m pytest tests/cli tests/integration/test_cli_end_to_end.py -v`
Run: `conda run -n 3dmolecule ecloudflow doctor --json`
Expected: tests PASS; doctor reports installed/missing optional tools without crashing.

- [ ] **Step 5: Commit the unified user interface**

```bash
git add src/ecloudflow/cli pyproject.toml tests/cli tests/integration/test_cli_end_to_end.py
git commit -m "feat: add unified ECloudFlow command line workflow"
```

---

### Task 20: Server Distributed Validation and Performance Regression Harness

**Files:**
- Create: `scripts/run_h100_smoke.sh`
- Create: `scripts/benchmark_scaling.sh`
- Create: `src/ecloudflow/training/benchmark.py`
- Create: `tests/distributed/test_nccl_training.py`
- Create: `tests/distributed/test_shard_coverage.py`
- Create: `tests/performance/test_sampling_nfe.py`
- Create: `tests/performance/test_memory_regression.py`
- Create: `configs/experiment/h100_smoke.yaml`
- Create: `configs/experiment/h100_large.yaml`
- Modify: `src/ecloudflow/config/schema.py`

**Interfaces:**
- Consumes: training and sampling CLIs plus server data/checkpoints.
- Produces: `BenchmarkConfig`, `ScalingReport`, one-command 4×H100 smoke, 1/2/4 GPU scaling JSON/CSV, NFE report, and memory regression records. `BenchmarkConfig` owns warm-up/measurement steps, fixed global workload, device counts, tolerances, and artifact destination.

- [ ] **Step 1: Write failing distributed coverage and benchmark-schema tests**

```python
def test_scaling_report_contains_all_required_measurements(tmp_path):
    report = benchmark_scaling(
        devices=[1],
        steps=2,
        config="experiment=h100_smoke",
        output_dir=tmp_path,
    )
    row = report.rows[0]
    assert row.devices == 1
    assert row.samples_per_second > 0
    assert row.peak_memory_bytes > 0
    assert row.speedup == 1.0
    assert row.scaling_efficiency == 1.0


def test_sampling_profiles_record_expected_nfe():
    assert measured_stub_nfe("fast") == 20
    assert measured_stub_nfe("balanced") == 82
    assert measured_stub_nfe("quality") == 208
```

- [ ] **Step 2: Run local performance-schema tests and verify missing harness**

Run: `conda run -n 3dmolecule python -m pytest tests/performance tests/distributed/test_shard_coverage.py -v`
Expected: FAIL because benchmark utilities do not exist.

- [ ] **Step 3: Implement reproducible server scripts and measurements**

`run_h100_smoke.sh` must use `set -euo pipefail`, print Git/config/data hashes, run `torchrun --nproc_per_node=4`, and write results under an explicit output directory. `benchmark_scaling.sh` runs identical global workloads at 1, 2, and 4 GPUs. Measure steady-state examples/s, optimizer steps/s, peak allocated/reserved GPU memory, communication time when available, speedup, efficiency, NFE, valid yield, and GPU-hours.

```python
def benchmark_scaling(
    devices: Sequence[int],
    steps: int,
    config: str,
    output_dir: Path,
) -> ScalingReport:
    """Benchmark identical global work across requested GPU counts.

    :param devices: Positive GPU counts, normally ``[1, 2, 4]`` on H100.
    :param steps: Measured optimizer steps after warm-up.
    :param config: Resolved experiment override used for every run.
    :param output_dir: Destination for raw logs, JSON, CSV, and environment data.
    :return: Throughput, memory, speedup, and scaling-efficiency rows.
    :rtype: ScalingReport
    :raises BenchmarkError: If a run fails or changes global workload shape.

    Warm-up samples are excluded. Global batch and model remain fixed so the
    result measures strong scaling rather than a changing scientific workload.
    """
```

- [ ] **Step 4: Pass local harness tests, then run server-only NCCL commands**

Local run: `conda run -n 3dmolecule python -m pytest tests/performance tests/distributed/test_shard_coverage.py -v`
Server run: `bash scripts/run_h100_smoke.sh --config experiment=h100_smoke --output runs/h100-smoke`
Server run: `bash scripts/benchmark_scaling.sh --config experiment=h100_large --output runs/scaling`
Expected: local tests PASS; server commands produce 4-rank NCCL success and 1/2/4 GPU reports.

- [ ] **Step 5: Commit distributed validation harness**

```bash
git add scripts src/ecloudflow/training/benchmark.py src/ecloudflow/config/schema.py tests/distributed tests/performance configs/experiment
git commit -m "test: add H100 distributed and performance validation"
```

---

### Task 21: README, Theory, Operational Documentation, Attribution, and Final Verification

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `docs/theory.md`
- Create: `docs/data.md`
- Create: `docs/training.md`
- Create: `docs/sampling.md`
- Create: `docs/evaluation.md`
- Create: `docs/configuration.md`
- Create: `docs/distributed.md`
- Create: `docs/visualization.md`
- Create: `docs/reproducibility.md`
- Create: `examples/python_api.py`
- Create: `tests/documentation/test_docs.py`
- Create: `tests/documentation/test_readme_commands.py`

**Interfaces:**
- Consumes: every implemented CLI/API and the approved design specification.
- Produces: complete GitHub-style usage and mathematical documentation with executable commands.

- [ ] **Step 1: Write failing documentation completeness and command tests**

```python
from pathlib import Path


def test_required_docs_exist_and_have_no_incomplete_markers():
    required = [
        "README.md", "docs/theory.md", "docs/data.md", "docs/training.md",
        "docs/sampling.md", "docs/evaluation.md", "docs/configuration.md",
        "docs/distributed.md", "docs/visualization.md", "docs/reproducibility.md",
        "THIRD_PARTY_NOTICES.md",
    ]
    forbidden = ("T" + "ODO", "T" + "BD", "FIX" + "ME", "implement " + "later")
    for name in required:
        text = Path(name).read_text(encoding="utf-8")
        assert len(text.splitlines()) >= 20
        assert not any(marker in text for marker in forbidden)


def test_readme_commands_are_recognized_by_cli():
    for command in extract_ecloudflow_commands(Path("README.md")):
        result = run_command_help_variant(command)
        assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run documentation tests and verify missing deliverables**

Run: `conda run -n 3dmolecule python -m pytest tests/documentation -v`
Expected: FAIL because operational documents do not exist.

- [ ] **Step 3: Write complete README, derivations, workflows, and notices**

README order: model summary, installation, `doctor`, tiny quick start, data preparation, four training stages, 4×H100 launch, de novo sampling, grow/link/replace/merge examples, `-n/--num-molecules`, docking ranking and IDs, evaluation/reporting, Python API, config overrides, troubleshooting, repository map, limitations, citations.

`docs/theory.md` derives the radial-spherical density representation, SE(3) contract, continuous velocity and score targets, categorical path, graph–field cycle, chemistry losses, fragment invariants, solver/corrector, CP-SAT graph constraints, and guidance. Each operational document uses commands verified against the CLI. Write `LICENSE` with the full MIT license text for original ECloudFlow code. `THIRD_PARTY_NOTICES.md` records adapted DiffGui, ECloudGen, PropMolFlow, JODO, and CoCoGraph elements by file and license/source; copied or modified upstream files retain their original headers where required.

```python
# examples/python_api.py
from ecloudflow import ECloudFlowPipeline

pipeline = ECloudFlowPipeline.from_pretrained("checkpoints/ecloudflow-large.ckpt")
result = pipeline.generate(
    pocket="examples/3ztx_pocket.pdb",
    fragment=None,
    num_molecules=100,
    profile="balanced",
    output_dir="runs/3ztx",
)
result.to_excel("runs/3ztx/summary.xlsx")
```

- [ ] **Step 4: Run the complete verification matrix**

Run: `conda run -n 3dmolecule python -m pytest tests/unit tests/integration tests/cli tests/documentation -v`
Run: `conda run -n 3dmolecule python -m pytest tests/distributed/test_shard_coverage.py tests/performance -v`
Run: `conda run -n 3dmolecule python tools/check_python_docs.py src tests`
Run: `conda run -n 3dmolecule python -m ruff check src tests tools`
Run: `conda run -n 3dmolecule python -m mypy src/ecloudflow`
Run: `conda run -n 3dmolecule ecloudflow doctor --json`
Expected: all non-external local checks PASS. External xTB/Vina and 4×H100 checks report their explicit status and are run on the server before publishing measured results.

- [ ] **Step 5: Perform manual artifact review**

Open one `report.html`, one molecule/electron viewer, `summary.xlsx`, `ranked.sdf`, and every SVG/PDF figure. Verify IDs, ordering, fragment highlighting, raw/relaxed separation, axis labels/units, confidence intervals, and download links. Record the review in `runs/verification/manual-review.json` with artifact hashes.

- [ ] **Step 6: Commit complete documentation and verification assets**

```bash
git add README.md LICENSE THIRD_PARTY_NOTICES.md docs examples tests/documentation
git commit -m "docs: document complete ECloudFlow workflows"
```

## Specification Coverage Map

| Approved design area | Implementation tasks | Verification evidence |
|---|---:|---|
| Product scope, CLI, output IDs, and bounded sample count | 1, 15, 16, 19, 21 | Config, pipeline, ranking, CLI, and README command tests |
| Canonical tensors, pocket frames, SE(3) equivariance, and binding-pose output | 2, 6, 10, 13, 15 | Frame round-trip, equivariance, fragment-invariance, and global-coordinate tests |
| Pocket/ligand electron fields and equivariant latent tokens | 4, 5, 8, 10, 11 | Rotation, density conservation, cache provenance, autoencoder, and loss tests |
| Joint flow, categorical paths, diffusion score correction, and fast profiles | 9, 10, 11, 13, 20 | Path endpoint, solver order, NFE, objective, and performance tests |
| De novo plus grow/link/replace/merge fragment optimization | 2, 6, 13, 15, 19 | Mode construction, bitwise clamping, pipeline, and CLI tests |
| Chemical validity during trajectories and exact final graph decoding | 3, 13, 14 | Valence projection, hook-order, CP-SAT, sanitize, and raw/relaxed tests |
| PDBBind/CrossDocked preprocessing, leakage control, and distributed data | 5, 6, 7, 12, 20 | Parser, grouped split, shard coverage, resume, and multi-rank tests |
| Four-stage BF16/DDP training on 4×H100 with reproducible checkpoints | 11, 12, 20 | Training-step, resume equivalence, Gloo/NCCL, and scaling reports |
| Docking, SA/QED/SMILES tables, deterministic sorting, and multi-format outputs | 15, 16 | Ranking tie-break, ID, workbook, Parquet/CSV/SDF/JSON tests |
| Seven-domain evaluation with uncertainty and honest unavailable states | 16, 17, 20 | Metric registry, macro bootstrap, missing-tool, and efficiency tests |
| Molecule/electron-cloud visualization and paper-quality reports | 18, 19, 21 | Deterministic SVG, self-contained HTML, artifact, and manual-review gates |
| English code comments and detailed explanatory core docstrings | 1 and every task | AST/token documentation gate plus Ruff/Mypy/pytest matrix |
| Complete theory, formulas, workflows, attribution, and reproducibility docs | 21 | Documentation completeness, executable README commands, and final matrix |

Every acceptance item in the approved specification maps to at least one automated test or an explicitly named server/manual verification artifact; there are no intentionally deferred implementation areas in this plan.

## Final Completion Gate

Before claiming completion:

1. Run `git status --short` and ensure only intentional user-owned files remain.
2. Run the Task 21 verification matrix from a clean checkout in `3dmolecule`.
3. Run the tiny train → sample → evaluate → report workflow and capture artifact paths.
4. On the server, run the 4-rank NCCL smoke and 1/2/4 H100 scaling harness.
5. Compare every acceptance item in the design specification against a passing test, command output, or generated artifact.
6. Report empirical model quality only from completed checkpoints and recorded benchmark runs; label unrun full-scale experiments as pending experiments rather than results.
