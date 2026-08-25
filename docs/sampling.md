# Sampling and fragment workflows

Sampling starts from a cavity/graph prior and integrates the joint state in a
centered pocket frame. The constrained pipeline supports de novo, grow, link,
replace, and merge modes through one API. Every attempt is classified as
`valid`, `rejected`, or `failed` with a reason code.

```bash
ecloudflow sample pocket.pdb --checkpoint checkpoints/ecloudflow-large.ckpt \
  --num-molecules 100 --profile balanced --output-dir runs/pocket
ecloudflow sample pocket.pdb --fragment hit_fragment.sdf --mode grow -n 100 \
  --profile balanced --output-dir runs/grow
ecloudflow sample pocket.pdb --fragment fragment_a.sdf \
  --fragment fragment_b.sdf --mode link -n 100
```

`--smoke --docking deterministic` is an explicit wiring fixture and does not
represent docking physics. Normal runs use `--docking auto` or `--docking vina`
when a Vina executable and receptor preparation are available.

## Profiles and budgets

`fast` uses Euler with 20 steps, `balanced` uses Heun with 40 steps plus a
terminal two-step corrector (nominal NFE 82), and `quality` uses 100 Heun steps
with a stronger corrector (nominal NFE 208). The target is valid unique
molecules; attempts are bounded by five times the target by default. A
shortfall is published with duplicate and rejection statistics, and
`--strict-count` turns it into a command failure after artifacts are written.

## Fragment invariants

The positioned fragment(s) are represented by a boolean atom mask. Fixed atom
types, charges, internal bonds, and coordinates are restored after every
integrator, score corrector, and projection call. New attachment edges are
allowed only through the attachment mask. For an SDF fragment, the default
mask is fail-closed: atoms with available implicit/explicit hydrogens are
inferred as sites, saturated atoms are blocked, and optional atom properties
`ecloudflow_attachment`, `attachment_site`, or `attachment` can explicitly
mark a site (`true`/`false`). This prevents a complete candidate graph from
connecting to arbitrary interior atoms. Exact clamping is tested by tensor
equality, not by a tolerance-based post-hoc repair.

## Pocket electron field

Checkpoint-backed inference builds the deterministic density, partial-charge,
donor, acceptor, hydrophobic, and aromatic pocket field from the canonical
pocket graph. The same framed field conditions atom-count prediction and every
flow/score network evaluation. Field construction is deterministic and does
not invoke xTB; optional ligand QM fields remain training supervision.

## Artifacts

Raw pose SDF files and optional MMFF/UFF relaxed SDF files are separate. The
ranker assigns `<POCKET_ID>-<RANK:06d>` IDs after docking and deterministic
tie-breaking. Failed docking and invalid attempts remain inspectable in
`failed.csv` and `generation.json`.
