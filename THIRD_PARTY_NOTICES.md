# Third-party notices

ECloudFlow is original research and engineering code released under the MIT
license in this repository. Dependencies remain under their own licenses; the
lockfile/environment and each dependency's distribution metadata are the
authoritative record for installed versions.

The design and implementation are informed by the following published systems.
These are conceptual references, not bundled source code:

| Reference | Use in ECloudFlow | Attribution |
| --- | --- | --- |
| DiffGui | pocket/ligand preprocessing and diffusion framing | Hu et al., *Nature Communications* 16, 7928 (2025), DOI 10.1038/s41467-025-63245-0 |
| ECloudGen | electron-cloud latent representation | Zhang et al., *Nature Computational Science* 5, 1017-1028 (2025), DOI 10.1038/s43588-025-00886-7 |
| PropMolFlow | flow-matching and property-conditioning ideas | *Nature Computational Science* (2025), DOI 10.1038/s43588-025-00946-y |
| JODO | joint 2D/3D categorical and coordinate generation | Huang et al., arXiv:2305.12347 |
| CoCoGraph | constrained graph diffusion and feasibility checks | Ruiz-Botella et al., arXiv:2505.16365 |
| FLOWR | interaction- and fragment-conditioned flow perspective | Cremer et al., *Nature Computational Science* 6, 565-574 (2026), DOI 10.1038/s43588-026-00998-8 |

No upstream repository files are copied into the source tree. If an adapted
file is added in a future release, its original copyright header and license
will remain in that file and a row identifying the path, source commit, and
license will be added here. RDKit, PyTorch, PyTorch Geometric, e3nn, Lightning,
Hydra, Pydantic, Typer, NumPy, SciPy, Biopython, WebDataset, Matplotlib,
Plotly, Py3Dmol, pandas, pyarrow, and openpyxl are used through their public
APIs and are not relicensed by this project. Vina, xTB, PoseBusters, and
OpenBabel are optional external executables with separate terms; users must
install and review those terms before enabling them.

Attribution is included for scientific context and does not imply endorsement,
compatibility, or performance equivalence. Consult the cited publications and
upstream licenses when redistributing a trained checkpoint or dataset.
