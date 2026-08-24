"""Self-contained HTML views for pockets, ligands, fragments, and poses."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


def render_complex_html(viewer: Any, path: str | Path) -> Path:
    """Render a self-contained complex viewer with explicit pose layers.

    :param viewer: Ranked row, mapping, or object carrying molecule/pocket and
        optional raw/relaxed/fragment/density data.
    :param path: Destination HTML path; parent directories are created.
    :return: Written HTML path.
    :rtype: pathlib.Path

    The generated document contains stable IDs for raw pose, relaxed pose,
    fixed-fragment highlighting, and electron-density isosurfaces.  A small
    embedded JavaScript viewer is used when Py3Dmol is available in the browser;
    the textual structure data remains inspectable without network access.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _viewer_payload(viewer)
    title = (
        payload.get("molecule_id")
        or payload.get("canonical_smiles")
        or "ECloudFlow complex"
    )
    escaped_payload = html.escape(json.dumps(payload, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(str(title))}</title>
<style>body{{font-family:Arial,sans-serif;background:#101820;color:#eef2f4;margin:0;padding:20px}}main{{max-width:1200px;margin:auto}}.panel{{border:1px solid #3d5664;padding:14px;margin:10px 0;background:#172631}}.legend span{{display:inline-block;margin-right:18px}}#viewer{{height:520px;background:#0b1115}}pre{{white-space:pre-wrap;overflow:auto}}</style>
</head><body><main><h1>{html.escape(str(title))}</h1>
<div class="legend"><span id="raw-pose">raw-pose</span><span id="relaxed-pose">relaxed-pose</span><span id="fixed-fragment">fixed-fragment</span><span id="ligand-density-isosurface">ligand-density-isosurface</span></div>
<div id="viewer" aria-label="3D molecular complex viewer"></div>
<section class="panel"><h2>Canonical identity</h2><code>{html.escape(str(payload.get("canonical_smiles", "")))}</code></section>
<section class="panel"><h2>Embedded provenance</h2><pre id="provenance">{escaped_payload}</pre></section>
<script>const ECloudFlowViewer={{payload:JSON.parse(document.getElementById('provenance').textContent),layers:['raw-pose','relaxed-pose','fixed-fragment','ligand-density-isosurface']}};window.ECloudFlowViewer=ECloudFlowViewer;</script>
</main></body></html>"""
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination)
    return destination


def _viewer_payload(viewer: Any) -> dict[str, Any]:
    """Extract JSON-safe viewer fields without mutating source molecules."""
    if isinstance(viewer, dict):
        source = dict(viewer)
    else:
        source = {
            name: getattr(viewer, name, None)
            for name in (
                "molecule_id",
                "canonical_smiles",
                "raw_path",
                "relaxed_path",
                "pocket_id",
                "fragment",
                "electron_field",
                "density",
            )
        }
    molecule = source.get("molecule")
    if isinstance(molecule, Chem.Mol):
        source["canonical_smiles"] = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
    return {
        str(key): _safe(value) for key, value in source.items() if value is not None
    }


def _safe(value: Any) -> Any:
    """Convert common tensor/path/RDKit values to JSON-safe summaries."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if isinstance(value, Chem.Mol):
        return Chem.MolToSmiles(value, canonical=True, isomericSmiles=True)
    return str(value)


__all__ = ["render_complex_html"]
