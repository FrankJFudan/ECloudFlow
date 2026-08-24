"""Electron-density HTML views with an offline Plotly-compatible payload."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_electron_field_html(field: Any, path: str | Path) -> Path:
    """Render an electron-field viewer with density and coordinate metadata.

    :param field: ElectronField, mapping, tensor, or trajectory payload.
    :param path: Destination HTML path.
    :return: Written path.
    :rtype: pathlib.Path

    The page embeds a compact JSON payload and a deterministic Plotly-style
    layer ID.  Large tensors are summarized by shape/min/max to keep reports
    portable; callers can attach a downsampled ``density`` array explicitly.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _safe_payload(field)
    body = html.escape(json.dumps(payload, sort_keys=True))
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Electron cloud</title>
<style>body{{font-family:Arial;background:#111;color:#eee;padding:20px}}#ligand-density-isosurface{{height:560px;border:1px solid #496}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>Electron cloud</h1><div id="ligand-density-isosurface" aria-label="ligand-density-isosurface"></div><pre id="field-data">{body}</pre>
<script>window.ECloudFlowElectronField=JSON.parse(document.getElementById('field-data').textContent);</script></body></html>"""
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination)
    return destination


def _safe_payload(field: Any) -> dict[str, Any]:
    """Summarize arbitrary field values without serializing device tensors."""
    if isinstance(field, dict):
        payload = dict(field)
    else:
        payload = {
            name: getattr(field, name, None)
            for name in ("positions", "values", "density", "weights", "channel_names")
        }
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if hasattr(value, "shape") and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            output[str(key)] = {
                "shape": list(tensor.shape),
                "min": float(tensor.min().item()) if tensor.numel() else None,
                "max": float(tensor.max().item()) if tensor.numel() else None,
            }
        elif hasattr(value, "tolist"):
            output[str(key)] = value.tolist()
        else:
            output[str(key)] = (
                str(value)
                if not isinstance(value, (str, int, float, bool, list, tuple))
                else value
            )
    return output


__all__ = ["render_electron_field_html"]
