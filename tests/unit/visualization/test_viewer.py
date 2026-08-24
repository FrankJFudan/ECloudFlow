from ecloudflow.visualization import render_complex_html, render_electron_field_html


def test_complex_viewer_contains_ranked_molecule_fragment_and_density(tmp_path):
    path = render_complex_html(
        {
            "molecule_id": "3ZTX-000001",
            "canonical_smiles": "CCO",
            "fragment": "fixed",
            "density": {"shape": [8, 8, 8]},
            "raw_path": "raw.sdf",
            "relaxed_path": "relaxed.sdf",
        },
        tmp_path / "complex.html",
    )
    html = path.read_text(encoding="utf-8")
    assert "3ZTX-000001" in html
    assert "fixed-fragment" in html
    assert "ligand-density-isosurface" in html
    assert "raw-pose" in html and "relaxed-pose" in html


def test_electron_field_viewer_embeds_field_payload(tmp_path):
    path = render_electron_field_html(
        {"density": [0.1, 0.2], "channel_names": ["rho"]},
        tmp_path / "field.html",
    )
    text = path.read_text(encoding="utf-8")
    assert "ligand-density-isosurface" in text
    assert "channel_names" in text
