import hashlib

from ecloudflow.visualization import plot_quality_speed_pareto


def _normalized_hash(path):
    data = path.read_bytes().replace(b"first", b"same").replace(b"second", b"same")
    return hashlib.sha256(data).hexdigest()


def test_publication_plot_is_deterministic_and_exports_vector(tmp_path):
    fixture = [{"wall_time": 1.0, "quality": 0.5}, {"wall_time": 2.0, "quality": 0.8}]
    first = plot_quality_speed_pareto(fixture, tmp_path / "first.svg", seed=3)
    second = plot_quality_speed_pareto(fixture, tmp_path / "second.svg", seed=3)
    assert first.exists() and second.exists()
    assert _normalized_hash(first) == _normalized_hash(second)
