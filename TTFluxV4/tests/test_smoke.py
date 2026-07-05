from ttflux.core.contracts import AnalysisRun
from ttflux.core.paths import ensure_project_layout


def test_analysis_run_contract(tmp_path):
    ensure_project_layout(tmp_path)
    run = AnalysisRun.new(root=tmp_path, input_video=None, status="test")
    data = run.to_dict()

    assert data["status"] == "test"
    assert data["input_video"] is None
    assert "ball_track" in data["outputs"]
    assert data["outputs"]["metrics"].endswith("metrics.json")
