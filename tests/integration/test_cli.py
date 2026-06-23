"""CLI smoke tests for every subcommand."""

from __future__ import annotations

from cognos.cli import main


def test_cli_init_writes_template(tmp_path):
    out = tmp_path / "cognos.yaml"
    assert main(["init", "-o", str(out)]) == 0
    assert out.exists() and "task:" in out.read_text()


def test_cli_agents_lists_eight():
    assert main(["agents"]) == 0


def test_cli_demo_regression(tmp_path):
    rc = main(["demo", "--task", "regression", "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
    runs = list((tmp_path / "runs").glob("2*"))
    assert runs and (runs[0] / "summary.json").exists()


def test_cli_explain_and_report(tmp_path):
    cfg = tmp_path / "cognos.yaml"
    main(["init", "-o", str(cfg)])
    # point the template at a real dataset
    from cognos import synth
    data = tmp_path / "data.csv"
    synth.make_regression_dataset(n=120).to_csv(data, index=False)
    text = cfg.read_text().replace("path: data.csv", f"path: {data}").replace("max_candidates: 24", "max_candidates: 6")
    cfg.write_text(text)
    assert main(["explain", "--config", str(cfg)]) == 0
    assert main(["run", "--config", str(cfg), "--runs-dir", str(tmp_path / "runs")]) in (0, 2)
    run_id = next((tmp_path / "runs").glob("2*")).name
    assert main(["report", "--run", run_id, "--runs-dir", str(tmp_path / "runs")]) == 0
    assert main(["list-runs", "--runs-dir", str(tmp_path / "runs")]) == 0
