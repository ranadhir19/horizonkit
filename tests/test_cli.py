import numpy as np

from horizonkit.cli import main
from horizonkit.models import survival


def make_log(tmp_path):
    rng = np.random.default_rng(4)
    x = np.exp(rng.uniform(np.log(1), np.log(500), 600))
    y = (rng.random(len(x)) < survival("weibull", {"kappa": 0.8, "eta": 70.0}, x)).astype(int)
    groups = np.array([f"p{i % 30}" for i in range(len(x))])
    p = tmp_path / "runs.csv"
    with p.open("w") as fh:
        fh.write("horizon,success,task_id\n")
        for xi, yi, g in zip(x, y, groups):
            fh.write(f"{xi:.5f},{yi},{g}\n")
    return p


def test_cli_fit_and_identify(tmp_path, capsys):
    p = make_log(tmp_path)
    assert main(["fit", str(p)]) == 0
    out = capsys.readouterr().out
    assert "What this tool does not establish" in out
    assert main(["identify", str(p)]) == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out and "One-directional" in out


def test_cli_passk_and_extrapolate(tmp_path, capsys):
    p = make_log(tmp_path)
    assert main(["passk", str(p), "-k", "3"]) == 0
    assert "pass^3" in capsys.readouterr().out
    assert main(["extrapolate", str(p)]) == 0
    out = capsys.readouterr().out
    assert "T99" in out and "law-relative" in out


def test_cli_bad_file_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "bad.csv"
    p.write_text("horizon,success\n-5,1\n")
    assert main(["fit", str(p)]) == 2
    assert "horizonkit:" in capsys.readouterr().err
