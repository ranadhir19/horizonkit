import json

import numpy as np
import pytest

from horizonkit import load_runs
from horizonkit.errors import InvalidRunLogError


def test_generic_csv_and_aggregate(tmp_path):
    p = tmp_path / "runs.csv"
    p.write_text("horizon,success,task_id\n10,1,a\n10,0,a\n20,1,b\n")
    d = load_runs(p)
    assert d.n == 3 and d.n_success == 2
    agg = d.aggregate()
    assert agg.trials.tolist() == [2, 1]
    assert agg.successes.tolist() == [1, 1]
    assert d.group_counts() == [("a", 2, 1), ("b", 1, 1)]


def test_aggregate_rows_expand(tmp_path):
    p = tmp_path / "agg.csv"
    p.write_text("horizon,trials,successes\n5,10,7\n15,10,3\n")
    d = load_runs(p)
    assert d.n == 20 and d.n_success == 10


def test_metr_jsonl(tmp_path):
    p = tmp_path / "metr.jsonl"
    rows = [
        {"alias": "m1", "task_id": "t1", "human_minutes": 12.5, "score_binarized": 1},
        {"alias": "m1", "task_id": "t2", "human_minutes": 90, "score_binarized": 0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    d = load_runs(p, format="metr")
    assert d.source_format == "metr"
    assert d.horizons.tolist() == [12.5, 90.0]


def test_hal_style_json_container(tmp_path):
    p = tmp_path / "hal.json"
    p.write_text(json.dumps({"rollouts": [
        {"duration_seconds": 30, "success": True},
        {"steps": 100, "score": 0},
    ]}))
    d = load_runs(p, format="hal")
    assert d.n == 2 and d.n_success == 1


@pytest.mark.parametrize("text", [
    "horizon,success\n-1,1\n",
    "horizon,success\n10,2\n",
    "length,outcome\n10,maybe\n",
])
def test_invalid_rows_are_refused(tmp_path, text):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(InvalidRunLogError):
        load_runs(p)
