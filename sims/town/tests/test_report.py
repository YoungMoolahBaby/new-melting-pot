"""report.py puts each run's numbers under its own name, and never merges runs that
are different experiments."""
import csv
import json
import os

import report
import train


def fake_run(root, name, river="hughes", seed=0, clean=0.1, updates=3):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"town": "neighbours", "players": "selfish", "seed": seed, "river": river,
                   "alpha": 0.0, "beta": 0.0, "game": "jaxtown"}, f)
    with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=train.METRIC_FIELDS)
        w.writeheader()
        for u in range(1, updates + 1):
            w.writerow({**{k: 0 for k in train.METRIC_FIELDS}, "update": u,
                        "env_steps": 1000 * u, "clean_per_step": clean})
    open(os.path.join(d, "episodes.jsonl"), "w").close()
    return d


def table(out):
    with open(os.path.splitext(out)[0] + ".md") as f:
        return [ln for ln in f if ln.startswith("| neighbours") or ln.startswith("| **")]


def test_empty_run_does_not_shift_rows(tmp_path):
    runs = [fake_run(tmp_path, "a", seed=0, updates=0), fake_run(tmp_path, "b", seed=1, clean=0.2),
            fake_run(tmp_path, "c", seed=2, clean=0.3)]
    out = str(tmp_path / "chart.png")
    report.main([*runs, "--group", "--out", out])
    rows = table(out)
    assert any("seed 1 |" in r and "| 0.200 |" in r for r in rows)
    assert any("seed 2 |" in r and "| 0.300 |" in r for r in rows)
    assert not any("seed 0 |" in r for r in rows)          # no updates: no row


def test_different_rivers_are_never_merged(tmp_path):
    runs = [fake_run(tmp_path, "h", river="hughes"), fake_run(tmp_path, "m", river="melting_pot")]
    out = str(tmp_path / "chart.png")
    report.main([*runs, "--group", "--out", out])
    rows = table(out)
    assert len(rows) == 2 and not any("mean" in r for r in rows)
    assert any("river hughes" in r for r in rows) and any("river melting_pot" in r for r in rows)
