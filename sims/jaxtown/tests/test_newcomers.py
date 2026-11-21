"""Newcomers: a move takes exactly three players from the other town (weights and optimizer
state), keeps the host's other four, and trains on in its own folder; the plan covers every
move both ways plus the controls. Training runs on the CPU.

    cd /mnt/c/dev/new-melting-pot && JAX_PLATFORMS=cpu ~/venvs/meltingpot/bin/python -m pytest \\
        sims/jaxtown/tests/test_newcomers.py -q
"""
import json
import os
import pickle
import sys

import jax
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import newcomers as nc  # noqa: E402
import train_jax as jtrain  # noqa: E402

TINY = ["--envs", "4", "--minibatches", "2", "--rollout", "8", "--ckpt-minutes", "0",
        "--river", "hughes"]
UPD = 4 * 8


def leaves(state, key):
    return jax.tree_util.tree_leaves(state[key])


def ckpt(folder):
    return pickle.load(open(os.path.join(folder, "ckpt.pkl"), "rb"))


def test_the_plan_is_every_move_both_ways_plus_a_control_per_town():
    moves = nc.plan()
    assert len(moves) == 64 and len({m.name for m in moves}) == 64
    controls = [m for m in moves if m.control]
    assert len(controls) == 16
    # Newcomers never share the hosts' starting weights, in a control or a real move.
    assert all(m.origin_seed != m.seed for m in moves)
    real = [m for m in moves if not m.control]
    for p in nc.PLAYER_TYPES:
        for s in nc.SEEDS:
            pairs = {(m.host, m.origin) for m in real if (m.players, m.seed) == (p, s)}
            assert len(pairs) == 12 and all((b, a) in pairs for a, b in pairs)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Tiny finished runs: strangers and isolating, selfish, seeds 0 and 1, in two
    folders (as the four towns' seeds are on this PC)."""
    root = tmp_path_factory.mktemp("towns")
    a, b = str(root / "a"), str(root / "b")
    jtrain.main(["--runs", "strangers:selfish:0", "isolating:selfish:0", "--out-root", a,
                 "--steps", str(UPD), *TINY])
    jtrain.main(["--runs", "strangers:selfish:1", "isolating:selfish:1", "--out-root", b,
                 "--steps", str(UPD), *TINY])
    return [a, b]


def test_a_move_takes_three_newcomers_and_trains_on_in_its_own_folder(sources, tmp_path):
    out = str(tmp_path / "newcomers")
    move = nc.Move("strangers", "isolating", "selfish", 0, 1)
    folder, made = nc.make_folder(move, sources, out, steps=2 * UPD)
    assert made and os.path.basename(folder) == "strangers_selfish_s0_from_isolating_s1"
    host = ckpt(os.path.join(sources[0], "strangers_selfish_s0"))
    origin = ckpt(os.path.join(sources[1], "isolating_selfish_s1"))
    new = ckpt(folder)
    for key in ("params", "opt"):
        assert len(leaves(new, key)) == len(leaves(host, key)) > 1
        for n, h, o in zip(leaves(new, key), leaves(host, key), leaves(origin, key)):
            assert np.array_equal(n[:3], o[:3]) and np.array_equal(n[3:], h[3:])
    assert any(not np.array_equal(h, o) for h, o in    # the two towns' players really differ
               zip(leaves(host, "params"), leaves(origin, "params")))
    assert np.array_equal(new["key"], host["key"]) and (new["env_steps"], new["update"]) == (0, 0)
    cfg = json.load(open(os.path.join(folder, "config.json")))
    assert (cfg["town"], cfg["seed"], cfg["newcomers"]["origin"]) == ("strangers", 0, "isolating")
    assert cfg["newcomers"]["steps_before"] == UPD and not cfg["newcomers"]["control"]
    # It trains on, logs from step 0, and keeps its settings.
    jtrain.main(["--folders", folder, "--resume", "--steps", str(UPD), *TINY])
    rows = open(os.path.join(folder, "metrics.csv")).read().split()
    assert len(rows) == 2 and rows[1].split(",")[:2] == ["1", str(UPD)]
    after = ckpt(folder)
    assert after["env_steps"] == UPD and after["town"] == "strangers"
    assert not np.array_equal(leaves(after, "params")[0], leaves(new, "params")[0])
    with pytest.raises(SystemExit, match="different settings"):
        jtrain.main(["--folders", folder, "--resume", "--steps", str(2 * UPD), *TINY[:-2]])
    # Making it again leaves the trained folder alone.
    assert nc.make_folder(move, sources, out) == (folder, False)
    assert ckpt(folder)["env_steps"] == UPD


def test_folders_need_resume_and_refuse_runs(tmp_path):
    with pytest.raises(SystemExit):
        jtrain.parse(["--folders", str(tmp_path)])
    with pytest.raises(SystemExit):
        jtrain.parse(["--folders", str(tmp_path), "--resume", "--runs", "strangers:fair"])


def test_unlike_runs_are_not_mixed(sources, tmp_path):
    move = nc.Move("strangers", "isolating", "selfish", 0, 0)
    # Newcomers from a run that trained longer than the host: refused.
    host_only = tmp_path / "host_only"
    host_only.mkdir()
    os.symlink(os.path.join(sources[0], move.host_run), host_only / move.host_run)
    longer = str(tmp_path / "longer")
    jtrain.main(["--runs", "isolating:selfish:0", "--out-root", longer,
                 "--steps", str(2 * UPD), *TINY])
    with pytest.raises(SystemExit, match="env_steps"):
        nc.make_folder(move, [str(host_only), longer], str(tmp_path / "x"))
    # A run found in two source folders: refused (which one?).
    with pytest.raises(SystemExit, match="exactly one"):
        nc.make_folder(move, [sources[0], sources[0]], str(tmp_path / "y"))
    assert not os.path.exists(tmp_path / "x") or not os.listdir(tmp_path / "x")


def test_train_runs_every_folder_a_batch_at_a_time_and_skips_finished_ones(sources, tmp_path):
    out = str(tmp_path / "newcomers")
    moves = nc.plan(player_types=("selfish",), towns=("strangers", "isolating"))
    assert len(moves) == 8 and sum(m.control for m in moves) == 4
    assert all(m.origin_seed != m.seed for m in moves)
    nc.make(sources, out, moves, steps=UPD, log=lambda s: None)
    nc.train(out, steps=UPD, batch=4, moves=moves, ckpt_minutes=0, dry_run=True)
    assert not os.path.exists(os.path.join(out, "queue.log"))      # a preview logs nothing
    nc.train(out, steps=UPD, batch=4, moves=moves, ckpt_minutes=0)
    for m in moves:
        assert ckpt(os.path.join(out, m.name))["env_steps"] == UPD
    queue = open(os.path.join(out, "queue.log")).read()
    assert queue.count("exited with code 0") == 2 and "every batch finished" in queue
    nc.train(out, steps=UPD, batch=4, moves=moves, ckpt_minutes=0)   # all done: no batch runs
    assert open(os.path.join(out, "queue.log")).read().count("starting") == 2


def _games(folder, games):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "episodes.jsonl"), "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")


def _game(steps, cleaned, ate, length=1000):
    return {"env_steps": steps, "length": length, "cleaned": cleaned, "ate": ate}


def test_report_measures_newcomers_and_locals_before_and_after(tmp_path):
    host, home = str(tmp_path / "isolating_selfish_s0"), str(tmp_path / "strangers_selfish_s1")
    # Before the move only the last 10% of each run counts (steps 90 to 100 here).
    _games(host, [_game(10, [99] * 7, [9] * 7), _game(95, [10] * 7, [0] * 7)])
    _games(home, [_game(10, [0] * 7, [0] * 7), _game(95, [80] * 3 + [1] * 4, [5] * 7, 2000)])
    move = nc.Move("isolating", "strangers", "selfish", 0, 1)
    folder = os.path.join(str(tmp_path / "newcomers"), move.name)
    _games(folder, [_game(s, [c] * 3 + [10] * 4, [1] * 7)
                    for s, c in ((1e6, 40), (2e6, 30), (12e6, 99), (19e6, 10), (20e6, 10))])
    json.dump({"newcomers": {"host_run": host, "newcomers_run": home, "steps_before": 100}},
              open(os.path.join(folder, "config.json"), "w"))
    (r,) = nc.summarise(str(tmp_path / "newcomers"), [move])
    assert r["kind"] == "into the isolating town"
    assert (r["home_new_clean"], r["host_local_clean"], r["host_apples"]) == (40, 10, 0)
    assert r["early"]["new_clean"] == 35 and r["late"]["new_clean"] == 10    # 0-2M, 18-20M
    assert r["late"]["local_clean"] == 10 and r["late"]["apples"] == 7
    assert nc.gap_closed(r) == pytest.approx(1.0)        # all the way to the locals' 10
    groups, per_move = nc.report_tables([r])
    assert "| into the isolating town | 1 | 40, 35, 10 | 10, 10 | 0.0, 7.0, 7.0 | 100%" in groups
    nc.report_chart([r], str(tmp_path / "chart.png"))
    assert os.path.getsize(tmp_path / "chart.png") > 10_000
