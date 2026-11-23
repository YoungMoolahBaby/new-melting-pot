"""Losing one cleaner: the plan, who leaves and who comes in, that the move trains on, and the
report's measures. Runs on the CPU.

    cd /mnt/c/dev/new-melting-pot && JAX_PLATFORMS=cpu ~/venvs/meltingpot/bin/python -m pytest \\
        sims/jaxtown/tests/test_lose_cleaner.py -q
"""
import json
import os
import pickle
import sys

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))
import lose_cleaner as lc  # noqa: E402
import train_jax as jtrain  # noqa: E402

TINY = ["--envs", "4", "--minibatches", "2", "--rollout", "8", "--ckpt-minutes", "0",
        "--river", "hughes"]
UPD = 4 * 8


def test_the_plan():
    moves = lc.plan()
    assert len(moves) == 8 and len({m.name for m in moves}) == 8
    assert all(m.origin_seed != m.seed for m in moves)
    assert [m.kind for m in moves] == [lc.KINDS[0]] * 2 + [lc.KINDS[1]] * 2 + [lc.KINDS[2]] * 4
    assert {(m.host_run, m.origin_run) for m in moves if m.share == 0.1} == {
        ("connected_selfish_s0_coop10", "connected_selfish_s1_coop10"),
        ("connected_selfish_s1_coop10", "connected_selfish_s0_coop10"),
        ("connected_selfish_s2_coop10", "connected_selfish_s3_coop10"),
        ("connected_selfish_s3_coop10", "connected_selfish_s2_coop10")}
    assert moves[0].name == "coop25_s0_from_s1_eater"
    assert moves[0].host_run == "connected_selfish_s0_coop25"


def fake_games(run_dir, cleaned, steps):
    with open(os.path.join(run_dir, "episodes.jsonl"), "w") as f:
        for _ in range(3):
            f.write(json.dumps({"env_steps": steps, "length": 1000, "cleaned": cleaned,
                                "ate": [1] * 7}) + "\n")


def test_the_top_cleaner_leaves_and_the_move_trains_on(tmp_path):
    src = str(tmp_path / "runs")
    jtrain.main(["--runs", "connected:selfish:0:melting_pot:0.25",
                 "connected:selfish:1:melting_pot:0.25", "--out-root", src,
                 "--steps", str(UPD), *TINY])
    s0, s1 = (os.path.join(src, f"connected_selfish_s{s}_coop25") for s in (0, 1))
    fake_games(s0, [0, 0, 6, 9, 0, 1, 0], UPD)     # top: seat 3; least: seat 6 (ties: last)
    fake_games(s1, [4, 1, 0, 0, 0, 0, 8], UPD)     # top: seat 6; least: seat 5
    out = str(tmp_path / "moves")
    eater, cleaner = lc.Move(0.25, 0, 1, "eater"), lc.Move(0.25, 0, 1, "cleaner")
    lc.make([src], out, moves=[eater, cleaner], steps=UPD, log=lambda line: None)
    for move, origin_slot in ((eater, 5), (cleaner, 6)):
        cfg = json.load(open(os.path.join(out, move.name, "config.json")))
        info = cfg["newcomers"]
        assert cfg["coop_share"] == 0.25 and info["kind"] == move.kind
        assert (info["newcomer_slots"], info["origin_slots"]) == ([3], [origin_slot])
        assert info["host_rank"] == [3, 2, 5, 0, 1, 4, 6]
        moved = pickle.load(open(os.path.join(out, move.name, "ckpt.pkl"), "rb"))
        host = pickle.load(open(os.path.join(s0, "ckpt.pkl"), "rb"))
        home = pickle.load(open(os.path.join(s1, "ckpt.pkl"), "rb"))
        pi = lambda s: s["params"]["params"]["pi"]["kernel"]  # noqa: E731
        assert np.array_equal(pi(moved)[3], pi(home)[origin_slot])
        for s in (0, 1, 2, 4, 5, 6):            # the six locals are the host's own
            assert np.array_equal(pi(moved)[s], pi(host)[s])
        assert moved["env_steps"] == 0
    # It trains on with the host's co-op.
    jtrain.main(["--folders", os.path.join(out, eater.name), "--resume", "--steps", str(UPD),
                 *TINY])
    assert pickle.load(open(os.path.join(out, eater.name, "ckpt.pkl"), "rb"))["env_steps"] == UPD


def test_the_report_measures_the_newcomer_the_other_cleaner_and_the_locals(tmp_path):
    def run(name, games):
        d = tmp_path / name
        d.mkdir(parents=True)
        with open(d / "episodes.jsonl", "w") as f:
            for steps, cleaned, ate, ret in games:
                f.write(json.dumps({"env_steps": steps, "length": 1000, "cleaned": cleaned,
                                    "ate": ate, "return": ret, "feed": [0] * 7}) + "\n")
        return str(d)
    z = [0] * 7
    # Before the move: seats 3 and 2 clean (the last 10% is the game at step 100).
    host = run("host", [(5, [9] * 7, z, z),
                        (100, [0, 0, 4, 6, 0, 0, 0], [5, 5, 0, 0, 5, 5, 5], [5, 5, 2, 3, 5, 5, 5])])
    home = run("home", [(100, [0, 0, 0, 0, 0, 1, 7], z, z)])
    move = lc.plan()[0]
    d = run(f"moves/{move.name}", [
        (1, [0, 0, 8, 2, 0, 0, 0], [1] * 7, [1] * 7),                          # just after
        (50, z, z, z),
        (90, [0, 0, 5, 0, 1, 0, 0], [500, 500, 0, 300, 500, 500, 500],         # at the end
         [500, 500, 40, 300, 500, 500, 500])])
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"newcomers": {"host_run": host, "newcomers_run": home,
                                 "newcomer_slots": [3], "origin_slots": [5],
                                 "host_rank": [3, 2, 0, 1, 4, 5, 6],
                                 "steps_before": 100}}, f)
    (r,) = lc.summarise(str(tmp_path / "moves"), [move])
    assert r["apples"] == (pytest.approx(25), pytest.approx(7), pytest.approx(2800))
    assert r["held"]
    assert r["new_home"] == pytest.approx(1)              # home seat 5, last 10%
    assert r["left_clean"] == pytest.approx(6)            # host seat 3 before
    assert r["new_clean"] == (pytest.approx(2), pytest.approx(0))
    assert r["other_clean"] == (pytest.approx(4), pytest.approx(8), pytest.approx(5))
    assert r["local_clean"] == (pytest.approx(4), pytest.approx(8), pytest.approx(6))
    # At the end seat 2 cleans 5 of 6 and seat 4 the other 1 (both at least a tenth); seat 2
    # was paid 40 and ate nothing, seat 4 ate 500 and was paid nothing.
    assert r["cleaners_before"][0] == 2
    assert r["cleaners_end"] == (2, pytest.approx(20), pytest.approx(250))
    a, b, c = lc.report_tables([r])
    assert "2,800 | 1 of 1 |" in a and "| 6 | 4; 8; 5 | 2; 2 | 20; 250 |" in b
    assert move.name in c and "| yes |" in c
    # A 10% town has no second cleaner.
    m10 = lc.plan()[4]
    assert m10.share == 0.1 and m10.kind == lc.KINDS[2]
