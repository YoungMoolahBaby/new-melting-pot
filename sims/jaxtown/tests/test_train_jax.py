"""The GPU trainer: a tiny run checkpoints and resumes, two runs keep their own
players, and the files match what sims/town/report.py reads.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python -m pytest sims/jaxtown/tests -q
"""
import json
import os
import pickle
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train_jax as jtrain  # noqa: E402

TINY = ["--envs", "4", "--minibatches", "2", "--rollout", "8", "--ckpt-minutes", "0"]


def test_both_trainers_load_in_one_session():
    # sims/town/train.py stays importable as `train` next to the GPU trainer.
    import train
    assert hasattr(train, "Run") and train.__file__.endswith(os.path.join("town", "train.py"))
    assert jtrain.town_train is train


def rows(path):
    with open(path) as f:
        return f.read().strip().splitlines()


def test_tiny_run_checkpoints_resumes_and_refuses_to_mix(tmp_path):
    out = str(tmp_path / "run")
    base = ["--town", "connected", "--players", "fair", "--out", out, *TINY]
    jtrain.main(base + ["--steps", "32"])
    assert len(rows(os.path.join(out, "metrics.csv"))) == 1 + 1          # header + 1 update
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert (cfg["game"], cfg["beta"], cfg["ppo"]["encoder"]) == ("jaxtown", 0.05, "cells")
    jtrain.main(base + ["--steps", "64", "--resume"])
    last = rows(os.path.join(out, "metrics.csv"))
    assert len(last) == 1 + 2 and last[-1].split(",")[1] == "64"
    with pytest.raises(SystemExit, match="already holds a run"):
        jtrain.main(base + ["--steps", "64"])
    with pytest.raises(SystemExit, match="different settings"):
        jtrain.main(base + ["--steps", "64", "--resume", "--envs", "8"])


def test_two_runs_keep_their_own_players(tmp_path):
    jtrain.main(["--runs", "isolating:selfish", "strangers:fair:3", "--steps", "32",
                 "--out-root", str(tmp_path), *TINY])
    a = pickle.load(open(tmp_path / "isolating_selfish_s0" / "ckpt.pkl", "rb"))
    b = pickle.load(open(tmp_path / "strangers_fair_s3" / "ckpt.pkl", "rb"))
    assert (a["beta"], b["beta"]) == (0.0, 0.05)
    assert a["env_steps"] == b["env_steps"] == 32
    import jax
    wa = jax.tree_util.tree_leaves(a["params"])[0]
    wb = jax.tree_util.tree_leaves(b["params"])[0]
    assert not np.allclose(wa, wb)


def test_shared_players_get_the_average_of_everyones_apples():
    reward = np.array([[1.0, 0, 0, 0, 0, 0, 0.4], [0, 0, 0, 0, 0, 0, 0]], np.float32)
    shared = np.asarray(jtrain.team_reward(reward, True))
    assert shared.shape == reward.shape and np.allclose(shared[0], 0.2) and not shared[1].any()
    assert np.array_equal(np.asarray(jtrain.team_reward(reward, False)), reward)


def test_dirt_rule_and_shared_players_are_set_per_run(tmp_path):
    jtrain.main(["--runs", "neighbours:shared", "neighbours:fair:0:capped", "--steps", "32",
                 "--out-root", str(tmp_path), *TINY])
    a = json.load(open(tmp_path / "neighbours_shared_s0" / "config.json"))
    b = json.load(open(tmp_path / "neighbours_fair_s0_capped" / "config.json"))
    assert (a["players"], a["dirt"], a["beta"]) == ("shared", "melting_pot", 0.0)
    assert (b["players"], b["dirt"], b["beta"]) == ("fair", "capped", 0.05)
    out = str(tmp_path / "neighbours_fair_s0_capped")
    with pytest.raises(SystemExit, match="different settings"):
        jtrain.main(["--players", "fair", "--out", out, "--steps", "64", "--resume", *TINY])


def test_finished_games_are_logged_like_melting_pots(tmp_path):
    # Every game ends by frame 5,000: 4 games x 128 steps x 40 updates finishes each.
    out = str(tmp_path / "run")
    jtrain.main(["--town", "isolating", "--out", out, "--steps", str(4 * 128 * 40), *TINY,
                 "--rollout", "128"])
    eps = [json.loads(ln) for ln in rows(os.path.join(out, "episodes.jsonl"))]
    assert len(eps) >= 4 and all(e["length"] >= 1099 for e in eps)
    for key in ("collective_return", "equality", "cleaned_total", "zaps_total",
                "top2_clean_share", "feed", "sustainability"):
        assert key in eps[0]
