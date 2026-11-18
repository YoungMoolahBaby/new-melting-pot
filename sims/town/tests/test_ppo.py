"""Tests for the learner: the fairness reward, one PPO update, and a tiny training
run that checkpoints and resumes. Run inside WSL:
    ~/venvs/meltingpot/bin/python -m pytest sims/town/tests -q
"""
import json
import os

import numpy as np

import envs
import ppo


def test_inequity_reward_matches_hand_calculation():
    r = np.array([1.0, 0.0, 0.0])
    e = np.array([2.0, 1.0, 0.0])
    # Player 0 is ahead of both: guilt = beta/2 * ((2-1) + (2-0)) = 1.5 beta.
    # Player 2 is behind both: envy = alpha/2 * ((2-0) + (1-0)) = 1.5 alpha.
    u = ppo.inequity_reward(r, e, alpha=5.0, beta=0.05)
    assert np.allclose(u, [1.0 - 0.075, 0.0 - (5.0 * 0.5 + 0.05 * 0.5), 0.0 - 7.5])
    # Selfish players feel nothing.
    assert np.allclose(ppo.inequity_reward(r, e, 0.0, 0.0), r)
    # Smoothing: e_t = gamma * lambda * e_{t-1} + r_t.
    assert np.allclose(ppo.smooth_rewards(e, r, 0.99, 0.95), 0.9405 * e + r)
    # The same on JAX arrays, inside jit (the GPU trainer uses it there).
    import jax
    import jax.numpy as jnp
    uj = jax.jit(ppo.inequity_reward)(jnp.asarray(r), jnp.asarray(e), 5.0, 0.05)
    assert np.allclose(np.asarray(uj), u, atol=1e-6)


def test_downsample_is_block_average():
    rgb = np.random.default_rng(0).integers(0, 256, size=(2, 7, 88, 88, 3), dtype=np.uint8)
    ref = rgb.reshape(2, 7, 22, 4, 22, 4, 3).astype(float).mean(axis=(3, 5))
    out = envs.downsample(rgb)
    assert out.shape == (2, 7, 22, 22, 3) and out.dtype == np.uint8
    assert np.abs(out - ref).max() <= 0.5 + 1e-9


def test_one_update_changes_every_player_and_stays_finite():
    import jax
    import jax.numpy as jnp
    cfg = ppo.Config(epochs=1, minibatches=2)
    learner = ppo.Learner(cfg)
    params, opt = learner.init(jax.random.PRNGKey(0))
    T, E, P = 6, 4, cfg.num_players
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.integers(0, 256, size=(T, E, P, 22, 22, 3), dtype=np.uint8))
    ready = jnp.ones((T, E, P))
    start = jnp.zeros((T, E), bool).at[0].set(True)
    h = learner.initial_state(E)
    h0 = h
    acts, logps, values = [], [], []
    for t in range(T):
        h, a, logp, v, _ = learner.act(params, h, obs[t], ready[t], start[t],
                                       jax.random.PRNGKey(t))
        assert a.shape == (P, E) and int(a.max()) < cfg.num_actions
        acts.append(a), logps.append(logp), values.append(v)
    batch = {"obs": obs, "ready": ready, "start": start, "action": jnp.stack(acts),
             "logp": jnp.stack(logps), "value": jnp.stack(values),
             "reward": jnp.asarray(rng.normal(size=(T, P, E)), jnp.float32),
             "done": jnp.zeros((T, E), bool), "h0": h0, "last_value": values[-1]}
    new_params, _, stats = learner.update(params, opt, batch, jax.random.PRNGKey(9))
    assert np.isfinite(np.asarray(stats)).all()
    # Each player's own weights moved (per-player networks, nothing shared).
    w0 = jax.tree_util.tree_leaves(params)[0]
    w1 = jax.tree_util.tree_leaves(new_params)[0]
    moved = np.abs(np.asarray(w1 - w0)).reshape(P, -1).max(axis=1)
    assert (moved > 0).all()


def test_tiny_training_run_checkpoints_and_resumes(tmp_path):
    import train
    out = str(tmp_path / "run")
    base = ["--town", "connected", "--players", "fair", "--workers", "2",
            "--envs-per-worker", "2", "--rollout", "8", "--out", out, "--ckpt-minutes", "0"]
    train.main(base + ["--steps", "32"])
    assert os.path.exists(os.path.join(out, "ckpt.pkl"))
    with open(os.path.join(out, "metrics.csv")) as f:
        assert len(f.read().strip().splitlines()) == 1 + 1   # header + 1 update
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["beta"] == 0.05 and cfg["alpha"] == 0.0
    train.main(base + ["--steps", "64", "--resume"])
    with open(os.path.join(out, "metrics.csv")) as f:
        rows = f.read().strip().splitlines()
    assert len(rows) == 1 + 2 and rows[-1].split(",")[1] == "64"

    # Never two runs in one folder, and a resume keeps its settings.
    import pytest
    with pytest.raises(SystemExit, match="already holds a run"):
        train.main(base + ["--steps", "64"])
    with pytest.raises(SystemExit, match="different settings"):
        train.main(base + ["--steps", "64", "--resume", "--river", "hughes"])

    # Rows logged after the last checkpoint (a hard kill) are dropped on resume.
    with open(os.path.join(out, "metrics.csv"), "a") as f:
        f.write("99" + ",0" * (len(train.METRIC_FIELDS) - 1) + "\n")
    with open(os.path.join(out, "episodes.jsonl"), "a") as f:
        f.write(json.dumps({"update": 99}) + "\n")
    train.main(base + ["--steps", "64", "--resume"])
    with open(os.path.join(out, "metrics.csv")) as f:
        assert len(f.read().strip().splitlines()) == 1 + 2
    with open(os.path.join(out, "episodes.jsonl")) as f:
        assert all(json.loads(ln)["update"] <= 2 for ln in f if ln.strip())


def test_two_runs_in_one_process_keep_separate_players(tmp_path):
    import pickle
    import train
    train.main(["--runs", "isolating:selfish", "strangers:fair:3", "--workers", "1",
                "--envs-per-worker", "2", "--rollout", "8", "--steps", "16",
                "--out-root", str(tmp_path)])
    a = pickle.load(open(tmp_path / "isolating_selfish_s0" / "ckpt.pkl", "rb"))
    b = pickle.load(open(tmp_path / "strangers_fair_s3" / "ckpt.pkl", "rb"))
    assert (a["beta"], b["beta"]) == (0.0, 0.05)
    assert a["env_steps"] == b["env_steps"] == 16
    import jax
    wa, wb = jax.tree_util.tree_leaves(a["params"])[0], jax.tree_util.tree_leaves(b["params"])[0]
    assert not np.allclose(wa, wb)
