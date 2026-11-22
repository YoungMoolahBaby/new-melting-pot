"""Newcomers in a co-op town: the plan, who moves, and the relabelling of a newcomer who
lands in another seat (exact: the same network on relabelled views). Runs on the CPU.

    cd /mnt/c/dev/new-melting-pot && JAX_PLATFORMS=cpu ~/venvs/meltingpot/bin/python -m pytest \\
        sims/jaxtown/tests/test_coop_newcomers.py -q
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))
import cleanup as cu  # noqa: E402
import coop_newcomers as cn  # noqa: E402
import newcomers as nc  # noqa: E402
import ppo  # noqa: E402
import train_jax as jtrain  # noqa: E402

TINY = ["--envs", "4", "--minibatches", "2", "--rollout", "8", "--ckpt-minutes", "0",
        "--river", "hughes"]
UPD = 4 * 8
IDS = np.arange(cu.CH["player0"], cu.CH["player0"] + 7)


def player(tree, s):
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[s], tree)


def test_the_plan():
    moves = cn.plan()
    assert len(moves) == 16 and len({m.name for m in moves}) == 16
    assert all(m.origin_seed != m.seed for m in moves)
    for v in cn.VARIANTS:
        assert sorted(m.kind for m in moves if m.variant == v) == sorted(cn.KINDS * 2)
    into = [m for m in moves if m.kind == "into the co-op"]
    assert {(m.host_run, m.origin_run) for m in into} == {
        ("connected_selfish_s0_coop10", "connected_selfish_s2"),
        ("connected_selfish_s1_coop10", "connected_selfish_s3")}
    assert moves[0].name == "coop_s0_from_none_s2_eaters"
    assert [m.variant for m in moves] == ["eaters"] * 8 + ["cleaners"] * 8   # two batches


def test_seats_map_newcomers_home_and_locals_to_the_seats_left():
    assert nc.seat_map([5, 2, 0], [1, 6, 3]) == [3, 0, 6, 2, 4, 1, 5]
    assert nc.seat_map([0, 1, 2], [0, 1, 2]) == list(range(7))
    with pytest.raises(ValueError):
        nc.seat_map([0, 0, 1], [1, 2, 3])


def test_a_reseated_newcomer_is_the_same_network_on_relabelled_views():
    learner = ppo.Learner(ppo.Config(encoder="cells", obs_shape=(cu.VIEW, cu.VIEW,
                                                                  cu.NUM_CHANNELS)))
    host, host_opt = learner.init(jax.random.PRNGKey(0))
    home, home_opt = learner.init(jax.random.PRNGKey(1))
    slots, origin_slots = [4, 6, 1], [0, 2, 5]
    perm = nc.seat_map(slots, origin_slots)
    new = nc.reseat(nc.swap_in(host, home, slots, origin_slots), slots, origin_slots)
    new_opt = nc.reseat(nc.swap_in(host_opt, home_opt, slots, origin_slots), slots,
                        origin_slots)
    # A view in the host town: random cells, others marked by their seats' channels.
    rng = np.random.default_rng(0)
    view = (rng.random((cu.VIEW, cu.VIEW, cu.NUM_CHANNELS)) < 0.2).astype(np.uint8)
    view[..., IDS] = 0
    for h in range(7):
        r, c = rng.integers(0, cu.VIEW, 2)
        view[r, c, IDS[h]] = 1
    at_home = view.copy()           # the same view as the home town labelled it
    at_home[..., IDS[perm]] = view[..., IDS]
    args = (np.zeros((1, 128), np.float32), np.ones(1, np.float32), np.ones(1, bool))
    for seat, home_seat in zip(slots, origin_slots):
        _, logits, value = learner.net.apply(player(new, seat), args[0], view[None], *args[1:])
        _, l0, v0 = learner.net.apply(player(home, home_seat), args[0], at_home[None],
                                      *args[1:])
        np.testing.assert_allclose(logits, l0, atol=1e-5)
        np.testing.assert_allclose(value, v0, atol=1e-5)
        # Without the relabelling it would act differently.
        _, raw, _ = learner.net.apply(player(nc.swap_in(host, home, slots, origin_slots),
                                             seat), args[0], view[None], *args[1:])
        assert not np.allclose(raw, l0, atol=1e-5)
    # Locals untouched; a newcomer's other layers and optimizer moments came with it.
    for a, b in zip(jax.tree_util.tree_leaves(new), jax.tree_util.tree_leaves(host)):
        assert np.array_equal(np.asarray(a)[[0, 2, 3, 5]], np.asarray(b)[[0, 2, 3, 5]])
    assert np.array_equal(new["params"]["fc1"]["kernel"][4], home["params"]["fc1"]["kernel"][0])
    mu, mu_home = new_opt[1][0].mu["params"]["conv1"]["kernel"], \
        home_opt[1][0].mu["params"]["conv1"]["kernel"]
    assert np.array_equal(np.asarray(mu)[4][:, :, IDS], np.asarray(mu_home)[0][:, :, IDS[perm]])
    # Same seats: nothing relabelled (experiment 1's moves).
    same = nc.swap_in(host, home)
    assert nc.reseat(same, nc.NEWCOMER_SLOTS, nc.NEWCOMER_SLOTS) is same


def test_the_report_measures_newcomers_in_their_seats(tmp_path):
    def run(name, games):
        d = tmp_path / name
        d.mkdir(parents=True)
        with open(d / "episodes.jsonl", "w") as f:
            for steps, cleaned, ate in games:
                f.write(json.dumps({"env_steps": steps, "length": 1000, "cleaned": cleaned,
                                    "ate": ate}) + "\n")
        return str(d)
    host = run("host", [(5, [9] * 7, [1] * 7), (100, [10, 10, 1, 1, 1, 1, 1], [50] * 7)])
    home = run("home", [(100, [0, 0, 0, 0, 3, 6, 9], [5] * 7)])
    move = cn.plan()[0]
    out = tmp_path / "moves"
    d = run(f"moves/{move.name}", [(1, [2, 4, 6, 0, 0, 0, 0], [0] * 7),
                                    (90, [8, 8, 8, 1, 1, 1, 1], [100] * 7)])
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"newcomers": {"host_run": host, "newcomers_run": home,
                                 "newcomer_slots": [0, 1, 2], "origin_slots": [4, 5, 6],
                                 "steps_before": 100}}, f)
    (r,) = cn.summarise(str(out), [move])
    assert r["home_clean"] == pytest.approx(6)                 # home seats 4-6, last 10%
    assert r["replaced_clean"] == pytest.approx(7)             # host seats 0-2 before
    assert r["local_before"] == pytest.approx(1)
    assert r["apples_before"] == pytest.approx(350)
    assert r["new_clean"] == (pytest.approx(4), pytest.approx(8))
    assert r["local_clean"] == (pytest.approx(0), pytest.approx(1))
    assert r["apples"] == (pytest.approx(0), pytest.approx(700))
    groups, every = cn.report_tables([r])
    assert "6; 4; 8" in groups and move.name in every


def fake_games(run_dir, cleaned, steps):
    with open(os.path.join(run_dir, "episodes.jsonl"), "w") as f:
        for _ in range(3):
            f.write(json.dumps({"env_steps": steps, "length": 1000, "cleaned": cleaned,
                                "ate": [1] * 7}) + "\n")


def test_who_moves_and_the_move_trains_on(tmp_path):
    src = str(tmp_path / "runs")
    jtrain.main(["--runs", "connected:selfish:0:melting_pot:0.1", "connected:selfish:2",
                 "--out-root", src, "--steps", str(UPD), *TINY])
    coop, none = os.path.join(src, "connected_selfish_s0_coop10"), \
        os.path.join(src, "connected_selfish_s2")
    fake_games(coop, [0, 9, 1, 2, 8, 3, 7], UPD)     # cleaned most: seats 1, 4, 6
    fake_games(none, [5, 0, 6, 1, 2, 3, 4], UPD)     # seats 2, 0, 6; least: 3, 1 (and 4)
    assert cn.cleaning_rank(coop, UPD) == [1, 4, 6, 5, 3, 2, 0]
    out = str(tmp_path / "moves")
    into = cn.Move(True, 0, False, 2, "cleaners")
    away = cn.Move(False, 2, True, 0, "eaters")
    cn.make([src], out, moves=[into, away], steps=UPD, log=lambda line: None)
    cfg = json.load(open(os.path.join(out, into.name, "config.json")))
    assert cfg["coop_share"] == 0.1 and cfg["newcomers"]["kind"] == "into the co-op"
    assert (cfg["newcomers"]["newcomer_slots"], cfg["newcomers"]["origin_slots"]) == \
        ([1, 4, 6], [2, 0, 6])
    cfg = json.load(open(os.path.join(out, away.name, "config.json")))
    assert cfg["coop_share"] == 0.0 and cfg["newcomers"]["kind"] == "out of the co-op"
    assert (cfg["newcomers"]["newcomer_slots"], cfg["newcomers"]["origin_slots"]) == \
        ([4, 3, 1], [3, 2, 0])
    moved = pickle.load(open(os.path.join(out, into.name, "ckpt.pkl"), "rb"))
    home = pickle.load(open(os.path.join(none, "ckpt.pkl"), "rb"))
    assert np.array_equal(moved["params"]["params"]["pi"]["kernel"][4],
                          home["params"]["params"]["pi"]["kernel"][0])
    # It trains on with the host's co-op.
    jtrain.main(["--folders", os.path.join(out, into.name), "--resume", "--steps", str(UPD),
                 *TINY])
    assert pickle.load(open(os.path.join(out, into.name, "ckpt.pkl"), "rb"))["env_steps"] == UPD
