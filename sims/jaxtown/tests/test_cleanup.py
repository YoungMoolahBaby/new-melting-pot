"""Our JAX Clean Up follows Melting Pot's rules: checked on small hand-made maps.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python -m pytest sims/jaxtown/tests -q

The side-by-side statistics against real Melting Pot are in sims/jaxtown/compare.py.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cleanup as cu  # noqa: E402

OPEN = [
    "WWWWWWWWWWW",
    "W         W",
    "W         W",
    "W         W",
    "W         W",
    "W         W",
    "W         W",
    "W         W",
    "W         W",
    "WPPP      W",
    "WWWWWWWWWWW",
]
N, E, S, W = 0, 1, 2, 3


def setup(rows=OPEN, players=2, town="neighbours", **put):
    env = cu.CleanUp(rows, num_players=players)
    tw = env.town(town, river=None)
    s = env.reset(jax.random.PRNGKey(0), tw)
    return env, tw, place(s, **put), jax.jit(env.step)


def place(s, pos=None, ori=None, **fields):
    if pos is not None:
        fields["pos"] = jnp.asarray(pos, jnp.int32)
    if ori is not None:
        fields["ori"] = jnp.asarray(ori, jnp.int32)
    return s.replace(**fields)


def act(step, tw, s, actions, seed=0):
    return step(jax.random.PRNGKey(seed), s, jnp.asarray(actions), tw)


# --- the real map -------------------------------------------------------------------------

def test_real_map():
    env = cu.CleanUp()
    assert (env.num_river, env.num_apple, len(env.spawn_cells)) == (147, 122, 19)
    starts = {r: int(env.town("neighbours", r).start_dirt.sum())
              for r in ("melting_pot", "hughes", "clean")}
    assert starts == {"melting_pot": 79, "hughes": 60, "clean": 0}
    assert int(env.feed_cells.sum()) == 7
    assert bool(env.town("isolating").feed) and not bool(env.town("connected").feed)
    s = env.reset(jax.random.PRNGKey(1), env.town())
    obs, ready = env.observe(s, env.town())
    assert obs.shape == (7, 11, 11, cu.NUM_CHANNELS) and obs.dtype == jnp.uint8
    assert np.allclose(ready, 1.0)


# --- beams ----------------------------------------------------------------------------

BEAM_NORTH = {(5, 4), (4, 4), (3, 4), (5, 6), (4, 6), (3, 6), (4, 5), (3, 5), (2, 5)}


@pytest.mark.parametrize("facing", [N, E])
def test_zap_beam_reaches_exactly_melting_pots_cells(facing):
    env, tw, s, step = setup()
    hit = set()
    for r in range(1, 10):
        for c in range(1, 10):
            if (r, c) == (5, 5):
                continue
            s1 = place(s, pos=[(5, 5), (r, c)], ori=[facing, N])
            s2, *_ = act(step, tw, s1, [cu.ZAP, cu.NOOP])
            if not bool(s2.alive[1]):
                hit.add((r, c))
    expected = BEAM_NORTH if facing == N else {(5 + (c - 5), 5 + (5 - r)) for r, c in BEAM_NORTH}
    assert hit == expected


def test_zap_stops_at_first_player_and_at_walls():
    env, tw, s, step = setup(players=3, pos=[(5, 5), (4, 5), (3, 5)], ori=[N, N, N])
    s2, *_ = act(step, tw, s, [cu.ZAP, cu.NOOP, cu.NOOP])
    assert list(np.asarray(s2.alive)) == [True, False, True]
    walled = OPEN.copy()
    walled[4] = "W    W    W"
    env, tw, s, step = setup(rows=walled, pos=[(5, 5), (3, 5)], ori=[N, N])
    s2, *_ = act(step, tw, s, [cu.ZAP, cu.NOOP])
    assert bool(s2.alive[1])


def test_zapped_player_is_out_for_50_frames_then_respawns():
    env, tw, s, step = setup(pos=[(5, 5), (4, 5)], ori=[N, N])
    s, *_ = act(step, tw, s, [cu.ZAP, cu.NOOP])
    out = 1
    while not bool(s.alive[1]):
        s, obs, *_ = act(step, tw, s, [cu.NOOP, cu.NOOP], seed=out)
        out += 1
    assert out == cu.RESPAWN_FRAMES + 1          # gone after frames 1..50, back after 51
    assert tuple(np.asarray(s.pos[1])) in {tuple(x) for x in env.spawn_cells}


def test_zap_recharges_in_10_frames_and_ready_shows_it():
    env, tw, s, step = setup(pos=[(5, 5), (1, 1)], ori=[N, N])
    fired, ready = [], []
    for f in range(1, 30):
        s, obs, rd, *_ = act(step, tw, s, [cu.ZAP, cu.NOOP], seed=f)
        if int(s.zap_cd[0]) == cu.ZAP_COOLDOWN:
            fired.append(f)
        ready.append(float(rd[0]))
    assert fired == [1, 12, 23]
    assert ready[0] == 0.0 and ready[1] == pytest.approx(0.1) and ready[10] == 1.0


def test_clean_beam_recharges_in_2_and_stops_at_first_dirt():
    river = OPEN.copy()
    river[2] = "W    F    W"
    river[3] = "W    F    W"
    river[4] = "W   F F   W"
    env, tw, s, step = setup(rows=river, pos=[(5, 5), (9, 9)], ori=[N, N])
    fired, cleaned = [], []
    for f in range(1, 9):
        s, obs, rd, rew, done, info = act(step, tw, s, [cu.CLEAN, cu.NOOP], seed=f)
        if int(s.clean_cd[0]) == cu.CLEAN_COOLDOWN:
            fired.append(f)
            cleaned.append(int(info["cleaned"][0]))
    assert fired == [1, 4, 7]
    # Shot 1: both side lines hit their dirt (4,4) and (4,6); centre hits (3,5) first.
    # Shot 2: centre reaches (2,5). Then nothing is left in reach.
    assert cleaned == [3, 1, 0]
    assert int(s.dirt.sum()) == 0


# --- moving and eating -------------------------------------------------------------------

def test_moves_are_relative_to_facing():
    env, tw, s, step = setup(pos=[(5, 5), (9, 9)], ori=[E, N])
    expect = [(cu.FORWARD, (5, 6), E), (cu.STEP_RIGHT, (6, 6), E), (cu.BACKWARD, (6, 5), E),
              (cu.STEP_LEFT, (5, 5), E), (cu.TURN_RIGHT, (5, 5), S), (cu.FORWARD, (6, 5), S),
              (cu.TURN_LEFT, (6, 5), E)]
    for a, pos, ori in expect:
        s, *_ = act(step, tw, s, [a, cu.NOOP])
        assert (tuple(np.asarray(s.pos[0])), int(s.ori[0])) == (pos, ori)


def test_bumping_walls_and_players():
    env, tw, s, step = setup(pos=[(1, 1), (9, 9)], ori=[N, N])
    s2, *_ = act(step, tw, s, [cu.FORWARD, cu.NOOP])                 # wall ahead
    assert tuple(np.asarray(s2.pos[0])) == (1, 1)
    # A swap fails both ways; two players into one cell: exactly one gets it.
    s2, *_ = act(step, tw, place(s, pos=[(5, 5), (5, 6)], ori=[E, W]), [cu.FORWARD, cu.FORWARD])
    assert [tuple(x) for x in np.asarray(s2.pos)] == [(5, 5), (5, 6)]
    winners = set()
    for seed in range(20):
        s2, *_ = act(step, tw, place(s, pos=[(5, 4), (5, 6)], ori=[E, W]),
                     [cu.FORWARD, cu.FORWARD], seed=seed)
        at = [tuple(x) for x in np.asarray(s2.pos)]
        assert at.count((5, 5)) == 1
        winners.add(at.index((5, 5)))
    assert winners == {0, 1}                     # players go in random order
    # Following: the one behind gets through only if the leader went first.
    behind_moved = set()
    for seed in range(20):
        s2, *_ = act(step, tw, place(s, pos=[(5, 4), (5, 5)], ori=[E, E]),
                     [cu.FORWARD, cu.FORWARD], seed=seed)
        assert tuple(np.asarray(s2.pos[1])) == (5, 6)
        behind_moved.add(tuple(np.asarray(s2.pos[0])) == (5, 5))
    assert behind_moved == {True, False}


def test_walking_onto_an_apple_eats_it():
    orchard = OPEN.copy()
    orchard[1] = "WFFFFFFFFFW"     # an all-dirty river, so no apple regrows mid-test
    orchard[4] = "W    B    W"
    env, tw, s, step = setup(rows=orchard, pos=[(5, 5), (9, 9)], ori=[N, N])
    s = s.replace(apple=jnp.ones_like(s.apple))
    s2, obs, rd, reward, done, info = act(step, tw, s, [cu.FORWARD, cu.NOOP])
    assert float(reward[0]) == 1.0 and float(info["ate"][0]) == 1.0
    assert not bool(s2.apple[0])
    # An apple that appears under a standing player is eaten at once, as in Melting Pot.
    s3 = s2.replace(apple=jnp.ones_like(s2.apple))
    s4, obs, rd, reward, *_ = act(step, tw, s3, [cu.NOOP, cu.NOOP])
    assert float(reward[0]) == 1.0 and not bool(s4.apple[0])


def test_apples_growing_under_a_standing_player_get_eaten():
    orchard = OPEN.copy()
    orchard[5] = "W    B    W"     # no river: dirt fraction 0, apples grow at 5% a frame
    env, tw, s, step = setup(rows=orchard, pos=[(5, 5), (9, 9)], ori=[N, N])
    ate = 0.0
    for f in range(1, 201):
        s, obs, rd, reward, done, info = act(step, tw, s, [cu.TURN_LEFT, cu.NOOP], seed=f)
        ate += float(info["ate"][0])
        assert not bool(s.apple[0])                 # never left sitting under the player
    assert ate >= 1


# --- the river and the orchard ------------------------------------------------------------

def test_apples_grow_at_5_percent_on_a_clean_river_and_not_past_40_percent():
    env = cu.CleanUp()
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)

    def one_frame(key, town):
        s = env.reset(key, town)
        s2, *_ = env.step(jax.random.fold_in(key, 1), s, jnp.zeros(7, jnp.int32), town)
        return s2.apple.sum()

    grown = jax.jit(jax.vmap(one_frame, in_axes=(0, None)))
    clean = float(grown(keys, env.town("neighbours", "clean")).mean())
    assert clean == pytest.approx(122 * cu.MAX_GROWTH, rel=0.05)      # 6.1 per frame
    assert float(grown(keys, env.town("neighbours", "hughes")).max()) == 0   # 41% dirty


def test_dirt_starts_spawning_after_frame_50():
    river = OPEN.copy()
    for r in (1, 2, 3, 4):
        river[r] = "WHHHHHHHHHW"                                # 36 clean river cells
    env, tw, s, step = setup(rows=river, pos=[(8, 1), (9, 9)], ori=[N, N])
    dirt = []
    for f in range(1, 81):
        s, *_ = act(step, tw, s, [cu.NOOP, cu.NOOP], seed=f)
        dirt.append(int(s.dirt.sum()))
    assert dirt[49] == 0                                        # after frame 50: none yet
    assert 7 <= dirt[-1] <= 24                                  # ~0.5 a frame for 30 frames
    assert int(env.town("neighbours", river="clean").dirt_spawns) == 0


def test_capped_dirt_stops_at_40_percent_and_melting_pots_does_not():
    # From a clean river with dirt arriving, nobody cleaning, for 800 frames (games
    # last at least 1,099). Capped: dirt stops at 59 of 147 cells, the first count at
    # or past 40%. Melting Pot's: it keeps coming (~0.5 a frame for 750 frames).
    env = cu.CleanUp()
    noop = jnp.zeros(7, jnp.int32)

    def dirt_counts(town, key):
        s = env.reset(key, town)

        def frame(s, k):
            s, *_ = env.step(k, s, noop, town)
            return s, s.dirt.sum()

        return jax.lax.scan(frame, s, jax.random.split(key, 800))[1]

    run = jax.jit(jax.vmap(dirt_counts, in_axes=(None, 0)))
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    counts = {}
    for dirt in ("capped", "melting_pot"):
        town = env.town("neighbours", "clean", dirt=dirt).replace(dirt_spawns=jnp.asarray(True))
        counts[dirt] = np.asarray(run(town, keys))
    assert counts["capped"].max() == 59 and (counts["capped"][:, -1] == 59).all()
    assert counts["melting_pot"][:, -1].min() > 120
    assert float(env.town().dirt_cap) == 1.0                      # Melting Pot's by default
    with pytest.raises(ValueError, match="dirt must be one of"):
        env.town(dirt="gentle")


def test_game_length_follows_melting_pot():
    keys = jax.random.split(jax.random.PRNGKey(3), 20000)
    ends = jax.vmap(cu.episode_ends, in_axes=(0, None))
    rate = {f: float(ends(keys, jnp.int32(f)).mean()) for f in (1098, 1099, 1100, 1199, 5000)}
    assert rate[1098] == 0 and rate[1100] == 0 and rate[5000] == 1
    assert rate[1099] == pytest.approx(0.2, abs=0.02) and rate[1199] == pytest.approx(0.2, abs=0.02)


# --- what players see, and the towns ------------------------------------------------------------

@pytest.mark.parametrize("facing", [N, E, S, W])
def test_view_turns_with_you(facing):
    orchard = OPEN.copy()
    # An apple 2 ahead and 1 to the right of a player at (5,5), for each facing.
    ahead, right = cu.DIRS[facing], cu.DIRS[(facing + 1) % 4]
    r, c = np.array([5, 5]) + 2 * ahead + right
    orchard[r] = orchard[r][:c] + "B" + orchard[r][c + 1:]
    env, tw, s, step = setup(rows=orchard, pos=[(5, 5), (9, 9)], ori=[facing, N])
    s = s.replace(apple=jnp.ones_like(s.apple))
    obs, _ = env.observe(s, tw)
    v = np.asarray(obs[0])
    assert v[cu.SELF_ROW - 2, cu.SELF_COL + 1, cu.CH["apple"]] == 1
    assert v[..., cu.CH["apple"]].sum() == 1
    assert v[cu.SELF_ROW, cu.SELF_COL, cu.CH["self"]] == 1
    assert v[..., cu.CH["self"]].sum() == 1


def test_others_have_identity_except_in_strangers():
    for town, expect_id in (("neighbours", 1), ("strangers", 0)):
        env, tw, s, step = setup(town=town, pos=[(5, 5), (3, 5)], ori=[N, N])
        v = np.asarray(env.observe(s, tw)[0][0])
        other = (cu.SELF_ROW - 2, cu.SELF_COL)
        assert v[other][cu.CH["someone"]] == 1
        assert v[other][cu.CH["player1"]] == expect_id
        assert v[cu.SELF_ROW, cu.SELF_COL, cu.CH["someone"]] == 0       # you're "self"


def test_badge_shows_for_50_frames_after_cleaning_only_in_connected_town():
    river = OPEN.copy()
    river[4] = "W    F    W"
    for town, expect in (("connected", 50), ("neighbours", 0)):
        env, tw, s, step = setup(rows=river, town=town, pos=[(5, 5), (7, 5)], ori=[N, N])
        shown, first = 0, None
        for f in range(1, 70):
            a = cu.CLEAN if f == 1 else cu.NOOP
            s, obs, *_ = act(step, tw, s, [a, cu.NOOP], seed=f)
            # Player 1 stands 2 behind player 0, facing the same way: player 0 is 2 ahead.
            if np.asarray(obs[1])[cu.SELF_ROW - 2, cu.SELF_COL, cu.CH["badge"]]:
                shown += 1
                first = first or f
        assert shown == expect
        if expect:
            assert first == 2          # the badge lags the cleaning by a frame, as in Lua


def test_feed_pays_every_4_frames_only_in_isolating_town():
    env = cu.CleanUp()
    r, c = cu.spec.FEED_CELLS[0]
    for town, expect in (("isolating", 1.0), ("neighbours", 0.0)):
        tw = env.town(town)
        s = env.reset(jax.random.PRNGKey(0), tw)
        # Player 0 one step south of the tile, facing it; everyone else far away.
        pos = np.asarray(s.pos).copy()
        pos[0] = (r + 1, c)
        s = place(s, pos=pos, ori=[N] * 7)
        step = jax.jit(env.step)
        paid = []
        for f in range(1, 42):
            a = [cu.FORWARD if f == 1 else cu.NOOP] + [cu.NOOP] * 6
            s, obs, rd, rew, done, info = act(step, tw, s, a, seed=f)
            if float(info["feed"][0]):
                paid.append((f, float(info["feed"][0])))
        # Arrive in frame 1; the countdown starts that frame, as in Melting Pot.
        assert sum(x for _, x in paid) == pytest.approx(expect)
        if expect:
            assert [f for f, _ in paid] == list(range(4, 41, 4))


def test_coop_payout_moves_a_share_of_each_apple_to_recent_cleaners():
    ate = jnp.array([1.0, 0, 0, 1, 0, 0, 0])
    lit = jnp.array([False, True, True, True, False, False, False])
    got = np.asarray(cu.coop_payout(ate, lit, 0.3))
    # A pot of 0.6 split by the 3 lit players, 0.2 each; each eater pays 0.3 (player 3
    # eats and is lit, so pays 0.3 and gets 0.2 back).
    np.testing.assert_allclose(got, [-0.3, 0.2, 0.2, -0.1, 0, 0, 0], atol=1e-6)
    assert abs(got.sum()) < 1e-6
    assert not np.asarray(cu.coop_payout(ate, jnp.zeros(7, bool), 0.3)).any()  # nobody lit
    assert not np.asarray(cu.coop_payout(ate, lit, 0.0)).any()
    with pytest.raises(ValueError, match="coop_share"):
        cu.CleanUp().town("connected", coop_share=1.5)


def test_coop_changes_rewards_only_never_the_game():
    env, n, frames = cu.CleanUp(), 8, 1500

    def play(tw):
        def one(es, k):
            ka, ks = jax.random.split(k)
            acts = jax.random.randint(ka, (n, env.num_players), 0, 9)
            out = jax.vmap(env.step, (0, 0, 0, None))(jax.random.split(ks, n), es, acts, tw)
            return out[0], out
        es = jax.vmap(env.reset, (0, None))(jax.random.split(jax.random.PRNGKey(3), n), tw)
        keys = jax.random.split(jax.random.PRNGKey(4), frames)
        return jax.device_get(jax.jit(lambda e: jax.lax.scan(one, e, keys))(es)[1])

    plain = play(env.town("connected", "hughes"))
    coop = play(env.town("connected", "hughes", coop_share=0.5))
    for i in (0, 1, 2, 4, 5):            # state, views, ready, done, info: the same game
        for a, b in zip(jax.tree_util.tree_leaves(plain[i]), jax.tree_util.tree_leaves(coop[i])):
            assert np.array_equal(a, b)
    r0, r1 = np.asarray(plain[3]), np.asarray(coop[3])
    np.testing.assert_allclose(r0.sum(-1), r1.sum(-1), atol=1e-5)   # nothing made or lost
    assert np.abs(r0 - r1).sum() > 1                                 # but apples changed hands
