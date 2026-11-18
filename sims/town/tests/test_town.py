"""Checks that the four towns are what they claim to be.

Run inside WSL after sims/town/install_lua.sh:
    ~/venvs/meltingpot/bin/python -m pytest sims/town/tests -q
"""
import numpy as np
import pytest

import town_config as tc

BADGE_RGB = np.array(tc.BADGE_PALETTE["*"][:3], dtype=np.uint8)
FEED_RGB = np.array(tc.FEED_PALETTE["o"][:3], dtype=np.uint8)
FEED_PIXELS_PER_TILE = sum(row.count("o") for row in tc.FEED_SHAPE.strip().split("\n"))


def count_colour(img, rgb):
    return int(np.all(img == rgb, axis=-1).sum())


@pytest.mark.parametrize("town", tc.TOWNS)
def test_town_builds_and_steps(town):
    env = tc.build_town(town)
    ts = env.reset()
    assert len(ts.observation) == tc.NUM_PLAYERS
    for name in tc.METRIC_OBS:
        assert name in ts.observation[0]
    for _ in range(20):
        ts = env.step([0] * tc.NUM_PLAYERS)
    env.close()


def avatar_palettes(town):
    config = tc.get_config(town)
    settings = tc.build(config.default_player_roles, config)
    palettes = []
    for obj in settings["simulation"]["gameObjects"]:
        if obj["name"].startswith("avatar"):
            for comp in obj["components"]:
                if comp["component"] == "Appearance":
                    palettes.append(str(comp["kwargs"]["palettes"]))
    return palettes


def test_strangers_all_look_alike_neighbours_dont():
    assert len(set(avatar_palettes("strangers"))) == 1
    assert len(set(avatar_palettes("neighbours"))) == tc.NUM_PLAYERS


def test_feed_tiles_only_in_isolating_town():
    for town in tc.TOWNS:
        env = tc.build_town(town)
        world = env.reset().observation[0]["WORLD.RGB"]
        expected = 7 * FEED_PIXELS_PER_TILE if town == "isolating" else 0
        assert count_colour(world, FEED_RGB) == expected, town
        env.close()


def feed_income(town, steps=3000, seed=0):
    """Reward that isn't apples, from random players (feed pays every frame here)."""
    env = tc.build_town(town, feed_reward=0.1, feed_period=1)
    env.reset()
    rng = np.random.default_rng(seed)
    extra = 0.0
    for _ in range(steps):
        ts = env.step(list(rng.integers(0, len(tc.clean_up.ACTION_SET), size=tc.NUM_PLAYERS)))
        for i in range(tc.NUM_PLAYERS):
            extra += float(ts.reward[i]) - float(ts.observation[i]["PLAYER_ATE_APPLE"])
        if ts.last():
            env.reset()
    env.close()
    return extra


def test_feed_pays_only_in_isolating_town():
    assert feed_income("isolating") > 0
    assert abs(feed_income("neighbours")) < 1e-9


def badge_counts(town, steps=400):
    """Badge pixels per frame while DeepMind's trained cleaner bots play."""
    from meltingpot import bot
    from meltingpot.scenario import PERMITTED_OBSERVATIONS
    env = tc.build_town(town)
    names = ["clean_up__cleaner_0", "clean_up__cleaner_1"] * 4
    policies = [bot.build(n) for n in names[:tc.NUM_PLAYERS]]
    states = [p.initial_state() for p in policies]
    ts = env.reset()
    counts = []
    for _ in range(steps):
        actions = []
        for i, p in enumerate(policies):
            obs = {k: v for k, v in ts.observation[i].items() if k in PERMITTED_OBSERVATIONS}
            a, states[i] = p.step(ts._replace(observation=obs, reward=ts.reward[i]), states[i])
            actions.append(int(a))
        ts = env.step(actions)
        counts.append(count_colour(ts.observation[0]["WORLD.RGB"], BADGE_RGB))
    for p in policies:
        p.close()
    env.close()
    return np.array(counts)


def test_badges_appear_and_fade_only_in_connected_town():
    counts = badge_counts("connected")
    assert counts.max() > 0, "no badge ever appeared"
    assert np.any(np.diff(counts) < 0), "badges never faded"
    assert badge_counts("neighbours", steps=200).max() == 0


def test_shared_map_is_melting_pots():
    # spec.py keeps a copy so the JAX version needn't import Melting Pot.
    assert tc.spec.ASCII_MAP.strip("\n") == tc.clean_up.ASCII_MAP.strip("\n")


def test_river_starts():
    """Starting dirt: 79 dirty river cells as shipped, 60 Hughes-style, 0 clean."""
    for river, dirty in (("melting_pot", 79), ("hughes", tc.HUGHES_DIRT), ("clean", 0)):
        m = tc.town_map("isolating", river)
        assert m.count("F") == dirty, river
        assert m.count("F") + m.count("H") == 147, river   # same river, only cleaner
        assert m.count("Z") == len(tc.FEED_CELLS)


def random_play(river, steps=600, seed=0):
    """Apples eaten and dirt cleaned by 7 random players."""
    env = tc.build_town("neighbours", river=river)
    env.reset()
    rng = np.random.default_rng(seed)
    ate = cleaned = 0.0
    for _ in range(steps):
        ts = env.step(list(rng.integers(0, len(tc.clean_up.ACTION_SET), size=tc.NUM_PLAYERS)))
        ate += sum(float(o["PLAYER_ATE_APPLE"]) for o in ts.observation)
        cleaned += sum(float(o["PLAYER_CLEANED"]) for o in ts.observation)
        if ts.last():
            env.reset()
    env.close()
    return ate, cleaned


def test_clean_river_grows_apples_and_never_gets_dirty():
    ate_clean, cleaned_clean = random_play("clean")
    ate_shipped, cleaned_shipped = random_play("melting_pot")
    # As shipped, random players can't clean enough for apples to grow.
    assert ate_clean > ate_shipped
    # Random players clean whatever dirt there is; on the clean river none ever appears.
    assert cleaned_shipped > 0 and cleaned_clean == 0
