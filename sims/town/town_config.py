"""The four towns of experiment 1, as Melting Pot configs built on Clean Up.

    strangers   everyone looks the same (you still see yourself as blue)
    neighbours  everyone has their own colour (normal Clean Up)
    connected   neighbours + the connecting AI: a gold badge rides on anyone who
                cleaned the river lately, so everyone can see who's pitching in
    isolating   neighbours + the isolating AI: "feed" tiles in the living area pay
                a small private reward every few frames to whoever stands on one

Reuses meltingpot.configs.substrates.clean_up (map, prefabs, scene, avatars) and
the clean_up_town Lua level in sims/town/lua (install it with install_lua.sh).

    from town_config import build_town
    env = build_town("connected")
"""
from __future__ import annotations

import copy

from meltingpot import substrate
from meltingpot.configs.substrates import clean_up
from meltingpot.utils.substrates import shapes, specs
from ml_collections import config_dict

import spec
from spec import FEED_CELLS, HUGHES_DIRT, NUM_PLAYERS, RIVERS, TOWNS  # noqa: F401  (re-exported)

# Per-player numbers we log (never given to the learning players).
METRIC_OBS = ("PLAYER_CLEANED", "PLAYER_ATE_APPLE", "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP")

# Everyone else's colour in the strangers town: Clean Up's first non-self colour.
STRANGER_COLOUR = clean_up.human_readable_colors[0]

BADGE_SHAPE = """
x*x**x*x
x******x
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""
BLANK_SHAPE = "\n" + "\n".join(["xxxxxxxx"] * 8) + "\n"
BADGE_PALETTE = {"x": (0, 0, 0, 0), "*": (255, 210, 40, 255)}

FEED_SHAPE = """
xxxxxxxx
x######x
x#oooo#x
x#o##o#x
x#oooo#x
x#oooo#x
x######x
xxxxxxxx
"""
FEED_PALETTE = {"x": (0, 0, 0, 0), "#": (40, 40, 48, 255), "o": (220, 70, 220, 255)}


def town_map(town: str, river: str = "melting_pot") -> str:
    """Clean Up's map, with the river's starting dirt set by `river` and feed tiles
    ('Z') added for the isolating town (see spec.map_rows)."""
    return "\n" + "\n".join(spec.map_rows(town, river)) + "\n"


def feed_prefab(reward: float, period: int) -> dict:
    return {
        "name": "feed",
        "components": [
            {"component": "StateManager",
             "kwargs": {"initialState": "feed",
                        "stateConfigs": [{"state": "feed", "layer": "lowerPhysical",
                                          "sprite": "Feed"}]}},
            {"component": "Transform"},
            {"component": "Appearance",
             "kwargs": {"renderMode": "ascii_shape", "spriteNames": ["Feed"],
                        "spriteShapes": [FEED_SHAPE], "palettes": [FEED_PALETTE],
                        "noRotates": [True]}},
            {"component": "Feed", "kwargs": {"rewardAmount": reward, "period": period}},
        ],
    }


def badge_object(player_idx: int, frames_visible: int) -> dict:
    """A badge that rides on avatar `player_idx` (0-based) after it cleans."""
    lua_idx = player_idx + 1
    return {
        "name": f"badge{lua_idx}",
        "components": [
            {"component": "StateManager",
             "kwargs": {"initialState": "badgeWait",
                        "stateConfigs": [
                            {"state": "badgeOn", "layer": "badgeLayer", "sprite": "Badge"},
                            {"state": "badgeOff", "layer": "badgeLayer", "sprite": "BadgeOff"},
                            {"state": "badgeWait"},
                        ]}},
            {"component": "Transform"},
            {"component": "Appearance",
             "kwargs": {"renderMode": "ascii_shape", "spriteNames": ["Badge", "BadgeOff"],
                        "spriteShapes": [BADGE_SHAPE, BLANK_SHAPE],
                        "palettes": [BADGE_PALETTE, BADGE_PALETTE],
                        "noRotates": [True, True]}},
            {"component": "AvatarConnector",
             "kwargs": {"playerIndex": lua_idx, "aliveState": "badgeOff",
                        "waitState": "badgeWait"}},
            {"component": "CleanerBadge",
             "kwargs": {"playerIndex": lua_idx, "onState": "badgeOn", "offState": "badgeOff",
                        "waitState": "badgeWait", "framesVisible": frames_visible}},
        ],
    }


def avatar_objects(town: str, num_players: int) -> list:
    # Clean Up reads this module flag when it builds each avatar; it adds the
    # per-player metrics we log (PLAYER_CLEANED etc.). Our process only.
    clean_up._ENABLE_DEBUG_OBSERVATIONS = True
    avatars = []
    for i in range(num_players):
        avatar = copy.deepcopy(clean_up.create_avatar_object(i, clean_up.TARGET_SPRITE_SELF))
        if town == "strangers":
            for comp in avatar["components"]:
                if comp["component"] == "Appearance":
                    comp["kwargs"]["palettes"] = [shapes.get_palette(STRANGER_COLOUR)]
        avatars.append(avatar)
    return avatars


def build(roles, config):
    """lab2d settings for one town (Melting Pot calls this with roles + config)."""
    town = config.town
    n = len(roles)
    objects = avatar_objects(town, n)
    if town == "connected":
        objects += [badge_object(i, config.badge_frames) for i in range(n)]
    prefabs = dict(clean_up.create_prefabs())
    char_map = dict(clean_up.CHAR_PREFAB_MAP)
    if town == "isolating":
        prefabs["feed"] = feed_prefab(config.feed_reward, config.feed_period)
        char_map["Z"] = {"type": "all", "list": ["sand", "feed"]}
    scene = clean_up.create_scene()
    if config.river == "clean":  # no dirt ever spawns
        spawners = [c for c in scene["components"] if c["component"] == "DirtSpawner"]
        assert len(spawners) == 1, "Clean Up's scene changed: no single DirtSpawner to switch off"
        spawners[0]["kwargs"]["dirtSpawnProbability"] = 0.0
    return dict(
        levelName="clean_up_town",
        levelDirectory="meltingpot/lua/levels",
        numPlayers=n,
        maxEpisodeLengthFrames=5000,
        spriteSize=8,
        topology="BOUNDED",
        simulation={
            "map": town_map(town, config.river),
            "gameObjects": objects,
            "scene": scene,
            "prefabs": prefabs,
            "charPrefabMap": char_map,
        },
    )


def get_config(town: str = "neighbours", *, feed_reward: float = 0.1, feed_period: int = 4,
               badge_frames: int = 50, river: str = "melting_pot",
               world_rgb: bool = True, positions: bool = False) -> config_dict.ConfigDict:
    """Config for one town. Feed pays `feed_reward` every `feed_period` frames
    (default 0.025 per frame; an apple is 1.0). `river` sets the starting dirt
    (see RIVERS). `world_rgb=False` drops the bird's-eye frame (only clips need
    it), which makes training steps cheaper. `positions=True` adds each player's
    POSITION (x=column, y=row) and ORIENTATION, for checking the JAX version."""
    if town not in TOWNS:
        raise ValueError(f"town must be one of {TOWNS}, not {town!r}")
    if river not in RIVERS:
        raise ValueError(f"river must be one of {RIVERS}, not {river!r}")
    config = clean_up.get_config()
    with config.unlocked():
        config.town = town
        config.river = river
        config.feed_reward = float(feed_reward)
        config.feed_period = int(feed_period)
        config.badge_frames = int(badge_frames)
        config.individual_observation_names = [
            "RGB", "READY_TO_SHOOT", "NUM_OTHERS_WHO_CLEANED_THIS_STEP", *METRIC_OBS]
        obs_spec = {
            "RGB": specs.OBSERVATION["RGB"],
            "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
            "NUM_OTHERS_WHO_CLEANED_THIS_STEP": specs.float64(),
            **{name: specs.float64() for name in METRIC_OBS},
        }
        if positions:
            config.individual_observation_names += ["POSITION", "ORIENTATION"]
            obs_spec["POSITION"] = specs.OBSERVATION["POSITION"]
            obs_spec["ORIENTATION"] = specs.OBSERVATION["ORIENTATION"]
        if world_rgb:
            obs_spec["WORLD.RGB"] = specs.rgb(168, 240)
        else:
            config.global_observation_names = []
        config.timestep_spec = specs.timestep(obs_spec)
        config.lab2d_settings_builder = build
    return config


def build_town(town: str = "neighbours", **kwargs) -> substrate.substrate.Substrate:
    config = get_config(town, **kwargs)
    return substrate.build_from_config(config, roles=config.default_player_roles)
