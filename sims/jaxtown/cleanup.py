"""Clean Up in JAX, with Melting Pot's rules and our four towns.

Built from SocialJax's Clean Up (third_party/socialjax, Apache 2.0, commit 18e2171):
the same idea (the whole map as arrays, each player's view a rotated slice of it,
thousands of games at once on the GPU), rewritten so the rules match DeepMind's
Melting Pot Clean Up (dm-meltingpot 2.4.1), which our first runs and the published
results use.

One frame, in Melting Pot's order (updater priorities, higher first; within one
priority the players go in random order, as dmlab2d's grid.cc shuffles them):

    450  badge timers (our CleanerBadge, sims/town/lua/clean_up_town)
    150  turn or move (avatar_library Avatar); a move into a wall or another player
         fails; walking onto an apple eats it (+1); feed tiles pay (our Feed)
    140  zap beams, then clean beams (Zapper, Cleaner); recharge 10 and 2 frames
    135  zapped players come back after 50 frames, on a free spawn point
    100  apples grow (0.05 x how far the river is below 40% dirty), and one that
         grows under a player is eaten at once; one dirt cell may appear (p 0.5,
         from frame 51); the game may end

Beams (dmlab2d Grid::HitBeamActual, length 3, radius 1): the 3 cells straight
ahead, and the cell on each side of the player plus the 2 ahead of it. Each of the
3 lines stops at a wall; a zap line stops at the first player it hits, a clean line
at the first dirt it cleans.

What players see: an 11x11 grid (5 cells left and right, 9 ahead, 1 behind, as
Melting Pot's view), turned so ahead is up, one 0/1 channel per thing (CHANNELS).
Melting Pot gives pixels instead; same information, no picture-reading to learn.
Others' facing isn't shown (Melting Pot's avatars don't rotate). In the strangers
town the per-player channels stay empty, so others are just "someone".

Changed from SocialJax: relative movement and turning; Melting Pot's view offsets;
beam shape and blocking; zap timeout (50) and recharges (10, 2); dirt fraction over
the 147 river cells only; independent random draws for apples, dirt and spawning
(SocialJax reuses one key); random game length as Melting Pot (ends at frame 1,099
or a multiple of 100 after it, 20% each time, at most 5,000); each player's own
apples (SocialJax shares by default); river starts; the four towns.

One rule of our own, off by default: town(dirt="capped") stops dirt arriving while 40%
or more of the river is dirty, as the open Sequential Social Dilemma Games Cleanup does
(spec.DIRT_RULES). Melting Pot's dirt keeps coming until the river is full.

And a co-op, also ours and off by default: town(coop_share=f) has the connecting AI take
a share f of every apple eaten and split it evenly between the players who cleaned in the
last 50 frames (the badge's record, coop_payout). It changes rewards only, never the game.

Known simplifications, kept small: zap beams all fire before clean beams (Melting
Pot interleaves them in random order); while out, a zapped player still sees the
live world from where they were hit (what Melting Pot shows an off-board player is
unverified); respawned players keep their facing (dmlab2d's default, unverified).
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "town"))
import spec  # noqa: E402

# Melting Pot's ACTION_SET order.
NOOP, FORWARD, BACKWARD, STEP_LEFT, STEP_RIGHT, TURN_LEFT, TURN_RIGHT, ZAP, CLEAN = range(9)
NUM_ACTIONS = 9

# Facing: 0 north, 1 east, 2 south, 3 west (rows grow southward).
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], np.int32)
# Where a move goes, relative to facing (0 ahead, 1 right, 2 back, 3 left); -1 no move.
MOVE_REL = np.array([-1, 0, 2, 3, 1, -1, -1, -1, -1], np.int32)
TURN = np.array([0, 0, 0, 0, 0, -1, 1, 0, 0], np.int32)

VIEW_AHEAD, VIEW_BEHIND, VIEW_SIDE = 9, 1, 5
VIEW = 2 * VIEW_SIDE + 1                       # 11 (and 9 + 1 + 1 rows)
SELF_ROW, SELF_COL = VIEW_AHEAD, VIEW_SIDE     # where you are in your own view
PAD = 10                                       # map padding for view slices

BEAM_LENGTH, BEAM_RADIUS = 3, 1
ZAP_COOLDOWN, CLEAN_COOLDOWN, RESPAWN_FRAMES = 10, 2, 50
DIRT_PROB, DIRT_DELAY = 0.5, 50
DEPLETION, RESTORATION, MAX_GROWTH = 0.4, 0.0, 0.05
END_FIRST, END_EVERY, END_PROB, END_MAX = 1099, 100, 0.2, 5000

CHANNELS = ("wall", "apple", "dirt", "water", "grass", "feed", "self", "badge", "someone",
            *[f"player{k}" for k in range(spec.NUM_PLAYERS)])
CH = {name: i for i, name in enumerate(CHANNELS)}
NUM_CHANNELS = len(CHANNELS)


def _view_offsets() -> np.ndarray:
    """[4 facings, 11, 11, 2]: map offset of each view cell; view row 0 is farthest ahead."""
    off = np.zeros((4, VIEW, VIEW, 2), np.int32)
    for o in range(4):
        ahead, right = DIRS[o], DIRS[(o + 1) % 4]
        for i in range(VIEW):
            for j in range(VIEW):
                off[o, i, j] = (VIEW_AHEAD - i) * ahead + (j - VIEW_SIDE) * right
    return off


def _beam_cells() -> np.ndarray:
    """[4 facings, 3 lines, BEAM_LENGTH, 2]: the cells each beam line visits, in order."""
    assert BEAM_RADIUS == 1
    cells = np.zeros((4, 3, BEAM_LENGTH, 2), np.int32)
    for o in range(4):
        ahead, right = DIRS[o], DIRS[(o + 1) % 4]
        starts = (-right, right, ahead)          # dmlab2d: west side, east side, centre
        for line, start in enumerate(starts):
            for k in range(BEAM_LENGTH):          # side lines: length - r + 1 = 3 cells
                cells[o, line, k] = start + k * ahead
    return cells


VIEW_OFFSETS = _view_offsets()
BEAM_CELLS = _beam_cells()


class Town(struct.PyTreeNode):
    """Everything that differs between towns, as data: one compiled step serves all."""
    identity: jnp.ndarray       # bool: others have their own colour (not in strangers)
    badge: jnp.ndarray          # bool: connecting AI
    feed: jnp.ndarray           # bool: isolating AI
    feed_reward: jnp.ndarray    # float
    feed_period: jnp.ndarray    # int
    badge_frames: jnp.ndarray   # int
    start_dirt: jnp.ndarray     # [R] bool: which river cells start dirty
    dirt_spawns: jnp.ndarray    # bool: False on the clean river (sanity check)
    dirt_cap: jnp.ndarray       # float: no new dirt while this share is dirty (1.0: never stops)
    coop_share: jnp.ndarray     # float: share of each apple paid to recent cleaners (0: none)


class State(struct.PyTreeNode):
    pos: jnp.ndarray            # [P, 2] int32 (row, col)
    ori: jnp.ndarray            # [P] int32 facing
    alive: jnp.ndarray          # [P] bool (False while zapped out)
    zapped_at: jnp.ndarray      # [P] int32 frame of the last zap
    zap_cd: jnp.ndarray         # [P] int32 frames until the zap beam is ready
    clean_cd: jnp.ndarray       # [P] int32
    dirt: jnp.ndarray           # [R] bool, the river's dirt-capable cells
    apple: jnp.ndarray          # [A] bool, the orchard's apple cells
    badge_t: jnp.ndarray        # [P] int32 frames of badge left
    cleaned_prev: jnp.ndarray   # [P] bool cleaned last frame (the badge lags a frame)
    feed_t: jnp.ndarray         # [P] int32 frames until the next feed payout
    t: jnp.ndarray              # int32 frames played


def coop_payout(ate, lit, share):
    """The co-op's change to each player's reward this frame, [P] (it sums to 0): every
    apple eaten pays `share` of itself into a pot, split evenly between the `lit` players
    (cleaned recently), an eater among them included. While nobody is lit, eaters keep
    their apples whole."""
    n = lit.sum()
    pot = share * ate.sum()
    return jnp.where(n > 0, pot * lit / jnp.maximum(n, 1) - share * ate, 0.0)


def episode_ends(key, frame):
    """Melting Pot's StochasticIntervalEpisodeEnding, as measured from 19,918 games:
    the game ends after frame 1,099 + 100k with probability 0.2 each time, or at 5,000."""
    check = (frame >= END_FIRST) & ((frame - END_FIRST) % END_EVERY == 0)
    return (frame >= END_MAX) | (check & (jax.random.uniform(key) < END_PROB))


class CleanUp:
    """Clean Up on one map, for `num_players` players. Use `town()` for the settings
    of each town, then reset/step (vmap them for many games at once)."""

    def __init__(self, rows: list[str] | None = None, num_players: int = spec.NUM_PLAYERS):
        rows = rows or spec.map_rows("neighbours", "melting_pot")
        self.rows = rows
        self.num_players = num_players
        h, w = len(rows), max(len(r) for r in rows)
        self.shape = (h, w)
        grid = np.array([list(r.ljust(w)) for r in rows])
        self.wall = grid == "W"
        self.river_cells = np.argwhere(np.isin(grid, ["H", "F"])).astype(np.int32)
        self.apple_cells = np.argwhere(np.isin(grid, ["B", "T"])).astype(np.int32)
        self.spawn_cells = np.argwhere(grid == "P").astype(np.int32)
        self.water = np.isin(grid, ["H", "F", "S", "~"])
        self.grass = np.isin(grid, ["B", "s"])
        feed = np.zeros((h, w), bool)
        for r, c in spec.FEED_CELLS:
            if r < h and c < w:
                feed[r, c] = grid[r, c] in (" ", "Z")
        feed |= grid == "Z"
        self.feed_cells = feed
        assert len(self.spawn_cells) >= num_players, "need a spawn point per player"
        self.num_river, self.num_apple = len(self.river_cells), len(self.apple_cells)
        # Lookup grids: index into dirt/apple arrays, -1 elsewhere. Padded for beams.
        river_idx = -np.ones((h, w), np.int32)
        river_idx[tuple(self.river_cells.T)] = np.arange(self.num_river)
        apple_idx = -np.ones((h, w), np.int32)
        apple_idx[tuple(self.apple_cells.T)] = np.arange(self.num_apple)
        bp = BEAM_LENGTH + 1   # beams never reach further than this past a wall
        self._bp = bp
        self._wall_beam = jnp.asarray(np.pad(self.wall, bp, constant_values=True))
        self._river_beam = jnp.asarray(np.pad(river_idx, bp, constant_values=-1))
        self._apple_idx = jnp.asarray(apple_idx)
        self._wall = jnp.asarray(self.wall)
        self._feed = jnp.asarray(self.feed_cells)
        self._spawns = jnp.asarray(self.spawn_cells)
        self._river = jnp.asarray(self.river_cells)
        self._apples = jnp.asarray(self.apple_cells)
        static = np.zeros((h, w, NUM_CHANNELS), np.uint8)
        static[..., CH["wall"]] = self.wall
        static[..., CH["grass"]] = self.grass
        self._static = jnp.asarray(static)
        self._water = jnp.asarray(self.water)
        self._view_off = jnp.asarray(VIEW_OFFSETS)
        self._beam = jnp.asarray(BEAM_CELLS)
        self._start_dirt_rows = np.array([grid[r, c] == "F" for r, c in self.river_cells], bool)

    # --- towns ------------------------------------------------------------------

    def town(self, name: str = "neighbours", river: str | None = "melting_pot",
             feed_reward: float = spec.FEED_REWARD, feed_period: int = spec.FEED_PERIOD,
             badge_frames: int = spec.BADGE_FRAMES, dirt: str = "melting_pot",
             coop_share: float = 0.0) -> Town:
        """Settings for one of spec.TOWNS. `river=None` keeps the dirt drawn in the
        map rows given to the constructor (for test maps). `dirt`: one of spec.DIRT_RULES.
        `coop_share`: see coop_payout."""
        if name not in spec.TOWNS:
            raise ValueError(f"town must be one of {spec.TOWNS}, not {name!r}")
        if dirt not in spec.DIRT_RULES:
            raise ValueError(f"dirt must be one of {spec.DIRT_RULES}, not {dirt!r}")
        if not 0.0 <= coop_share <= 1.0:
            raise ValueError(f"coop_share must be between 0 and 1, not {coop_share!r}")
        if river is None:
            start = self._start_dirt_rows
        else:
            rows = spec.map_rows(name, river)
            start = np.array([rows[r][c] == "F" for r, c in self.river_cells], bool)
        return Town(identity=jnp.asarray(name != "strangers"),
                    badge=jnp.asarray(name == "connected"),
                    feed=jnp.asarray(name == "isolating"),
                    feed_reward=jnp.asarray(feed_reward, jnp.float32),
                    feed_period=jnp.asarray(feed_period, jnp.int32),
                    badge_frames=jnp.asarray(badge_frames, jnp.int32),
                    start_dirt=jnp.asarray(start),
                    dirt_spawns=jnp.asarray(river != "clean"),
                    dirt_cap=jnp.asarray(spec.DIRT_CAP[dirt], jnp.float32),
                    coop_share=jnp.asarray(coop_share, jnp.float32))

    # --- reset and step -----------------------------------------------------------

    def reset(self, key, town: Town) -> State:
        p = self.num_players
        k1, k2 = jax.random.split(key)
        pos = self._spawns[jax.random.permutation(k1, len(self.spawn_cells))[:p]]
        zeros = jnp.zeros(p, jnp.int32)
        return State(pos=pos, ori=jax.random.randint(k2, (p,), 0, 4, jnp.int32),
                     alive=jnp.ones(p, bool), zapped_at=zeros - RESPAWN_FRAMES,
                     zap_cd=zeros, clean_cd=zeros, dirt=town.start_dirt,
                     apple=jnp.zeros(self.num_apple, bool), badge_t=zeros,
                     cleaned_prev=jnp.zeros(p, bool), feed_t=zeros,
                     t=jnp.asarray(0, jnp.int32))

    def step(self, key, state: State, actions, town: Town):
        """One frame. actions [P] in Melting Pot's ACTION_SET order.

        Returns (state, obs, ready, reward, done, info). When the game ends the state
        (and obs) are already the next game's; info and reward are this frame's.
        info: per player `ate`, `cleaned`, `zaps` (others zapped), `feed` (feed reward).
        """
        k_frame, k_end, k_reset = jax.random.split(key, 3)
        nxt, info = self._frame(k_frame, state, jnp.asarray(actions, jnp.int32), town)
        # Lit badges as players see them this frame (cleaned in the last badge_frames).
        coop = coop_payout(info["ate"], nxt.badge_t > 0, town.coop_share)
        done = episode_ends(k_end, nxt.t)
        fresh = self.reset(k_reset, town)
        nxt = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, nxt)
        obs, ready = self.observe(nxt, town)
        reward = info["ate"] + info["feed"] + coop
        return nxt, obs, ready, reward, done, info

    def _frame(self, key, s: State, act, town: Town):
        p = self.num_players
        players = jnp.arange(p)
        frame = s.t + 1
        k_move, k_zap, k_clean, k_resp, k_apple, k_dirt = jax.random.split(key, 6)

        # 450: badge timers (cleaning shows on the badge from the next frame).
        badge_t = jnp.where(s.cleaned_prev, town.badge_frames, jnp.maximum(s.badge_t - 1, 0))

        # 150: turn, then move, players in random order; eat apples walked onto.
        alive = s.alive
        ori = jnp.where(alive, (s.ori + jnp.asarray(TURN)[act]) % 4, s.ori)
        rel = jnp.asarray(MOVE_REL)[act]
        wants = alive & (rel >= 0)
        target = s.pos + jnp.asarray(DIRS)[(ori + jnp.maximum(rel, 0)) % 4]

        def move_one(k, carry):
            pos, apple, moved, ate = carry
            i = order[k]
            t_i = target[i]
            taken = jnp.any(alive & jnp.all(pos == t_i, axis=-1) & (players != i))
            ok = wants[i] & ~self._wall[t_i[0], t_i[1]] & ~taken
            pos = pos.at[i].set(jnp.where(ok, t_i, pos[i]))
            if self.num_apple:   # (test maps may have no orchard)
                a_idx = self._apple_idx[t_i[0], t_i[1]]                      # -1: not orchard
                eat = ok & (a_idx >= 0) & apple[jnp.maximum(a_idx, 0)]
                apple = apple.at[jnp.maximum(a_idx, 0)].set(apple[jnp.maximum(a_idx, 0)] & ~eat)
            else:
                eat = jnp.zeros((), bool)
            return pos, apple, moved.at[i].set(ok), ate.at[i].set(eat)

        order = jax.random.permutation(k_move, p)
        pos, apple, moved, ate = jax.lax.fori_loop(
            0, p, move_one, (s.pos, s.apple, jnp.zeros(p, bool), jnp.zeros(p, bool)))

        # 150: feed tiles pay whoever stays on one, every feed_period frames. As in
        # Melting Pot, the countdown starts in the frame you arrive (the Feed updater
        # runs after the move; compare.py measured it), so the first pay comes 3 later.
        on_feed = alive & town.feed & self._feed[pos[:, 0], pos[:, 1]]
        feed_t = jnp.where(on_feed & moved, town.feed_period, s.feed_t)
        feed_t = jnp.where(on_feed, feed_t - 1, feed_t)
        pay = on_feed & (feed_t <= 0)
        feed_t = jnp.where(pay, town.feed_period, feed_t)
        feed = jnp.where(pay, town.feed_reward, 0.0)

        # 140: zap beams (recharge 10), players in random order.
        bp = self._bp

        def line_cells(i):
            return pos[i] + self._beam[ori[i]]                          # [3 lines, 3, 2]

        def zap_one(k, carry):
            alive, zapped_at, zap_cd, zaps = carry
            i = zap_order[k]
            live, cd = alive[i], zap_cd[i]
            fire = live & (cd == 0) & (act[i] == ZAP)
            zap_cd = zap_cd.at[i].set(
                jnp.where(live, jnp.where(cd > 0, cd - 1, jnp.where(fire, ZAP_COOLDOWN, 0)), cd))
            cells = line_cells(i)
            wall = self._wall_beam[cells[..., 0] + bp, cells[..., 1] + bp]           # [3, 3]
            occ = (alive & (players != i))[None, None, :] & jnp.all(
                cells[:, :, None, :] == pos[None, None, :, :], axis=-1)            # [3, 3, P]
            block = wall | occ.any(-1)
            first = jnp.argmax(block, axis=1)                                      # [3]
            # The player in each line's first blocked cell (none if a wall, or if the
            # line is clear: then argmax is 0 and that cell holds nobody).
            hit_line = jnp.take_along_axis(occ, first[:, None, None], axis=1)[:, 0, :]
            hit = fire & hit_line.any(0)                                           # [P]
            return (alive & ~hit, jnp.where(hit, frame, zapped_at), zap_cd,
                    zaps.at[i].add(hit.sum()))

        zap_order = jax.random.permutation(k_zap, p)
        alive, zapped_at, zap_cd, zaps = jax.lax.fori_loop(
            0, p, zap_one, (alive, s.zapped_at, s.zap_cd, jnp.zeros(p, jnp.int32)))

        # 140: clean beams (recharge 2): each line cleans the first dirt it meets.
        def clean_one(k, carry):
            dirt, clean_cd, cleaned = carry
            i = clean_order[k]
            live, cd = alive[i], clean_cd[i]
            fire = live & (cd == 0) & (act[i] == CLEAN)
            clean_cd = clean_cd.at[i].set(
                jnp.where(live, jnp.where(cd > 0, cd - 1, jnp.where(fire, CLEAN_COOLDOWN, 0)), cd))
            cells = line_cells(i)
            wall = self._wall_beam[cells[..., 0] + bp, cells[..., 1] + bp]
            r_idx = self._river_beam[cells[..., 0] + bp, cells[..., 1] + bp]       # [3, 3]
            if self.num_river:   # (test maps may have no river)
                dirty = (r_idx >= 0) & dirt[jnp.maximum(r_idx, 0)]
            else:
                dirty = jnp.zeros(r_idx.shape, bool)
            block = wall | dirty
            first = jnp.argmax(block, axis=1)
            hit = fire & jnp.take_along_axis(dirty, first[:, None], axis=1)[:, 0]   # [3]
            idx = jnp.where(hit, jnp.take_along_axis(r_idx, first[:, None], axis=1)[:, 0],
                            self.num_river)
            gone = jnp.zeros(self.num_river + 1, bool).at[idx].set(True)[:-1]
            return dirt & ~gone, clean_cd, cleaned.at[i].set(hit.sum())

        clean_order = jax.random.permutation(k_clean, p)
        dirt, clean_cd, cleaned = jax.lax.fori_loop(
            0, p, clean_one, (s.dirt, s.clean_cd, jnp.zeros(p, jnp.int32)))

        # 135: players zapped 50 frames ago come back on a free spawn point.
        resp_keys = jax.random.split(k_resp, p)

        def respawn_one(i, carry):
            pos, alive = carry
            taken = jnp.any(alive[:, None] & jnp.all(
                pos[:, None, :] == self._spawns[None, :, :], axis=-1), axis=0)     # [S]
            free = ~taken
            # (No free spawn point, only possible on tiny test maps: wait a frame.)
            due = ~alive[i] & (frame - zapped_at[i] >= RESPAWN_FRAMES) & free.any()
            pick = jax.random.choice(resp_keys[i], len(self.spawn_cells),
                                     p=free / jnp.maximum(free.sum(), 1))
            pos = pos.at[i].set(jnp.where(due, self._spawns[pick], pos[i]))
            return pos, alive.at[i].set(alive[i] | due)

        pos, alive = jax.lax.fori_loop(0, p, respawn_one, (pos, alive))

        # 100: apples grow, dirt may spawn.
        frac = dirt.sum() / max(self.num_river, 1)
        interp = jnp.minimum((frac - DEPLETION) / (RESTORATION - DEPLETION), 1.0)
        grow = jax.random.uniform(k_apple, (self.num_apple,)) < MAX_GROWTH * interp
        apple = apple | grow
        ate = ate.astype(jnp.int32)
        if self.num_apple:
            # An apple that appears under a player is eaten at once: in Melting Pot the
            # new apple "enters" the player's cell (measured: players only turning on
            # the spot still ate 181 apples in 1,950 frames).
            a_idx = self._apple_idx[pos[:, 0], pos[:, 1]]
            under = alive & (a_idx >= 0) & apple[jnp.maximum(a_idx, 0)]
            ate = ate + under
            gone = jnp.zeros(self.num_apple + 1, bool).at[
                jnp.where(under, a_idx, self.num_apple)].set(True)[:-1]
            apple = apple & ~gone
        if self.num_river:
            k_spawn, k_cell = jax.random.split(k_dirt)
            clean_cells = ~dirt
            # (frac < 1.0 is the same as "any cell clean": Melting Pot's rule unchanged.)
            spawn = (town.dirt_spawns & (frame > DIRT_DELAY) & clean_cells.any()
                     & (frac < town.dirt_cap) & (jax.random.uniform(k_spawn) < DIRT_PROB))
            cell = jax.random.choice(k_cell, self.num_river,
                                     p=clean_cells / jnp.maximum(clean_cells.sum(), 1))
            dirt = dirt.at[cell].set(dirt[cell] | spawn)

        nxt = State(pos=pos, ori=ori, alive=alive, zapped_at=zapped_at, zap_cd=zap_cd,
                    clean_cd=clean_cd, dirt=dirt, apple=apple, badge_t=badge_t,
                    cleaned_prev=cleaned > 0, feed_t=feed_t, t=frame)
        info = {"ate": ate.astype(jnp.float32), "cleaned": cleaned.astype(jnp.float32),
                "zaps": zaps.astype(jnp.float32), "feed": feed.astype(jnp.float32),
                "length": frame}
        return nxt, info

    # --- what each player sees ----------------------------------------------------------

    def observe(self, s: State, town: Town):
        """obs [P, 11, 11, NUM_CHANNELS] uint8 (ahead is up), ready [P] float32:
        Melting Pot's READY_TO_SHOOT (1 when the zap beam is charged, 0 while out)."""
        p = self.num_players
        h, w = self.shape
        g = self._static
        a = self._apples
        g = g.at[a[:, 0], a[:, 1], CH["apple"]].set(s.apple.astype(jnp.uint8))
        r = self._river
        g = g.at[r[:, 0], r[:, 1], CH["dirt"]].set(s.dirt.astype(jnp.uint8))
        dirt_grid = g[..., CH["dirt"]] > 0
        g = g.at[..., CH["water"]].set((self._water & ~dirt_grid).astype(jnp.uint8))
        g = g.at[..., CH["feed"]].set((self._feed & town.feed).astype(jnp.uint8))
        alive = s.alive.astype(jnp.uint8)
        rows, cols = s.pos[:, 0], s.pos[:, 1]
        g = g.at[rows, cols, CH["someone"]].max(alive)
        g = g.at[rows, cols, CH["badge"]].max(alive * ((s.badge_t > 0) & town.badge))
        g = g.at[rows, cols, CH["player0"] + jnp.arange(p)].max(alive * town.identity)
        g = jnp.pad(g, ((PAD, PAD), (PAD, PAD), (0, 0)))

        def view(pos, ori, live):
            off = self._view_off[ori]                                    # [11, 11, 2]
            v = g[pos[0] + PAD + off[..., 0], pos[1] + PAD + off[..., 1]]  # [11, 11, C]
            me = v[SELF_ROW, SELF_COL]
            me = me.at[CH["someone"]].set(0).at[CH["player0"]:].set(0)
            me = me.at[CH["self"]].set(live)
            return v.at[SELF_ROW, SELF_COL].set(me)

        obs = jax.vmap(view)(s.pos, s.ori, alive)
        ready = jnp.where(s.alive, jnp.maximum(1.0 - s.zap_cd / ZAP_COOLDOWN, 0.0), 0.0)
        return obs, ready.astype(jnp.float32)
