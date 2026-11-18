"""Many copies of one town, stepping in parallel worker processes.

Each worker process runs `envs_per_worker` games one after another; all workers
run at once. The learner sends one action per player per game and gets back:

    obs     [E, 7, 22, 22, 3] uint8   each player's own view, 4x smaller
    ready   [E, 7] float32            READY_TO_SHOOT (zap beam charged)
    reward  [E, 7] float32            the game's reward (apples, feed)
    done    [E] bool                  the game ended this step (obs is from the new one)
    ate     [E, 7] float32            apples eaten this step (for logging)
    cleaned [E, 7] float32            dirt cleaned this step (for logging)

plus a list of finished episodes, each a dict of per-player totals. The players
never see `ate`, `cleaned` or the zap counts; they are only logged.

This module doesn't import Melting Pot or JAX at the top, so the learner process
stays light and the workers never touch the GPU.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np

NUM_PLAYERS = 7
DOWN = 4          # 88x88 view -> 22x22 (2x2 pixels per map cell)
VIEW = 88 // DOWN


def downsample(rgb: np.ndarray) -> np.ndarray:
    """Average 4x4 pixel blocks: [..., 88, 88, 3] -> [..., 22, 22, 3] uint8.

    Strided adds, rows then columns: ~12x faster than a reshape-and-sum, which
    matters because this runs for every player on every step. (Not OpenCV: just
    importing cv2 makes the Melting Pot game itself ~8x slower, measured
    2026-11-16: 1,800 -> 230 steps/s.)
    """
    r = rgb.astype(np.uint16)
    s = r[..., 0::4, :, :] + r[..., 1::4, :, :] + r[..., 2::4, :, :] + r[..., 3::4, :, :]
    s = s[..., 0::4, :] + s[..., 1::4, :] + s[..., 2::4, :] + s[..., 3::4, :]
    return ((s + 8) >> 4).astype(np.uint8)


def episode_summary(length, ret, ate, cleaned, zaps, feed, t_reward, n_reward,
                    cleaned_by_quarter=None) -> dict:
    """One finished game, from per-player totals (shared with sims/jaxtown/train_jax.py).
    t_reward / n_reward: sum of the frames at which each player got a reward, and how
    many such frames (for Hughes et al.'s sustainability metric)."""
    ret, t_reward, n_reward = (np.asarray(x, float) for x in (ret, t_reward, n_reward))
    n = len(ret)
    # Hughes et al. 2018's equality metric: 1 - Gini over players' returns.
    total = ret.sum()
    gini = np.abs(ret[:, None] - ret[None, :]).sum() / (2 * n * total) if total > 0 else 0.0
    sustain = np.where(n_reward > 0, t_reward / np.maximum(n_reward, 1), np.nan)
    return {
        "length": int(length),
        "return": ret.tolist(),
        "ate": np.asarray(ate, float).tolist(),
        "cleaned": np.asarray(cleaned, float).tolist(),
        "zaps": np.asarray(zaps, float).tolist(),
        "feed": np.asarray(feed, float).tolist(),
        "collective_return": float(total),
        "equality": float(1.0 - gini) if total > 0 else float("nan"),  # nobody ate: undefined
        "sustainability": float(np.nanmean(sustain)) if np.isfinite(sustain).any() else 0.0,
        "cleaned_by_quarter": cleaned_by_quarter,
    }


class _Game:
    """One town and its running episode totals."""

    def __init__(self, town: str, town_kwargs: dict):
        import town_config as tc  # imported here, inside the worker
        self.env = tc.build_town(town, world_rgb=False, **town_kwargs)
        self.t = 0
        self._zero()

    def _zero(self):
        n = NUM_PLAYERS
        self.t = 0
        self.tot = {k: np.zeros(n) for k in ("ret", "ate", "cleaned", "zaps", "feed",
                                             "t_reward", "n_reward")}
        # All players' cleaning on each step, to see whether it lasts through the game.
        self.clean_t = []

    @staticmethod
    def _unpack(ts):
        obs = ts.observation
        rgb = np.stack([o["RGB"] for o in obs])
        ready = np.array([float(o["READY_TO_SHOOT"]) for o in obs], np.float32)
        ate = np.array([float(o["PLAYER_ATE_APPLE"]) for o in obs], np.float32)
        cleaned = np.array([float(o["PLAYER_CLEANED"]) for o in obs], np.float32)
        zaps = np.array([float(o["NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP"]) for o in obs],
                        np.float32)
        return rgb, ready, ate, cleaned, zaps

    def reset(self):
        self._zero()
        ts = self.env.reset()
        rgb, ready, _, _, _ = self._unpack(ts)
        return rgb, ready

    def step(self, actions):
        ts = self.env.step([int(a) for a in actions])
        self.t += 1
        rgb, ready, ate, cleaned, zaps = self._unpack(ts)
        reward = np.asarray(ts.reward, np.float32)
        tot = self.tot
        tot["ret"] += reward
        tot["ate"] += ate
        tot["cleaned"] += cleaned
        tot["zaps"] += zaps
        tot["feed"] += reward - ate
        pos = reward > 0
        tot["t_reward"] += self.t * pos
        tot["n_reward"] += pos
        self.clean_t.append(float(cleaned.sum()))
        episode = None
        done = bool(ts.last())
        if done:
            episode = self.summary()
            rgb, ready = self.reset()
        return rgb, ready, reward, done, ate, cleaned, episode

    def summary(self) -> dict:
        tot = self.tot
        return episode_summary(
            self.t, tot["ret"], tot["ate"], tot["cleaned"], tot["zaps"], tot["feed"],
            tot["t_reward"], tot["n_reward"],
            # Quarters of this game's own length (games end at random after 1000+ steps).
            [float(q.sum()) for q in np.array_split(self.clean_t, 4)])


def _worker(conn, town, town_kwargs, n_envs):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C goes to the learner, which closes us
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # meltingpot/__init__.py imports its bots and scenarios, which load TensorFlow
    # (~0.5 GB per process). The games don't need them, so stand in empty modules.
    import sys
    import types
    for name in ("meltingpot.bot", "meltingpot.scenario"):
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        games = [_Game(town, town_kwargs) for _ in range(n_envs)]
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                outs = [g.reset() for g in games]
                conn.send(("ok", (downsample(np.stack([o[0] for o in outs])),
                                  np.stack([o[1] for o in outs]))))
            elif cmd == "step":
                outs = [g.step(a) for g, a in zip(games, data)]
                rgb = np.stack([o[0] for o in outs])
                conn.send(("ok", (
                    downsample(rgb),
                    np.stack([o[1] for o in outs]),
                    np.stack([o[2] for o in outs]),
                    np.array([o[3] for o in outs]),
                    np.stack([o[4] for o in outs]),
                    np.stack([o[5] for o in outs]),
                    [o[6] for o in outs if o[6] is not None],
                )))
            elif cmd == "close":
                for g in games:
                    g.env.close()
                conn.send(("ok", None))
                return
    except Exception:  # send the error to the learner instead of hanging it
        conn.send(("error", traceback.format_exc()))


class VecTown:
    """`num_workers` x `envs_per_worker` copies of one town."""

    def __init__(self, town: str, num_workers: int = 8, envs_per_worker: int = 6,
                 town_kwargs: dict | None = None):
        self.num_envs = num_workers * envs_per_worker
        self.epw = envs_per_worker
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        # Children copy the environment at start: keep them off the GPU.
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            for _ in range(num_workers):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, town, town_kwargs or {},
                                                      envs_per_worker), daemon=True)
                p.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(p)
        finally:
            if saved is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def _gather(self):
        # Read every worker's reply before raising, so no stale reply is left in a
        # pipe for close() to mistake for its acknowledgement.
        replies = [c.recv() for c in self.conns]
        errors = [data for status, data in replies if status == "error"]
        if errors:
            raise RuntimeError("env worker failed:\n" + errors[0])
        return [data for _, data in replies]

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        outs = self._gather()
        return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])

    def step_async(self, actions: np.ndarray):
        """Send actions [E, 7] ints; the workers start stepping right away."""
        for i, c in enumerate(self.conns):
            c.send(("step", actions[i * self.epw:(i + 1) * self.epw]))

    def step_wait(self):
        outs = self._gather()
        cat = [np.concatenate([o[k] for o in outs]) for k in range(6)]
        episodes = [ep for o in outs for ep in o[6]]
        return (*cat, episodes)

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=10)
