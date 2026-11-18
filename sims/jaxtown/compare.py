"""Our JAX Clean Up side by side with DeepMind's Melting Pot (plan step 3, gate B).

Two checks:

  beam   Melting Pot itself, with players zapping a lot: where, relative to the
         zapper, were the players it hit? It must be exactly our beam's 9 cells.
  games  The same random players in both versions, for each town and river start:
         game length, and apples, cleaning, zaps and feed per step. Melting Pot
         runs in CPU worker processes, ours on the GPU. Only each game's own
         totals count, and only finished games (first game per JAX env, so long
         games aren't cut short).

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/compare.py

Writes out/jaxtown/compare.md (table) and compare.json (all numbers).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOWN_DIR = os.path.join(os.path.dirname(HERE), "town")
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TOWN_DIR)
sys.path.insert(0, HERE)

import spec  # noqa: E402  (light)

# Action probabilities in Melting Pot's ACTION_SET order:
# noop, forward, backward, step left, step right, turn left, turn right, zap, clean.
POLICIES = {
    "random": [1 / 9] * 9,
    "busy": [0.05, 0.2, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2],   # more beams, more walking
}
ZAPPY = [0.05, 0.15, 0.05, 0.05, 0.05, 0.1, 0.1, 0.4, 0.05]
METRICS = ("length", "apples_per_step", "cleaned_per_step", "zaps_per_step", "feed_per_step")
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])   # facing N, E, S, W as (row, col)


# --- Melting Pot side (CPU worker processes) ---------------------------------------------------

def _mp_setup():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import types
    for name in ("meltingpot.bot", "meltingpot.scenario"):   # skip TensorFlow
        sys.modules.setdefault(name, types.ModuleType(name))


def _mp_games(job):
    town, river, policy, n_games, seed = job
    _mp_setup()
    import envs
    game = envs._Game(town, {"river": river})
    game.reset()
    rng = np.random.default_rng(seed)
    p = POLICIES[policy]
    games = []
    while len(games) < n_games:
        *_, ep = game.step(rng.choice(9, size=spec.NUM_PLAYERS, p=p))
        if ep is not None:
            games.append(_totals(ep["length"], ep["ate"], ep["cleaned"], ep["zaps"], ep["feed"]))
    game.env.close()
    return games


def _totals(length, ate, cleaned, zaps, feed):
    n = float(length)
    return {"length": n, "apples_per_step": float(np.sum(ate)) / n,
            "cleaned_per_step": float(np.sum(cleaned)) / n,
            "zaps_per_step": float(np.sum(zaps)) / n, "feed_per_step": float(np.sum(feed)) / n}


def _mp_beam(job):
    """Relative (ahead, right) cell of every player hit by a zap, in Melting Pot."""
    seed, steps = job
    _mp_setup()
    import town_config as tc
    env = tc.build_town("neighbours", world_rgb=False, positions=True)
    events = []
    env.observables().events.subscribe(on_next=events.append)
    env.reset()
    rng = np.random.default_rng(seed)
    seen = collections.Counter()
    for _ in range(steps):
        events.clear()
        ts = env.step(rng.choice(9, size=spec.NUM_PLAYERS, p=ZAPPY))
        obs = ts.observation
        for name, data in events:
            if name != "zap":
                continue
            src, tgt = int(data[2]) - 1, int(data[4]) - 1          # Lua counts from 1
            (sx, sy), (tx, ty) = obs[src]["POSITION"], obs[tgt]["POSITION"]
            facing = int(obs[src]["ORIENTATION"])
            d = np.array([ty - sy, tx - sx])                        # (row, col)
            seen[(int(d @ DIRS[facing]), int(d @ DIRS[(facing + 1) % 4]))] += 1
        if ts.last():
            env.reset()
    env.close()
    return seen


# --- JAX side (GPU) --------------------------------------------------------------------------

_JAX = {}


def jax_games(town, river, policy, num_envs, seed=0):
    """First game of each of `num_envs` games (all run to the end: 5,000 frames max).
    Compiled once: the town and policy are data, not code."""
    import jax
    import jax.numpy as jnp
    import cleanup as cu
    if num_envs not in _JAX:
        _JAX[num_envs] = _jax_runner(cu, jax, jnp, num_envs)
    env, run = _JAX[num_envs]
    out = run(jax.random.PRNGKey(seed), env.town(town, river),
              jnp.log(jnp.asarray(POLICIES[policy])))
    done, length, acc = (np.asarray(x) for x in out)
    t, e = np.nonzero(done)
    return [_totals(length[i, j], *acc[i, j]) for i, j in zip(t, e)]


def _jax_runner(cu, jax, jnp, num_envs):
    env = cu.CleanUp()

    def run(key, tw, logits):
        s = jax.vmap(env.reset, (0, None))(jax.random.split(key, num_envs), tw)

        def body(carry, _):
            s, acc, first, key = carry
            key, ka, ks = jax.random.split(key, 3)
            a = jax.random.categorical(ka, logits, shape=(num_envs, env.num_players))
            s, _, _, _, done, info = jax.vmap(env.step, (0, 0, 0, None))(
                jax.random.split(ks, num_envs), s, a, tw)
            acc = acc + jnp.stack([info[k].sum(-1) for k in ("ate", "cleaned", "zaps", "feed")],
                                  -1)
            out = (done & first, info["length"], acc)
            return (s, acc, first & ~done, key), out

        init = (s, jnp.zeros((num_envs, 4)), jnp.ones(num_envs, bool), key)
        _, out = jax.lax.scan(body, init, None, cu.END_MAX)
        return out

    return env, jax.jit(run)


# --- comparing -----------------------------------------------------------------------------

def stats(games, key):
    x = np.array([g[key] for g in games])
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def agree(a, b, tol=0.10):
    """Within 10% of Melting Pot, or the 95% ranges overlap."""
    (ma, ca), (mb, cb) = a, b
    return abs(ma - mb) <= tol * abs(mb) + 1e-12 or abs(ma - mb) <= ca + cb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp-games", type=int, default=120, help="Melting Pot games per setting")
    ap.add_argument("--jax-envs", type=int, default=1024, help="JAX games per setting")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--beam-steps", type=int, default=3000, help="Melting Pot steps per worker")
    ap.add_argument("--quick", action="store_true", help="one setting only (smoke test)")
    ap.add_argument("--rivers", nargs="+", default=list(spec.RIVERS), choices=spec.RIVERS)
    ap.add_argument("--towns", nargs="+", default=list(spec.TOWNS), choices=spec.TOWNS)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "jaxtown"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    settings = [(t, r, "random") for t in args.towns for r in args.rivers]
    settings += [(t, "hughes", "busy") for t in args.towns if "hughes" in args.rivers]
    if args.quick:
        settings = [("isolating", "hughes", "busy")]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        beam_jobs = pool.map_async(_mp_beam, [(s, args.beam_steps) for s in range(args.workers)])
        per = -(-args.mp_games // args.workers)
        game_jobs = {s: pool.map_async(_mp_games, [(*s, per, 1000 * i + k)
                                                   for k in range(args.workers)])
                     for i, s in enumerate(settings)}
        # Meanwhile the GPU plays the JAX games.
        jx = {s: jax_games(*s, num_envs=args.jax_envs, seed=i) for i, s in enumerate(settings)}
        print(f"JAX games done ({time.time() - t0:.0f} s); waiting for Melting Pot...", flush=True)
        mpg = {s: [g for chunk in job.get() for g in chunk] for s, job in game_jobs.items()}
        seen = sum(beam_jobs.get(), collections.Counter())

    import cleanup as cu
    ours = {(int(-r), int(c)) for r, c in cu.BEAM_CELLS[0].reshape(-1, 2)}   # facing north
    beam_ok = set(seen) == ours
    lines = [f"Melting Pot vs our JAX Clean Up, random players ({time.time() - t0:.0f} s).", "",
             f"**Beam:** Melting Pot zapped {sum(seen.values()):,} players at "
             f"{len(seen)} distinct cells relative to the zapper; ours covers {len(ours)}. "
             + ("Identical." if beam_ok else
                f"DIFFERENT: only Melting Pot {sorted(set(seen) - ours)}, "
                f"only ours {sorted(ours - set(seen))}."), "",
             "| town | river | players | measure | Melting Pot | JAX | agree |",
             "|---|---|---|---|---|---|---|"]
    results, all_ok = {}, beam_ok
    for s in settings:
        rows = {}
        for key in METRICS:
            a, b = stats(jx[s], key), stats(mpg[s], key)
            ok = agree(a, b)
            all_ok &= ok
            rows[key] = {"mp": b, "jax": a, "agree": ok}
            fmt = (lambda m: f"{m[0]:,.0f} ± {m[1]:,.0f}") if key == "length" else (
                lambda m: f"{m[0]:.4f} ± {m[1]:.4f}")
            lines.append(f"| {s[0]} | {s[1]} | {s[2]} | {key} | {fmt(b)} | {fmt(a)} | "
                         f"{'yes' if ok else '**NO**'} |")
        results["/".join(s)] = {"rows": rows, "games": {"mp": len(mpg[s]), "jax": len(jx[s])}}
    lines += ["", f"Games per setting: Melting Pot {len(mpg[settings[0]])}, "
              f"JAX {len(jx[settings[0]])}. ± is a 95% range. Agree = within 10% of Melting "
              "Pot or overlapping ranges.", "", f"**Gate B: {'PASS' if all_ok else 'FAIL'}**"]
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.out, "compare.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out, "compare.json"), "w") as f:
        json.dump({"beam": {str(k): v for k, v in seen.items()}, "beam_ok": beam_ok,
                   "settings": results}, f, indent=1)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
