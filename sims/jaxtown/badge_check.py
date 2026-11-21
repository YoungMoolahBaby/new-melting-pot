"""Do the connected town's players use the badge at all? (2026-11-21)

Plays trained players with their weights frozen (no learning), with the same luck, twice: in
their town as trained (badges shown) and with the badges hidden, and compares cleaning,
apples and zaps. It also measures:

- how often each player's badge is lit (cleaned in the last 50 frames);
- how often another player's lit badge is in a player's view;
- how much a player's choice of action moves when the badges in its view (its own
  included) are taken away, at the same moment and with the same memory (total variation:
  0 = no change, 1 = a completely different choice), on the frames with a badge in view;
  and how often its most likely action changes ("flip").

For reference, the neighbours town's players are played in the connected town too: they
never saw a badge in training, so their eyes' weights for it are still the random ones
they started with. Badges matter to the connected players only if they move their choices
more than that.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/badge_check.py \\
        --runs out/jaxtown/towns out/colab/towns --out out/jaxtown/badge_check
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))

import numpy as np  # noqa: E402

FIELDS = ("ate", "cleaned", "zaps")


def load(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "ckpt.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    return cfg, ckpt["params"], ckpt["env_steps"]


def learner_for(cfg):
    import ppo
    p = dict(cfg["ppo"], obs_shape=tuple(cfg["ppo"]["obs_shape"]))
    return ppo.Learner(ppo.Config(**p))


def make_play(env, learner, num_envs, frames):
    """A compiled call: `frames` frames of `num_envs` games with frozen players."""
    import jax
    import jax.numpy as jnp
    import cleanup as cu
    P, n, badge = env.num_players, num_envs, cu.CH["badge"]
    me = (cu.SELF_ROW, cu.SELF_COL)

    def policy(params, h, obs, ready, start):
        obs, ready = jnp.swapaxes(obs, 0, 1), jnp.swapaxes(ready, 0, 1)   # [P, E, ...]
        return jax.vmap(learner.net.apply, in_axes=(0, 0, 0, 0, None))(
            params, h, obs, ready, start)

    def play(params, town, key):
        key, k_env = jax.random.split(key)
        es = jax.vmap(env.reset, (0, None))(jax.random.split(k_env, n), town)
        obs, ready = jax.vmap(env.observe, (0, None))(es, town)
        h = learner.initial_state(n)

        def one(carry, _):
            es, obs, ready, start, h, acc, tot, st, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            h2, logits, _ = policy(params, h, obs, ready, start)
            a = jax.random.categorical(k_act, logits)                          # [P, E]
            # The same moment with the badges taken out of view (same memory h).
            _, logits_nb, _ = policy(params, h, obs.at[..., badge].set(0), ready, start)
            pr, pr_nb = jax.nn.softmax(logits), jax.nn.softmax(logits_nb)
            tv = 0.5 * jnp.abs(pr - pr_nb).sum(-1)                              # [P, E]
            flip = (logits.argmax(-1) != logits_nb.argmax(-1)).astype(jnp.float32)
            seen = jnp.swapaxes(obs[..., badge].any(axis=(2, 3)), 0, 1)         # [P, E]
            others = obs[..., badge].at[:, :, me[0], me[1]].set(0).any(axis=(2, 3))
            lit = (es.badge_t > 0) & es.alive                                   # [E, P]
            st = st + jnp.stack([
                (seen * tv).sum(), (seen * flip).sum(), seen.sum(),
                others.sum(), jnp.float32(n * P)]).astype(jnp.float32)
            es, obs2, ready2, _, done, info = jax.vmap(env.step, (0, 0, 0, None))(
                jax.random.split(k_env, n), es, a.T, town)
            acc = acc + jnp.stack([info[f] for f in FIELDS], -1)                # [E, P, F]
            d = done[:, None, None]
            tot = {"sum": tot["sum"] + (d * acc).sum(0),
                   "lit": tot["lit"] + lit.sum(0),
                   "frames": tot["frames"] + jnp.where(done, info["length"], 0).sum(),
                   "games": tot["games"] + done.sum()}
            acc = jnp.where(d, 0.0, acc)
            return (es, obs2, ready2, done, h2, acc, tot, st, key), None

        tot = {"sum": jnp.zeros((P, len(FIELDS))), "lit": jnp.zeros(P),
               "frames": jnp.zeros((), jnp.int32), "games": jnp.zeros((), jnp.int32)}
        carry = (es, obs, ready, jnp.ones(n, bool), h, jnp.zeros((n, P, len(FIELDS))), tot,
                 jnp.zeros(5), key)
        carry, _ = jax.lax.scan(one, carry, None, frames)
        return carry[6], carry[7]

    return jax.jit(play)


def summary(tot, st, num_envs, frames):
    s, f = np.asarray(tot["sum"]), max(int(tot["frames"]), 1)
    st = np.asarray(st)
    per_player_clean = 1000 * s[:, 1] / f
    return {"games": int(tot["games"]),
            "clean": float(per_player_clean.mean()),            # per player per 1,000 frames
            "clean_by_player": [round(float(x), 1) for x in per_player_clean],
            "apples": float(1000 * s[:, 0].sum() / f),           # all 7, per 1,000 frames
            "zaps_per_game": float(s[:, 2].sum() / max(int(tot["games"]), 1)),
            "lit": [round(float(x), 3) for x in np.asarray(tot["lit"]) / (num_envs * frames)],
            "badge_in_view": float(st[2] / st[4]),               # any badge, own included
            "other_badge_in_view": float(st[3] / st[4]),
            "tv": float(st[0] / max(st[2], 1)),
            "flip": float(st[1] / max(st[2], 1))}


def find_runs(roots, town, players):
    out = {}
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, f"{town}_{players}_s*"))):
            if os.path.exists(os.path.join(d, "ckpt.pkl")):
                with open(os.path.join(d, "config.json")) as f:
                    seed = json.load(f)["seed"]
                out[seed] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="folders holding the town runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--frames", type=int, default=12000, help="frames per game slot")
    ap.add_argument("--seed", type=int, default=7, help="the luck shared by both versions")
    args = ap.parse_args(argv)
    import jax
    import jax.numpy as jnp
    import cleanup as cu

    env = cu.CleanUp()
    os.makedirs(args.out, exist_ok=True)
    play, net, results = None, None, []
    for players in ("selfish", "fair"):
        conn = find_runs(args.runs, "connected", players)
        nbrs = find_runs(args.runs, "neighbours", players)
        for seed in sorted(conn):
            for label, run_dir, hide in (("connected, badge shown", conn[seed], False),
                                         ("connected, badge hidden", conn[seed], True),
                                         ("neighbours players, badge shown", nbrs.get(seed),
                                          False)):
                if run_dir is None:
                    continue
                cfg, params, steps = load(run_dir)
                if play is None:   # one compiled call serves every run of the same network
                    play, net = make_play(env, learner_for(cfg), args.envs, args.frames), cfg["ppo"]
                if cfg["ppo"] != net:
                    raise SystemExit(f"{run_dir}: a different network from the first run")
                town = env.town("connected", cfg["river"], cfg["feed_reward"],
                                cfg["feed_period"], cfg["badge_frames"], cfg["dirt"])
                if hide:
                    town = town.replace(badge=jnp.asarray(False))
                t0 = time.time()
                tot, st = jax.device_get(play(jax.tree.map(jnp.asarray, params), town,
                                              jax.random.PRNGKey(args.seed)))
                r = {"players": players, "seed": seed, "version": label, "run": run_dir,
                     "trained_steps": steps,
                     **summary(tot, st, args.envs, args.frames)}
                results.append(r)
                print(f"{players} s{seed} {label}: clean {r['clean']:.1f} apples "
                      f"{r['apples']:.1f} zaps/game {r['zaps_per_game']:.1f} | other badge in "
                      f"view {r['other_badge_in_view']:.0%} | tv {r['tv']:.3f} flip "
                      f"{r['flip']:.1%} | lit {np.mean(r['lit']):.0%} | {r['games']} games, "
                      f"{time.time() - t0:.0f} s", flush=True)
                with open(os.path.join(args.out, "results.json"), "w") as f:
                    json.dump(results, f, indent=1)
    print("wrote", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
