"""Train our players in the JAX Clean Up, all on the GPU (plan step 4).

The same players and learner as sims/town/train.py: ppo.py, each player its own
network with GRU memory, PPO, and Hughes et al. 2018's fairness reward. Same flags,
same output files (config.json, metrics.csv, episodes.jsonl, ckpt.pkl), so
sims/town/report.py charts runs from both games. The game is sims/jaxtown/cleanup.py;
one rollout in every game plus its PPO update is a single compiled call, and the
town and fairness settings are data, so one compile serves every run.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/train_jax.py \\
        --runs neighbours:selfish neighbours:fair --river hughes --steps 1e7

Ctrl-C stops after the current update and saves checkpoints; --resume carries on.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import signal
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # share the GPU
# Under WSL no single GPU allocation can reach 4 GiB (measured 2026-11-17: 3.9 GiB ok,
# 4.5 GiB fails, 3 x 3 GiB ok). JAX's default pool grows in ever bigger blocks and
# then fails; asking the driver for each buffer avoids that. The compiled step's
# working memory must stay under ~3.9 GiB too: see --minibatches.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
HERE = os.path.dirname(os.path.abspath(__file__))
TOWN_DIR = os.path.join(os.path.dirname(HERE), "town")
sys.path.insert(0, HERE)
sys.path.insert(0, TOWN_DIR)

import numpy as np  # noqa: E402

import spec  # noqa: E402
import train as town_train  # noqa: E402  (sims/town/train.py: shared flags, logs, checks)
from envs import episode_summary  # noqa: E402

METRIC_FIELDS = town_train.METRIC_FIELDS
# (alpha, beta) as sims/town/train.py, plus "shared" (JAX only): every player's reward is
# the average of all seven players' apples, so anyone's apple counts for everyone.
PLAYERS = {**town_train.PLAYERS, "shared": (0.0, 0.0)}
SHARED = ("shared",)
# Batch shape and the dirt rule change what's learned too.
SAME_ON_RESUME = town_train.SAME_ON_RESUME + ("envs", "minibatches", "dtype", "dirt")
# Per-player episode totals kept during a game: reward, apples, cleaned, zaps, feed,
# sum of frames with a reward, number of frames with a reward.
ACC = ("ret", "ate", "cleaned", "zaps", "feed", "t_reward", "n_reward")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="town:players[:seed[:dirt]] ..., e.g. neighbours:selfish "
                         "neighbours:fair:0:capped (seed and dirt default to --seed, --dirt)")
    ap.add_argument("--town", choices=spec.TOWNS, default="neighbours")
    ap.add_argument("--players", choices=sorted(PLAYERS), default="selfish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=1e7, help="game steps per run")
    ap.add_argument("--envs", type=int, default=512, help="games played at once per run")
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatches", type=int, default=16,
                    help="pieces each PPO epoch is split into; at 512 games, 16 keeps the "
                         "step's GPU working memory at 2.85 GiB (8: 4.86 GiB, too big under "
                         "WSL)")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                    help="number type for the network's eyes (conv and dense layers): "
                         "bfloat16 trains ~1.5x faster here (29k vs 19k steps/s, 2026-11-17)")
    ap.add_argument("--alpha", type=float, default=None, help="override envy (disadvantageous)")
    ap.add_argument("--beta", type=float, default=None, help="override guilt (advantageous)")
    ap.add_argument("--ia-gamma", type=float, default=0.99)
    ap.add_argument("--ia-lambda", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--feed-reward", type=float, default=spec.FEED_REWARD)
    ap.add_argument("--feed-period", type=int, default=spec.FEED_PERIOD)
    ap.add_argument("--badge-frames", type=int, default=spec.BADGE_FRAMES)
    ap.add_argument("--river", choices=spec.RIVERS, default="melting_pot")
    ap.add_argument("--dirt", choices=spec.DIRT_RULES, default="melting_pot",
                    help="how dirt arrives (spec.DIRT_RULES): melting_pot until the river is "
                         "full, capped stops at 40%% dirty")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--out-root", default=os.path.join("out", "jaxtown"))
    ap.add_argument("--out", default=None, help="exact output folder (one run only)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    args.specs = []
    for s in args.runs or [f"{args.town}:{args.players}:{args.seed}:{args.dirt}"]:
        parts = s.split(":")
        if (not 2 <= len(parts) <= 4 or parts[0] not in spec.TOWNS or parts[1] not in PLAYERS
                or (len(parts) == 4 and parts[3] not in spec.DIRT_RULES)):
            ap.error(f"bad run {s!r}: write town:players[:seed[:dirt]], town in {spec.TOWNS}, "
                     f"players in {sorted(PLAYERS)}, dirt in {spec.DIRT_RULES}")
        args.specs.append((parts[0], parts[1], int(parts[2]) if len(parts) > 2 else args.seed,
                           parts[3] if len(parts) > 3 else args.dirt))
    if args.out and len(args.specs) > 1:
        ap.error("--out names one folder; use --out-root with several runs")
    if args.envs % args.minibatches:
        ap.error("--envs must be a multiple of --minibatches")
    if args.rollout >= 1099:   # so at most one game per env ends in a rollout (see finish)
        ap.error("--rollout must be shorter than the shortest game (1,099 frames)")
    return args


def team_reward(reward, share):
    """What each player is paid, [envs, players]: their own apples (and feed), or with
    `share` the average of everyone's in that game."""
    import jax.numpy as jnp
    return jnp.where(share, reward.mean(-1, keepdims=True), reward)


def make_train_step(env, learner, num_envs, rollout, ia_gamma, ia_lambda):
    """One compiled call: `rollout` steps in all games, then the PPO update."""
    import jax
    import jax.numpy as jnp
    import ppo

    def train_step(rs, town, alpha, beta, share):
        def one_step(carry, _):
            es, obs, ready, start, h, smoothed, acc, fin, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            h2, a, logp, value, _ = learner._act(rs["params"], h, obs, ready, start, k_act)
            es, obs2, ready2, reward, done, info = jax.vmap(env.step, (0, 0, 0, None))(
                jax.random.split(k_env, num_envs), es, a.T, town)
            paid = team_reward(reward, share)
            smoothed = ppo.smooth_rewards(smoothed, paid, ia_gamma, ia_lambda)
            subjective = ppo.inequity_reward(paid, smoothed, alpha, beta)
            smoothed = jnp.where(done[:, None], 0.0, smoothed)
            got = (reward > 0).astype(jnp.float32)
            acc = acc + jnp.stack([reward, info["ate"], info["cleaned"], info["zaps"],
                                   info["feed"], info["length"][:, None] * got, got], -1)
            # Keep each finished game's totals (at most one per env per rollout, since
            # a rollout is shorter than the shortest game): only these go to the host.
            ended, length, totals = fin
            fin = (ended | done, jnp.where(done, info["length"], length),
                   jnp.where(done[:, None, None], acc, totals))
            acc = jnp.where(done[:, None, None], 0.0, acc)
            trans = {"obs": obs, "ready": ready, "start": start, "action": a, "logp": logp,
                     "value": value, "reward": subjective.T, "done": done}
            sums = jnp.stack([info["cleaned"].sum(), info["ate"].sum(), info["feed"].sum(),
                              subjective.sum()])
            return (es, obs2, ready2, done, h2, smoothed, acc, fin, key), (trans, sums)

        key, k_last, k_upd = jax.random.split(rs["key"], 3)
        fin = (jnp.zeros(num_envs, bool), jnp.zeros(num_envs, jnp.int32),
               jnp.zeros_like(rs["acc"]))
        carry = (rs["env"], rs["obs"], rs["ready"], rs["start"], rs["h"], rs["smoothed"],
                 rs["acc"], fin, key)
        carry, (batch, sums) = jax.lax.scan(one_step, carry, None, rollout)
        es, obs, ready, start, h, smoothed, acc, ended, key = carry
        _, _, _, last_value, _ = learner._act(rs["params"], h, obs, ready, start, k_last)
        batch["h0"], batch["last_value"] = rs["h"], last_value
        params, opt, stats = learner._update(rs["params"], rs["opt"], batch, k_upd)
        new = {"params": params, "opt": opt, "env": es, "obs": obs, "ready": ready,
               "start": start, "h": h, "smoothed": smoothed, "acc": acc, "key": key}
        return new, (ended, sums.sum(0), stats)

    return jax.jit(train_step, donate_argnums=(0,))


class Run:
    """One town with its own 7 learning players, all its games on the GPU."""

    def __init__(self, run_spec, args, env, learner, train_step):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        self.town, self.players, self.seed, self.dirt = run_spec
        alpha, beta = PLAYERS[self.players]
        self.alpha = alpha if args.alpha is None else args.alpha
        self.beta = beta if args.beta is None else args.beta
        # (Melting Pot's dirt rule adds nothing to the name, so older runs still resume.)
        self.name = f"{self.town}_{self.players}_s{self.seed}" + (
            "" if self.dirt == "melting_pot" else f"_{self.dirt}")
        self.out = args.out or os.path.join(args.out_root, self.name)
        self.args, self.env, self.learner, self.train_step = args, env, learner, train_step
        town_train.check_folder(self.out, args, {"town": self.town, "players": self.players,
                                                 "seed": self.seed, "alpha": self.alpha,
                                                 "beta": self.beta, "dirt": self.dirt},
                                SAME_ON_RESUME)
        self.tw = env.town(self.town, args.river, args.feed_reward, args.feed_period,
                           args.badge_frames, self.dirt)
        self.reward_args = (jnp.float32(self.alpha), jnp.float32(self.beta),
                            jnp.asarray(self.players in SHARED))

    def setup(self):
        jax, jnp, args, env = self.jax, self.jnp, self.args, self.env
        os.makedirs(self.out, exist_ok=True)
        self.ckpt_path = os.path.join(self.out, "ckpt.pkl")
        self.env_steps, self.update = 0, 0
        if args.resume and os.path.exists(self.ckpt_path):
            with open(self.ckpt_path, "rb") as f:
                state = pickle.load(f)
            params, opt, key = state["params"], state["opt"], jnp.asarray(state["key"])
            self.env_steps, self.update = state["env_steps"], state["update"]
            town_train.trim_logs(self.out, self.update, self.name)
            print(f"{self.name}: resumed at {self.env_steps:,} steps (update {self.update})",
                  flush=True)
        else:
            key, k = jax.random.split(jax.random.PRNGKey(self.seed))
            params, opt = self.learner.init(k)
            config = {k: v for k, v in vars(args).items() if k not in ("specs", "runs")}
            config.update(game="jaxtown", town=self.town, players=self.players, seed=self.seed,
                          dirt=self.dirt, alpha=self.alpha, beta=self.beta, num_envs=args.envs,
                          ppo=self.learner.cfg._asdict())
            with open(os.path.join(self.out, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        key, k_env = jax.random.split(key)
        n, p = args.envs, env.num_players
        es = jax.vmap(env.reset, (0, None))(jax.random.split(k_env, n), self.tw)
        obs, ready = jax.vmap(env.observe, (0, None))(es, self.tw)
        self.rs = {"params": params, "opt": opt, "env": es, "obs": obs, "ready": ready,
                   "start": jnp.ones(n, bool), "h": self.learner.initial_state(n),
                   "smoothed": jnp.zeros((n, p)), "acc": jnp.zeros((n, p, len(ACC))),
                   "key": key}
        metrics_path = os.path.join(self.out, "metrics.csv")
        new_file = not os.path.exists(metrics_path)
        self.metrics_f = open(metrics_path, "a", newline="")
        self.metrics = csv.DictWriter(self.metrics_f, fieldnames=METRIC_FIELDS)
        if new_file:
            self.metrics.writeheader()
        self.episodes_f = open(os.path.join(self.out, "episodes.jsonl"), "a")
        self.recent = []
        self.t_begin = time.time()

    @property
    def done_training(self):
        return self.env_steps >= self.args.steps

    def dispatch(self):
        """Queue this run's next rollout + update on the GPU (returns at once). Until
        finish() logs it, the weights are one update ahead of the counters."""
        self.rs, self.pending = self.train_step(self.rs, self.tw, *self.reward_args)
        self.in_flight = True

    def wait(self):
        """Block until the queued update is done; bring its small results to the host."""
        self.result = self.jax.device_get(self.pending)

    def finish(self, round_s):
        """Log the update. round_s: wall time of the round in which every active run did
        one update (they share the GPU), so sps is this run's rate while sharing."""
        (done, length, acc), sums, stats = self.result
        finished = []
        for e in np.nonzero(done)[0]:
            tot = dict(zip(ACC, acc[e].T))
            finished.append(episode_summary(
                length[e], tot["ret"], tot["ate"], tot["cleaned"], tot["zaps"], tot["feed"],
                tot["t_reward"], tot["n_reward"]))
        steps_now = self.args.envs * self.args.rollout
        town_train.log_update(self, finished, sums, np.asarray(stats).mean(axis=0), steps_now,
                              sps=steps_now / round_s, t_update=round(round_s, 2))
        self.in_flight = False

    def checkpoint(self):
        town_train.save_ckpt(self.ckpt_path, town_train.ckpt_state(
            self, self.rs["params"], self.rs["opt"], self.rs["key"]))

    def close(self):
        for f in ("metrics_f", "episodes_f"):
            if hasattr(self, f):
                getattr(self, f).close()


_COMPILED = {}   # same settings in one process (tests, notebooks): compile once


def build(args):
    import cleanup as cu
    import ppo
    key = (args.lr, args.ent_coef, args.minibatches, args.envs, args.rollout, args.ia_gamma,
           args.ia_lambda, args.dtype)
    if key not in _COMPILED:
        env = cu.CleanUp()
        cfg = ppo.Config(lr=args.lr, ent_coef=args.ent_coef, minibatches=args.minibatches,
                         encoder="cells", obs_shape=(cu.VIEW, cu.VIEW, cu.NUM_CHANNELS),
                         compute_dtype=args.dtype)
        learner = ppo.Learner(cfg)
        _COMPILED[key] = (env, learner, make_train_step(
            env, learner, args.envs, args.rollout, args.ia_gamma, args.ia_lambda))
    return _COMPILED[key]


def main(argv=None):
    args = parse(argv)
    env, learner, train_step = build(args)
    runs = [Run(s, args, env, learner, train_step) for s in args.specs]
    for r in runs:
        r.setup()
    print("runs:", ", ".join(f"{r.name} (alpha {r.alpha}, beta {r.beta}, dirt {r.dirt})"
                             for r in runs),
          f"| {args.envs} games each, river {args.river} | target {int(args.steps):,} steps "
          "each", flush=True)
    stop = {"now": False}

    def on_signal(signum, frame):
        stop["now"] = True
        print("stopping after this update (checkpoints first)...", flush=True)

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    last_ckpt = time.time()
    try:
        while not stop["now"]:
            active = [r for r in runs if not r.done_training]
            if not active:
                break
            t_round = time.time()
            for r in active:       # queue every run's work, then collect: the GPU stays busy
                r.dispatch()
            for r in active:
                r.wait()
            round_s = time.time() - t_round
            for r in active:
                r.finish(round_s)
            if time.time() - last_ckpt > 60 * args.ckpt_minutes:
                for r in runs:
                    r.checkpoint()
                last_ckpt = time.time()
    finally:
        for r in runs:
            if hasattr(r, "rs"):
                try:
                    if getattr(r, "in_flight", False):   # weights ahead of counters: log it
                        r.wait()
                        r.finish(float("nan"))
                    r.checkpoint()
                    print(f"saved {r.ckpt_path} at {r.env_steps:,} steps", flush=True)
                except Exception as e:   # e.g. the GPU failed mid-update: keep its error
                    print(f"{r.name}: could not save a checkpoint ({type(e).__name__}); "
                          "it resumes from its last saved one", flush=True)
            r.close()
        for s, handler in old.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    main()
