"""Train groups of 7 players from scratch, one group per town, all in one process.

Run inside WSL from the repo root (a login shell, so the GPU libraries are found):
    ~/venvs/meltingpot/bin/python sims/town/train.py --runs neighbours:selfish neighbours:fair

Each run is "town:players[:seed]". Every run has its own games, its own 7 players
and its own output folder; they share only the GPU and this process. Sharing one
process matters on this PC: two JAX processes on one GPU under WSL took turns so
badly that each step got ~30x slower, while one process keeps the GPU busy and
lets one run's games step while it chooses the other run's actions.

players:
    selfish   each player's reward is its own apples (and feed, in the isolating town)
    fair      the same, minus Hughes et al. 2018's advantageous inequity aversion
              ("guilt": alpha 0, beta 0.05), which made Cleanup players cooperate there

Writes to out/town/<town>_<players>_s<seed>/:
    config.json     everything needed to rerun it
    metrics.csv     one row per update: steps, speed, cleaning and apples per step, losses
    episodes.jsonl  one line per finished game: per-player returns, cleaning, apples, zaps
    ckpt.pkl        weights + optimizer, saved every few minutes; --resume continues it
Ctrl-C (or SIGTERM) finishes the current update, saves checkpoints and stops.
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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from envs import NUM_PLAYERS, VecTown  # noqa: E402  (light: no Melting Pot, no JAX)
from spec import RIVERS, TOWNS  # noqa: E402  (light too)

PLAYERS = {"selfish": (0.0, 0.0), "fair": (0.0, 0.05)}   # (alpha, beta)
METRIC_FIELDS = ["update", "env_steps", "wall_s", "sps", "clean_per_step", "apples_per_step",
                 "feed_per_step", "subjective_per_step", "episodes", "collective_return",
                 "pg_loss", "value_loss", "entropy", "approx_kl", "t_act", "t_wait",
                 "t_update"]


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="town:players[:seed] ... (default: --town/--players/--seed)")
    ap.add_argument("--town", choices=TOWNS, default="neighbours")
    ap.add_argument("--players", choices=sorted(PLAYERS), default="selfish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=2e7, help="game steps per run")
    ap.add_argument("--workers", type=int, default=6, help="game processes per run")
    ap.add_argument("--envs-per-worker", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=None, help="override envy (disadvantageous)")
    ap.add_argument("--beta", type=float, default=None, help="override guilt (advantageous)")
    ap.add_argument("--ia-gamma", type=float, default=0.99)
    ap.add_argument("--ia-lambda", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--feed-reward", type=float, default=0.1)
    ap.add_argument("--feed-period", type=int, default=4)
    ap.add_argument("--badge-frames", type=int, default=50)
    ap.add_argument("--river", choices=RIVERS, default="melting_pot",
                    help="starting dirt: as shipped (54%%), just past the tipping point as in "
                         "Hughes et al. 2018 (41%%), or clean with no dirt ever (sanity check)")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--out-root", default=os.path.join("out", "town"))
    ap.add_argument("--out", default=None, help="exact output folder (one run only)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    specs = args.runs or [f"{args.town}:{args.players}:{args.seed}"]
    args.specs = []
    for s in specs:
        parts = s.split(":")
        if len(parts) < 2:
            ap.error(f"bad run {s!r}: write town:players[:seed], e.g. neighbours:fair")
        town, players = parts[0], parts[1]
        seed = int(parts[2]) if len(parts) > 2 else args.seed
        if town not in TOWNS or players not in PLAYERS:
            ap.error(f"bad run {s!r}: town in {TOWNS}, players in {sorted(PLAYERS)}")
        args.specs.append((town, players, seed))
    if args.out and len(args.specs) > 1:
        ap.error("--out names one folder; use --out-root with several runs")
    return args


# Settings that change what's learned: a resumed run must keep the ones it started with.
SAME_ON_RESUME = ("town", "players", "seed", "alpha", "beta", "river", "feed_reward",
                  "feed_period", "badge_frames", "lr", "ent_coef", "ia_gamma", "ia_lambda",
                  "rollout")


def check_folder(out, args, run_settings, same_on_resume=SAME_ON_RESUME):
    """Never mix two runs in one folder: a new run needs a folder with no run in it,
    and a resumed run must keep the settings it started with. (Shared with
    sims/jaxtown/train_jax.py.)"""
    has_ckpt = os.path.exists(os.path.join(out, "ckpt.pkl"))
    has_logs = any(os.path.exists(os.path.join(out, n)) for n in ("metrics.csv", "episodes.jsonl"))
    if args.resume and has_ckpt:
        with open(os.path.join(out, "config.json")) as f:
            saved = json.load(f)
        saved.setdefault("river", "melting_pot")   # runs from before --river existed
        saved.setdefault("dirt", "melting_pot")    # JAX runs from before --dirt existed
        saved.setdefault("coop_share", 0.0)        # JAX runs from before the co-op existed
        now = {**vars(args), **run_settings}
        diff = {k: (saved.get(k), now.get(k)) for k in same_on_resume
                if saved.get(k) != now.get(k)}
        if diff:
            raise SystemExit(f"{out}: --resume with different settings than the run "
                             f"started with, (was, now): {diff}")
    elif has_ckpt:
        raise SystemExit(f"{out} already holds a run: use --resume to carry on, "
                         "or pick another --out-root")
    elif has_logs:
        raise SystemExit(f"{out} holds logs but no checkpoint (stopped before its "
                         "first save?): pick another --out-root")


def trim_logs(out, update, name):
    """On resume: drop log rows written after the checkpoint (a hard kill loses up to
    --ckpt-minutes of training, which is about to be trained again)."""
    for fname in ("metrics.csv", "episodes.jsonl"):
        path = os.path.join(out, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = f.readlines()
        if fname == "metrics.csv":   # first column is the update number
            head, rows = lines[:1], lines[1:]
            keep = [ln for ln in rows if int(ln.split(",", 1)[0]) <= update]
        else:
            head, rows = [], [ln for ln in lines if ln.strip()]
            keep = [ln for ln in rows if json.loads(ln)["update"] <= update]
        if len(keep) < len(rows):
            with open(path, "w") as f:
                f.writelines(head + keep)
            print(f"{name}: dropped {len(rows) - len(keep)} rows of {fname} logged "
                  "after the checkpoint", flush=True)


def log_update(run, finished, sums, stats, steps_now, sps, t_act="", t_wait="", t_update=""):
    """After one update: count it, log its finished games and its metrics row, print
    progress every 10 updates. Shared with sims/jaxtown/train_jax.py, so both trainers'
    files mean the same thing. `run` has update, env_steps, t_begin, name, recent and
    the open metrics / episodes files; sums = cleaned, ate, feed, subjective."""
    run.env_steps += steps_now
    run.update += 1
    for ep in finished:
        run.episodes_f.write(json.dumps(episode_line(ep, run.update, run.env_steps)) + "\n")
    run.episodes_f.flush()
    run.recent = (run.recent + [ep["collective_return"] for ep in finished])[-50:]
    n = steps_now
    row = {"update": run.update, "env_steps": run.env_steps,
           "wall_s": round(time.time() - run.t_begin),
           "sps": round(sps) if np.isfinite(sps) else "",
           "clean_per_step": sums[0] / n, "apples_per_step": sums[1] / n,
           "feed_per_step": sums[2] / n, "subjective_per_step": sums[3] / n,
           "episodes": len(finished),
           "collective_return": (np.mean([ep["collective_return"] for ep in finished])
                                 if finished else ""),
           "pg_loss": stats[0], "value_loss": stats[1], "entropy": stats[2],
           "approx_kl": stats[3], "t_act": t_act, "t_wait": t_wait, "t_update": t_update}
    run.metrics.writerow(row)
    run.metrics_f.flush()
    if run.update % 10 == 0:
        cr = f"{np.mean(run.recent):.1f}" if run.recent else "-"
        sps_txt = f"{row['sps']:,}" if row["sps"] != "" else "?"
        print(f"{run.name} upd {run.update} steps {run.env_steps:,} ({sps_txt}/s) "
              f"clean/step {row['clean_per_step']:.4f} apples/step "
              f"{row['apples_per_step']:.4f} return (last {len(run.recent)} games) {cr} "
              f"entropy {stats[2]:.3f}", flush=True)


def ckpt_state(run, params, opt, key):
    """What a checkpoint holds (shared with sims/jaxtown/train_jax.py)."""
    import jax
    return {"params": jax.device_get(params), "opt": jax.device_get(opt),
            "key": np.asarray(key), "env_steps": run.env_steps, "update": run.update,
            "town": run.town, "players": run.players, "alpha": run.alpha, "beta": run.beta}


def save_ckpt(path, state):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def episode_line(ep, update, env_steps):
    cleaned = np.array(ep["cleaned"])
    top2 = np.sort(cleaned)[-2:].sum() / cleaned.sum() if cleaned.sum() > 0 else float("nan")
    return {"update": update, "env_steps": env_steps, **ep,
            "cleaned_total": float(cleaned.sum()), "ate_total": float(np.sum(ep["ate"])),
            "zaps_total": float(np.sum(ep["zaps"])), "top2_clean_share": float(top2)}


class Run:
    """One town with its own 7 learning players."""

    def __init__(self, spec, args):
        self.town, self.players, self.seed = spec
        alpha, beta = PLAYERS[self.players]
        self.alpha = alpha if args.alpha is None else args.alpha
        self.beta = beta if args.beta is None else args.beta
        self.name = f"{self.town}_{self.players}_s{self.seed}"
        self.out = args.out or os.path.join(args.out_root, self.name)
        self.args = args
        self._check_folder()   # before starting any game workers
        town_kwargs = dict(feed_reward=args.feed_reward, feed_period=args.feed_period,
                           badge_frames=args.badge_frames, river=args.river)
        self.venv = VecTown(self.town, args.workers, args.envs_per_worker, town_kwargs)
        self.num_envs = self.venv.num_envs

    def _check_folder(self):
        check_folder(self.out, self.args, {"town": self.town, "players": self.players,
                                           "seed": self.seed, "alpha": self.alpha,
                                           "beta": self.beta})

    def _trim_logs(self):
        trim_logs(self.out, self.update, self.name)

    def setup(self, learner):
        """After JAX is imported: weights (new or resumed), files, first observations."""
        import jax
        import jax.numpy as jnp
        import ppo
        self.jax, self.jnp, self.ppo, self.learner = jax, jnp, ppo, learner
        args = self.args
        os.makedirs(self.out, exist_ok=True)
        self.ckpt_path = os.path.join(self.out, "ckpt.pkl")
        self.env_steps, self.update = 0, 0
        if args.resume and os.path.exists(self.ckpt_path):
            with open(self.ckpt_path, "rb") as f:
                state = pickle.load(f)
            self.params, self.opt = state["params"], state["opt"]
            self.key = jnp.asarray(state["key"])
            self.env_steps, self.update = state["env_steps"], state["update"]
            self._trim_logs()
            print(f"{self.name}: resumed at {self.env_steps:,} steps (update {self.update})",
                  flush=True)
        else:
            self.key, k = jax.random.split(jax.random.PRNGKey(self.seed))
            self.params, self.opt = learner.init(k)
            config = {k: v for k, v in vars(args).items() if k not in ("specs", "runs")}
            config.update(town=self.town, players=self.players, seed=self.seed,
                          alpha=self.alpha, beta=self.beta, num_envs=self.num_envs,
                          ppo=learner.cfg._asdict())
            with open(os.path.join(self.out, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
        metrics_path = os.path.join(self.out, "metrics.csv")
        new_file = not os.path.exists(metrics_path)
        self.metrics_f = open(metrics_path, "a", newline="")
        self.metrics = csv.DictWriter(self.metrics_f, fieldnames=METRIC_FIELDS)
        if new_file:
            self.metrics.writeheader()
        self.episodes_f = open(os.path.join(self.out, "episodes.jsonl"), "a")
        self.obs, self.ready = self.venv.reset()
        self.start = np.ones(self.num_envs, bool)
        self.h = learner.initial_state(self.num_envs)
        self.smoothed = np.zeros((self.num_envs, NUM_PLAYERS), np.float32)
        self.recent = []
        self.t_begin = time.time()

    @property
    def done_training(self):
        return self.env_steps >= self.args.steps

    def begin_rollout(self):
        self.h0 = self.h
        self.buf = {k: [] for k in ("obs", "ready", "start", "action", "logp", "value",
                                    "reward", "done")}
        self.sums = np.zeros(4)  # cleaned, ate, feed, subjective
        self.finished = []
        self.t_act = self.t_wait = 0.0
        self.t0 = time.time()

    def act_and_send(self):
        ta = time.time()
        # One explicit transfer: passing numpy straight into the jitted act was
        # ~10x slower here (measured 2026-11-16, WSL GPU).
        prev = self.jax.device_put((self.obs, self.ready, self.start))
        self.h, a, logp, value, self.key = self.learner.act(self.params, self.h, *prev, self.key)
        self.venv.step_async(np.asarray(a).T)                       # [E, P]
        self.pending = (prev, a, logp, value)
        self.t_act += time.time() - ta

    def receive(self):
        tw = time.time()
        obs, ready, reward, done, ate, cleaned, episodes = self.venv.step_wait()
        self.t_wait += time.time() - tw
        (o, r, s), a, logp, value = self.pending
        self.smoothed = self.ppo.smooth_rewards(self.smoothed, reward, self.args.ia_gamma,
                                                self.args.ia_lambda)
        subjective = self.ppo.inequity_reward(reward, self.smoothed, self.alpha, self.beta)
        self.smoothed[done] = 0.0
        for name, x in (("obs", o), ("ready", r), ("start", s), ("action", a),
                        ("logp", logp), ("value", value), ("reward", subjective.T),
                        ("done", done)):
            self.buf[name].append(x)
        self.obs, self.ready, self.start = obs, ready, done
        self.sums += [cleaned.sum(), ate.sum(), (reward - ate).sum(), subjective.sum()]
        self.finished += episodes

    def dispatch_update(self):
        """Queue the PPO update on the GPU (returns at once; JAX runs it async)."""
        jax, jnp = self.jax, self.jnp
        self.t_roll = time.time() - self.t0
        # Value of where each game stands now (the new h from this call is not kept).
        _, _, _, last_value, self.key = self.learner.act(
            self.params, self.h, *jax.device_put((self.obs, self.ready, self.start)), self.key)
        # Device arrays are stacked on the GPU; the host ones go over in one transfer each.
        batch = {name: (np.stack(xs) if isinstance(xs[0], np.ndarray) else jnp.stack(xs))
                 for name, xs in self.buf.items()}
        batch["reward"] = batch["reward"].astype(np.float32)
        batch["h0"], batch["last_value"] = self.h0, last_value
        self.key, k = jax.random.split(self.key)
        self.params, self.opt, self.stats = self.learner.update(self.params, self.opt, batch, k)

    def finish_update(self):
        stats = np.asarray(self.stats).mean(axis=0)   # waits for the GPU
        steps_now = self.num_envs * self.args.rollout
        log_update(self, self.finished, self.sums, stats, steps_now,
                   sps=steps_now / (time.time() - self.t0), t_act=round(self.t_act, 2),
                   t_wait=round(self.t_wait, 2),
                   t_update=round(time.time() - self.t0 - self.t_roll, 2))

    def checkpoint(self):
        save_ckpt(self.ckpt_path, ckpt_state(self, self.params, self.opt, self.key))

    def close(self):
        for f in ("metrics_f", "episodes_f"):
            if hasattr(self, f):
                getattr(self, f).close()
        self.venv.close()


def main(argv=None):
    args = parse(argv)
    # Start every run's game workers before JAX touches the GPU.
    runs = [Run(spec, args) for spec in args.specs]

    import ppo
    learner = ppo.Learner(ppo.Config(lr=args.lr, ent_coef=args.ent_coef))
    for r in runs:
        r.setup(learner)
    print("runs:", ", ".join(f"{r.name} (alpha {r.alpha}, beta {r.beta}, {r.num_envs} games)"
                             for r in runs), f"| target {int(args.steps):,} steps each",
          flush=True)

    stop = {"now": False}

    def on_signal(signum, frame):
        stop["now"] = True
        print("stopping after this update (checkpoints first)...", flush=True)

    old_handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    last_ckpt = time.time()
    try:
        while not stop["now"]:
            active = [r for r in runs if not r.done_training]
            if not active:
                break
            for r in active:
                r.begin_rollout()
            # Pipelined: while one run's games step, we choose the next run's actions.
            for t in range(args.rollout):
                for r in active:
                    if t > 0:
                        r.receive()
                    r.act_and_send()
            for r in active:
                r.receive()
            for r in active:
                r.dispatch_update()
            for r in active:
                r.finish_update()
            if time.time() - last_ckpt > 60 * args.ckpt_minutes:
                for r in runs:
                    r.checkpoint()
                last_ckpt = time.time()
    finally:
        for r in runs:
            if hasattr(r, "params"):
                r.checkpoint()
                print(f"saved {r.ckpt_path} at {r.env_steps:,} steps", flush=True)
            r.close()
        for s, handler in old_handlers.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    main()
