"""Newcomers (experiment 1): 3 of a town's 7 trained players move to another town and live
there with its 4 locals, and everyone carries on learning. Do the newcomers come to act
like the locals, or the locals like them?

The moves: every town to every other town, both ways, for selfish and for fair players
and both seeds: 4 x 3 x 2 x 2 = 48. Players stay who they are: selfish players move to the
selfish version of the other town and fair ones to the fair version, so their fairness
setting comes with them. Plus a control per town (4 x 2 x 2 = 16): 3 players from the same
town's other seed, so a new town can be told apart from just new faces. 64 runs.

Newcomers always come from the other seed's run. Runs with the same seed start from the
same weights (train_jax seeds the players' first weights with --seed), so same-seed players
are siblings of the hosts while the controls' are not: the first try (2026-11-20, kept as
"siblings" folders) mixed the two, and a sibling town's newcomers fitted in where every
other newcomer broke the town's apple supply.

How a move is made: the run starts from the host run's checkpoint (its 7 players' weights
and optimizer state, and its random key, so every move into one host plays out with the
same luck), and the newcomers' weights and optimizer state replace slots 0-2. Slots 3-6
are the locals. Its counters start at 0: steps since the move.

    cd /mnt/c/dev/new-melting-pot
    # 1. make the 64 run folders from the finished four towns (both seeds)
    ~/venvs/meltingpot/bin/python sims/jaxtown/newcomers.py make \\
        --sources out/jaxtown/towns out/colab/towns --out-root out/jaxtown/newcomers
    # 2. train them 8 at a time (train_jax.py --folders ... --resume); safe to re-run
    ~/venvs/meltingpot/bin/python sims/jaxtown/newcomers.py train --out-root out/jaxtown/newcomers
    # 3. chart + tables (docs/experiments/01-newcomers.md)
    ~/venvs/meltingpot/bin/python sims/jaxtown/newcomers.py report --out-root \\
        out/jaxtown/newcomers --out out/jaxtown/writeup/newcomers.png
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from typing import NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))

import numpy as np  # noqa: E402

import cloud_vm  # noqa: E402  (ckpt_steps: a checkpoint's step count without JAX)
import spec  # noqa: E402

TRAIN = os.path.join(HERE, "train_jax.py")
NEWCOMER_SLOTS = (0, 1, 2)
SEEDS = (0, 1)
PLAYER_TYPES = ("selfish", "fair")
STEPS = 2e7
# Settings the training after a move keeps from before it (train_jax flags of the same
# name, with - for _). Host and newcomers must agree on them, and on the dirt rule.
KEEP = ("river", "feed_reward", "feed_period", "badge_frames", "lr", "ent_coef", "ia_gamma",
        "ia_lambda", "rollout", "envs", "minibatches", "dtype")


class Move(NamedTuple):
    host: str          # the town the newcomers move to (its run gives the 4 locals)
    origin: str        # the town they come from
    players: str       # selfish or fair, the same on both sides
    seed: int          # the host run's seed
    origin_seed: int   # the newcomers' run's seed (in plan(): always the other seed)

    @property
    def control(self):
        return self.host == self.origin

    @property
    def name(self):
        return (f"{self.host}_{self.players}_s{self.seed}"
                f"_from_{self.origin}_s{self.origin_seed}")

    @property
    def host_run(self):
        return f"{self.host}_{self.players}_s{self.seed}"

    @property
    def origin_run(self):
        return f"{self.origin}_{self.players}_s{self.origin_seed}"


def plan(seeds=SEEDS, player_types=PLAYER_TYPES, towns=spec.TOWNS):
    """Every move between two towns (both ways), and a control per town, for each player
    type and seed, the newcomers always from the other seed (see above). Ordered by player
    type, seed, host, origin."""
    if len(seeds) != 2:
        raise ValueError("newcomers come from the other of exactly two seeds")
    moves = []
    for p in player_types:
        for s in seeds:
            other = seeds[1 - seeds.index(s)]
            for host in towns:
                for origin in towns:
                    moves.append(Move(host, origin, p, s, other))
    return moves


def find_run(sources, name):
    hits = [os.path.join(r, name) for r in sources
            if os.path.exists(os.path.join(r, name, "ckpt.pkl"))]
    if len(hits) != 1:
        raise SystemExit(f"{name}: a finished run must be in exactly one of --sources "
                         f"{list(sources)}, found {len(hits)}")
    return hits[0]


def load(run_dir):
    with open(os.path.join(run_dir, "ckpt.pkl"), "rb") as f:
        state = pickle.load(f)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    return state, config


def swap_in(host_tree, newcomer_tree, slots=NEWCOMER_SLOTS, origin_slots=None):
    """The host's weights (or optimizer state) with `slots` taken from the newcomers' run:
    host slot slots[k] gets the newcomers' run's player origin_slots[k] (default: the same
    slots). Every leaf has the player axis first (ppo.Learner vmaps over players)."""
    import jax
    idx = list(slots)
    src = list(slots if origin_slots is None else origin_slots)

    def put(h, o):
        h, o = np.asarray(h), np.asarray(o)
        if h.shape != o.shape or h.dtype != o.dtype or h.ndim == 0 or h.shape[0] != 7:
            raise ValueError(f"can't swap players between leaves {h.shape} {h.dtype} and "
                             f"{o.shape} {o.dtype}")
        x = h.copy()
        x[idx] = o[src]
        return x
    return jax.tree_util.tree_map(put, host_tree, newcomer_tree)


def seat_map(slots, origin_slots, players=7):
    """For newcomers seated in host `slots` who sat in `origin_slots` at home: for each host
    seat, the home seat whose player its occupant stands in for. Each newcomer maps to its
    own home seat; the locals' seats map to the home seats left over, in order."""
    slots, origin_slots = list(slots), list(origin_slots)
    if sorted(set(slots)) != sorted(slots) or sorted(set(origin_slots)) != sorted(origin_slots) \
            or len(slots) != len(origin_slots):
        raise ValueError(f"bad seats {slots} <- {origin_slots}")
    seats = [None] * players
    for h, o in zip(slots, origin_slots):
        seats[h] = o
    rest = iter(s for s in range(players) if s not in origin_slots)
    return [s if s is not None else next(rest) for s in seats]


def reseat(tree, slots, origin_slots):
    """Weights (or Adam moments) with the newcomers in `slots` relabelled for their new seats.
    Players tell each other apart by seat (cleanup's player0-6 channels, which a player never
    sees on itself). A newcomer moved to a different seat would see someone in its old seat's
    channel, which it never saw on anyone else, and so has untrained weights for. Its first
    layer (a 1x1 conv, one weight row per input channel) is permuted instead: it sees the host
    player in seat h as it saw the home player in seat seat_map(...)[h]. That's exact: the same
    network on relabelled views. Fellow newcomers look as they did at home; locals look like
    the home players who stayed behind. Same seats: nothing changes."""
    import jax
    import cleanup
    perm = seat_map(slots, origin_slots)
    if perm == list(range(len(perm))):
        return tree
    first = cleanup.CH["player0"]
    ids = np.arange(first, first + len(perm))

    def put(path, x):
        keys = [getattr(k, "key", None) for k in path]
        if keys[-2:] != ["conv1", "kernel"]:
            return x
        x = np.asarray(x).copy()
        if x.ndim != 5 or x.shape[0] != len(perm) or x.shape[3] != cleanup.NUM_CHANNELS:
            raise ValueError(f"unexpected first layer {x.shape}")
        for s in slots:
            k = x[s]                                  # [1, 1, channels, out], a view
            k[:, :, ids, :] = k[:, :, ids[perm], :]   # seat h's row <- home seat perm[h]'s
        return x
    return jax.tree_util.tree_map_with_path(put, tree)


def make_folder(move, sources, out_root, steps=STEPS):
    """Write `out_root/<move.name>/` (ckpt.pkl + config.json) for train_jax --folders.
    Returns (folder, made); a folder that already exists is never touched."""
    out = os.path.join(out_root, move.name)
    if os.path.exists(out):
        return out, False
    host_dir, origin_dir = find_run(sources, move.host_run), find_run(sources, move.origin_run)
    info = {"host": move.host, "origin": move.origin, "seed": move.seed,
            "origin_seed": move.origin_seed, "control": move.control,
            "newcomer_slots": list(NEWCOMER_SLOTS)}
    write_folder(out, host_dir, origin_dir, info, out_root, steps)
    return out, True


def write_folder(out, host_dir, origin_dir, info, out_root, steps=STEPS, slots=NEWCOMER_SLOTS,
                 origin_slots=None):
    """Write the run folder `out` (ckpt.pkl + config.json): the host run's checkpoint with
    the newcomers' run's players origin_slots (default: the same seats) in seats `slots`,
    relabelled for their new seats (reseat). `info` goes into config.json's "newcomers"."""
    name = os.path.basename(out)
    origin_slots = list(slots if origin_slots is None else origin_slots)
    (host, cfg_h), (origin, cfg_o) = load(host_dir), load(origin_dir)
    for k in ("players", "alpha", "beta", "env_steps"):
        if host[k] != origin[k]:
            raise SystemExit(f"{name}: host and newcomers differ in {k} "
                             f"({host[k]} vs {origin[k]})")
    diff = {k: (cfg_h.get(k), cfg_o.get(k)) for k in KEEP + ("dirt",)
            if cfg_h.get(k) != cfg_o.get(k)}
    if diff:
        raise SystemExit(f"{name}: host and newcomers trained differently: {diff}")

    def move_in(h, o):
        return reseat(swap_in(h, o, slots, origin_slots), slots, origin_slots)
    state = {"params": move_in(host["params"], origin["params"]),
             "opt": move_in(host["opt"], origin["opt"]), "key": host["key"],
             "env_steps": 0, "update": 0, "town": host["town"], "players": host["players"],
             "alpha": host["alpha"], "beta": host["beta"]}
    config = {**cfg_h, "steps": steps, "out": None, "out_root": out_root, "resume": True,
              "newcomers": {"host_run": host_dir, "newcomers_run": origin_dir, **info,
                            "steps_before": host["env_steps"],
                            "updates_before": host["update"]}}
    part = out + ".part"
    if os.path.exists(part):
        raise SystemExit(f"{part} is left from an earlier try: look at it, remove it, re-run")
    os.makedirs(part)
    with open(os.path.join(part, "ckpt.pkl"), "wb") as f:
        pickle.dump(state, f)
    with open(os.path.join(part, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    os.replace(part, out)        # the folder appears whole or not at all
    return out


def make(sources, out_root, moves=None, steps=STEPS, log=print):
    os.makedirs(out_root, exist_ok=True)
    made = 0
    for m in moves or plan():
        _, new = make_folder(m, sources, out_root, steps)
        made += new
    log(f"made {made} run folders in {out_root} ({len(moves or plan()) - made} were there)")


def train(out_root, steps=STEPS, batch=8, moves=None, ckpt_minutes=10.0, python=None,
          dry_run=False):
    """Train every move's folder, `batch` runs per train_jax process, one process after
    another. Finished batches are skipped, so it carries on after a stop."""
    folders = [os.path.join(out_root, m.name) for m in moves or plan()]
    missing = [f for f in folders if not os.path.exists(os.path.join(f, "ckpt.pkl"))]
    if missing:
        raise SystemExit(f"{len(missing)} run folders missing (run make first), "
                         f"e.g. {missing[0]}")
    queue = os.path.join(out_root, "queue.log")

    def note(text):
        with open(queue, "a") as f:
            f.write(f"{time.strftime('%F %T')} {text}\n")
    batches = [folders[i:i + batch] for i in range(0, len(folders), batch)]
    for i, b in enumerate(batches):
        if all(cloud_vm.ckpt_steps(os.path.join(f, "ckpt.pkl"))[0] >= steps for f in b):
            continue
        cfgs = [json.load(open(os.path.join(f, "config.json"))) for f in b]
        flags = []
        for k in KEEP:
            values = {json.dumps(c[k]) for c in cfgs}
            if len(values) != 1:
                raise SystemExit(f"batch {i}: its runs differ in {k}: {sorted(values)}")
            flags += [f"--{k.replace('_', '-')}", str(cfgs[0][k])]
        argv = [python or sys.executable, TRAIN, "--folders", *b, "--resume",
                "--steps", str(int(steps)), "--ckpt-minutes", str(ckpt_minutes), *flags]
        if dry_run:
            print(" ".join(argv))
            continue
        note(f"newcomers batch {i + 1} of {len(batches)}: starting {len(b)} runs")
        with open(os.path.join(out_root, f"train_batch{i}.log"), "a") as log:
            code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        note(f"newcomers batch {i + 1} of {len(batches)}: exited with code {code}")
        if code:
            note("stopping here: re-run train to carry on")
            raise SystemExit(code)
    if not dry_run:
        note("newcomers: every batch finished")


# ---- Report: what the newcomers and the locals did -----------------------------------------

LOCAL_SLOTS = tuple(s for s in range(7) if s not in NEWCOMER_SLOTS)
KINDS = ("into the isolating town", "out of the isolating town", "between ordinary towns",
         "control, ordinary towns", "control, isolating town")
EARLY = 0.1   # "just after the move": the first 10% of the steps after it (2M of 20M)
LATE = 0.1    # "at the end": the last 10% (as sims/town/report.py), and before the move
BIN = 5e5     # chart: steps per point


def kind(move):
    if move.control:
        return KINDS[4] if move.host == "isolating" else KINDS[3]
    if move.host == "isolating":
        return KINDS[0]
    return KINDS[1] if move.origin == "isolating" else KINDS[2]


def games(run_dir, start=0.0):
    """A run's finished games (episodes.jsonl) from `start` env steps on."""
    out = []
    with open(os.path.join(run_dir, "episodes.jsonl")) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["env_steps"] >= start:
                    out.append(e)
    return out


def per_1000(eps, key, slots):
    """`key` (cleaned, ate, ...) of the players in `slots` together, per 1,000 game steps
    (frames; games differ in length); nan without games."""
    frames = sum(e["length"] for e in eps)
    if not frames:
        return float("nan")
    return 1000 * sum(float(np.sum(np.asarray(e[key])[list(slots)])) for e in eps) / frames


def measures(eps):
    """Newcomers' and locals' cleaning per player, and the whole town's apples, all per
    1,000 game steps."""
    return {"new_clean": per_1000(eps, "cleaned", NEWCOMER_SLOTS) / len(NEWCOMER_SLOTS),
            "local_clean": per_1000(eps, "cleaned", LOCAL_SLOTS) / len(LOCAL_SLOTS),
            "apples": per_1000(eps, "ate", range(7))}


def summarise(out_root, moves=None):
    """One row per move: before it (the host's and the newcomers' own runs, last 10%),
    just after it and at the end; and each move's measures over steps since the move."""
    cache = {}

    def before(run_dir, steps):
        if run_dir not in cache:
            cache[run_dir] = measures(games(run_dir, (1 - LATE) * steps))
        return cache[run_dir]

    rows = []
    for m in moves or plan():
        folder = os.path.join(out_root, m.name)
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)["newcomers"]
        host = before(cfg["host_run"], cfg["steps_before"])
        home = before(cfg["newcomers_run"], cfg["steps_before"])
        eps = games(folder)
        done = max(e["env_steps"] for e in eps)
        early = measures([e for e in eps if e["env_steps"] <= EARLY * done])
        late = measures([e for e in eps if e["env_steps"] >= (1 - LATE) * done])
        edges = np.arange(0, done + BIN, BIN)
        idx = np.searchsorted(edges, [e["env_steps"] for e in eps], side="left")
        series = {k: [] for k in early}
        for b in range(1, len(edges)):
            got = measures([e for e, i in zip(eps, idx) if i == b])
            for k, v in got.items():
                series[k].append(v)
        rows.append({"move": m, "kind": kind(m), "steps": done,
                     # The newcomers at home are slots 0-2 of their own run.
                     "home_new_clean": home["new_clean"], "host_local_clean": host["local_clean"],
                     "host_apples": host["apples"], "early": early, "late": late,
                     "x": edges[1:] - BIN / 2, "series": series})
    return rows


def gap_closed(r):
    """How far the newcomers' cleaning moved from their home level toward the locals' level
    before the move (1 = all the way); nan when the two levels are close."""
    gap = r["host_local_clean"] - r["home_new_clean"]
    return (r["late"]["new_clean"] - r["home_new_clean"]) / gap if abs(gap) > 10 else float("nan")


def report_tables(rows):
    def mean(xs):
        xs = np.asarray(xs, float)
        return float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")

    def rng(xs):
        xs = np.asarray(xs, float)
        xs = xs[np.isfinite(xs)]
        return f"{xs.min():.0%} to {xs.max():.0%}" if len(xs) else ""

    lines = ["| Move | Runs | Newcomers' cleaning: at home, just after, at the end | "
             "Locals' cleaning: before, at the end | Town's apples: before, just after, at "
             "the end | Newcomers' gap to the locals closed |",
             "|---|---|---|---|---|---|"]
    for k in KINDS:
        g = [r for r in rows if r["kind"] == k]
        if not g:
            continue
        f = lambda get: mean([get(r) for r in g])  # noqa: E731
        closed = [gap_closed(r) for r in g]
        lines.append(
            f"| {k} | {len(g)} | {f(lambda r: r['home_new_clean']):.0f}, "
            f"{f(lambda r: r['early']['new_clean']):.0f}, {f(lambda r: r['late']['new_clean']):.0f}"
            f" | {f(lambda r: r['host_local_clean']):.0f}, {f(lambda r: r['late']['local_clean']):.0f}"
            f" | {f(lambda r: r['host_apples']):.1f}, {f(lambda r: r['early']['apples']):.1f}, "
            f"{f(lambda r: r['late']['apples']):.1f} | "
            # (Only where the two levels are far apart: moves to or from the isolating town.)
            + (f"{mean(closed):.0%} ({rng(closed)})" if k in KINDS[:2] else "")
            + " |")
    groups = "\n".join(lines)
    lines = ["| Host town | Players | Seed | Newcomers from | Newcomers' cleaning: at home, at "
             "the end | Locals' cleaning: before, at the end | Town's apples: before, just after, "
             "at the end |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["move"]
        lines.append(f"| {m.host} | {m.players} | {m.seed} | {m.origin} (seed {m.origin_seed}) "
                     f"| {r['home_new_clean']:.0f}, {r['late']['new_clean']:.0f} | "
                     f"{r['host_local_clean']:.0f}, {r['late']['local_clean']:.0f} | "
                     f"{r['host_apples']:.1f}, {r['early']['apples']:.1f}, "
                     f"{r['late']['apples']:.1f} |")
    return groups, "\n".join(lines)


def report_chart(rows, out):
    """Three panels over steps since the move: the newcomers' cleaning, the locals', and
    the town's apples; one line per kind of move (mean, band: the middle half of its runs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import report as town_report   # sims/town/report.py: palette and direct labels
    P = town_report
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2,
                         "xtick.color": P.MUTED, "ytick.color": P.MUTED, "text.color": P.INK})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=150, facecolor=P.SURFACE, sharex=True)
    panels = [("new_clean", "The newcomers' cleaning", "dirt cleaned per player per 1,000 steps"),
              ("local_clean", "The locals' cleaning", "dirt cleaned per player per 1,000 steps"),
              ("apples", "The town's apples", "apples eaten per 1,000 steps, all 7")]
    ends = [[] for _ in axes]
    for i, k in enumerate(KINDS[:4]):        # the isolating controls stay flat: table only
        g = [r for r in rows if r["kind"] == k]
        if not g:
            continue
        colour = P.SERIES[i]
        n = min(len(r["x"]) for r in g)
        x = g[0]["x"][:n] / 1e6
        for p, (key, _, _) in enumerate(panels):
            ys = np.array([r["series"][key][:n] for r in g], float)
            ok = np.isfinite(ys).all(0)
            mean = ys[:, ok].mean(0)
            lo, hi = np.percentile(ys[:, ok], [25, 75], axis=0)
            axes[p].fill_between(x[ok], lo, hi, color=colour, alpha=0.15, linewidth=0)
            axes[p].plot(x[ok], mean, color=colour, linewidth=1.5, label=f"{k} ({len(g)} runs)",
                         solid_capstyle="round")
            ends[p].append((x[ok][-1], mean[-1], k))
    # Before the move: where the locals of each kind of town stood.
    ordinary = [r["host_local_clean"] for r in rows if r["move"].host != "isolating"]
    isolating = [r["host_local_clean"] for r in rows if r["move"].host == "isolating"]
    for ax, (key, name, ylabel) in zip(axes, panels):
        ax.set_facecolor(P.SURFACE)
        ax.set_title(name, loc="left", fontsize=11, color=P.INK, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("game steps since the move (millions)")
        ax.grid(True, color=P.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if key != "apples":   # dotted: the locals before the move, named in the first panel
            for levels, text in ((ordinary, "ordinary towns' locals before the move"),
                                 (isolating, "isolating town's locals before the move")):
                if not levels:
                    continue
                level = np.mean(levels)
                ax.axhline(level, color=P.MUTED, linewidth=1, linestyle=(0, (2, 2)))
                if key == "new_clean":
                    ax.text(0.2, level + 1, text, fontsize=8, color=P.MUTED, va="bottom")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.1)
    for ax, panel_ends in zip(axes, ends):
        P.direct_labels(fig, ax, panel_ends, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Newcomers: 3 of 7 players move to another town (20M steps after the move)",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=P.INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.94), frameon=False,
               ncol=2, fontsize=9, labelcolor=P.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, facecolor=P.SURFACE)
    plt.close(fig)


def report(out_root, out, moves=None):
    rows = summarise(out_root, moves)
    groups, per_move = report_tables(rows)
    report_chart(rows, out)
    text = (f"Per 1,000 game steps: cleaning per player, apples for all 7 together. Before:"
            f" the last {LATE:.0%} of the runs the players came from; just after: the "
            f"first {EARLY:.0%} of the steps after the move; at the end: the last {LATE:.0%}.\n\n"
            f"{groups}\n\n{per_move}\n")
    with open(os.path.splitext(out)[0] + ".md", "w") as f:
        f.write(text)
    print(groups)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="make the run folders from finished runs")
    m.add_argument("--sources", nargs="+", required=True,
                   help="folders of finished runs (<town>_<players>_s<seed>)")
    t = sub.add_parser("train", help="train every run folder, a batch at a time")
    t.add_argument("--batch", type=int, default=8, help="runs per process (they share the GPU)")
    t.add_argument("--ckpt-minutes", type=float, default=10.0)
    t.add_argument("--dry-run", action="store_true", help="print the commands only")
    r = sub.add_parser("report", help="chart + tables of what the newcomers and locals did")
    r.add_argument("--out", required=True, help="PNG path; a .md of tables is written beside it")
    for p in (m, t, r):
        p.add_argument("--out-root", default=os.path.join("out", "jaxtown", "newcomers"))
    for p in (m, t):
        p.add_argument("--steps", type=float, default=STEPS, help="steps after the move")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        make(args.sources, args.out_root, steps=args.steps)
    elif args.cmd == "train":
        train(args.out_root, args.steps, args.batch, ckpt_minutes=args.ckpt_minutes,
              dry_run=args.dry_run)
    else:
        report(args.out_root, args.out)


if __name__ == "__main__":
    main()
