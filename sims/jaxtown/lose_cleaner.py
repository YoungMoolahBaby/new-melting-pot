"""Losing one cleaner (experiment 2, part 3): a co-op town's top cleaner leaves, one player
from another co-op town takes its seat, and everyone carries on learning for 20M steps. Does
a 25% town, where two players share the cleaning, survive losing one? Was part 2's collapse
about losing the cleaner, or about who came in (there, players from towns where cleaning
didn't pay)? Plan and write-up: docs/experiments/02c-losing-a-cleaner.md.

The towns are experiment 2's 20M-step runs, the connected town with selfish players: the 25%
and 10% co-op, seeds 0 and 1 (out/jaxtown/coop_pilot), and the 10% co-op, seeds 2 and 3
(out/jaxtown/coop_round2). Each town's top cleaner (most cleaning over the run's last 10% of
games before the move) leaves. In its seat:

    25%, an eater comes in      25% s0 <- 25% s1's least cleaner, 25% s1 <- 25% s0's
    25%, control                25% s0 <- 25% s1's top cleaner,   25% s1 <- 25% s0's
    10%, an eater comes in      10% s0 <-> s1 and s2 <-> s3, each the other's least cleaner

("Least": ties go to the higher seat, as cleaning_rank orders ties lower seat first.) As in
part 2, a newcomer in another seat than its home one has its view relabelled for its new seat
(newcomers.reseat): exact, and it changes nothing it does. 8 runs, one batch.

    cd /mnt/c/dev/new-melting-pot
    ~/venvs/meltingpot/bin/python sims/jaxtown/lose_cleaner.py make \\
        --sources out/jaxtown/coop_pilot out/jaxtown/coop_round2 --out-root out/jaxtown/lose_cleaner
    ~/venvs/meltingpot/bin/python sims/jaxtown/lose_cleaner.py train --out-root out/jaxtown/lose_cleaner
    ~/venvs/meltingpot/bin/python sims/jaxtown/lose_cleaner.py report \\
        --out-root out/jaxtown/lose_cleaner --out out/jaxtown/writeup/lose-cleaner.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NamedTuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coop_newcomers as cn  # noqa: E402
import coop_report as cr  # noqa: E402
import newcomers as nc  # noqa: E402

TOWN, PLAYERS = cn.TOWN, cn.PLAYERS
STEPS = nc.STEPS
LATE, EARLY, BIN = nc.LATE, nc.EARLY, nc.BIN
HOLDS = cr.TAKE_OFF     # apples per 1,000 steps at the end: the town held (decided in the plan)
KINDS = ("25%, an eater comes in", "25%, control: a cleaner comes in",
         "10%, an eater comes in")


class Move(NamedTuple):
    share: float        # the co-op share, the same in both towns
    seed: int           # the host run's seed
    origin_seed: int    # the newcomer's run's seed
    who: str            # who comes in: "eater" (its town's least cleaner) or "cleaner" (top)

    @staticmethod
    def run(share, seed):
        return f"{TOWN}_{PLAYERS}_s{seed}_coop{share * 100:g}"

    @property
    def host_run(self):
        return self.run(self.share, self.seed)

    @property
    def origin_run(self):
        return self.run(self.share, self.origin_seed)

    @property
    def name(self):
        return f"coop{self.share * 100:g}_s{self.seed}_from_s{self.origin_seed}_{self.who}"

    @property
    def kind(self):
        if self.share == 0.1:
            return KINDS[2]
        return KINDS[1] if self.who == "cleaner" else KINDS[0]


def plan():
    """The 8 moves, in KINDS order."""
    moves = [Move(0.25, a, b, who) for who in ("eater", "cleaner") for a, b in ((0, 1), (1, 0))]
    return moves + [Move(0.1, a, b, "eater") for a, b in ((0, 1), (1, 0), (2, 3), (3, 2))]


def make(sources, out_root, moves=None, steps=STEPS, log=print):
    os.makedirs(out_root, exist_ok=True)
    made = 0
    for m in moves or plan():
        out = os.path.join(out_root, m.name)
        if os.path.exists(out):
            continue
        host_dir = nc.find_run(sources, m.host_run)
        origin_dir = nc.find_run(sources, m.origin_run)
        before = nc.load(host_dir)[0]["env_steps"]
        host_rank = cn.cleaning_rank(host_dir, before)
        origin_rank = cn.cleaning_rank(origin_dir, before)
        slot = host_rank[0]
        origin_slot = origin_rank[0] if m.who == "cleaner" else origin_rank[-1]
        info = {"experiment": "losing a cleaner", "kind": m.kind, "who": m.who,
                "seed": m.seed, "origin_seed": m.origin_seed,
                "newcomer_slots": [slot], "origin_slots": [origin_slot],
                "host_rank": host_rank, "origin_rank": origin_rank}
        nc.write_folder(out, host_dir, origin_dir, info, out_root, steps, [slot], [origin_slot])
        log(f"{m.name}: seat {slot} <- {origin_slot}")
        made += 1
    log(f"made {made} run folders in {out_root}")


# ---- Report ---------------------------------------------------------------------------------

def clean(eps, seats):
    """Dirt cleaned by the players in `seats` together, per 1,000 game steps."""
    return nc.per_1000(eps, "cleaned", seats)


def cleaners(eps):
    """How many players did the cleaning (at least a tenth of it each), and what each of them
    got from the co-op and ate, per 1,000 game steps (nan without cleaners)."""
    m = cr.measure(eps) if eps else None
    if m is None:
        return np.nan, np.nan, np.nan
    return m["cleaners"], m["cleaners_got"] - m["cleaners_ate"], m["cleaners_ate"]


def summarise(out_root, moves=None):
    """One row per move: before it (the host's run and the newcomer's own run, each over its
    last 10%), just after it (first 10% since) and at the end (last 10%), and a series over
    the steps since the move. The newcomer is measured in its seat; "the other cleaner" is the
    host's second cleaner (25% towns only); the locals are the six who stayed, together."""
    cache = {}

    def last(run_dir, steps):
        if (run_dir, steps) not in cache:
            cache[run_dir, steps] = nc.games(run_dir, (1 - LATE) * steps)
        return cache[run_dir, steps]

    rows = []
    for m in moves or plan():
        folder = os.path.join(out_root, m.name)
        with open(os.path.join(folder, "config.json")) as f:
            info = json.load(f)["newcomers"]
        new, home = info["newcomer_slots"], info["origin_slots"]
        local = [s for s in range(7) if s not in new]
        other = [info["host_rank"][1]] if m.share != 0.1 else None
        host = last(info["host_run"], info["steps_before"])
        eps = nc.games(folder)
        if not eps:
            continue
        done = max(e["env_steps"] for e in eps)
        early = [e for e in eps if e["env_steps"] <= EARLY * done]
        late = [e for e in eps if e["env_steps"] >= (1 - LATE) * done]
        edges = np.arange(0, done + BIN, BIN)
        idx = np.searchsorted(edges, [e["env_steps"] for e in eps], side="left")
        bins = [[e for e, i in zip(eps, idx) if i == b] for b in range(1, len(edges))]
        three = lambda f: (f(host), f(early), f(late))  # noqa: E731
        apples = three(cn.apples)
        rows.append({
            "move": m, "steps": done,
            "apples": apples, "held": apples[2] >= HOLDS,
            "new_home": clean(last(info["newcomers_run"], info["steps_before"]), home),
            "new_clean": (clean(early, new), clean(late, new)),
            "left_clean": clean(host, new),
            "other_clean": three(lambda e: clean(e, other)) if other else None,
            "local_clean": three(lambda e: clean(e, local)),
            "cleaners_before": cleaners(host), "cleaners_end": cleaners(late),
            "x": edges[1:] - BIN / 2,
            "series": {"apples": [cn.apples(b) for b in bins],
                       "new_clean": [clean(b, new) for b in bins],
                       "local_clean": [clean(b, local) for b in bins]}})
    return rows


def _both(vals, fmt="{:,.0f}"):
    """Mean of the runs, then each run in brackets."""
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.mean(vals))} ({', '.join(fmt.format(v) for v in vals)})"


def report_tables(rows):
    """Three markdown tables: the town and its players by kind of move, the cleaners and
    their pay by kind, and every run."""
    groups = [(k, [r for r in rows if r["move"].kind == k]) for k in KINDS]
    groups = [(k, g) for k, g in groups if g]

    def col(g, get, fmt="{:,.0f}"):
        return _both([get(r) for r in g], fmt)

    def three(g, key):
        return "; ".join(col(g, lambda r, i=i: r[key][i]) for i in range(3))

    a = ["| Move | Town's apples: before; just after; at the end | Held (runs) | "
         "Newcomer's cleaning: at home; just after; at the end | Locals' cleaning, the six "
         "together: before; just after; at the end |", "|---|---|---|---|---|"]
    b = ["| Move | The cleaner who left, before | The other cleaner: before; just after; at "
         "the end | Players who did the cleaning: before; at the end | Each of them at the "
         "end: paid by the co-op; apples eaten |", "|---|---|---|---|---|"]
    for k, g in groups:
        held = f"{sum(r['held'] for r in g)} of {len(g)}"
        a.append(f"| {k} | {three(g, 'apples')} | {held} | {col(g, lambda r: r['new_home'])}; "
                 f"{col(g, lambda r: r['new_clean'][0])}; {col(g, lambda r: r['new_clean'][1])}"
                 f" | {three(g, 'local_clean')} |")
        other = (three(g, "other_clean") if all(r["other_clean"] for r in g) else "-")
        b.append(f"| {k} | {col(g, lambda r: r['left_clean'])} | {other} | "
                 f"{col(g, lambda r: r['cleaners_before'][0], '{:.0f}')}; "
                 f"{col(g, lambda r: r['cleaners_end'][0], '{:.0f}')} | "
                 f"{col(g, lambda r: r['cleaners_end'][1])}; "
                 f"{col(g, lambda r: r['cleaners_end'][2])} |")
    c = ["| Run | Steps since the move | Town's apples: before, just after, at the end | Held | "
         "Newcomer's cleaning: at home, at the end | Locals' cleaning: before, at the end | "
         "Players who did the cleaning at the end |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ap, lc, n = r["apples"], r["local_clean"], r["cleaners_end"][0]
        c.append(f"| {r['move'].name} | {r['steps'] / 1e6:.1f}M | {ap[0]:,.0f}, {ap[1]:,.0f}, "
                 f"{ap[2]:,.0f} | {'yes' if r['held'] else 'no'} | {r['new_home']:,.0f}, "
                 f"{r['new_clean'][1]:,.0f} | {lc[0]:,.0f}, {lc[2]:,.0f} | "
                 f"{'-' if not np.isfinite(n) else f'{n:.0f}'} |")
    return "\n".join(a), "\n".join(b), "\n".join(c)


def report_chart(rows, out):
    """Three panels over steps since the move: the town's apples (with the line for holding),
    the newcomer's cleaning, the six locals' cleaning together. One line per kind of move
    (mean of its runs; band: lowest run to highest)."""
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import report as P

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2,
                         "xtick.color": P.MUTED, "ytick.color": P.MUTED, "text.color": P.INK})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150, facecolor=P.SURFACE,
                             sharex=True)
    panels = (("apples", "Town's apples", "apples per 1,000 steps (all 7)"),
              ("new_clean", "The newcomer's cleaning", "dirt per 1,000 steps"),
              ("local_clean", "The six locals' cleaning, together", "dirt per 1,000 steps"))
    for j, (key, title, ylabel) in enumerate(panels):
        ax = axes[j]
        for c, k in enumerate(KINDS):
            g = [r for r in rows if r["move"].kind == k]
            if not g:
                continue
            n = min(len(r["x"]) for r in g)
            ys = np.array([r["series"][key][:n] for r in g], float)
            x = g[0]["x"][:n] / 1e6
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi, mean = np.nanmin(ys, 0), np.nanmax(ys, 0), np.nanmean(ys, 0)
            ax.fill_between(x, lo, hi, color=P.SERIES[c], alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=P.SERIES[c], linewidth=1.5, label=k, solid_capstyle="round")
        if key == "apples":
            ax.axhline(HOLDS, color=P.MUTED, linewidth=0.8, linestyle="--")
            ax.annotate("holds: 1,000 at the end", (0, HOLDS), xytext=(2, 3),
                        textcoords="offset points", fontsize=8, color=P.MUTED)
        ax.set_facecolor(P.SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=P.INK, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("game steps since the move (millions)")
        ax.grid(True, color=P.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Losing one cleaner: line, mean of the runs; band, lowest run to highest",
                 x=0.01, y=0.99, ha="left", fontsize=12, color=P.INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False,
               ncol=3, fontsize=9, labelcolor=P.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, facecolor=P.SURFACE)
    plt.close(fig)


def report(out_root, out, moves=None):
    rows = summarise(out_root, moves)
    a, b, c = report_tables(rows)
    text = (f"**By kind of move** (per 1,000 game steps; mean of the runs, each run in "
            f"brackets)\n\n{a}\n\n**The cleaners**\n\n{b}\n\n**Every move**\n\n{c}\n")
    print(text)
    report_chart(rows, out)
    with open(os.path.splitext(out)[0] + ".md", "w") as f:
        f.write(text)
    print(f"chart: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="make the 8 run folders")
    mk.add_argument("--sources", nargs="+", required=True)
    mk.add_argument("--out-root", required=True)
    tr = sub.add_parser("train", help="train them, one batch of 8 (safe to re-run)")
    tr.add_argument("--out-root", required=True)
    tr.add_argument("--dry-run", action="store_true")
    rp = sub.add_parser("report", help="tables (printed, and <out>.md) and a chart")
    rp.add_argument("--out-root", required=True)
    rp.add_argument("--out", required=True, help="the chart (.png)")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        make(args.sources, args.out_root)
    elif args.cmd == "train":
        nc.train(args.out_root, STEPS, moves=plan(), dry_run=args.dry_run)
    else:
        report(args.out_root, args.out)


if __name__ == "__main__":
    main()
