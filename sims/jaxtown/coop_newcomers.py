"""Newcomers in a co-op town (experiment 2, part 2): 3 of a town's 7 players move between
a town with the 10% co-op and one without it, and everyone carries on learning for 20M steps.
Do newcomers who never had the co-op take up cleaning, or ride free and break it? Do co-op
players keep cleaning in a town where it no longer pays?

The towns are experiment 2's 20M-step runs, the connected town with selfish players: with
the co-op, seeds 0 and 1 (out/jaxtown/coop_pilot); without it, seeds 2 and 3
(out/jaxtown/coop_round2). Newcomers always come from another seed (see newcomers.py).

    into the co-op       co-op s0 <- none s2, co-op s1 <- none s3
    out of the co-op     none s2 <- co-op s0, none s3 <- co-op s1
    control, co-op       co-op s0 <- co-op s1, co-op s1 <- co-op s0
    control, no co-op    none s2 <- none s3, none s3 <- none s2

In a co-op town one player does all the cleaning, so who moves matters. Each move runs twice:
    eaters      the 3 players who cleaned least move, into the seats of the host's 3 who
                cleaned least (the cleaners stay on both sides);
    cleaners    the 3 who cleaned most move, into the seats of the host's 3 who cleaned most
                (from or into a co-op town: its one cleaner and two of its eaters).
"Cleaned most" is over each run's last 10% of games before the move; players are paired by
rank (the top cleaner takes the top cleaner's seat). A newcomer can land in another seat than
it had at home, so its view is relabelled for its new seat (newcomers.reseat): exact, and it
changes nothing it does. 16 runs.

    cd /mnt/c/dev/new-melting-pot
    ~/venvs/meltingpot/bin/python sims/jaxtown/coop_newcomers.py make \\
        --sources out/jaxtown/coop_pilot out/jaxtown/coop_round2 --out-root out/jaxtown/coop_newcomers
    ~/venvs/meltingpot/bin/python sims/jaxtown/coop_newcomers.py train --out-root out/jaxtown/coop_newcomers
    ~/venvs/meltingpot/bin/python sims/jaxtown/coop_newcomers.py report \\
        --out-root out/jaxtown/coop_newcomers --out out/jaxtown/writeup/coop-newcomers.png
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
import newcomers as nc  # noqa: E402

TOWN, PLAYERS, SHARE = "connected", "selfish", 0.1
COOP_SEEDS, NONE_SEEDS = (0, 1), (2, 3)
VARIANTS = ("eaters", "cleaners")
KINDS = ("into the co-op", "out of the co-op", "control, co-op", "control, no co-op")
STEPS = nc.STEPS
LATE = nc.LATE


class Move(NamedTuple):
    host_coop: bool     # does the host town have the co-op
    seed: int           # the host run's seed
    origin_coop: bool   # did the newcomers' town
    origin_seed: int    # the newcomers' run's seed
    variant: str        # who moves: "eaters" or "cleaners"

    @staticmethod
    def run(coop, seed):
        return f"{TOWN}_{PLAYERS}_s{seed}" + (f"_coop{SHARE * 100:g}" if coop else "")

    @property
    def host_run(self):
        return self.run(self.host_coop, self.seed)

    @property
    def origin_run(self):
        return self.run(self.origin_coop, self.origin_seed)

    @property
    def name(self):
        town = lambda coop: "coop" if coop else "none"  # noqa: E731
        return (f"{town(self.host_coop)}_s{self.seed}_from_{town(self.origin_coop)}"
                f"_s{self.origin_seed}_{self.variant}")

    @property
    def kind(self):
        if self.host_coop == self.origin_coop:
            return KINDS[2] if self.host_coop else KINDS[3]
        return KINDS[0] if self.host_coop else KINDS[1]


def plan():
    """The 16 moves, ordered by variant then kind."""
    pairs = []
    for c, n in zip(COOP_SEEDS, NONE_SEEDS):
        pairs += [(True, c, False, n), (False, n, True, c)]
    for a, b in ((COOP_SEEDS, True), (NONE_SEEDS, False)):
        pairs += [(b, a[0], b, a[1]), (b, a[1], b, a[0])]
    moves = [Move(*p, v) for v in VARIANTS for p in pairs]
    return sorted(moves, key=lambda m: (VARIANTS.index(m.variant), KINDS.index(m.kind)))


def cleaning_rank(run_dir, steps_before):
    """Seats ordered from the player who cleaned most to least, over the run's last 10% of
    games (ties: lower seat first)."""
    eps = nc.games(run_dir, (1 - LATE) * steps_before)
    if not eps:
        raise SystemExit(f"{run_dir}: no games in its last 10%")
    total = np.sum([e["cleaned"] for e in eps], axis=0)
    return [int(s) for s in np.argsort(-total, kind="stable")]


def seats(rank, variant):
    """The 3 seats that move (or are taken), in rank order."""
    return rank[:3] if variant == "cleaners" else rank[-3:]


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
        slots = seats(cleaning_rank(host_dir, before), m.variant)
        origin_slots = seats(cleaning_rank(origin_dir, before), m.variant)
        info = {"experiment": "coop newcomers", "kind": m.kind, "variant": m.variant,
                "host_coop_share": SHARE if m.host_coop else 0.0,
                "origin_coop_share": SHARE if m.origin_coop else 0.0,
                "seed": m.seed, "origin_seed": m.origin_seed,
                "newcomer_slots": slots, "origin_slots": origin_slots}
        nc.write_folder(out, host_dir, origin_dir, info, out_root, steps, slots, origin_slots)
        log(f"{m.name}: seats {slots} <- {origin_slots}")
        made += 1
    log(f"made {made} run folders in {out_root}")


# ---- Report ---------------------------------------------------------------------------------

EARLY = nc.EARLY    # "just after the move": the first 10% of the steps after it
BIN = nc.BIN        # chart: steps per point
VARIANT_TEXT = {"eaters": "the three who cleaned least move",
                "cleaners": "the three who cleaned most move"}


def clean(eps, seats):
    """Dirt cleaned per player in `seats`, per 1,000 game steps."""
    return nc.per_1000(eps, "cleaned", seats) / len(seats)


def apples(eps):
    return nc.per_1000(eps, "ate", range(7))


def summarise(out_root, moves=None):
    """One row per move: before it (the host's run and the newcomers' own run, each over its
    last 10%), just after it (first 10% since) and at the end (last 10%), and a series over
    the steps since the move. Newcomers are measured in their seats, locals in the others."""
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
        rows.append({
            "move": m, "steps": done,
            "home_clean": clean(last(info["newcomers_run"], info["steps_before"]), home),
            "replaced_clean": clean(host, new), "local_before": clean(host, local),
            "apples_before": apples(host),
            "new_clean": (clean(early, new), clean(late, new)),
            "local_clean": (clean(early, local), clean(late, local)),
            "apples": (apples(early), apples(late)),
            "x": edges[1:] - BIN / 2,
            "series": {"new_clean": [clean(b, new) for b in bins],
                       "local_clean": [clean(b, local) for b in bins],
                       "apples": [apples(b) for b in bins]}})
    return rows


def report_tables(rows):
    def both(vals, fmt="{:.0f}"):
        """Mean of the runs, then each run in brackets."""
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return "-"
        if len(vals) == 1:
            return fmt.format(vals[0])
        return f"{fmt.format(np.mean(vals))} ({', '.join(fmt.format(v) for v in vals)})"

    head = ("| Who moves | Move | Newcomers' cleaning: at home, just after, at the end | "
            "Players they replaced, before | Locals' cleaning: before, at the end | "
            "Town's apples: before, just after, at the end |")
    lines = [head, "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for k in KINDS:
            g = [r for r in rows if r["move"].variant == v and r["move"].kind == k]
            if not g:
                continue
            col = lambda get: [get(r) for r in g]  # noqa: E731
            lines.append(
                f"| {v} | {k} | {both(col(lambda r: r['home_clean']))}; "
                f"{both(col(lambda r: r['new_clean'][0]))}; {both(col(lambda r: r['new_clean'][1]))}"
                f" | {both(col(lambda r: r['replaced_clean']))}"
                f" | {both(col(lambda r: r['local_before']))}; {both(col(lambda r: r['local_clean'][1]))}"
                f" | {both(col(lambda r: r['apples_before']), '{:,.0f}')}; "
                f"{both(col(lambda r: r['apples'][0]), '{:,.0f}')}; "
                f"{both(col(lambda r: r['apples'][1]), '{:,.0f}')} |")
    groups = "\n".join(lines)
    lines = ["| Run | Steps since the move | Newcomers' cleaning: at home, at the end | "
             "Replaced, before | Locals' cleaning: before, at the end | Town's apples: "
             "before, just after, at the end |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['move'].name} | {r['steps'] / 1e6:.1f}M | {r['home_clean']:.0f}, "
                     f"{r['new_clean'][1]:.0f} | {r['replaced_clean']:.0f} | "
                     f"{r['local_before']:.0f}, {r['local_clean'][1]:.0f} | "
                     f"{r['apples_before']:,.0f}, {r['apples'][0]:,.0f}, {r['apples'][1]:,.0f} |")
    return groups, "\n".join(lines)


def report_chart(rows, out):
    """Rows: who moves (eaters, cleaners). Columns: the newcomers' cleaning, the locals', the
    town's apples, over steps since the move. One line per kind of move (mean of its two
    runs; band: one run to the other)."""
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import report as P

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2,
                         "xtick.color": P.MUTED, "ytick.color": P.MUTED, "text.color": P.INK})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), dpi=150, facecolor=P.SURFACE,
                             sharex=True)
    panels = (("new_clean", "Newcomers' cleaning", "dirt per player per 1,000 steps"),
              ("local_clean", "Locals' cleaning", "dirt per player per 1,000 steps"),
              ("apples", "Town's apples", "apples per 1,000 steps (all 7)"))
    for i, v in enumerate(VARIANTS):
        for j, (key, title, ylabel) in enumerate(panels):
            ax = axes[i, j]
            for c, k in enumerate(KINDS):
                g = [r for r in rows if r["move"].variant == v and r["move"].kind == k]
                if not g:
                    continue
                n = min(len(r["x"]) for r in g)
                ys = np.array([r["series"][key][:n] for r in g], float)
                x = g[0]["x"][:n] / 1e6
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lo, hi, mean = np.nanmin(ys, 0), np.nanmax(ys, 0), np.nanmean(ys, 0)
                ax.fill_between(x, lo, hi, color=P.SERIES[c], alpha=0.15, linewidth=0)
                ax.plot(x, mean, color=P.SERIES[c], linewidth=1.5, label=k,
                        solid_capstyle="round")
            ax.set_facecolor(P.SURFACE)
            ax.set_title(f"{title}: {VARIANT_TEXT[v]}" if j == 0 else title, loc="left",
                         fontsize=10, color=P.INK, pad=6)
            ax.set_ylabel(ylabel)
            ax.grid(True, color=P.GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 1:
                ax.set_xlabel("game steps since the move (millions)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Newcomers in a co-op town: line, mean of two runs; band, one run to the "
                 "other", x=0.01, y=0.995, ha="left", fontsize=12, color=P.INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), frameon=False,
               ncol=4, fontsize=9, labelcolor=P.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, facecolor=P.SURFACE)
    plt.close(fig)


def report(out_root, out, moves=None):
    rows = summarise(out_root, moves)
    groups, every = report_tables(rows)
    text = (f"**By kind of move** (per 1,000 game steps; mean of the runs, each run in "
            f"brackets)\n\n{groups}\n\n**Every move**\n\n{every}\n")
    print(text)
    report_chart(rows, out)
    with open(os.path.splitext(out)[0] + ".md", "w") as f:
        f.write(text)
    print(f"chart: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="make the 16 run folders")
    mk.add_argument("--sources", nargs="+", required=True)
    mk.add_argument("--out-root", required=True)
    tr = sub.add_parser("train", help="train them, 8 runs per batch (safe to re-run)")
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
