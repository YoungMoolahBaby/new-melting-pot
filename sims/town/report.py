"""Charts and a summary for training runs made by train.py.

    ~/venvs/meltingpot/bin/python sims/town/report.py out/town/check1/neighbours_selfish_s0 \
        out/town/check1/neighbours_fair_s0 --out out/town/check1/check.png

Three panels, one per measure, each run a line (game steps on the x axis):
    river cleaning   dirt cleaned per game step, all 7 players together
    apples           apples eaten per game step, all 7 players together
    return per game  everyone's reward over a whole game (rolling mean)
and a table of the last stretch of each run, printed and saved as <out>.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

# Reference palette (dataviz skill, slots 1-2 then the rest in order), light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")


def load(run_dir):
    with open(os.path.join(run_dir, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    metrics = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
               for k in ("env_steps", "clean_per_step", "apples_per_step", "feed_per_step")}
    episodes = []
    path = os.path.join(run_dir, "episodes.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            episodes = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    return metrics, episodes, config


def rolling(x, n):
    if len(x) == 0:
        return x
    n = max(1, min(n, len(x)))
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    out = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, np.nan), out])


def label_for(config):
    who = {"selfish": "selfish players", "fair": "fair-minded players"}.get(
        config["players"], config["players"])
    game = " (JAX)" if config.get("game") == "jaxtown" else ""
    return f"{config['town']}, {who}{game}"


def summary(metrics, episodes, frac=0.1):
    steps = metrics["env_steps"]
    if len(steps) == 0:
        return {}
    tail = steps >= steps[-1] * (1 - frac)
    eps = [e for e in episodes if e["env_steps"] >= steps[-1] * (1 - frac)] or episodes[-48:]
    pick = lambda k: float(np.mean([e[k] for e in eps])) if eps else float("nan")  # noqa: E731
    return {
        "steps": int(steps[-1]),
        "clean_per_step": float(np.nanmean(metrics["clean_per_step"][tail])),
        "apples_per_step": float(np.nanmean(metrics["apples_per_step"][tail])),
        "games": len(eps),
        "return_per_game": pick("collective_return"),
        "cleaned_per_game": pick("cleaned_total"),
        "zaps_per_game": pick("zaps_total"),
        # A game where nobody ate isn't "equal"; leave it out of equality.
        "equality": (float(np.mean([e["equality"] for e in eps if e["collective_return"] > 0]))
                     if any(e["collective_return"] > 0 for e in eps) else float("nan")),
        "top2_clean_share": (float(np.nanmean([e["top2_clean_share"] for e in eps]))
                             if eps else float("nan")),
    }


def run_series(metrics, episodes):
    """The three panels' (x in millions of steps, y) for one run."""
    x = metrics["env_steps"] / 1e6
    window = max(1, len(x) // 40)
    out = [(x, rolling(metrics["clean_per_step"], window)),
           (x, rolling(metrics["apples_per_step"], window))]
    if episodes:
        ex = np.array([e["env_steps"] for e in episodes]) / 1e6
        out.append((ex, rolling([e["collective_return"] for e in episodes], 48)))
    else:
        out.append((np.array([]), np.array([])))
    return out


def combine(series):
    """Several runs' (x, y) -> common x, mean y, and the lowest and highest run."""
    valid = [(x[np.isfinite(y)], y[np.isfinite(y)]) for x, y in series if np.isfinite(y).any()]
    if len(valid) <= 1:
        x, y = valid[0] if valid else (np.array([]), np.array([]))
        return x, y, None, None
    lo, hi = max(x[0] for x, _ in valid), min(x[-1] for x, _ in valid)
    grid = np.linspace(lo, hi, 200)
    ys = np.array([np.interp(grid, x, y) for x, y in valid])
    return grid, ys.mean(0), ys.min(0), ys.max(0)


def chart(groups, out, title):
    """groups: [(label, short label, [runs])]; several runs in a group are drawn as
    their mean with a band from the lowest to the highest run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=150, facecolor=SURFACE, sharex=True)
    panels = [("River cleaning", "dirt cleaned per game step"),
              ("Apples", "apples eaten per game step"),
              ("Everyone's reward per game", "apples (+ feed) per game, all 7")]
    for ax, (name, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("game steps (millions)")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for i, (lab, short, runs) in enumerate(groups):
        colour = SERIES[i % len(SERIES)]
        per_run = [run_series(m, e) for m, e, _ in runs]
        for p, ax in enumerate(axes):
            sx, sy, ylo, yhi = combine([s[p] for s in per_run])
            if ylo is not None:
                ax.fill_between(sx, ylo, yhi, color=colour, alpha=0.15, linewidth=0)
            ax.plot(sx, sy, color=colour, linewidth=1.5, label=lab, solid_capstyle="round")
            ok = np.isfinite(sy)
            if ok.any():  # direct label at the line's end, in text ink
                ax.annotate(short, (sx[ok][-1], sy[ok][-1]), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK2)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.margins(x=0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.94), frameon=False,
               ncol=min(len(groups), 4), fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True, help="PNG path; a .md table is written beside it")
    ap.add_argument("--title", default="Clean Up: how the players learn")
    ap.add_argument("--group", action="store_true",
                    help="draw runs of the same town, players and game (seeds) as one mean "
                         "line with a band from the lowest to the highest run")
    args = ap.parse_args(argv)
    runs = [load(r) for r in args.runs]
    # Settings that make two runs different experiments (seeds aside). Any that differ
    # between the runs given here are added to the labels.
    setting = {"river": "melting_pot", "alpha": None, "beta": None, "feed_reward": None,
               "feed_period": None, "badge_frames": None}
    get = lambda cfg, k: cfg.get(k, setting[k])  # noqa: E731
    varying = [k for k in setting if len({str(get(cfg, k)) for _, _, cfg in runs}) > 1]
    def run_label(cfg):
        return label_for(cfg) + "".join(f", {k.replace('_', ' ')} {get(cfg, k)}"
                                        for k in varying)

    groups = {}
    for i, run in enumerate(runs):
        cfg = run[2]
        lab = run_label(cfg)
        if args.group:   # same experiment, any seed
            key = (cfg.get("game"), cfg["town"], cfg["players"],
                   *(str(get(cfg, k)) for k in setting))
        else:
            key, lab = i, f"{lab}, seed {cfg.get('seed', 0)}"
        short = cfg["players"] + (" JAX" if cfg.get("game") == "jaxtown" else "")
        groups.setdefault(key, (lab, short, []))[2].append(run)
    groups = [(lab + (f" ({len(rs)} seeds)" if len(rs) > 1 else ""), short, rs)
              for lab, short, rs in groups.values()]
    chart(groups, args.out, args.title)
    lines = ["| run | steps | dirt cleaned / step | apples / step | reward / game | "
             "cleaned / game | top-2 share of cleaning | equality | zaps / game | games |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, s):
        return (f"| {name} | {s['steps']:,} | {s['clean_per_step']:.3f} | "
                f"{s['apples_per_step']:.3f} | {s['return_per_game']:.1f} | "
                f"{s['cleaned_per_game']:.0f} | {s['top2_clean_share']:.2f} | "
                f"{s['equality']:.2f} | {s['zaps_per_game']:.1f} | {s['games']} |")

    for lab, _, rs in groups:
        pairs = [(cfg, s) for (m, e, cfg), s in zip(rs, (summary(m, e) for m, e, _ in rs))
                 if s]   # a run with no updates yet has no summary: skip it, not its neighbour
        sums = [s for _, s in pairs]
        for cfg, s in pairs:
            lines.append(row(f"{run_label(cfg)}, seed {cfg.get('seed', 0)}", s))
        if len(sums) > 1:
            def nanmean(xs):   # all blank (e.g. equality when nobody ate): blank, no warning
                xs = np.asarray(xs, float)
                return float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")
            mean = {k: (nanmean([s[k] for s in sums]) if k != "steps"
                        else int(np.mean([s[k] for s in sums]))) for k in sums[0]}
            mean["games"] = int(sum(s["games"] for s in sums))
            lines.append(row(f"**{lab}, mean**", mean))
    table = "\n".join(lines)
    print(table)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("Last 10% of each run's steps (games finished in that stretch).\n\n" + table + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
