"""Experiment 2's tables and chart: the co-op runs next to the same towns without it.

    ~/venvs/meltingpot/bin/python sims/jaxtown/coop_report.py --runs out/jaxtown/coop_pilot \
        out/jaxtown/coop_round2 out/jaxtown/towns out/colab/towns out/jaxtown/dirt \
        --out out/jaxtown/writeup/coop-20m.png

Finds every JAX run in the --runs folders (a folder holding config.json and episodes.jsonl)
that started from the hughes river under Melting Pot's dirt rule, keys it by town, players,
co-op share and seed, and looks at its first 20M steps (runs without the co-op that went on to
100M are cut at 20M). Per 1,000 game steps: apples eaten by all 7,
dirt cleaned per player, the reward the co-op moved (what the players it paid gained), how
many players did the cleaning (at least a tenth of it each), and what they and the others each
earned and ate. Prints the tables (markdown) and saves them next to the chart as <out>.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))
import report as P  # noqa: E402  (the palette and chart style of the other charts)

STEPS = 20e6
END = (18e6, 20e6)          # the last 10% of 20M steps
TAKE_OFF = 1000             # apples per 1,000 steps: "the town took off"
CLEANER = 0.1               # a cleaner did at least this share of the town's cleaning
SHARES = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
# The 10% co-op in other settings: (label, town, players, seeds).
ELSEWHERE = (("connected town, selfish, seeds 0-1", "connected", "selfish", (0, 1)),
             ("connected town, selfish, new seeds 2-3", "connected", "selfish", (2, 3)),
             ("connected town, fair-minded", "connected", "fair", (0, 1)),
             ("neighbours town (no badge), selfish", "neighbours", "selfish", (0, 1)))


def find_runs(roots):
    runs = {}
    for root in roots:
        for cfg_path in sorted(glob.glob(os.path.join(root, "*", "config.json"))):
            d = os.path.dirname(cfg_path)
            if not os.path.exists(os.path.join(d, "episodes.jsonl")):
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            # Only runs like these: our JAX game, the hughes river start, Melting Pot's dirt.
            if (cfg.get("game") != "jaxtown" or cfg.get("river") != "hughes"
                    or cfg.get("dirt", "melting_pot") != "melting_pot"):
                continue
            key = (cfg["town"], cfg["players"], float(cfg.get("coop_share", 0.0)), cfg["seed"])
            if key in runs:
                raise SystemExit(f"two runs for {key}: {runs[key]} and {d}")
            runs[key] = d
    return runs


def games(run_dir, upto=STEPS):
    with open(os.path.join(run_dir, "episodes.jsonl")) as f:
        eps = [json.loads(x) for x in f if x.strip()]
    return [e for e in eps if e["env_steps"] <= upto]


def measure(eps):
    """Per 1,000 game steps, over the games in `eps` (None if there are none)."""
    frames = sum(e["length"] for e in eps)
    if not frames:
        return None
    k = 1000 / frames
    out = {"games": len(eps),
           "apples": k * sum(sum(e["ate"]) for e in eps),
           "clean": k * sum(sum(e["cleaned"]) for e in eps) / 7,
           "moved": k * sum(sum(max(r - a - f, 0.0) for r, a, f in
                                zip(e["return"], e["ate"], e["feed"])) for e in eps)}
    # The town's cleaners: the players who did at least a tenth of its cleaning over these
    # games. (Not "each game's top two": where one player does all the cleaning, that pairs
    # the cleaner with an eater.)
    cleaned = np.sum([e["cleaned"] for e in eps], axis=0)
    ret = np.sum([e["return"] for e in eps], axis=0)
    ate = np.sum([e["ate"] for e in eps], axis=0)
    who = (cleaned >= CLEANER * cleaned.sum() if cleaned.sum() > 0
           else np.zeros(len(cleaned), bool))
    out["cleaners"] = int(who.sum())
    for name, idx in (("cleaners", who), ("others", ~who)):
        n = idx.sum()
        out[f"{name}_got"] = k * ret[idx].sum() / n if n else np.nan
        out[f"{name}_ate"] = k * ate[idx].sum() / n if n else np.nan
    return out


def window(eps, lo, hi):
    return measure([e for e in eps if lo < e["env_steps"] <= hi])


def took_off(eps, level=TAKE_OFF, bin_steps=1e6):
    """The end of the first 1M-step stretch in which the town ate `level` apples per 1,000
    steps or more (None: never, within the games given)."""
    top = max((e["env_steps"] for e in eps), default=0)
    for hi in np.arange(bin_steps, top + bin_steps, bin_steps):
        m = window(eps, hi - bin_steps, hi)
        if m and m["apples"] >= level:
            return float(hi)
    return None


def series(eps, bin_steps=0.5e6, upto=STEPS):
    """Apples per 1,000 steps in each stretch of `bin_steps`: (x in M steps, y)."""
    xs, ys = [], []
    for hi in np.arange(bin_steps, upto + 1, bin_steps):
        m = window(eps, hi - bin_steps, hi)
        xs.append((hi - bin_steps / 2) / 1e6)
        ys.append(np.nan if m is None else m["apples"])
    return np.array(xs), np.array(ys)


def pct(share):
    return "none" if share == 0 else f"{share * 100:g}%"


def seeds_text(values, fmt="{:,.0f}"):
    """Mean, then each seed in brackets: '2,399 (2,396-2,401)'."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.mean(vals))} ({fmt.format(min(vals))}-{fmt.format(max(vals))})"


def tables(runs):
    lines = []
    end = {k: window(games(d), *END) for k, d in runs.items()}

    lines += ["**How big a share** (connected town, selfish players, seeds 0 and 1; 18M-20M steps)",
              "",
              "| Co-op share | Apples per 1,000 steps (all 7; seeds) | Dirt cleaned per player | "
              "Paid out by the co-op | Players who did the cleaning (seeds) | First reached "
              f"{TAKE_OFF:,} apples (seeds) |",
              "|---|---|---|---|---|---|"]
    for s in SHARES:
        ks = [("connected", "selfish", s, seed) for seed in (0, 1) if
              ("connected", "selfish", s, seed) in runs]
        if not ks:
            continue
        ms = [end[k] for k in ks]
        offs = [took_off(games(runs[k])) for k in ks]
        off = ", ".join("not by 20M" if o is None else f"{o / 1e6:.0f}M" for o in offs)
        lines.append(f"| {pct(s)} | {seeds_text([m['apples'] for m in ms])} | "
                     f"{np.mean([m['clean'] for m in ms]):.0f} | "
                     f"{np.mean([m['moved'] for m in ms]):,.0f} | "
                     f"{', '.join(str(m['cleaners']) for m in ms)} | {off} |")

    sharing = ("neighbours", "shared", 0.0, 0)
    if sharing in runs:
        off = took_off(games(runs[sharing]))
        lines += ["", "For scale: players who share all their apples (neighbours town, seed 0) "
                  f"ate {end[sharing]['apples']:,.0f} apples per 1,000 steps at 18M-20M steps; "
                  f"they first reached {TAKE_OFF:,} "
                  + ("after 20M." if off is None else f"at {off / 1e6:.0f}M.")]

    lines += ["", "**The 10% co-op elsewhere** (18M-20M steps)", "",
              "| Setting | Apples without the co-op (seeds) | Apples with 10% (seeds) | "
              "Dirt cleaned per player, without and with |",
              "|---|---|---|---|"]
    for label, town, players, seeds in ELSEWHERE:
        without = [end.get((town, players, 0.0, s)) for s in seeds]
        with_ = [end.get((town, players, 0.1, s)) for s in seeds]
        if not all(without) or not all(with_):
            continue
        lines.append(f"| {label} | {seeds_text([m['apples'] for m in without])} | "
                     f"{seeds_text([m['apples'] for m in with_])} | "
                     f"{np.mean([m['clean'] for m in without]):.0f}, "
                     f"{np.mean([m['clean'] for m in with_]):.0f} |")

    lines += ["", "**Who earns what** (18M-20M steps; per player, per 1,000 steps; the players "
              "who did at least a tenth of the cleaning against the others)", "",
              "| Setting | Co-op share | Players who did the cleaning (seeds) | Each of them: "
              "reward, apples eaten | Each of the others: reward, apples eaten |",
              "|---|---|---|---|---|"]
    rows = [("connected town, selfish", "connected", "selfish", s, (0, 1)) for s in SHARES]
    rows += [(label, town, players, 0.1, seeds) for label, town, players, seeds in ELSEWHERE[1:]]
    for label, town, players, s, seeds in rows:
        ms = [end.get((town, players, s, seed)) for seed in seeds]
        if not all(ms):
            continue
        avg = {k: np.mean([m[k] for m in ms]) for k in ms[0]}
        lines.append(f"| {label} | {pct(s)} | {', '.join(str(m['cleaners']) for m in ms)} | "
                     f"{avg['cleaners_got']:.0f}, {avg['cleaners_ate']:.0f} | "
                     f"{avg['others_got']:.0f}, {avg['others_ate']:.0f} |")

    lines += ["", "**Every run** (18M-20M steps, per 1,000 steps)", "",
              "| Town | Players | Co-op share | Seed | Apples (all 7) | Dirt cleaned per player | "
              f"Paid out by the co-op | Players who did the cleaning | First reached {TAKE_OFF:,} "
              "apples | Games |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    shown = {(t, p) for _, t, p, _ in ELSEWHERE}
    for k in sorted(k for k in runs if (k[0], k[1]) in shown):
        m, off = end[k], took_off(games(runs[k]))
        if m is None:
            continue
        lines.append(f"| {k[0]} | {k[1]} | {pct(k[2])} | {k[3]} | {m['apples']:,.0f} | "
                     f"{m['clean']:.0f} | {m['moved']:,.0f} | {m['cleaners']} | "
                     f"{'not by 20M' if off is None else f'{off / 1e6:.0f}M'} | {m['games']} |")
    return "\n".join(lines)


def chart(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2,
                         "xtick.color": P.MUTED, "ytick.color": P.MUTED, "text.color": P.INK})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150, facecolor=P.SURFACE, sharey=True)

    def draw(ax, keys, colour, label, ends, width=1.5):
        curves = [series(games(runs[k]))[1] for k in keys if k in runs]
        if not curves:
            return
        x = series([])[0]
        ys = np.vstack(curves)
        with warnings.catch_warnings():     # stretches before any game ended: blank
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi, mean = np.nanmin(ys, 0), np.nanmax(ys, 0), np.nanmean(ys, 0)
        if len(curves) > 1:
            ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=colour, linewidth=width, label=label, solid_capstyle="round")
        ends.append((x[-1], mean[-1], label))

    # Each share keeps one colour; on the right, the setting the left panel shows at 10%
    # keeps the 10% colour, and the other settings take the colours no share uses.
    colours = {s: (P.MUTED if s == 0 else P.SERIES[i - 1]) for i, s in enumerate(SHARES)}
    elsewhere_colours = [colours[0.1], P.SERIES[6], P.SERIES[7], P.SERIES[3]]
    share_ends = []
    for s in SHARES:
        draw(axes[0], [("connected", "selfish", s, seed) for seed in (0, 1)], colours[s],
             "no co-op" if s == 0 else pct(s), share_ends)
    axes[0].set_title("How big a share: connected town, selfish players (seeds 0-1)",
                      loc="left", fontsize=11, color=P.INK, pad=8)
    for (label, town, players, seeds), colour in zip(ELSEWHERE, elsewhere_colours):
        draw(axes[1], [(town, players, 0.1, s) for s in seeds], colour, label, [])
        draw(axes[1], [(town, players, 0.0, s) for s in seeds], colour, None, [], width=0.8)
    axes[1].set_title("The 10% co-op elsewhere (thin lines: the same without it)", loc="left",
                      fontsize=11, color=P.INK, pad=8)

    for ax in axes:
        ax.set_facecolor(P.SURFACE)
        ax.set_xlabel("game steps (millions)")
        ax.grid(True, color=P.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, STEPS / 1e6)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("apples per 1,000 steps (all 7)")
    P.direct_labels(fig, axes[0], share_ends)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=P.INK2, ncol=2)
    axes[1].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=P.INK2)
    fig.suptitle("The co-op: apples eaten over the first 20M steps (line: mean of the seeds; "
                 "band: one seed to the other)", x=0.01, y=0.99, ha="left", fontsize=12,
                 color=P.INK)
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, facecolor=P.SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="folders holding run folders")
    ap.add_argument("--out", required=True, help="the chart (.png); the tables go to <out>.md")
    args = ap.parse_args(argv)
    runs = find_runs(args.runs)
    text = tables(runs)
    print(text)
    chart(runs, args.out)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(text + "\n")
    print(f"\nchart: {args.out}")


if __name__ == "__main__":
    main()
