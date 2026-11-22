# Experiment 2, part 3 (planned): losing one cleaner

**Planned, not run (2026-11-22).** Nothing here has been built or trained yet. When it has, this
page becomes the write-up.

## The question

In [part 2](02b-coop-newcomers.md) a 10% co-op town rested on one player, who did all the
cleaning. When that player left with two of the town's eaters, and three players from a town
without the co-op took their seats, the clean river was gone from the first games and didn't come
back in 20M steps. Two questions follow:

- **Does a town with two cleaners survive losing one?** At 25%, two players shared the cleaning
  ([part 1](02-coop.md#who-does-the-work-who-earns)).
- **Was part 2's collapse about losing the cleaner, or about who came in?** There, the
  newcomers brought habits from towns where cleaning didn't pay. Here, the cleaner's seat goes
  to an eater from another co-op town.

## The set-up

**The towns** are part 1's 20M-step runs: the connected town, selfish players, the river starting
41% dirty (`hughes`). Over their last 10% of steps (part 1's runs, per 1,000 steps):

- **25% co-op, seed 0:** two cleaners, 260 and 240 dirt, and neither ate. 1,996 apples.
- **25% co-op, seed 1:** two cleaners, 349 and 133 dirt; the second also ate 157 apples.
  1,721 apples.
- **10% co-op, seeds 0-3:** one cleaner each, about 500 dirt (what arrives), who ate nothing.
  2,016-2,401 apples.

**The move.** One player moves: the town's top cleaner (most cleaning over the last 10% before
the move) leaves, and someone takes its seat. Then everyone keeps learning for 20M steps.

| Runs | Town | Who leaves | Who takes its seat |
|---|---|---|---|
| 2 | 25% co-op, seeds 0 and 1 | its top cleaner | the player who cleaned least in the other 25% town |
| 2 | 25% co-op, seeds 0 and 1 (control) | its top cleaner | the other 25% town's top cleaner |
| 4 | 10% co-op, seeds 0-3 | its one cleaner | the player who cleaned least in another 10% town |

The pairs are seed 0 with seed 1, and (at 10%) seed 2 with seed 3, each way round. A newcomer in
a different seat from its home one has its view relabelled, as in part 2. That's exact, and the
game and the learning are unchanged.

## What we'll measure

Per 1,000 game steps, before the move (its last 10%), just after it (first 2M steps) and at the
end (last 2M), and over time:

- the town's apples;
- at 25%, the cleaner who stayed: does it take on the other's share?
- the newcomer's cleaning, and the other locals';
- how many players did the cleaning at the end (at least a tenth of it each), and what the co-op
  paid them.

**What counts as holding:** the town eats at least 1,000 apples per 1,000 steps over the last 2M
steps, part 1's line for a town that took off. Decided now, before the runs.

- If the 25% towns hold and the 10% towns don't, a second cleaner gives a town slack.
- If the 10% towns hold with a co-op eater in the cleaner's seat, part 2's collapse was about
  who came in. If they don't, it was about losing the cleaner.

## Cost

8 runs in one batch, about 1¾ hours on the RTX 4070 Ti (part 2's batches of 8 took 1¾ hours
each). The code is part 2's `sims/jaxtown/coop_newcomers.py`, extended to move one player, with
tests on the CPU first. Before training, check with the other sessions that use the GPU.

## Not in this round

- **The co-op ending in place:** the same town losing its co-op.
- **Longer runs:** whether part 2's town that lost its cleaner recovers after 20M steps.
- **Newcomers from towns without the co-op:** part 2 did that.
