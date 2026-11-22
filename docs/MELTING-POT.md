# Melting Pot

[DeepMind's Melting Pot](https://github.com/google-deepmind/meltingpot) (`dm-meltingpot` 2.4.1,
Apache 2.0) is a set of small multiplayer games built around social dilemmas, plus pretrained bot
populations to play them with. Each **scenario** puts the agents you're testing in a game with
bots they've never met, so it measures whether they can get along with strangers.

## What it gives us

- 24 games in 49 versions, and about 312 test scenarios.
- Pretrained bots (cooperative, selfish, and others).
- People can play the games themselves, alongside bots (`meltingpot/human_players`).
- We can build our own games (Python settings plus Lua game logic).

What it doesn't have: models of human behaviour, or any game about platforms or companies. Those we
build or bring.

## The games

| Family | Game (versions) | What happens |
|---|---|---|
| Sharing a resource | Commons Harvest (3) | Apples regrow only near other apples. Take the last one and the patch dies |
| | Clean Up | Apples grow only while someone cleans a polluted river, which pays the cleaner nothing |
| | Externality Mushrooms (2) | Some mushrooms reward the eater and cost everyone else |
| | Allelopathic Harvest (2) | Replant berries in your favourite colour; the most common colour grows fastest |
| | Factory of the Commons (2) | Shared factory machines turn raw materials into food |
| | Chemistry (4) | Two self-running reaction cycles stay alive only if players bring their by-products together |
| Teamwork | Collaborative Cooking (8 kitchens) | Overcooked: two cooks make soup, the classic human-AI teamwork test |
| | Coop Mining | Gold only comes out if two miners hit it at the same moment |
| | Boat Race (2) | A boat needs two rowers in sync |
| | Daycare | One player has to help another get food it can't reach alone |
| Trade and trust | Fruit Market (2) | Apple and banana farmers trade so each can eat what they like |
| | Gift Refinements | A refined item turns into three and can be given away |
| | Coins | Grabbing coins of another player's colour hurts that player |
| | "In the Matrix" (7 games, 17 versions) | Classic game theory as rooms: Prisoner's Dilemma, Stag Hunt, Chicken, Bach or Stravinsky, Pure Coordination, Rationalizable Coordination, Running with Scissors |
| Conflict and deception | Territory (4) | Claim ground by painting it; you can break other players' walls |
| | Paintball (2) | Capture the flag, king of the hill |
| | Predator Prey (5) | Prey survive by working together |
| | Hidden Agenda | Crewmates against a hidden impostor, with a vote to throw someone out |

Closest to our questions: Commons Harvest (squeezing until collapse), Clean Up (paying to heal),
Externality Mushrooms (profit that dumps costs on others), Hidden Agenda (a helper secretly working
for someone else), Fruit Market (a real market), Collaborative Cooking and Daycare (people and AI
working together).

## Running it on Windows (WSL2)

- Melting Pot's engine, **dmlab2d**, ships only for Linux and macOS (PyPI wheels; its README says
  the build rules are Linux and macOS only). No native Windows.
- So we run it in **WSL2**. This PC (Ryzen 5 5600, ASUS ROG STRIX B550-F) has virtualization
  turned off, so first: restart, **Del** into the BIOS, **F7** Advanced Mode, **Advanced > CPU
  Configuration > SVM Mode > Enabled**, **F10** to save. Then in an admin PowerShell:
  `wsl --install --no-distribution` and restart. On this PC's Windows build (26300) that command
  only printed "not installed", so on 2026-11-16 we did it in two steps instead:
  `dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`,
  the same for `VirtualMachinePlatform`, then `winget install --id Microsoft.WSL` (3.0.5), then a
  restart.
- **Linux goes on C:.** After the restart: `wsl --install -d Ubuntu-24.04`.
- WSL2 turns on Windows' hypervisor at boot. If games feel different afterwards, the hypervisor
  can be switched off for gaming with `bcdedit /set hypervisorlaunchtype off` plus a restart
  (`auto` to turn it back on).
- Installing Melting Pot downloads about **1.2 GB** (472 MB of game assets and bots, plus
  TensorFlow).
- WSL2 also gives us the RTX 4070 Ti for training, and play windows that open on the Windows
  desktop.

### What's installed (2026-11-16)

- Ubuntu 24.04 on C:, WSL 3.0.5. `wsl --install -d Ubuntu-24.04 --no-launch`, then open it once
  to pick a username and password.
- uv in Ubuntu, and a venv at `~/venvs/meltingpot` (inside Linux, not on `/mnt/c`, because it's
  much faster there): Python 3.12, `dm-meltingpot` 2.4.1, dmlab2d 1.0.2, TensorFlow 2.21. The
  install took under 2 minutes.
- GPU: `tensorflow[and-cuda]==2.21.0` (2.8 GB of NVIDIA libraries). TensorFlow missed
  `libcusolver.so.11` until `LD_LIBRARY_PATH` listed the venv's `nvidia/*/lib` folders; that's now
  set in Ubuntu's `~/.profile`, so always go through a login shell (`bash -l`). It then sees the
  RTX 4070 Ti.
- Calling it from Windows: use `wsl -d Ubuntu-24.04 -e bash -lc '...'`. Without `-e`, Linux's
  shell re-reads the command and breaks on `$PATH` (Windows paths with `(x86)` in them).
- Check that passed: Clean Up with 7 players ran 200 steps in 0.1 s, and scenario `clean_up_0`
  (3 of our players among DeepMind's trained bots) ran 200 steps in 2.3 s. The absl
  "unexpected argument" warnings it prints are noise.
- Clean Up already gives each player `NUM_OTHERS_WHO_CLEANED_THIS_STEP`. Experiment 1 leaves it
  out: it's a count, not who, and "seeing who's been cleaning" is what the towns vary.

### Playing it yourself

`wsl -d Ubuntu-24.04 -e bash -lc '~/venvs/meltingpot/bin/python -m meltingpot.human_players.play_clean_up'`

Keys: **WASD** move, **Q / E** turn, **1** zap, **2** clean the river, **Tab** switch to another
player. The other players stand still. No bots in this mode; playing alongside bots is something
we'd build.

### Training our own players (experiment 1, 2026-11-16)

Code in `sims/town/`: `envs.py` (many games in worker processes), `ppo.py` (each player has its
own network; fairness reward from Hughes et al. 2018), `train.py`, `report.py`. Several runs share
one process:

```
wsl -d Ubuntu-24.04 -e bash -lc 'cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/town/train.py --runs neighbours:selfish neighbours:fair --steps 1e7'
```

Ctrl-C saves checkpoints; `--resume` carries on (it refuses if the settings changed, and a new run
refuses a folder that already holds one). Tests: `~/venvs/meltingpot/bin/python -m pytest
sims/town/tests -q` (18 tests, about 2.5 minutes; the JAX version's 56 are in
`sims/jaxtown/tests`, about half an hour on the CPU, most of it the Colab and Cloud runners'
drills; `pytest sims` runs all 74). The players only see their own view, shrunk to
22×22 (2×2 pixels per map cell), and whether their zap beam is ready. They don't get Clean Up's
`NUM_OTHERS_WHO_CLEANED_THIS_STEP` count, so the only way to know who cleaned is to see it.

`--river` sets how dirty the river starts. Apples only grow while under 40% of it is dirty.

- `melting_pot` (default): Clean Up as shipped, 79 of 147 river cells dirty (54%).
- `hughes`: just past the tipping point, as in Hughes et al. 2018, 60 of 147 (41%).
- `clean`: no dirt and none ever spawns, a sanity check that the players can learn at all.

Seven random players over 1,000 steps eat 0, 18 and 405 apples respectively. The first check run
(2026-11-16, 10M steps each, selfish vs fair, shipped river) barely learned: under 0.01 apples
per step in every update, and both groups alike. On the clean river the same learner went from 0.5 to 2.5 apples per step by 1.2M
steps. So the shipped start is too far past the tipping point for players learning from scratch to
ever find a reward.

Speed on this PC (Ryzen 5 5600, RTX 4070 Ti, WSL), learned the hard way:

- Importing OpenCV (`cv2`) in a process makes the Melting Pot game about 8× slower (1,800 →
  230 steps/s). The game workers never import it.
- Two JAX processes sharing the GPU under WSL take turns so badly that every step gets about
  30× slower. So one process trains several runs, and while one run's games step, it chooses
  the next run's actions.
- `jax.device_put` the inputs before calling a jitted function; passing numpy arrays straight in
  was about 10× slower here. Split random keys inside jit (outside, a split costs 1.5 ms).
- `meltingpot/__init__.py` loads TensorFlow for its bots; the game workers swap in empty
  `meltingpot.bot` / `meltingpot.scenario` modules first (1.0 → 0.7 GB per worker).
- Free-running game processes manage about 7,000 steps/s in total; in lockstep with the learner
  it's about 2,000-3,000 in total, because each step waits for the slowest worker.
- Never step a Clean Up game after its last step (`ts.last()`): the automatic reset crashes in
  Clean Up's dirt tracker. Reset it yourself.

The second check run (2026-11-17, `--river hughes`, 10M steps each) reproduced the paper's
direction: fair players cleaned about twice as much as selfish ones (323 vs 159 dirt per game).
It didn't reproduce the payoff: the river never got below the tipping point, so neither town ate
much. The paper and SocialJax train about 30× longer (around 300M steps). At ~1,000 steps/s per
run that would take 3½ days per run, which is why we built the JAX version below.

### Our JAX Clean Up (2026-11-17)

`sims/jaxtown/` runs Clean Up on the GPU, thousands of games at once. It's built from SocialJax's
Clean Up (ICLR 2026, Apache 2.0; a copy of 5 files at commit 18e2171 in
`third_party/socialjax/`, see its `SOURCE.md` and our `NOTICE`). It was rewritten so the rules
match Melting Pot's, read from Melting Pot's Lua and dmlab2d's `grid.cc`:

- movement relative to facing, with players moving in random order;
- Melting Pot's beam shape and blocking;
- the 50-frame zap timeout and recharge times;
- the dirt fraction over the 147 river cells;
- Melting Pot's random game length;
- each player's own apples;
- the four towns, as data, in `sims/town/spec.py`, which both versions read.

`cleanup.py`'s docstring has the details and the few known simplifications.

The one deliberate difference is what players see: an 11×11 grid (the same window as Melting
Pot's: 9 ahead, 1 behind, 5 to each side) of 0/1 channels per cell, instead of pixels. Each other
player has their own channel, except in the strangers town where everyone else is just "someone".
Others' facing isn't shown, because Melting Pot's players don't rotate.

How we know it matches (`sims/jaxtown/compare.py`, `out/jaxtown/compare.md`):

- **Beam:** with Melting Pot's own players zapping, 327 zaps hit exactly our beam's 9 cells.
- **Random play:** random players in every town and river start: 120 Melting Pot games against
  1,024 of ours each. All 80 comparisons agree within 10% or within the 95% ranges: game length,
  apples, cleaning, zaps and feed per step.
- **What the comparison caught along the way:**
  - In Melting Pot an apple that grows under a standing player is eaten at once. Players only
    turning on the spot ate 181 apples in 1,950 frames.
  - A feed tile's countdown starts in the frame you arrive.
- **Rule tests:** 21 hand-made-map tests in `sims/jaxtown/tests` cover beams, the timeout,
  recharges, moves and bumping, eating, growth, dirt, game length, the view, identity, the badge
  and the feed.

Training (`sims/jaxtown/train_jax.py`, same flags and output files as `sims/town/train.py`, so
`report.py` charts both):

```
wsl -d Ubuntu-24.04 -e bash -lc 'cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/train_jax.py --runs neighbours:selfish neighbours:fair --river hughes --steps 1e7'
```

Speed:

| What | Steps per second |
|---|---|
| Random play, 1,024 games | ~630,000 (SocialJax's version) |
| Random play, 2,048 games | ~750,000 (ours) |
| Training, 512 games, bfloat16 (`--dtype`) | ~28,000 |
| Training, same, float32 | ~19,000 |
| Melting Pot trainer, per run | ~1,000 |

Learning, not playing, takes most of the time: per update, 0.4 s plays the games and 1.8 s
trains. The clean-river sanity run learns as on Melting Pot: apples per step 0.2 → 2.3 by 3M
steps, against 0.5 → 2.5. It's a little slower per step because each update averages over 512
games.

GPU memory under WSL:

- No single GPU allocation can reach 4 GiB (3.9 GiB works, 4.5 GiB fails, 3 × 3 GiB works).
- JAX's default memory pool grows in ever bigger blocks and then fails, so `train_jax.py` uses
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
- One compiled training step must also stay under that cap. At 512 games, 16 minibatches need
  2.85 GiB; 8 would need 4.86 GiB.

### Selfish vs fair, on to 300M steps (2026-11-17 to 11-18)

The test: in the neighbours town with the `hughes` river start, do fair-minded players (Hughes et
al.'s guilt, beta 0.05) clean more and eat more than selfish ones, as in the paper? A result only
counts if it holds across seeds, so we compare against the published work, not against one run.

**At 10M steps, 4 seeds each:** fair players cleaned more on average, 283 dirt per game (range
221-368), against 213 for selfish players (116-330). The ranges overlap, and the selfish players
ate more (14.8 against 9.6 apples per game, all seven together). Melting Pot's single check run
above (fair 323, selfish 159) sits inside those ranges, so the two versions agree within the
noise. Too early to call.

**At 300M steps, 2 seeds each** (four runs sharing the GPU, about 7,200 steps/s each, 11½ hours):

![Cleaning, apples and reward over 300M steps, selfish and fair](img/long-run-300m.png)

| Players | Seed | Dirt cleaned / step | Apples / step | Reward / game (all 7) | Cleaned / game |
|---|---|---|---|---|---|
| selfish | 0 | 0.323 | 0.049 | 72.8 | 484 |
| selfish | 1 | 0.309 | 0.044 | 65.1 | 462 |
| fair | 0 | 0.322 | 0.043 | 64.2 | 485 |
| fair | 1 | 0.333 | 0.051 | 76.2 | 497 |
| random players | | 0.348 | 0.008 | | |

Last 10% of each run; random players from `compare.md`. Games last about 1,500 steps.

- **No difference at the end.** Both groups clean about as much as random players and eat about
  70 apples per game between the seven of them. A clean river feeds them about 2.3 apples per
  *step*.
- **Guilt slowed learning down.** Selfish players found the apples by about 60M steps; fair
  players took until about 280M to catch up.
- **So this check fails.** It doesn't match Hughes et al., where guilt brought more cleaning and
  more reward. It matches SocialJax only in part: there selfish players barely clean, and here
  the fair ones don't do better.

### Why the players stop learning (2026-11-18 to 11-19)

They eat almost only at the very start of each game: the average apple is eaten around frame
50-55, and in 9 games out of 10 before frame 70. The river starts 41% dirty, just over the 40%
line above which no apples grow, and dirt only starts arriving at frame 51. Early cleaning dips
the river under the line and apples grow for a moment. Then dirt arrives at 0.5 per frame, faster
than the players clean (about 0.32), and for the remaining ~1,400 frames the river stays silted
up. With no reward anywhere later in the game, nothing teaches them to keep it clean.

**Our first guess was the dirt rule. The test below says it isn't.** The three versions of
Cleanup differ in how dirt arrives:

- **Melting Pot:** its `DirtSpawner` (`lua/levels/clean_up/components.lua`) adds one dirt with
  probability 0.5 every frame as long as any river cell is clean, so the river can fill up
  completely. Once it tips, only a big group clean-up brings the apples back.
- **SocialJax:** the same; no check against the threshold.
- **The open Sequential Social Dilemma Games Cleanup**
  ([eugenevinitsky/sequential_social_dilemma_games](https://github.com/eugenevinitsky/sequential_social_dilemma_games),
  `cleanup.py`, same 0.4 / 0.5 / 0.05 numbers) stops adding dirt once the river is 40% dirty.
  The river can't get worse than just past the line, so a little cleaning always brings a few
  apples back.

Hughes et al.'s own code isn't public; the open version is widely used in later work. So we
guessed their result came from the gentler game, and tested that.

**The test** (100M steps, one seed each, `hughes` river start, three runs sharing the GPU, about
9,100 steps/s each, 3 hours):

- selfish and fair players, with dirt stopping at 40% (`--dirt capped`, the open version's rule);
- players who share their apples, on Melting Pot's own rule: each is paid the average of all
  seven players' apples (`shared`). Can a silted river be won back at all?

![The first 100M steps: players sharing apples, and selfish and fair players under each dirt rule](img/dirt-rule-100m.png)

| Players | Dirt rule | Dirt cleaned / step | Apples / step | Reward / game (all 7) | Cleaned / game | Zaps / game |
|---|---|---|---|---|---|---|
| sharing apples | Melting Pot's | 0.507 | 1.890 | 2,850 | 764 | 2.6 |
| selfish | stops at 40% | 0.028 | 0.049 | 73 | 42 | 34.7 |
| fair | stops at 40% | 0.023 | 0.014 | 22 | 35 | 37.1 |
| selfish (seed 0 above) | Melting Pot's | 0.318 | 0.051 | 76 | 479 | 26.3 |
| fair (seed 0 above) | Melting Pot's | 0.277 | 0.031 | 46 | 416 | 27.3 |

Steps 90M-100M of each run (the 300M runs above cut at 100M), about 6,600 games each.

- **Players who share keep the river clean, on Melting Pot's own rule.** They clean 0.51 dirt a
  step, just ahead of the 0.5 that arrives, and eat 1.9 apples a step, about four-fifths of the
  2.3 a river that never gets dirty gives. That's about 37 times what selfish players get.
- **They split the work.** The same three players (1, 2 and 6) clean in every game, 240-270 dirt
  each, and eat almost nothing; the other four eat 680-740 apples each and barely clean. They
  hardly zap each other (2.6 zaps a game, against 26-37). They learned it fast: 0.26 apples a
  step by 5M steps, 1.2 by 25M.
- **Stopping the dirt at 40% didn't help.** Selfish players ate about as much as under Melting
  Pot's rule, and fair players less, though fair players were slow to learn under either rule (as
  in the 300M runs above). With the river held at the line, where apples barely grow, and dirt
  coming straight back after each clean, a little cleaning still brings almost no apples; they
  cleaned about a tenth as much.
- **So the river isn't the problem. Who pays for cleaning is.** Cleaning costs the cleaner time
  they could spend eating, and feeds everyone, so each selfish player does better eating while others clean, and then
  nobody cleans. That's the social dilemma Clean Up was built to pose, and the players are
  stuck in it, not stopped by a broken game. Hughes et al.'s guilt (beta 0.05) didn't get our
  players out of it, at 100M or at 300M steps.
- **One seed each**, so a first look. The gap between sharing players and the rest (2,850 against
  at most 76) is far bigger than anything seeds changed in the runs above (64-76 at 300M).

For experiment 1 this sharpens the question: with the same game, the same river and the same
learner, a town that pulls together eats about 2,850 apples a game and one that doesn't eats about
70. The towns ask how much of that gap connection closes, and how much isolation widens it.

### Where the training time goes (2026-11-19)

One training step on the 4070 Ti (512 games x 128 steps, 2.3 s) splits into playing the game
(~4%), the players choosing their moves (~12%) and learning (~84%); within learning, the
players' memory (GRU) loop takes about a quarter. Speed-ups that keep the maths exactly the same
gave little: unrolling the memory loop -9% of the learning time, batching 4 runs in one call -13%,
the eyes as plain matrix products none. Fewer, bigger minibatches (which would change how the
players learn) gave only -8-10%, so they're not worth it. The work is the players' networks
themselves. Under WSL, the GPU's default graph recording matters a lot: turning it off made
learning 3.8x slower. (NVIDIA's profiler crashes under WSL; these are timings of the whole
learning step with parts swapped out.)

### On a Colab TPU (2026-11-19)

Same code on Google Colab's free TPU (v5e, one chip, JAX 0.11.2), checked with
`sims/jaxtown/tpu_check.py`:

| | Colab TPU v5e | RTX 4070 Ti |
|---|---|---|
| 256 random games x 1,500 frames | identical, every frame | |
| Clean-river practice run, apples per step at 1M / 2M / 3M steps | 1.57 / 2.14 / 2.35 | 1.58 / 2.11 / 2.32 |
| Training, 1 run | 22,100 steps/s | 28,600 steps/s |
| Training, 3 runs sharing the chip | 22,200 steps/s in all | 27,300 steps/s in all |
| Random play, 4,096 games | 165,000 steps/s | 2,130,000 steps/s |

- **Same game, bit for bit.** The game uses only integer maths and JAX's random numbers, which
  are the same on every machine.
- **Same learning, within about 2%.** Not bit for bit: the chips round floating-point numbers
  differently, and small differences grow during training. The curves match.
- **One TPU chip trains at about 80% of the 4070 Ti.** The game itself runs 13x slower on the
  TPU (its many small steps suit a GPU better), but it's only a few percent of training. Several
  chips, each running its own runs, is where TPUs would add speed.

### Training on Colab: `colab_runner.py`

`sims/jaxtown/colab_runner.py` runs `train_jax.py` on a Colab machine from this PC and keeps it
going. It needs the `colab` command (`uv tool install --with keyrings.alt google-colab-cli` in
WSL, then `colab whoami` once to sign in in the browser) and the current commit pushed to
GitHub, since the machine clones the public repo at that commit.

```
cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/colab_runner.py \
    --session towns -- --runs neighbours:selfish neighbours:fair --river hughes \
    --steps 1e8 --out-root out/colab/towns
```

- Every `--sync-minutes` (10) it copies each run back to `--out-root` here: the checkpoint first,
  then the logs, and only the new lines of the long game log.
- Colab doesn't publish its limits (people report ~12-hour sessions and ~90 minutes idle), and a
  machine can go at any time. When it does, the runner trims the logs here back to the
  checkpoint here, starts a new machine, uploads the copies and resumes. Training saves a
  checkpoint every 5 minutes on Colab, so a lost machine costs the training since the last
  checkpoint copied here (at most about 15 minutes with the defaults), plus a few minutes to
  start another.
- When every run is done it copies everything back and stops the machine; it stops it on any
  error or Ctrl-C too. Running the same command again carries on from the copies here. At the
  start it stops any machine left under the same `--session` name (say, after this PC
  restarted), so use one runner per session name.
- Tested against a pretend Colab (`tests/test_colab_runner.py`): the machine is taken away
  mid-run, a second one resumes, and both runs end with every update logged exactly once.

**Drill on a real Colab TPU (2026-11-19).** Two runs of 40 updates on the free v5e, checkpoints
every minute, copies every 90 seconds. Once the first checkpoints (update 1) were copied here,
the machine was stopped from outside, as when Colab takes one back:

- The runner saw the machine gone at its next check, started a second one within seconds,
  uploaded the checkpoints (about 40 s for each run's 52 MB) and resumed at update 1, about 4
  minutes after the machine went.
- Both runs finished all 40 updates; the copies here have every update logged exactly once and
  all 1,448 games of each run in order. The runner then stopped the machine.
- Uploading ran at about 1.2 MB/s, so a run's checkpoint takes ~40 s to send (downloads
  weren't timed on their own).

**Free Colab ran out (2026-11-19).** The four towns' seed 1 (8 runs) trained on the free v5e for
1 h 36 min, to 13.5M steps a run. Then Colab took the machine and answered every request for
another TPU with "Service Unavailable" for over five hours, most likely the free plan's usage
limit (Colab doesn't publish it). The runs carried on on the 4070 Ti from the copies here. Free
Colab suits tests and short runs; long runs need a TPU grant or paid compute.

### Training on Cloud TPUs: `cloud_runner.py` (built 2026-11-19, not yet run on Google Cloud)

`sims/jaxtown/cloud_runner.py` is the runner for Cloud TPU machines, such as those of a
[TPU Research Cloud](https://sites.research.google/trc/about/) grant. It needs `gcloud` (signed
in, a project set), a Cloud Storage bucket in the machines' region, and the current commit
pushed to GitHub.

```
cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/cloud_runner.py \
    --name towns --zone us-central2-b --type v4-8 --spot --machines 2 \
    --bucket gs://<bucket> -- --runs <8 runs> --river hughes --steps 1e8 \
    --out-root out/cloud/towns
```

- **Only what was granted.** Cloud TPUs are free only within a grant's quota, and anything else
  is billed. So the zone, the type and the kind (`--spot` or `--on-demand`) have no defaults: they
  must be copied from the grant's email. The runner also stops if the bucket isn't in the
  machines' region, since copies between regions are charged.
- **Many chips.** The runs are dealt out over `--machines` machines. On each one, `train_jax.py`
  spreads its runs over the machine's chips (4 on a v4-8, 8 on a v5e or v6e). Each run's maths
  is unchanged. On one GPU nothing changes, and a test checks that two chips train bit for bit
  as one does.
- **The bucket holds the copies.** Every 5 minutes each machine puts its runs in the bucket
  (`sims/jaxtown/cloud_vm.py`): logs first, then the checkpoint, so the bucket's logs are never
  older than its checkpoint. When a spot machine is taken back, the runner deletes it, starts
  another and carries on from the bucket. That costs the training since the last copy (up to
  ~10 minutes) plus the new machine's set-up.
- **Copies home.** Every `--sync-minutes` (10) it copies the small files to `--out-root` here
  (settings, metrics, the new lines of the game log, each machine's train log in
  `_machines/`). Checkpoints come home only when a machine's runs finish, because every copy out
  of Google Cloud is charged. `--bring-home` copies everything, checkpoints too, without starting
  any machine, so a run can carry on on the GPU here when a grant ends. A run that's further along
  here than in the bucket goes up at the start, so GPU runs can carry on in the cloud.
- **Always cleaned up.** A machine is deleted as soon as its runs finish. On an error or Ctrl-C,
  each machine is first asked to save and put its copies in the bucket, then everything is
  copied home and the machines are deleted.
- **Tested against a pretend Google Cloud** (`tests/test_cloud_runner.py`, on the CPU):
  - three runs on two machines, one machine taken back mid-run and replaced: every update logged
    once, and the copies here are the same as the bucket's;
  - a run started on this PC carried on in the cloud;
  - training that keeps failing gives up and deletes its machine;
  - Ctrl-C saves on the machine, copies home and deletes it.

  The real `gcloud` commands are written from Google's documentation but haven't run yet: that
  needs a Google Cloud project. The first real test will be a short drill, as on Colab.

### Experiment 1: the four towns and their newcomers (2026-11-19 to 11-21)

The results are in [experiments/01-newcomers.md](experiments/01-newcomers.md). How they ran:

- **The four towns**, selfish and fair-minded, two seeds, 100M steps each, `--river hughes`:
  back on the 4070 Ti once Colab's free TPUs ran out, eight runs sharing the GPU at about 3,300
  steps a second each, one seed after the other (about 8 hours each). Seed 1's first 13.5M steps
  ran on the Colab TPU; it carried on from those checkpoints.
- **The newcomers:** `sims/jaxtown/newcomers.py`. `make` builds each move's run folder from two
  finished runs: the locals' weights and optimizer state, with players 1-3 taken from the
  newcomers' run, and the locals' random key, so every move into one town plays out with the same
  luck. `train` trains the 64 folders eight at a time through `train_jax.py --folders ...
  --resume` (a run named after its folder), with the settings they trained with before, and
  carries on after a stop. `report` draws the chart and writes the tables. 13 hours in all.
- **A mistake caught early:** the first plan took a move's newcomers from the same seed as the
  locals, and a control's from the other seed. Runs with the same seed start from the same
  weights, so that compared siblings with strangers. The first batch showed it within minutes;
  newcomers now always come from the other seed.

### The co-op (2026-11-21)

A rule of our own, off by default: the connecting AI takes a share of every apple eaten and splits
it evenly between the players who cleaned in the last 50 frames (the badge's record). An eater who
cleaned recently gets some of it back. While nobody has cleaned recently, eaters keep their apples
whole. It moves reward between players and never changes the game. We tried it instead of envy
because Hughes et al. 2018 found envy ("disadvantageous inequity aversion") doesn't help in Clean
Up: punishing others "does not directly increase the attractiveness of waste cleaning".

- **How to run it:** a fifth part of a run, `town:players:seed:dirt:share`, e.g.
  `connected:selfish:0:melting_pot:0.25`, or `--coop-share 0.25` for every run. The run's folder
  gets `_coop25` and its `config.json` a `coop_share`. Runs from before the co-op resume as share 0.
- **Checked side by side** with the game before the co-op, 16 games × 3,000 frames of random
  moves in the connected, neighbours and isolating towns:
  - share 0 is identical bit for bit (state, views, rewards, game endings);
  - share 0.5 plays the identical game, and each frame's total reward stays the same, up to rounding.
- **Tests:** the payout sums to zero and pays the right players; share 0.5 changes rewards only;
  the share belongs to the run, and old runs still resume.
- **Fair players** compare what they get after the co-op.
- **Sustainability** (when apples are eaten) counts the game's own rewards, not the payouts.
- **The results** are in [experiments/02-coop.md](experiments/02-coop.md) (20 runs of 20M steps
  with `--river hughes`, against experiment 1's towns; `sims/jaxtown/coop_report.py` makes its
  chart and tables).

### Newcomers in a co-op town (2026-11-21)

Three of a town's seven players move between a connected selfish town with the 10% co-op (seeds 0
and 1) and one without it (seeds 2 and 3), and everyone trains on for 20M steps. Each move runs
twice: once with the three who cleaned least, once with the three who cleaned most (in a co-op
town, its one cleaner and two of its eaters). 16 runs. The results are in
[experiments/02b-coop-newcomers.md](experiments/02b-coop-newcomers.md).

- **The code:** `sims/jaxtown/coop_newcomers.py` (`make`, `train`, `report`), built on
  experiment 1's `newcomers.py`. Players are ranked by how much they cleaned in their run's last
  10% of games before the move, and each newcomer takes the seat of the local with the same rank.
- **A newcomer in another seat:** players tell each other apart by seat (one view channel per
  seat). A newcomer who moved seats would see someone in the channel of its old seat, which it
  never saw on anyone else. So its first layer is relabelled for the new seat (`reseat`): it sees
  the players around it as it saw the ones at home. That's exact: a test checks the relabelled
  network gives the same odds for each move and the same value on the relabelled view, and acts
  differently without it. Experiment 1's moves kept their seats, so they don't change.
- **Training:** two batches of eight runs on the 4070 Ti, about 3½ hours in all.
