"""Does our JAX Clean Up run the same, and how fast, on a Colab TPU?

    colab run --tpu v5e1 --timeout 2400 sims/jaxtown/tpu_check.py [--commit C] [--train-only]
        # on a fresh Colab TPU machine (clones the public repo at commit C, default main;
        # --timeout: training is quiet for minutes, and Colab gives up after 30 s by default)
    ~/venvs/meltingpot/bin/python sims/jaxtown/tpu_check.py --local .
        # parts 1-2a on this PC's GPU, to compare

1. Same game: 256 random games x 1,500 frames from fixed seeds. Every frame's state
   (positions, facing, dirt, apples) is hashed on the device with integer maths; the
   hashes are digested at frames 10, 100, 300, 1000, 1500, so two machines can be
   compared line by line. Plus averages (apples, cleaning, zaps, dirt).
2. Speed: random play, then training (one run, then three sharing the chip).
3. Same learning: the clean-river practice run (3M steps), apples per step over time.

Results 2026-11-19 (commit 1999372) are in docs/MELTING-POT.md.
"""
import hashlib
import os
import subprocess
import sys
import time

LOCAL = "--local" in sys.argv
COMMIT = sys.argv[sys.argv.index("--commit") + 1] if "--commit" in sys.argv else "main"
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


if LOCAL:
    REPO = sys.argv[sys.argv.index("--local") + 1]
else:
    REPO = "/content/new-melting-pot"
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return "none"

    # Colab's flax can be older than its JAX and fail: install the flax and optax we train
    # with (as colab_runner.PINS), with JAX pinned so the TPU setup isn't touched.
    jv, jlv = ver("jax"), ver("jaxlib")
    say(f"Colab came with jax {jv}, jaxlib {jlv}, flax {ver('flax')}, optax {ver('optax')}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flax==0.12.12",
                    "optax==0.2.9", f"jax=={jv}", f"jaxlib=={jlv}"], check=True)
    say(f"now jax {ver('jax')}, flax {ver('flax')}, optax {ver('optax')}")
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "-q", "https://github.com/YoungMoolahBaby/new-melting-pot",
                        REPO], check=True)
    subprocess.run(["git", "-C", REPO, "checkout", "-q", COMMIT], check=True)
sys.path[:0] = [f"{REPO}/sims/jaxtown", f"{REPO}/sims/town"]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:   # same random numbers on both sides (the default in newer JAX; the flag may be gone)
    jax.config.update("jax_threefry_partitionable", True)
except AttributeError:
    pass
import cleanup as cu  # noqa: E402

head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                      text=True).stdout.strip()
say(f"jax {jax.__version__} | {jax.devices()[0].device_kind} x{jax.device_count()} | "
    f"python {sys.version.split()[0]} | repo {head}")

TRAIN_ONLY = "--train-only" in sys.argv

# --- 1. same game -------------------------------------------------------------------
N_GAMES, FRAMES = 256, 1500
env = cu.CleanUp()
town = env.town("neighbours", "hughes")
W = jnp.asarray(np.random.default_rng(12345).integers(1, 2**31, size=7 * 3 + 147 + 122),
                jnp.uint32)


def game(key):
    k0, key = jax.random.split(key)
    s = env.reset(k0, town)

    def frame(s, k):
        ka, ks = jax.random.split(k)
        a = jax.random.randint(ka, (env.num_players,), 0, cu.NUM_ACTIONS)
        s, _, _, _, done, info = env.step(ks, s, a, town)
        fp = jnp.concatenate([s.pos.ravel(), s.ori, s.dirt, s.apple]).astype(jnp.uint32)
        stats = jnp.stack([info["ate"].sum(), info["cleaned"].sum(), info["zaps"].sum(),
                           s.dirt.sum(), done]).astype(jnp.float32)
        return s, ((fp * W).sum(), stats)

    return jax.lax.scan(frame, s, jax.random.split(key, FRAMES))[1]


if not TRAIN_ONLY:
    play = jax.jit(jax.vmap(game))
    hashes, stats = play(jax.random.split(jax.random.PRNGKey(0), N_GAMES))
    hashes, stats = np.asarray(hashes), np.asarray(stats)
    say(f"game check: {N_GAMES} random games x {FRAMES} frames")
    for k in (10, 100, 300, 1000, 1500):
        say(f"  digest of frames 1-{k:<5d} "
            f"{hashlib.sha256(hashes[:, :k].tobytes()).hexdigest()[:16]}")
    names = ("apples/frame", "cleaned/frame", "zaps/frame", "dirt cells", "games ended")
    say("  averages: " + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, stats.mean((0, 1)))))

    # --- 2a. speed: random play ---------------------------------------------------------
    for n in (1024, 4096):
        f = jax.jit(jax.vmap(game))
        keys = jax.random.split(jax.random.PRNGKey(1), n)
        jax.block_until_ready(f(keys))
        t = time.time()
        jax.block_until_ready(f(keys))
        say(f"random play, {n} games: {n * FRAMES / (time.time() - t):,.0f} steps/s")

if LOCAL:
    sys.exit(0)

# --- 2b, 3. training: the real trainer, in this process (a TPU takes one process only) ---
os.chdir(REPO)
import train_jax  # noqa: E402


def train(name, extra):
    out = f"/content/out/{name}"
    say(f"{name}: start (progress lines every 10 updates show steps/s)")
    t = time.time()
    try:
        train_jax.main(["--ckpt-minutes", "1000", "--out-root", out] + extra)
    except Exception as e:   # report and go on to the next test
        say(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}")
    say(f"{name}: {time.time() - t:.0f}s wall")
    return out


UPD = 512 * 128   # steps per update
train("speed1", ["--runs", "neighbours:selfish", "--river", "hughes", "--steps", str(20 * UPD)])
train("speed3", ["--runs", "neighbours:selfish", "neighbours:fair", "neighbours:shared",
                 "--river", "hughes", "--steps", str(10 * UPD)])
out = train("sanity", ["--runs", "neighbours:selfish", "--river", "clean", "--steps", "3e6"])

import csv  # noqa: E402

path = f"{out}/neighbours_selfish_s0/metrics.csv"
if os.path.exists(path):
    rows = list(csv.DictReader(open(path)))
    say("clean-river practice run, apples per step (mean of updates up to each mark):")
    steps = np.array([float(r["env_steps"]) for r in rows])
    apples = np.array([float(r["apples_per_step"]) for r in rows])
    for m in (0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6):
        k = (steps > m - 0.5e6) & (steps <= m)
        if k.any():
            say(f"  {m / 1e6:.1f}M  {apples[k].mean():.3f}")
say("done")
