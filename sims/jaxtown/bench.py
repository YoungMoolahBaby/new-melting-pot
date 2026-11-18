"""How fast does SocialJax's Clean Up run on this PC? (Step 0, gate A.)

Random players, 1,000 steps per game, many games at once on the GPU. Times only
the stepping (compiled ahead, then run once to warm up, then timed).

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/bench.py

For comparison: Melting Pot's Clean Up runs ~2,000 steps/s in one process here,
and our training gets ~1,000 steps/s per run (measured 2026-11-16).
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # share the GPU politely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "socialjax"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from socialjax.environments.cleanup.clean_up import Clean_up  # noqa: E402

NUM_AGENTS = 7
STEPS = 1000


def make_rollout(env, num_envs):
    def rollout(key):
        key, k = jax.random.split(key)
        _, state = jax.vmap(env.reset)(jax.random.split(k, num_envs))

        def step(carry, _):
            state, key = carry
            key, ka, ks = jax.random.split(key, 3)
            actions = jax.random.randint(ka, (num_envs, NUM_AGENTS), 0, 9)
            obs, state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(ks, num_envs), state, actions)
            # Keep something from obs and reward so XLA can't skip computing them.
            return (state, key), (reward.sum(), obs.sum(dtype=jnp.float32))

        (_, _), (rewards, obs_sums) = jax.lax.scan(step, (state, key), None, STEPS)
        return rewards.sum(), obs_sums.sum()

    return jax.jit(rollout)


def main():
    print("devices:", jax.devices())
    env = Clean_up(num_agents=NUM_AGENTS, shared_rewards=False)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    print("obs per player:", obs.shape[1:], obs.dtype)
    rows = []
    for n in [int(x) for x in (sys.argv[1:] or ["1", "128", "512", "1024", "2048", "4096"])]:
        fn = make_rollout(env, n)
        t0 = time.perf_counter()
        compiled = fn.lower(jax.random.PRNGKey(0)).compile()
        t_compile = time.perf_counter() - t0
        jax.block_until_ready(compiled(jax.random.PRNGKey(1)))       # warm-up
        t0 = time.perf_counter()
        out = jax.block_until_ready(compiled(jax.random.PRNGKey(2)))
        dt = time.perf_counter() - t0
        sps = n * STEPS / dt
        rows.append((n, dt, sps, t_compile))
        print(f"{n:5d} games: {dt:7.2f} s for {n * STEPS:,} steps = {sps:12,.0f} steps/s "
              f"(compile {t_compile:.1f} s; apples eaten {float(out[0]):,.0f})", flush=True)
    print("\n| games at once | steps/s | vs Melting Pot (~2,000/process) |\n|---|---|---|")
    for n, dt, sps, _ in rows:
        print(f"| {n:,} | {sps:,.0f} | {sps / 2000:,.0f}x |")


if __name__ == "__main__":
    main()
