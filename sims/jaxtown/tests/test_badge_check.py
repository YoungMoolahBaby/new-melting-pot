"""Badge check: hiding the badge removes it from every view, and the sensitivity measure
reads 0 (to rounding) for players whose eyes give the badge no weight, and more for
players whose eyes do. Runs on the CPU.

    cd /mnt/c/dev/new-melting-pot && JAX_PLATFORMS=cpu ~/venvs/meltingpot/bin/python -m pytest \\
        sims/jaxtown/tests/test_badge_check.py -q
"""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))
import badge_check as bc  # noqa: E402
import cleanup as cu  # noqa: E402
import ppo  # noqa: E402


def test_hidden_badge_and_the_sensitivity_measure():
    env = cu.CleanUp()
    learner = ppo.Learner(ppo.Config(encoder="cells", obs_shape=(cu.VIEW, cu.VIEW,
                                                                  cu.NUM_CHANNELS)))
    params, _ = learner.init(jax.random.PRNGKey(0))
    kernel = params["params"]["conv1"]["kernel"]        # [players, 1, 1, channels, 16]
    blind = jax.tree.map(lambda x: x, params)
    blind["params"]["conv1"]["kernel"] = kernel.at[..., cu.CH["badge"], :].set(0.0)
    keen = jax.tree.map(lambda x: x, params)
    keen["params"]["conv1"]["kernel"] = kernel.at[..., cu.CH["badge"], :].set(5.0)
    town = env.town("connected", "hughes")
    play = bc.make_play(env, learner, num_envs=2, frames=200)
    key = jax.random.PRNGKey(1)

    def run(p, t):
        return bc.summary(*jax.device_get(play(p, t, key)), num_envs=2, frames=200)

    keen_shown = run(keen, town)
    assert keen_shown["badge_in_view"] > 0          # random players clean, so badges light
    # (A new network chooses almost evenly, so its choices move only a little.)
    assert keen_shown["tv"] > 1e-5
    assert run(blind, town)["tv"] < 1e-7             # no weight on the badge: no change
    hidden = run(keen, town.replace(badge=jnp.asarray(False)))
    assert hidden["badge_in_view"] == 0 and hidden["tv"] == 0.0
    assert all(0 <= x <= 1 for x in keen_shown["lit"])
