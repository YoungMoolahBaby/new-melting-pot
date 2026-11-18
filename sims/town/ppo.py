"""Per-player PPO in JAX: each of the 7 players has its own network and learns its
own habits. All 7 networks run together on the GPU (their weights are stacked on
a leading "player" axis and the network is vmapped over it), but no weights or
gradients are shared.

Network, per player (after Melting Pot's baseline agents, scaled to our 22x22 view):
    conv 16 (2x2, stride 2: one feature vector per map cell) -> conv 32 (3x3)
    -> dense 128 -> dense 128 -> GRU 128 (memory: who did what) -> policy + value

Also here: the fairness reward ("inequity aversion", Hughes et al. 2018, NeurIPS),
eq. 3-4 of the paper, as plain numpy so it's easy to test.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

VIEW = 22


class Config(NamedTuple):
    num_players: int = 7
    num_actions: int = 9
    hidden: int = 128
    # "pixels": Melting Pot's view shrunk to 22x22x3 (sims/town/envs.py);
    # "cells": our JAX Clean Up's 11x11 grid of 0/1 channels (sims/jaxtown/cleanup.py).
    encoder: str = "pixels"
    obs_shape: tuple = (VIEW, VIEW, 3)
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    # Number type for the eyes (conv and dense layers): "bfloat16" is faster on the
    # GPU; weights, memory (GRU) and outputs stay float32.
    compute_dtype: str = "float32"


# --- fairness ---------------------------------------------------------------

def smooth_rewards(e: np.ndarray, r: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Hughes et al. eq. 4: e_t = gamma * lambda * e_{t-1} + r_t (per player)."""
    return gamma * lam * e + r


def inequity_reward(r: np.ndarray, e: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Hughes et al. eq. 3, the subjective reward of each player.

    r, e: [..., N] this step's reward and the smoothed rewards of all N players.
    u_i = r_i - alpha/(N-1) * sum_j max(e_j - e_i, 0)   (envy: others are ahead)
              - beta /(N-1) * sum_j max(e_i - e_j, 0)   (guilt: I'm ahead)
    alpha = beta = 0 is a purely selfish player. Works on numpy and JAX arrays
    (inside jit too).
    """
    xp = jnp if isinstance(e, jax.Array) else np
    n = e.shape[-1]
    diff = e[..., None, :] - e[..., :, None]            # [..., i, j] = e_j - e_i
    behind = xp.maximum(diff, 0.0).sum(-1)
    ahead = xp.maximum(-diff, 0.0).sum(-1)
    return r - (alpha * behind + beta * ahead) / (n - 1)


# --- network ------------------------------------------------------------------

class Net(nn.Module):
    """Split into encode (eyes), core (memory) and heads, so training can run the
    eyes over a whole rollout in one batch and only step the memory in order."""
    num_actions: int
    hidden: int
    encoder: str = "pixels"
    compute_dtype: str = "float32"

    def setup(self):
        dt = jnp.dtype(self.compute_dtype)
        if self.encoder == "cells":   # already one vector per map cell
            self.conv1 = nn.Conv(16, (1, 1), dtype=dt)
        else:                         # 2x2 pixels per map cell -> one vector per cell
            self.conv1 = nn.Conv(16, (2, 2), strides=(2, 2), padding="VALID", dtype=dt)
        self.conv2 = nn.Conv(32, (3, 3), padding="SAME", dtype=dt)
        self.fc1 = nn.Dense(128, dtype=dt)
        self.fc2 = nn.Dense(128, dtype=dt)
        self.gru = nn.GRUCell(self.hidden)
        self.pi = nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))
        self.v = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))

    def encode(self, obs, ready):
        """obs [B, 22, 22, 3] uint8 pixels or [B, 11, 11, C] uint8 0/1 cells,
        ready [B] -> features [B, 128]."""
        x = obs.astype(jnp.float32) / (1.0 if self.encoder == "cells" else 255.0)
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = x.reshape(x.shape[:-3] + (-1,))
        x = jnp.concatenate([x, ready[..., None]], -1)
        return nn.relu(self.fc2(nn.relu(self.fc1(x)))).astype(jnp.float32)

    def core(self, h, x, start):
        h = jnp.where(start[..., None], 0.0, h)          # new episode: forget
        h, y = self.gru(h, x)
        return h, y

    def heads(self, y):
        return self.pi(y), self.v(y)[..., 0]

    def __call__(self, h, obs, ready, start):
        """h [B, H], obs [B, 22, 22, 3] uint8, ready [B], start [B] bool."""
        h, y = self.core(h, self.encode(obs, ready), start)
        logits, value = self.heads(y)
        return h, logits, value


class Learner:
    """Holds the jitted functions for one Config."""

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.net = Net(cfg.num_actions, cfg.hidden, cfg.encoder, cfg.compute_dtype)
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                              optax.adam(cfg.lr, eps=1e-5))
        self.act = jax.jit(self._act)
        self.update = jax.jit(self._update)

    # Weights and optimizer state, stacked over players.
    def init(self, key):
        cfg = self.cfg
        h = jnp.zeros((1, cfg.hidden))
        obs = jnp.zeros((1, *cfg.obs_shape), jnp.uint8)
        ready = jnp.zeros((1,))
        start = jnp.ones((1,), bool)
        keys = jax.random.split(key, cfg.num_players)
        params = jax.vmap(lambda k: self.net.init(k, h, obs, ready, start))(keys)
        opt = jax.vmap(self.tx.init)(params)
        return params, opt

    def initial_state(self, num_envs: int):
        return jnp.zeros((self.cfg.num_players, num_envs, self.cfg.hidden))

    def _act(self, params, h, obs, ready, start, key):
        """One step for every player in every game.

        h [P, E, H]; obs [E, P, 22, 22, 3]; ready [E, P]; start [E] (numpy is fine).
        Returns new h, actions [P, E], log-probs [P, E], values [P, E], next key.
        The key is split in here: outside jit a split costs ~1.5 ms per step (WSL GPU).
        """
        key, k = jax.random.split(key)
        obs = jnp.swapaxes(obs, 0, 1)
        ready = jnp.swapaxes(ready, 0, 1)
        h, logits, value = jax.vmap(self.net.apply, in_axes=(0, 0, 0, 0, None))(
            params, h, obs, ready, start)
        a = jax.random.categorical(k, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1)[..., 0]
        return h, a, logp, value, key

    def _unroll(self, params, h0, obs, ready, start):
        """One player over a rollout: obs [T, B, ...] -> logits [T, B, A], values [T, B]."""
        T, B = obs.shape[:2]
        x = self.net.apply(params, obs.reshape((T * B,) + obs.shape[2:]), ready.reshape(T * B),
                           method=Net.encode).reshape(T, B, -1)

        def step(h, xs):
            return self.net.apply(params, h, *xs, method=Net.core)

        _, y = jax.lax.scan(step, h0, (x, start))
        return self.net.apply(params, y, method=Net.heads)

    def _loss(self, params, h0, obs, ready, start, action, logp_old, adv, ret):
        cfg = self.cfg
        logits, v = self._unroll(params, h0, obs, ready, start)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, action[..., None], -1)[..., 0]
        ratio = jnp.exp(logp - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = 0.5 * jnp.square(v - ret).mean()
        entropy = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        total = pg + cfg.vf_coef * vloss - cfg.ent_coef * entropy
        kl = ((ratio - 1) - jnp.log(ratio)).mean()
        return total, jnp.stack([pg, vloss, entropy, kl])

    def gae(self, reward, value, done, last_value):
        """reward, value [T, P, E]; done [T, E]; last_value [P, E]."""
        cfg = self.cfg

        def step(carry, x):
            adv_next, v_next = carry
            r, v, d = x
            live = 1.0 - d[None, :].astype(jnp.float32)
            delta = r + cfg.gamma * v_next * live - v
            adv = delta + cfg.gamma * cfg.gae_lambda * live * adv_next
            return (adv, v), adv

        _, adv = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value),
                              (reward, value, done), reverse=True)
        return adv, adv + value

    def _update(self, params, opt, batch, key):
        """PPO epochs over one rollout.

        batch: obs [T, E, P, 22, 22, 3], ready [T, E, P], start [T, E], action,
        logp, value, reward [T, P, E], done [T, E], h0 [P, E, H], last_value [P, E].
        """
        cfg = self.cfg
        adv, ret = self.gae(batch["reward"], batch["value"], batch["done"],
                            batch["last_value"])
        # Player axis first, for vmapping over players: [P, T, E, ...].
        obs = jnp.moveaxis(batch["obs"], 2, 0)
        ready = jnp.moveaxis(batch["ready"], 2, 0)
        per_player = [jnp.moveaxis(x, 1, 0) for x in
                      (batch["action"], batch["logp"], adv, ret)]
        start = batch["start"]
        h0 = batch["h0"]
        num_envs = start.shape[1]
        n_mb = min(cfg.minibatches, num_envs)       # tiny test runs may have few games
        mb = num_envs // n_mb
        grad_fn = jax.vmap(jax.grad(self._loss, has_aux=True),
                           in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0))

        def minibatch(carry, idx):
            params, opt = carry
            pick = lambda x: x[:, :, idx]  # noqa: E731  [P, T, E, ...] -> envs idx
            grads, stats = grad_fn(params, h0[:, idx], pick(obs), pick(ready), start[:, idx],
                                   *[pick(x) for x in per_player])
            updates, opt = jax.vmap(self.tx.update)(grads, opt, params)
            params = optax.apply_updates(params, updates)
            return (params, opt), stats

        def epoch(carry, k):
            perm = jax.random.permutation(k, num_envs)[:mb * n_mb]
            return jax.lax.scan(minibatch, carry, perm.reshape(n_mb, mb))

        (params, opt), stats = jax.lax.scan(epoch, (params, opt),
                                            jax.random.split(key, cfg.epochs))
        # stats [epochs, minibatches, P, 4] -> per player [P, 4]
        return params, opt, stats.mean(axis=(0, 1))
