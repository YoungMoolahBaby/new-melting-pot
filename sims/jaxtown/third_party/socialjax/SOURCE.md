# SocialJax, Clean Up only

Copied 2026-11-17 from https://github.com/cooperativex/SocialJax at commit
`18e21718e52e659bd4cf9c29df2ddfc03dbaca3c` (main, 2026-10-17). Apache License 2.0; see
`LICENSE` in this folder. Paper: Guo et al., "SocialJax: An Evaluation Suite for Multi-agent
Reinforcement Learning in Sequential Social Dilemmas", ICLR 2026 (arXiv 2503.14576).

Files, at their original paths and unchanged:

- `socialjax/environments/cleanup/clean_up.py`
- `socialjax/environments/cleanup/rendering.py`
- `socialjax/environments/movement.py`
- `socialjax/environments/spaces.py`
- `socialjax/environments/multi_agent_env.py`

Our only additions here are the three `__init__.py` files, which are empty. The originals register
all SocialJax environments; we need only Clean Up. Nothing else from the repo is copied or
installed: it pins JAX 0.6, flax 0.10, mujoco, wandb and more.

Our own Clean Up, built from this one with Melting Pot's rules and our four towns, is
`sims/jaxtown/cleanup.py`. Its docstring lists what it changed and why.
