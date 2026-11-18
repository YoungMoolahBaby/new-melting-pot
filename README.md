# new-melting-pot

America has always called itself a melting pot. Now AI is joining it.

This project asks what kind of place lets people and AI live and work well together, and
undo **enshittification**: products and places that start out good, lock people in, and then
squeeze them. The tool is DeepMind's [Melting Pot](https://github.com/google-deepmind/meltingpot),
a set of small games where AI agents share, team up, or cheat each other.

**Status (2026-11-18): experiment 1 is built and being checked.** Four towns play Clean Up, a
game where a shared river must be kept clean for apples to grow: strangers, neighbours, a town
with an AI that connects people (it shows who has been cleaning), and a town with an AI that
isolates them (feed tiles pay a small private reward to whoever stands on them, like an
engagement feed). Players are selfish or fair-minded. It runs in
Melting Pot and in our own fast JAX copy of Clean Up, checked rule by rule against Melting Pot.

- [docs/PLAN.md](docs/PLAN.md): the idea, the ground rules, decisions. Start here.
- [docs/QUESTIONS.md](docs/QUESTIONS.md): every open question, parked until we take it up.
- [docs/MELTING-POT.md](docs/MELTING-POT.md): Melting Pot, how to run it, and experiment 1's
  code, checks and results so far.
- [docs/answers/](docs/answers/): five questions from
  [brownfield-compute](https://github.com/YoungMoolahBaby/brownfield-compute), each answered from
  real cases.

Licence: Apache 2.0 ([LICENSE](LICENSE)). It builds on Melting Pot and SocialJax, both Apache 2.0;
see [NOTICE](NOTICE).
