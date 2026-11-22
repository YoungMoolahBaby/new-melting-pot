# Plan

**Status (2026-11-21):** experiment 1 (four towns and their newcomers in Clean Up) has had its
first round; the write-up is [experiments/01-newcomers.md](experiments/01-newcomers.md). So has
experiment 2, a connecting AI that pays the cleaners
([experiments/02-coop.md](experiments/02-coop.md)). The code and checks are in
[MELTING-POT.md](MELTING-POT.md). This page is the thinking behind them.

## The idea

People and AI working together to undo enshittification. We want to find out what kind of place
keeps people and AI from learning to squeeze others (and each other), and what helps them unlearn
it. Simulated in DeepMind's Melting Pot ([MELTING-POT.md](MELTING-POT.md)).

## Where we want to end up

AI will change work and who holds power whether anyone plans for it or not. People's power has
always come from being needed (workers, soldiers, taxpayers); if AI does the work, that leverage
shrinks unless people keep it another way. Three pictures of where it could settle:

1. **Few owners:** a handful of companies own the AI and the computers; everyone else lives on
   what they're handed. Pullman at the size of the country.
2. **Everyone a maker:** AI is cheap and widely owned; each person with their AI is like a small
   company.
3. **Partnership:** big AI exists, but people keep a stake (shares, co-ops, a dividend like
   Alaska's oil cheque) and their AI stays loyal to them.

**We aim for a mix of 2 and 3.** The campus is a small version of that mix.

**Candidate core question (not decided):** what keeps a mix of 2 and 3 from sliding into 1?
Every success pulls toward 1: winners grow, buy others and lock people in, which is
enshittification again.

**The frame is the real country, not an imagined campus.** America
feels like "a town with a Costco and a massive landfill on fire". Costco is the good version that
exists at scale; the landfill is the enshittified rest. Innovation can feel like a crime because
the old system is fragile, and the new one looks set to steamroll it with no room for empathy.
Reworded candidate question: **how does America get through the AI steamroll with more Costcos,
less landfill, and its empathy intact?** The campus below stays as one idea, not the frame.

## Ground rules

1. **Nobody starts out bad.** Squeezing is learned from one's surroundings, by people and by AI,
   so it can be prevented and unlearned. (Not a full disease metaphor; Jon, 2026-11-16.)
2. **Weights at every level.** Upbringing, the situation, the team, the company, the culture. A
   2010 review of unethical choices at work ("bad apples, bad cases, and bad barrels") found all
   three levels matter. Never pin it on one.
3. **We study the squeeze only to find what stops it.** If the simulation turns up a sneakier
   squeeze, it is written up as a warning sign and how to spot it, never as a how-to.
4. **"Human" means a real person playing**, or a bot clearly labelled as a stand-in. Melting Pot
   has no models of human behaviour.
5. **As realistic as possible.** A result is trusted only after the model reproduces real cases.

## Costco and the landfill (2026-11-16)

- **Costco:** you come back because you want to (membership model, ~9 in 10 renew, the $1.50 hot
  dog since 1985). **The landfill:** you stay because you can't leave.
- **The landfill** is also what piles up inside people: old baggage, spite, pettiness, unhealthy
  habits, bad feeds (some feeds are good). People who feel left out turn bitter. Having an outlet
  beats stewing alone in the dark.
- **The fire:** leaders selling out the people they serve (a mayor under pressure sells the
  land cheap, so there's no room and the garbage burns).
- **The full toilet:** upkeep nobody does. Flush it, clean the bowl, wash the dishes. Housing
  prices out of reach while some own several homes, and each generation blaming the other.
- **What works:** places and habits people come back to by choice: Planet Fitness, Costco,
  men's-league hockey, video games, AI-assisted programming projects, some good feeds.

**Candidate root (2026-11-16): disconnection.** People stopped
knowing, needing and seeing each other, and the three feed each other in a loop: alone, people
stew (the landfill grows); nobody organized watches the leaders (the fire); nobody cleans what's
shared (the toilet); the place gets worse; people pull back further. Leads: trust in "most
people" in the US fell from about half (early 1970s) to about a third (General Social Survey, to
check exact figures); Elinor Ostrom (Nobel 2009) found communities that know each other, talk,
watch, and make their own rules keep shared resources healthy for centuries; the "what works"
list is mostly "third places". AI cuts both ways: it can reconnect people or isolate them further.

## Experiment 1 (candidate): strangers or neighbours? (Clean Up)

Chosen 2026-11-16. Built as four towns (strangers, neighbours, an AI that connects, an AI that
isolates); see [MELTING-POT.md](MELTING-POT.md).

Same agents, same Clean Up game, three versions:
- **A, strangers:** agents can't tell who's who.
- **B, neighbours:** agents can tell who's who and see who's been cleaning.
- **C, neighbours who can talk:** B plus a few simple signals.

Measure: how much cleaning happens, how many apples grow, how fairly they're shared, and whether
the cleaning lasts. Check first: selfish agents in A should leave the river dirty (the known
Clean Up result), and talking should help, as it did when people played commons games in the lab
(Ostrom, Walker & Gardner 1992, to check). Talking is the hardest piece to add; start with A and B.

Later, the AI twist: add an AI that either connects people (shows who's cleaning, brings them
together) or isolates them (keeps each one busy alone), and see which way the river goes.

## The campus idea (one idea, not the frame)

- A campus or town with a real American-dream feel: you see your coworkers, but you have your own
  space and your full rights. Even Ron Swanson would be comfortable there. Remote work feels too
  scattered.
- A **makerspace** at the centre: people make whatever they want, and the shared base software
  (CortenForge) gets better as they go. Blender works this way: its studio's films push the
  software forward.
- **You own what you make**, your AI companion included. It lives at home and can go with you
  anywhere.
- No judgement for what people make, as long as it hurts no one. Weapons: parked (D4).
- Shared gym, rec sports, greenhouses, a woodshop.
- **Feeling at home:** people know the place is doing its best for them. Touch and closeness
  matter; at work, psychological safety (it's safe to admit a mistake or ask for help).
- People will want their own ventures and take people with them; the company backs them.

## Candidate first test (not decided)

Two campuses, same people:
- **Pullman:** the company owns the homes, the gym and the AI.
- **Swanson:** people own their place and their AI; the company runs only the shared spaces.

Which one teaches people and AI to squeeze customers, and does that turn into squeezing each
other? Check first: the model should reproduce what went wrong in Pullman, Illinois (1894).
Not chosen yet: setting up a community around a company takes a huge amount of detail.

## Working with brownfield-compute (2026-11-16)

[brownfield-compute](https://github.com/YoungMoolahBaby/brownfield-compute) plans a village that
may one day become a city around a data center (one a retired-coal town would vote for; case
study Homer City, PA). The two projects work together so the village is planned around how
people actually behave.

- **Split:** brownfield owns the physical and economic village (sites, heat loop, housing, costs,
  the community benefits agreement). We own behaviour.
- **Boundary:** brownfield takes in only neutral findings cited to real cases. A game can point
  to which real cases to look at; the real cases are the evidence, never the simulation.
- **Answer format:** a few plain lines each: what happens, the real cases behind it, how sure we
  are, what it would change in the design.

Its questions, in the order its plan needs them:
1. **Enclave or mixing.** What layout, access and ownership keep a nicer new neighbourhood beside
   a poor town from becoming a resented enclave? Its amenities (food hall, courts, rink, fields,
   makerspace) are meant for all of Indiana County. Cases: Hershey vs Pullman, Columbia MD,
   Reston VA, Tulsa Remote.
2. **Running the shared places.** Who runs and pays for the makerspace, courts and fields, the
   winter garden and arboretum (co-op, township, operator, or a trust under the agreement), and
   what keeps them clean and fair? Also the ~3 m strips between narrow townhouses, which can
   become nobody's alley. This is the real Clean Up question.
3. **Who owns the heat loop.** Heat is metered and billed, so it isn't a free-rider problem. The
   risk is one operator controlling heat that homes depend on: trust and pricing, a power
   question. Cases: Danish district heating (mostly consumer co-ops or municipal) vs
   operator-owned networks.
4. **The construction-workforce surge**, before the flats (AWS's campus is to be built over
   2026-2036). Case: the North Dakota Bakken boom (housing squeeze, worker camps). It affects
   when each phase can happen.
5. **Does the agreement hold** once the operator is the most powerful party in town? It matters
   a lot, but it's further off for brownfield's plan.

## Decisions

- **D1 Melting Pot as it is, not a Rust/Bevy rewrite.** A rewrite would lose the pretrained bots
  and comparability with published results. Bevy comes later as a front end for people to play
  in (Windows, maybe VR), talking to an unchanged Melting Pot over a local socket.
- **D2 No Windows port for now.** Melting Pot's engine, dmlab2d, is Linux and macOS only; we run
  it in WSL2. A Windows build fix for dmlab2d is a possible side quest (lab2d issue #39, where the
  maintainer said they'd look at Windows support given a Windows setup).
- **D3 Framing:** learned, not born (ground rule 1).
- **D4 Weapons and defense parked.** Default: your home follows the law; the shared makerspace
  doesn't build weapons. See [QUESTIONS.md](QUESTIONS.md).

## Next

- Keep planning: take the questions in [QUESTIONS.md](QUESTIONS.md) one at a time.
- Melting Pot runs in WSL2 (2026-11-16, [MELTING-POT.md](MELTING-POT.md)). Experiment 1 is built
  there and in a faster JAX copy of Clean Up, checked against Melting Pot.
- brownfield's five questions are answered (2026-11-16), each from real cases, in
  [answers/](answers/): 01 enclave or mixing, 02 running the shared places, 03 who owns the heat
  loop, 04 the construction-workforce surge, 05 does the agreement hold. Each ends with where
  Melting Pot could help later.
