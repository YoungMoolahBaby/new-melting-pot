# To do

## SocialJax give-back

Fixes found while matching SocialJax's Clean Up to Melting Pot's, sent upstream.

Open, waiting for the maintainers:

- [ ] [cooperativex/SocialJax#17](https://github.com/cooperativex/SocialJax/pull/17): one PRNG key
  per draw in Clean Up; clear error for `cf=True`; debug print removed.
- [ ] [cooperativex/SocialJax#18](https://github.com/cooperativex/SocialJax/pull/18): agents start
  and respawn facing all four headings.

After they reply, not before, to keep it light:

- [ ] **Pull request:** the resets of coin_game, gift, harvest_common_open, mushrooms, pd_arena
  and territory_open draw the spawn shuffle and the start headings from one subkey, as Clean Up
  did. Same fix as #17; send it only if #17 is welcomed.
- [ ] **One issue with two questions.** A question, not a pull request: the intended values are
  unknown, and changing them would change the game their published results used.
  - Clean Up's dirt fraction also counts the 20 plain river cells, so apples stop at 67 dirty
    cells, against Melting Pot's 59. Intended?
  - The inequity-aversion smoothing decays by `0.99 * 0.01` per step, so the smoothed reward is
    almost just this step's reward. Intended?
