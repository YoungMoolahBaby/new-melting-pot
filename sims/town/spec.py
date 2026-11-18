"""What the four towns are, shared by both versions of the game:

    town_config.py   the towns in DeepMind's Melting Pot (Lua, CPU)
    sims/jaxtown     the same towns in our JAX Clean Up (GPU)

No Melting Pot or JAX imports here, so anything can read it cheaply.

    strangers   everyone looks the same (you still see yourself as yourself)
    neighbours  everyone has their own colour (normal Clean Up)
    connected   neighbours + the connecting AI: a gold badge rides on anyone who
                cleaned the river lately, so everyone can see who's pitching in
    isolating   neighbours + the isolating AI: "feed" tiles in the living area pay
                a small private reward every few frames to whoever stands on one
"""
from __future__ import annotations

TOWNS = ("strangers", "neighbours", "connected", "isolating")
NUM_PLAYERS = 7

# Clean Up's map, copied from meltingpot.configs.substrates.clean_up.ASCII_MAP
# (dm-meltingpot 2.4.1; tests/test_town.py checks they're still identical).
#   W wall   P spawn point   B orchard (apples can grow)   T orchard edge (apples too)
#   S ~ river water   H river, clean   F river, dirty   others: sand / grass / shadows
ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WHFFFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFHFFHHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFFFFFHFHFHFHFHFHFHHFHFFFHFW
W==============+~FHHHHHHf====W
W   P    P      ===+~SSf     W
W     P     P   P  <~Sf  P   W
W             P   P<~S>      W
W   P    P         <~S>   P  W
W               P  <~S>P     W
W     P           P<~S>      W
W           P      <~S> P    W
W  P             P <~S>      W
W^T^T^T^T^T^T^T^T^T;~S,^T^T^TW
WBBBBBBBBBBBBBBBBBBBssBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# Seven feed tiles in the sandy living area between river and orchard (row, col
# in ASCII_MAP), each on an empty sand cell, away from spawn points.
FEED_CELLS = ((8, 2), (8, 10), (10, 14), (11, 2), (13, 6), (8, 27), (12, 27))
FEED_REWARD = 0.1     # per payout; an apple is 1.0
FEED_PERIOD = 4       # frames between payouts while someone stays on the tile
BADGE_FRAMES = 50     # how long the badge stays up after the last cleaning

RIVERS = ("melting_pot", "hughes", "clean")
# How dirty the river starts. Apples only grow while under 40% of the river is dirty.
#   melting_pot  Clean Up as shipped: 79 of 147 river cells dirty (54%)
#   hughes       just past the tipping point, as in Hughes et al. 2018: 60 of 147 (41%)
#   clean        no dirt at all and none ever spawns, so apples always grow at full
#                speed: a sanity check that the players can learn to find apples
HUGHES_DIRT = 60


def map_rows(town: str = "neighbours", river: str = "melting_pot") -> list[str]:
    """ASCII_MAP as rows, with the river's starting dirt set by `river` and feed
    tiles ('Z') added for the isolating town."""
    if town not in TOWNS:
        raise ValueError(f"town must be one of {TOWNS}, not {town!r}")
    if river not in RIVERS:
        raise ValueError(f"river must be one of {RIVERS}, not {river!r}")
    rows = ASCII_MAP.strip("\n").split("\n")
    dirty = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "F"]
    keep = {"clean": 0, "hughes": HUGHES_DIRT}.get(river, len(dirty))
    # Clean the dirty cells evenly spread through the river, so the dirt that
    # stays is spread the same way as in the shipped map.
    n_clean = len(dirty) - keep
    for i in range(n_clean):
        r, c = dirty[(i * len(dirty)) // n_clean]
        rows[r] = rows[r][:c] + "H" + rows[r][c + 1:]
    if town == "isolating":
        for r, c in FEED_CELLS:
            assert rows[r][c] == " ", f"feed cell {(r, c)} is {rows[r][c]!r}, not sand"
            rows[r] = rows[r][:c] + "Z" + rows[r][c + 1:]
    return rows
