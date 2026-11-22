"""coop_report.py measures what it says on made-up runs."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import coop_report as cr  # noqa: E402


def game(env_steps, ate, cleaned, ret, length=1000):
    return {"env_steps": env_steps, "length": length, "ate": ate, "cleaned": cleaned,
            "feed": [0.0] * 7, "return": ret, "top2_clean_share": 1.0}


def make_run(root, name, town, players, share, seed, eps, **settings):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"game": "jaxtown", "town": town, "players": players, "seed": seed,
                   "river": "hughes", **({} if share is None else {"coop_share": share}),
                   **settings}, f)
    with open(os.path.join(d, "episodes.jsonl"), "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    return d


def test_measures_find_runs_and_take_off(tmp_path):
    # Players 0 and 1 clean and are paid 1 each by the co-op; the others eat 10 each and
    # give 0.4 each.
    ate = [0, 0, 10, 10, 10, 10, 10]
    ret = [1, 1, 9.6, 9.6, 9.6, 9.6, 9.6]
    good = game(19e6, ate, [30, 20, 0, 0, 0, 0, 0], ret)
    m = cr.measure([good, good])
    assert m["games"] == 2
    assert m["apples"] == pytest.approx(50)            # per 1,000 steps, all 7
    assert m["clean"] == pytest.approx(50 / 7)
    assert m["moved"] == pytest.approx(2)              # what the paid players gained
    assert m["cleaners"] == 2
    assert m["cleaners_got"] == pytest.approx(1) and m["cleaners_ate"] == 0
    assert m["others_got"] == pytest.approx(9.6) and m["others_ate"] == pytest.approx(10)
    assert cr.measure([]) is None

    # One player does all the cleaning and another cleans a little (under a tenth of it): one
    # cleaner, not averaged with an eater.
    one = game(19e6, [0, 10, 10, 10, 10, 10, 10], [95, 5, 0, 0, 0, 0, 0],
               [2, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6])
    m = cr.measure([one])
    assert m["cleaners"] == 1
    assert (m["cleaners_got"], m["cleaners_ate"]) == (pytest.approx(2), 0)
    assert (m["others_got"], m["others_ate"]) == (pytest.approx(9.6), pytest.approx(10))

    # A run without the co-op from before it existed (no coop_share) is share 0; games past
    # 20M steps are left out; the town "takes off" in the first 1M stretch at 1,000+ apples.
    big = game(0, [200] * 7, [5] * 7, [200] * 7)
    eps = [game(0.5e6, [1] * 7, [5] * 7, [1] * 7), dict(big, env_steps=3.5e6),
           dict(big, env_steps=30e6)]
    make_run(tmp_path, "old", "connected", "selfish", None, 0, eps)
    make_run(tmp_path, "new", "connected", "selfish", 0.1, 0, eps[:1])
    # Other river starts and dirt rules are other experiments: left out.
    make_run(tmp_path, "capped", "connected", "selfish", None, 0, eps, dirt="capped")
    make_run(tmp_path, "shipped", "connected", "selfish", None, 0, eps, river="melting_pot")
    runs = cr.find_runs([str(tmp_path)])
    assert set(runs) == {("connected", "selfish", 0.0, 0), ("connected", "selfish", 0.1, 0)}
    old = cr.games(runs[("connected", "selfish", 0.0, 0)])
    assert len(old) == 2
    assert cr.took_off(old) == 4e6
    assert cr.took_off(old[:1]) is None
    make_run(tmp_path, "again", "connected", "selfish", 0.0, 0, eps)
    with pytest.raises(SystemExit, match="two runs"):
        cr.find_runs([str(tmp_path)])
