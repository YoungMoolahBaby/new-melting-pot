"""Record a clip of DeepMind's trained bots playing Clean Up.

Run inside WSL (Ubuntu-24.04), from the repo root:
    ~/venvs/meltingpot/bin/python sims/clip_clean_up.py --steps 1000

Writes out/clean_up_bots.mp4, out/clean_up_bots.gif and out/clean_up_bots_frame.png.
"""
import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image
from meltingpot import bot, substrate
from meltingpot.scenario import PERMITTED_OBSERVATIONS

# 3 bots trained to clean the river, 4 trained to eat apples (DeepMind's own trained bots).
BOTS = [
    "clean_up__cleaner_0", "clean_up__cleaner_1", "clean_up__cleaner_0",
    "clean_up__consumer_0", "clean_up__consumer_1", "clean_up__consumer_0", "clean_up__consumer_1",
]
SCALE = 3
BAR = 54  # caption bar height in pixels


def bot_view(ts, i):
    """The timestep as bot i sees it: its own observation (permitted keys only) and reward."""
    obs = {k: v for k, v in ts.observation[i].items() if k in PERMITTED_OBSERVATIONS}
    return ts._replace(observation=obs, reward=ts.reward[i])


def frame(ts, step, cleaning, eaten):
    world = ts.observation[0]["WORLD.RGB"]
    big = cv2.resize(world, (world.shape[1] * SCALE, world.shape[0] * SCALE),
                     interpolation=cv2.INTER_NEAREST)
    bar = np.full((BAR, big.shape[1], 3), 20, np.uint8)
    line1 = f"step {step}   cleaning now: {cleaning}   apples eaten so far: {eaten}"
    line2 = "3 bots trained to clean, 4 trained to eat"
    cv2.putText(bar, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    return np.concatenate([big, bar], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    roles = substrate.get_config("clean_up").default_player_roles
    assert len(roles) == len(BOTS)
    env = substrate.build("clean_up", roles=roles)
    policies = [bot.build(name) for name in BOTS]
    states = [p.initial_state() for p in policies]

    ts = env.reset()
    frames, eaten, t0 = [], 0, time.time()
    for step in range(args.steps):
        actions = []
        for i, p in enumerate(policies):
            a, states[i] = p.step(bot_view(ts, i), states[i])
            actions.append(int(a))
        ts = env.step(actions)
        eaten += int(round(sum(float(r) for r in ts.reward)))
        # A player who cleaned sees k-1 others cleaning, everyone else sees k: so k = the max.
        cleaning = int(max(int(o["NUM_OTHERS_WHO_CLEANED_THIS_STEP"]) for o in ts.observation))
        frames.append(frame(ts, step + 1, cleaning, eaten))
        if ts.last():
            print(f"round ended by itself at step {step + 1}")
            break
    print(f"{len(frames)} steps in {time.time() - t0:.1f}s, apples eaten {eaten}")

    h, w = frames[0].shape[:2]
    mp4 = os.path.join(args.out, "clean_up_bots.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    gif = os.path.join(args.out, "clean_up_bots.gif")
    imgs = [Image.fromarray(f) for f in frames[::2]]  # every other step keeps the GIF small
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(2000 / args.fps), loop=0)

    Image.fromarray(frames[min(300, len(frames) - 1)]).save(
        os.path.join(args.out, "clean_up_bots_frame.png"))
    for p in policies:
        p.close()
    env.close()
    print("wrote", mp4, gif)


if __name__ == "__main__":
    main()
