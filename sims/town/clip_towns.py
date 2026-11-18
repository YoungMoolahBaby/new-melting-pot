"""Clip of the four towns side by side, played by DeepMind's trained Clean Up bots.

The bots were trained on normal Clean Up, so they don't react to badges or feeds;
this clip only shows what each town looks like. Run inside WSL from the repo root:
    ~/venvs/meltingpot/bin/python sims/town/clip_towns.py --steps 500
Writes out/town/towns.{gif,mp4} and out/town/towns_frame.png.
"""
import argparse
import os
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from meltingpot import bot  # noqa: E402
from meltingpot.scenario import PERMITTED_OBSERVATIONS  # noqa: E402

import town_config as tc  # noqa: E402

BOTS = ["clean_up__cleaner_0", "clean_up__cleaner_1", "clean_up__cleaner_0",
        "clean_up__consumer_0", "clean_up__consumer_1", "clean_up__consumer_0",
        "clean_up__consumer_1"]
TITLES = {
    "strangers": "1 Strangers: everyone looks the same",
    "neighbours": "2 Neighbours: everyone has their own colour",
    "connected": "3 Connecting AI: gold badge on recent cleaners",
    "isolating": "4 Isolating AI: purple feed tiles pay for staying home",
}
SCALE = 2
BAR = 44


class Town:
    def __init__(self, name, seed):
        self.name = name
        self.env = tc.build_town(name)
        self.policies = [bot.build(b) for b in BOTS]
        self.states = [p.initial_state() for p in self.policies]
        self.ts = self.env.reset()
        self.apples = 0
        self.feed = 0.0

    def step(self):
        acts = []
        for i, p in enumerate(self.policies):
            obs = {k: v for k, v in self.ts.observation[i].items() if k in PERMITTED_OBSERVATIONS}
            a, self.states[i] = p.step(self.ts._replace(observation=obs, reward=self.ts.reward[i]),
                                       self.states[i])
            acts.append(int(a))
        self.ts = self.env.step(acts)
        obs = self.ts.observation
        ate = sum(float(o["PLAYER_ATE_APPLE"]) for o in obs)
        self.apples += int(ate)
        self.feed += sum(float(r) for r in self.ts.reward) - ate
        cleaning = sum(1 for o in obs if float(o["PLAYER_CLEANED"]) > 0)
        return cleaning

    def panel(self, step, cleaning):
        world = self.ts.observation[0]["WORLD.RGB"]
        big = cv2.resize(world, (world.shape[1] * SCALE, world.shape[0] * SCALE),
                         interpolation=cv2.INTER_NEAREST)
        bar = np.full((BAR, big.shape[1], 3), 20, np.uint8)
        line2 = f"step {step}  cleaning now: {cleaning}  apples: {self.apples}"
        if self.name == "isolating":
            line2 += f"  feed: {self.feed:.1f}"
        cv2.putText(bar, TITLES[self.name], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, line2, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1,
                    cv2.LINE_AA)
        return np.concatenate([big, bar], axis=0)

    def close(self):
        for p in self.policies:
            p.close()
        self.env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="out/town")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    towns = [Town(name, seed=i) for i, name in enumerate(tc.TOWNS)]
    frames = []
    for step in range(1, args.steps + 1):
        panels = [t.panel(step, t.step()) for t in towns]
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        top = np.concatenate([panels[0], gap, panels[1]], axis=1)
        bottom = np.concatenate([panels[2], gap, panels[3]], axis=1)
        hgap = np.full((6, top.shape[1], 3), 255, np.uint8)
        frames.append(np.concatenate([top, hgap, bottom], axis=0))
    for t in towns:
        print(f"{t.name}: apples {t.apples}, feed {t.feed:.1f}")
        t.close()

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(os.path.join(args.out, "towns.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    imgs = [Image.fromarray(f) for f in frames[::2]]
    imgs[0].save(os.path.join(args.out, "towns.gif"), save_all=True, append_images=imgs[1:],
                 duration=int(2000 / args.fps), loop=0)
    Image.fromarray(frames[min(250, len(frames) - 1)]).save(
        os.path.join(args.out, "towns_frame.png"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
