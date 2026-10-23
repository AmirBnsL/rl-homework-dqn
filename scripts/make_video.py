#!/usr/bin/env python
"""Record the optional video: a random agent next to the best trained agent, on Pong and on
TreasureGrid, in the same clip.

    python scripts/make_video.py                        # both games if the artifacts exist
    python scripts/make_video.py --game treasure        # only the custom environment

Output: video.mp4 in the repository root (GIF fallback if ffmpeg is unavailable).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqnlib.envs import ENV_SPECS, make_env
from dqnlib.experiments import build_agent, load_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def best_run(env_key: str, variants, seeds=(0, 1, 2)):
    results = load_matrix(env_key, variants, seeds, RUNS)
    if not results:
        return None
    candidates = [(float(np.mean(run["eval_returns"])), run) for runs in results.values() for run in runs]
    return max(candidates, key=lambda item: item[0])[1]


def play(env, policy, seed: int, max_frames: int = 4000):
    state, info = env.reset(seed=seed)
    frames, total, done = [env.render()], 0.0, False
    while not done and len(frames) < max_frames:
        state, reward, terminated, truncated, info = env.step(policy(state))
        frames.append(env.render())
        total += float(reward)
        done = terminated or truncated
    return dict(frames=frames, return_=total, success=info.get("success"),
                steps=info.get("steps", len(frames) - 1))


def annotate(frame, title: str, subtitle: str, colour=(0, 0, 0)):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 34))
    draw.rectangle([0, 0, image.width, int(font.size * 2.8)], fill=(255, 255, 255))
    draw.text((8, 4), title, fill=colour, font=font)
    draw.text((8, 4 + int(font.size * 1.3)), subtitle, fill=(80, 80, 80), font=font)
    return np.asarray(image)


def side_by_side(env_key: str, seed: int, scale: int = 2, fps: int = 30) -> str | None:
    import imageio

    spec = ENV_SPECS[env_key]
    variants = ("dueling_double", "dueling", "double", "hard")
    run = best_run(env_key, variants)
    if run is None:
        print(f"skip {env_key}: no trained artifact")
        return None

    env = make_env(env_key, render_mode="rgb_array")
    agent = build_agent(env, env_key, run["variant"], seed=run["seed"])
    agent.load(run["pth_path"])
    rng = np.random.default_rng(0)

    random_episode = play(env, lambda s: int(rng.integers(env.action_space.n)), seed)
    greedy_episode = play(env, lambda s: agent.greedy_action(s), seed)
    env.close()

    frame_shape = random_episode["frames"][0].shape
    if len(frame_shape) == 2:  # grayscale 84x84 -> RGB, upscaled
        to_rgb = lambda f: np.repeat(f[:, :, None], 3, axis=2).repeat(scale, 0).repeat(scale, 1)
    else:
        to_rgb = lambda f: np.repeat(np.repeat(f, scale, 0), scale, 1)

    left = [to_rgb(f) for f in random_episode["frames"]]
    right = [to_rgb(f) for f in greedy_episode["frames"]]
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))

    separator = np.full((left[0].shape[0], 6, 3), 40, dtype=np.uint8)
    video = []
    for t in range(n):
        lframe = annotate(left[t], "random agent",
                          f"return {min(t, random_episode['steps'])}"
                          if env_key == "pong" else
                          f"steps {min(t, random_episode['steps'])}")
        rframe = annotate(right[t], f"trained agent ({run['variant']}, seed {run['seed']})",
                          f"return {min(t, greedy_episode['steps'])}"
                          if env_key == "pong" else
                          f"steps {min(t, greedy_episode['steps'])}", colour=(0, 110, 0))
        video.append(np.hstack([lframe, separator, rframe]))

    out = os.path.join(ROOT, f"video_{env_key}.mp4")
    imageio.mimsave(out, video, fps=fps, macro_block_size=1)
    print(f"wrote {out} | random return {random_episode['return_']:.1f} "
          f"vs trained {greedy_episode['return_']:.1f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="both", choices=["pong", "treasure", "both"])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    games = {"pong": "pong", "treasure": "treasure_v0"}
    chosen = games if args.game == "both" else {args.game: games[args.game]}
    outputs = [side_by_side(key, seed=100_000, scale=args.scale, fps=args.fps)
               for key in chosen.values()]
    outputs = [path for path in outputs if path]
    if len(outputs) > 1:  # concatenate the two clips
        import imageio.v2 as imageio
        clips = [imageio.mimread(path) for path in outputs]
        height = min(len(clip) for clip in clips)
        combined = [np.vstack([clip[i] for clip in clips]) for i in range(height)]
        imageio.mimsave(os.path.join(ROOT, "video.mp4"), combined, fps=args.fps, macro_block_size=1)
        print("wrote video.mp4 (both games)")


if __name__ == "__main__":
    main()
