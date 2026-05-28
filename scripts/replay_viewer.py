#!/usr/bin/env python3
"""Replay viewer for recorded Monopoly episodes.

Modes:
  Interactive playback:
    python scripts/replay_viewer.py path/to/episode.rollout.gz

  Record a new episode and view it:
    python scripts/replay_viewer.py --record [--seed 42]

  Export to video (headless):
    python scripts/replay_viewer.py path/to/episode.rollout.gz --export mp4 --output runs/visualizations/demo/
    python scripts/replay_viewer.py path/to/episode.rollout.gz --export gif --fps 2

Keyboard shortcuts (interactive):
  Space       Play / Pause
  Left/Right  Step back / forward
  +/-         Speed up / slow down
  Q           Quit
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pygame
from pathlib import Path

from Monopoly.envs.rollout import Rollout, load_rollout, save_rollout
from Monopoly.envs.renderer import MonopolyRenderer
from Monopoly.board import Board
from Monopoly.property import load_property_specs


def build_overlay_data(rollout: Rollout, idx: int) -> dict:
    """Build overlay_data dict from rollout history up to current index."""
    step = rollout.steps[idx]
    history = [s.net_worths for s in rollout.steps[:idx + 1]]
    return {
        'net_worths': step.net_worths,
        'action_label': step.action_label,
        'net_worth_history': history,
        'turn_number': step.state.turn_number,
        'total_turns': rollout.metadata.get('max_turns', 1000),
    }


def play_interactive(rollout: Rollout):
    """Launch interactive playback window."""
    board = Board.load_standard_board()
    specs = load_property_specs()
    renderer = MonopolyRenderer(board, specs, headless=False)
    renderer.init_playback_ui()

    idx = 0
    total = len(rollout.steps)
    clock = pygame.time.Clock()
    accumulator = 0.0

    try:
        while True:
            controls = renderer.handle_playback_events()
            if controls['quit']:
                break

            if controls['step_back'] and idx > 0:
                idx -= 1
            if controls['step_forward'] and idx < total - 1:
                idx += 1
            if controls['seek_to'] is not None:
                idx = int(controls['seek_to'] * (total - 1))
                idx = max(0, min(idx, total - 1))

            if not controls['paused']:
                dt = clock.get_time() / 1000.0
                accumulator += dt
                step_interval = 1.0 / max(0.1, controls['speed'])
                while accumulator >= step_interval and idx < total - 1:
                    idx += 1
                    accumulator -= step_interval

            step = rollout.steps[idx]
            overlay = build_overlay_data(rollout, idx)
            renderer.render(step.state, show_stats=True, overlay_data=overlay)
            renderer._draw_control_bar(idx, total)
            pygame.display.flip()
            clock.tick(30)
    finally:
        renderer.close()


def export_video(rollout: Rollout, output_path: str, fmt: str = 'mp4', fps: int = 4):
    """Export rollout to mp4 or gif (headless)."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("ERROR: imageio is required for export. Install with:")
        print("  pip install imageio imageio-ffmpeg")
        sys.exit(1)

    board = Board.load_standard_board()
    specs = load_property_specs()
    renderer = MonopolyRenderer(board, specs, headless=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == 'mp4':
        writer = imageio.get_writer(str(out), fps=fps, codec='libx264',
                                    output_params=['-pix_fmt', 'yuv420p'])
    else:
        writer = imageio.get_writer(str(out), fps=fps, mode='I', loop=0)

    total = len(rollout.steps)
    for idx, step in enumerate(rollout.steps):
        overlay = build_overlay_data(rollout, idx)
        renderer.render(step.state, show_stats=True, overlay_data=overlay)
        frame = renderer.capture_frame()

        # For gif, downsample to half resolution
        if fmt == 'gif':
            import numpy as np
            frame = frame[::2, ::2, :]

        writer.append_data(frame)
        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  Exported {idx + 1}/{total} frames...")

    writer.close()
    renderer.close()
    print(f"Saved {fmt.upper()} to {out}")


def record_episode(seed: int = 42, max_turns: int = 500) -> Rollout:
    """Run a single episode with random policy and return the rollout."""
    from Monopoly.envs.gym_env import MonopolyEnv

    env = MonopolyEnv(
        num_players=4,
        seed=seed,
        max_turns=max_turns,
        record_rollout=True,
    )
    obs, info = env.reset()

    done = False
    while not done:
        # Use legal mask to pick a random legal action
        legal = obs['legal_mask']
        legal_indices = [i for i, v in enumerate(legal) if v == 1]
        if not legal_indices:
            action = 89  # end turn fallback
        else:
            import random
            action = random.choice(legal_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    rollout = env.current_rollout
    env.close()
    return rollout


def main():
    parser = argparse.ArgumentParser(description="Monopoly Replay Viewer")
    parser.add_argument('rollout_path', nargs='?', help='Path to .rollout.gz file')
    parser.add_argument('--record', action='store_true',
                        help='Record a new episode with random policy')
    parser.add_argument('--seed', type=int, default=42, help='Seed for recording')
    parser.add_argument('--max-turns', type=int, default=500, help='Max turns for recording')
    parser.add_argument('--export', choices=['mp4', 'gif'], help='Export format')
    parser.add_argument('--fps', type=int, default=4, help='Export FPS')
    parser.add_argument('--output', type=str, help='Output directory or file path')
    args = parser.parse_args()

    if not args.rollout_path and not args.record:
        parser.error("Provide a rollout path or use --record")

    # Record mode
    if args.record:
        print(f"Recording episode (seed={args.seed}, max_turns={args.max_turns})...")
        rollout = record_episode(seed=args.seed, max_turns=args.max_turns)
        print(f"Recorded {len(rollout.steps)} steps")

        # Save the rollout
        out_dir = Path(args.output) if args.output else Path("runs/visualizations/demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = out_dir / f"episode_seed{args.seed}.rollout.gz"
        save_rollout(rollout, rollout_path)
        print(f"Saved rollout to {rollout_path}")

        if args.export:
            ext = args.export
            export_path = out_dir / f"episode_seed{args.seed}.{ext}"
            export_video(rollout, str(export_path), fmt=ext, fps=args.fps)
        else:
            play_interactive(rollout)
        return

    # Load existing rollout
    rollout_path = Path(args.rollout_path)
    if not rollout_path.exists():
        print(f"ERROR: File not found: {rollout_path}")
        sys.exit(1)

    print(f"Loading rollout from {rollout_path}...")
    rollout = load_rollout(rollout_path)
    print(f"Loaded {len(rollout.steps)} steps")

    if args.export:
        out_dir = Path(args.output) if args.output else rollout_path.parent
        ext = args.export
        export_path = out_dir / f"{rollout_path.stem.replace('.rollout', '')}.{ext}"
        export_video(rollout, str(export_path), fmt=ext, fps=args.fps)
    else:
        play_interactive(rollout)


if __name__ == '__main__':
    main()
