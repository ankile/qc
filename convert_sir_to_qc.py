#!/usr/bin/env python3
"""Convert self-improving-robots LeRobot datasets to QC-compatible HDF5.

Usage:
    python convert_sir_to_qc.py --output ~/.robomimic/sir_square/combined/low_dim.hdf5

This script:
1. Loads all specified LeRobot datasets from HuggingFace
2. Extracts transitions (states, actions, rewards, dones, next_states)
3. Groups by episode
4. Saves to HDF5 in robomimic-compatible format
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Default datasets to convert
DEFAULT_REPO_IDS = [
    "ankile/square-teleop-v5",
    "ankile/square-dagger-state-seq2-r1",
    "ankile/square-dagger-state-seq2-r2",
    "ankile/square-dagger-state-seq2-r3",
    "ankile/square-dagger-state-seq2-r4",
    "ankile/square-dagger-state-seq2-r5",
    "ankile/square-dagger-state-seq2-r6",
    "ankile/square-dagger-state-seq2-r7",
    "ankile/square-dagger-state-seq2-r8",
]


def load_lerobot_dataset(repo_id: str) -> dict:
    """Load a single LeRobot dataset and extract transitions.

    Args:
        repo_id: HuggingFace repo ID (e.g., "ankile/square-teleop-v5")

    Returns:
        Dict with arrays: states, actions, next_states, rewards, dones, episode_indices
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    table = dataset.hf_dataset.data

    # Extract columns via PyArrow (fast vectorized access)
    obs_state_raw = table.column("observation.state").to_numpy(zero_copy_only=False)
    obs_env_raw = table.column("observation.environment_state").to_numpy(zero_copy_only=False)
    actions_raw = table.column("action").to_numpy(zero_copy_only=False)

    # Stack into 2D arrays
    obs_state = np.stack(obs_state_raw).astype(np.float32)
    obs_env = np.stack(obs_env_raw).astype(np.float32)
    actions = np.stack(actions_raw).astype(np.float32)

    # CRITICAL FIX: Reorder environment_state to match robomimic 1.4.x convention
    # Robosuite 1.5.x order: [rel_pos(3), rel_quat(4), abs_pos(3), abs_quat(4)]
    # Robomimic 1.4.x order: [abs_pos(3), abs_quat(4), rel_pos(3), rel_quat(4)]
    obs_env_reordered = np.concatenate([
        obs_env[:, 7:10],   # abs_pos -> dims 0:3
        obs_env[:, 10:14],  # abs_quat -> dims 3:7
        obs_env[:, 0:3],    # rel_pos -> dims 7:10
        obs_env[:, 3:7],    # rel_quat -> dims 10:14
    ], axis=1)
    obs_env = obs_env_reordered
    print(f"  Reordered environment_state: [rel,abs] -> [abs,rel] (robomimic 1.4.x compat)")

    # Extract scalar columns
    rewards = table.column("reward").to_numpy(zero_copy_only=False).astype(np.float32)
    dones = table.column("done").to_numpy(zero_copy_only=False).astype(np.float32)
    is_valid = table.column("is_valid").to_numpy(zero_copy_only=False).astype(np.int64)
    episode_indices = table.column("episode_index").to_numpy(zero_copy_only=False).astype(np.int64)

    # Concatenate robot state and environment state
    states = np.concatenate([obs_state, obs_env], axis=1)

    # Compute next_states via rolling (padded frame format ensures correct terminal states)
    next_states = np.roll(states, -1, axis=0)

    # Filter to valid transitions only (is_valid=1)
    valid_mask = is_valid == 1
    n_total = len(states)
    n_valid = valid_mask.sum()
    print(f"  Filtered: {n_total:,} -> {n_valid:,} valid transitions")

    return {
        'states': states[valid_mask],
        'actions': actions[valid_mask],
        'next_states': next_states[valid_mask],
        'rewards': rewards[valid_mask],
        'dones': dones[valid_mask],
        'episode_indices': episode_indices[valid_mask],
    }


def split_into_episodes(data: dict) -> list[dict]:
    """Split flat arrays into per-episode dicts.

    Args:
        data: Dict with arrays (states, actions, etc.) and episode_indices

    Returns:
        List of dicts, one per episode
    """
    episodes = []
    unique_episodes = np.unique(data['episode_indices'])

    for ep_idx in unique_episodes:
        mask = data['episode_indices'] == ep_idx
        episodes.append({
            'observations': data['states'][mask],
            'actions': data['actions'][mask],
            'next_observations': data['next_states'][mask],
            'rewards': data['rewards'][mask],
            'dones': data['dones'][mask],
        })

    return episodes


def save_to_hdf5(episodes: list[dict], output_path: str):
    """Save episodes to robomimic-style HDF5.

    Structure:
        data/
            demo_0/
                obs/
                    state  (T, state_dim)
                next_obs/
                    state  (T, state_dim)
                actions    (T, action_dim)
                rewards    (T,)
                dones      (T,)
            demo_1/
                ...
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(episodes)} episodes to {output_path}")

    with h5py.File(output_path, 'w') as f:
        data_grp = f.create_group('data')

        total_timesteps = 0
        for i, ep in enumerate(tqdm(episodes, desc="Saving episodes")):
            demo_grp = data_grp.create_group(f'demo_{i}')

            # Store observations
            obs_grp = demo_grp.create_group('obs')
            obs_grp.create_dataset('state', data=ep['observations'], compression='gzip')

            # Store next observations
            next_obs_grp = demo_grp.create_group('next_obs')
            next_obs_grp.create_dataset('state', data=ep['next_observations'], compression='gzip')

            # Store actions, rewards, dones
            demo_grp.create_dataset('actions', data=ep['actions'], compression='gzip')
            demo_grp.create_dataset('rewards', data=ep['rewards'], compression='gzip')
            demo_grp.create_dataset('dones', data=ep['dones'], compression='gzip')

            total_timesteps += len(ep['actions'])

        # Store metadata
        f.attrs['total_timesteps'] = total_timesteps
        f.attrs['num_demos'] = len(episodes)

    print(f"Saved {total_timesteps:,} total timesteps across {len(episodes)} episodes")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot datasets to QC format")
    parser.add_argument(
        "--repo_ids",
        type=str,
        default=",".join(DEFAULT_REPO_IDS),
        help="Comma-separated list of HuggingFace repo IDs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.expanduser("~/.robomimic/sir_square/combined/low_dim.hdf5"),
        help="Output HDF5 file path"
    )
    args = parser.parse_args()

    repo_ids = [r.strip() for r in args.repo_ids.split(",")]

    print(f"Converting {len(repo_ids)} datasets:")
    for repo_id in repo_ids:
        print(f"  - {repo_id}")
    print()

    # Load all datasets
    all_episodes = []
    for repo_id in repo_ids:
        data = load_lerobot_dataset(repo_id)
        episodes = split_into_episodes(data)
        print(f"  -> {len(episodes)} episodes")
        all_episodes.extend(episodes)
        print()

    print(f"Total: {len(all_episodes)} episodes")

    # Save to HDF5
    save_to_hdf5(all_episodes, args.output)
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
