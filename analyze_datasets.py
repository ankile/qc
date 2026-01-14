#!/usr/bin/env python3
"""
Analyze and compare QC/RoboMimic vs SIR datasets.

Usage:
    python analyze_datasets.py
"""

import h5py
import numpy as np
import os
from pathlib import Path

# Paths
QC_PATH = os.path.expanduser("~/.robomimic/square/mh/low_dim_v15.hdf5")
SIR_PATH = os.path.expanduser("~/.robomimic/sir_square/combined/low_dim.hdf5")

# QC observation keys (in order they're concatenated)
QC_OBS_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']


def load_qc_data(path=QC_PATH, max_demos=None):
    """Load QC/RoboMimic data."""
    with h5py.File(path, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        if max_demos:
            demos = demos[:max_demos]

        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        episode_starts = [0]

        for demo in demos:
            # Concatenate observations in order
            obs_parts = [np.array(f[f'data/{demo}/obs/{k}']) for k in QC_OBS_KEYS]
            obs = np.concatenate(obs_parts, axis=-1)

            next_obs_parts = [np.array(f[f'data/{demo}/next_obs/{k}']) for k in QC_OBS_KEYS]
            next_obs = np.concatenate(next_obs_parts, axis=-1)

            actions = np.array(f[f'data/{demo}/actions'])
            rewards = np.array(f[f'data/{demo}/rewards'])

            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_obs.append(next_obs)
            episode_starts.append(episode_starts[-1] + len(obs))

        return {
            'observations': np.concatenate(all_obs).astype(np.float32),
            'actions': np.concatenate(all_actions).astype(np.float32),
            'rewards': np.concatenate(all_rewards).astype(np.float32),
            'next_observations': np.concatenate(all_next_obs).astype(np.float32),
            'episode_starts': np.array(episode_starts),
            'num_demos': len(demos),
        }


def load_sir_data(path=SIR_PATH, max_demos=None):
    """Load SIR converted data."""
    with h5py.File(path, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        if max_demos:
            demos = demos[:max_demos]

        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        episode_starts = [0]

        for demo in demos:
            obs = np.array(f[f'data/{demo}/obs/state'])
            next_obs = np.array(f[f'data/{demo}/next_obs/state'])
            actions = np.array(f[f'data/{demo}/actions'])
            rewards = np.array(f[f'data/{demo}/rewards'])

            all_obs.append(obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_obs.append(next_obs)
            episode_starts.append(episode_starts[-1] + len(obs))

        return {
            'observations': np.concatenate(all_obs).astype(np.float32),
            'actions': np.concatenate(all_actions).astype(np.float32),
            'rewards': np.concatenate(all_rewards).astype(np.float32),
            'next_observations': np.concatenate(all_next_obs).astype(np.float32),
            'episode_starts': np.array(episode_starts),
            'num_demos': len(demos),
        }


def print_stats(name, data, key):
    """Print statistics for a data array."""
    arr = data[key]
    print(f"\n{name} {key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min:  {arr.min(axis=0)}")
    print(f"  Max:  {arr.max(axis=0)}")
    print(f"  Mean: {arr.mean(axis=0)}")
    print(f"  Std:  {arr.std(axis=0)}")


def compare_single_demo(qc_data, sir_data, qc_demo_idx=0, sir_demo_idx=0):
    """Compare a single demo from each dataset."""
    print("\n" + "="*60)
    print("SINGLE DEMO COMPARISON")
    print("="*60)

    # Get episode slices
    qc_start = qc_data['episode_starts'][qc_demo_idx]
    qc_end = qc_data['episode_starts'][qc_demo_idx + 1]

    sir_start = sir_data['episode_starts'][sir_demo_idx]
    sir_end = sir_data['episode_starts'][sir_demo_idx + 1]

    print(f"\nQC demo {qc_demo_idx}: {qc_end - qc_start} timesteps")
    print(f"SIR demo {sir_demo_idx}: {sir_end - sir_start} timesteps")

    # First observation
    qc_obs0 = qc_data['observations'][qc_start]
    sir_obs0 = sir_data['observations'][sir_start]

    print(f"\nFirst observation (QC):  {qc_obs0}")
    print(f"First observation (SIR): {sir_obs0}")

    # First action
    qc_act0 = qc_data['actions'][qc_start]
    sir_act0 = sir_data['actions'][sir_start]

    print(f"\nFirst action (QC):  {qc_act0}")
    print(f"First action (SIR): {sir_act0}")

    # Observation breakdown
    print("\n--- Observation breakdown (first timestep) ---")
    print("Dim | Semantic            | QC Value   | SIR Value")
    print("-" * 55)
    semantics = [
        "eef_pos_x", "eef_pos_y", "eef_pos_z",
        "eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w",
        "gripper_left", "gripper_right",
        "obj_rel_x", "obj_rel_y", "obj_rel_z",
        "obj_rel_qx", "obj_rel_qy", "obj_rel_qz", "obj_rel_qw",
        "obj_abs_x", "obj_abs_y", "obj_abs_z",
        "obj_abs_qx", "obj_abs_qy", "obj_abs_qz", "obj_abs_qw",
    ]
    for i, sem in enumerate(semantics):
        print(f"{i:3d} | {sem:19s} | {qc_obs0[i]:10.4f} | {sir_obs0[i]:10.4f}")


def check_observation_ordering():
    """Check if observation ordering might be different."""
    print("\n" + "="*60)
    print("OBSERVATION ORDERING CHECK")
    print("="*60)

    print("\nQC concatenation order (from robomimic_utils.py):")
    print("  robot0_eef_pos (3) + robot0_eef_quat (4) + robot0_gripper_qpos (2) + object (14)")
    print("  = [0:3] eef_pos, [3:7] eef_quat, [7:9] gripper, [9:23] object")

    print("\nSIR concatenation order (from convert_sir_to_qc.py):")
    print("  observation.state (9) + observation.environment_state (14)")
    print("  = [0:9] robot state, [9:23] environment state")

    print("\nExpected SIR observation.state (9D) contents:")
    print("  eef_pos (3) + eef_quat (4) + gripper_qpos (2)")

    print("\nExpected SIR observation.environment_state (14D) contents:")
    print("  nut_to_eef_pos (3) + nut_to_eef_quat (4) + nut_pos (3) + nut_quat (4)")


def analyze_rewards(qc_data, sir_data):
    """Analyze reward structures."""
    print("\n" + "="*60)
    print("REWARD ANALYSIS")
    print("="*60)

    qc_r = qc_data['rewards']
    sir_r = sir_data['rewards']

    print(f"\nQC rewards:")
    print(f"  Unique values: {np.unique(qc_r)}")
    print(f"  Non-zero count: {(qc_r > 0).sum()} / {len(qc_r)}")

    print(f"\nSIR rewards:")
    print(f"  Unique values: {np.unique(sir_r)}")
    print(f"  Non-zero count: {(sir_r > 0).sum()} / {len(sir_r)}")


def analyze_actions_detailed(qc_data, sir_data):
    """Detailed action analysis."""
    print("\n" + "="*60)
    print("DETAILED ACTION ANALYSIS")
    print("="*60)

    qc_a = qc_data['actions']
    sir_a = sir_data['actions']

    semantics = [
        "delta_eef_x", "delta_eef_y", "delta_eef_z",
        "delta_rot_x", "delta_rot_y", "delta_rot_z",
        "gripper"
    ]

    print("\nPer-dimension statistics:")
    print("Dim | Semantic      | QC [min, max]        | SIR [min, max]       | QC std  | SIR std")
    print("-" * 95)
    for i, sem in enumerate(semantics):
        qc_min, qc_max = qc_a[:, i].min(), qc_a[:, i].max()
        sir_min, sir_max = sir_a[:, i].min(), sir_a[:, i].max()
        qc_std = qc_a[:, i].std()
        sir_std = sir_a[:, i].std()
        print(f"{i:3d} | {sem:13s} | [{qc_min:7.3f}, {qc_max:7.3f}] | [{sir_min:7.3f}, {sir_max:7.3f}] | {qc_std:7.4f} | {sir_std:7.4f}")


def check_quaternion_conventions(qc_data, sir_data):
    """Check quaternion conventions (wxyz vs xyzw)."""
    print("\n" + "="*60)
    print("QUATERNION CONVENTION CHECK")
    print("="*60)

    # Get first few quaternions from each dataset
    qc_quats = qc_data['observations'][:10, 3:7]  # eef quaternion
    sir_quats = sir_data['observations'][:10, 3:7]

    print("\nFirst 5 EEF quaternions (dims 3-6):")
    print("QC (expected xyzw):")
    for i in range(5):
        q = qc_quats[i]
        norm = np.linalg.norm(q)
        print(f"  {i}: {q} (norm={norm:.4f})")

    print("\nSIR:")
    for i in range(5):
        q = sir_quats[i]
        norm = np.linalg.norm(q)
        print(f"  {i}: {q} (norm={norm:.4f})")

    # Check object quaternions too
    qc_obj_quats = qc_data['observations'][:10, 12:16]  # First object quat (relative)
    sir_obj_quats = sir_data['observations'][:10, 12:16]

    print("\nFirst 5 object relative quaternions (dims 12-15):")
    print("QC:")
    for i in range(5):
        q = qc_obj_quats[i]
        norm = np.linalg.norm(q)
        print(f"  {i}: {q} (norm={norm:.4f})")

    print("\nSIR:")
    for i in range(5):
        q = sir_obj_quats[i]
        norm = np.linalg.norm(q)
        print(f"  {i}: {q} (norm={norm:.4f})")


def check_object_state_ordering(qc_data, sir_data):
    """Check if object state ordering might be different."""
    print("\n" + "="*60)
    print("OBJECT STATE ORDERING CHECK")
    print("="*60)

    # Get first episode
    qc_obs = qc_data['observations'][:100]  # First 100 timesteps
    sir_obs = sir_data['observations'][:100]

    # Extract object parts (dims 9-22)
    qc_obj = qc_obs[:, 9:23]
    sir_obj = sir_obs[:, 9:23]

    print("\nVariance analysis (high variance = changes during episode):")
    print("\nQC object state [9:23] variances:")
    print(f"  [9:12]  pos1:  {qc_obj[:, 0:3].var(axis=0).round(4)}")
    print(f"  [12:16] quat1: {qc_obj[:, 3:7].var(axis=0).round(4)}")
    print(f"  [16:19] pos2:  {qc_obj[:, 7:10].var(axis=0).round(4)}")
    print(f"  [19:23] quat2: {qc_obj[:, 10:14].var(axis=0).round(4)}")

    print("\nSIR object state [9:23] variances:")
    print(f"  [9:12]  pos1:  {sir_obj[:, 0:3].var(axis=0).round(4)}")
    print(f"  [12:16] quat1: {sir_obj[:, 3:7].var(axis=0).round(4)}")
    print(f"  [16:19] pos2:  {sir_obj[:, 7:10].var(axis=0).round(4)}")
    print(f"  [19:23] quat2: {sir_obj[:, 10:14].var(axis=0).round(4)}")

    print("\nMean values:")
    print("\nQC object state means:")
    print(f"  [9:12]  pos1:  {qc_obj[:, 0:3].mean(axis=0).round(4)}")
    print(f"  [12:16] quat1: {qc_obj[:, 3:7].mean(axis=0).round(4)}")
    print(f"  [16:19] pos2:  {qc_obj[:, 7:10].mean(axis=0).round(4)}")
    print(f"  [19:23] quat2: {qc_obj[:, 10:14].mean(axis=0).round(4)}")

    print("\nSIR object state means:")
    print(f"  [9:12]  pos1:  {sir_obj[:, 0:3].mean(axis=0).round(4)}")
    print(f"  [12:16] quat1: {sir_obj[:, 3:7].mean(axis=0).round(4)}")
    print(f"  [16:19] pos2:  {sir_obj[:, 7:10].mean(axis=0).round(4)}")
    print(f"  [19:23] quat2: {sir_obj[:, 10:14].mean(axis=0).round(4)}")

    print("\n--- Key observation ---")
    print("QC pos2 z-mean: {:.3f} (low = probably world-relative nut position)".format(
        qc_obj[:, 7:10].mean(axis=0)[2]))
    print("SIR pos2 z-mean: {:.3f} (high = probably world-relative nut position)".format(
        sir_obj[:, 7:10].mean(axis=0)[2]))
    print("\nThe z-coordinate difference suggests different coordinate systems!")


def main():
    print("="*60)
    print("QC vs SIR Dataset Analysis")
    print("="*60)

    # Check files exist
    if not os.path.exists(QC_PATH):
        print(f"ERROR: QC data not found at {QC_PATH}")
        return
    if not os.path.exists(SIR_PATH):
        print(f"ERROR: SIR data not found at {SIR_PATH}")
        return

    print(f"\nLoading QC data from: {QC_PATH}")
    qc_data = load_qc_data()
    print(f"  Demos: {qc_data['num_demos']}, Transitions: {len(qc_data['observations'])}")

    print(f"\nLoading SIR data from: {SIR_PATH}")
    sir_data = load_sir_data()
    print(f"  Demos: {sir_data['num_demos']}, Transitions: {len(sir_data['observations'])}")

    # Basic stats
    print_stats("QC", qc_data, 'observations')
    print_stats("SIR", sir_data, 'observations')

    print_stats("QC", qc_data, 'actions')
    print_stats("SIR", sir_data, 'actions')

    # Detailed analyses
    check_observation_ordering()
    compare_single_demo(qc_data, sir_data)
    analyze_rewards(qc_data, sir_data)
    analyze_actions_detailed(qc_data, sir_data)
    check_quaternion_conventions(qc_data, sir_data)
    check_object_state_ordering(qc_data, sir_data)

    print("\n" + "="*60)
    print("ROOT CAUSE IDENTIFIED")
    print("="*60)
    print("""
The OBJECT OBSERVATION ORDER is SWAPPED between QC and SIR!

QC order (robomimic 1.4.1):   [abs_pos, abs_quat, ???, ???]
SIR order (robosuite 1.5.x):  [rel_pos, rel_quat, abs_pos, abs_quat]

Evidence:
- QC obj[0:3] z-mean ≈ 0.88 (HIGH = absolute position)
- SIR obj[0:3] z-mean ≈ 0.05 (LOW = relative position)
- QC obj[7:10] z-mean ≈ 0.05 (LOW)
- SIR obj[7:10] z-mean ≈ 0.83 (HIGH = absolute position)

THE FIX: Reorder SIR environment_state from [rel, abs] to [abs, rel]

To interactively explore:
    from analyze_datasets import load_qc_data, load_sir_data
    qc = load_qc_data()
    sir = load_sir_data()

    # Compare object states
    qc['observations'][50, 9:23]   # QC object state
    sir['observations'][50, 9:23]  # SIR object state
""")

    return qc_data, sir_data


if __name__ == "__main__":
    main()
