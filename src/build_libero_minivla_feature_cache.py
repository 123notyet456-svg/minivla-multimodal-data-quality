from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from lerobot.datasets import LeRobotDataset


# ============================================================
# Configuration
# ============================================================

REPO_ID = "lerobot/libero_10"

MAIN_CAMERA_KEY = "observation.images.image"
WRIST_CAMERA_KEY = "observation.images.wrist_image"

STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task_index"
EPISODE_KEY = "episode_index"
FRAME_KEY = "frame_index"

CACHE_DIR = Path("cache") / "libero_minivla"
REPORT_DIR = Path("reports")

MAIN_FEATURE_FILE = CACHE_DIR / "main_vision_features.npy"
WRIST_FEATURE_FILE = CACHE_DIR / "wrist_vision_features.npy"

STATE_FILE = CACHE_DIR / "state.npy"
ACTION_FILE = CACHE_DIR / "action.npy"
TASK_FILE = CACHE_DIR / "task_index.npy"
EPISODE_FILE = CACHE_DIR / "episode_index.npy"
FRAME_FILE = CACHE_DIR / "frame_index.npy"

METADATA_FILE = CACHE_DIR / "metadata.json"
REPORT_FILE = REPORT_DIR / "libero_minivla_feature_cache_summary.csv"


# ============================================================
# Utility
# ============================================================

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def prepare_image_batch(images: torch.Tensor) -> torch.Tensor:
    """
    Convert image batch to:
        shape: [B, 3, H, W]
        dtype: float32
        range: approximately [0, 1]
    """

    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)

    if images.ndim != 4:
        raise ValueError(
            f"Expected 4D image batch, got shape={tuple(images.shape)}"
        )

    # BCHW
    if images.shape[1] in (1, 3, 4):
        pass

    # BHWC -> BCHW
    elif images.shape[-1] in (1, 3, 4):
        images = images.permute(0, 3, 1, 2)

    else:
        raise ValueError(
            f"Cannot determine channel dimension from shape={tuple(images.shape)}"
        )

    # RGB only
    if images.shape[1] == 4:
        images = images[:, :3]

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    else:
        images = images.float()

        max_value = float(images.max().item())

        if max_value > 1.5:
            images = images / 255.0

    images = torch.clamp(images, 0.0, 1.0)

    return images


def tensor_to_numpy(
    value,
    dtype=None,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if dtype is not None:
        value = value.astype(dtype)

    return value


def get_1d_index(batch: dict, key: str) -> np.ndarray:
    if key not in batch:
        raise KeyError(
            f"Missing key '{key}'. Available keys:\n"
            + "\n".join(sorted(batch.keys()))
        )

    value = tensor_to_numpy(batch[key])

    value = np.asarray(value).reshape(-1)

    return value.astype(np.int64)


# ============================================================
# Vision Encoder
# ============================================================

class MobileNetFeatureEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT

        backbone = mobilenet_v3_small(weights=weights)

        self.preprocess = weights.transforms()

        self.features = backbone.features
        self.pool = backbone.avgpool
        self.flatten = nn.Flatten(1)

        # MobileNetV3-Small pooled feature dimension
        self.feature_dim = 576

        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:

        images = prepare_image_batch(images)

        images = self.preprocess(images)

        x = self.features(images)
        x = self.pool(x)
        x = self.flatten(x)

        return x


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Feature extraction batch size.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for debugging.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing feature cache.",
    )

    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    expected_files = [
        MAIN_FEATURE_FILE,
        WRIST_FEATURE_FILE,
        STATE_FILE,
        ACTION_FILE,
        TASK_FILE,
        EPISODE_FILE,
        FRAME_FILE,
        METADATA_FILE,
    ]

    existing = [p for p in expected_files if p.exists()]

    if existing and not args.overwrite:
        print("=" * 80)
        print("Existing Mini-VLA cache detected.")
        print("=" * 80)

        for path in existing:
            print(path)

        print()
        print(
            "Use --overwrite if you intentionally want to rebuild the cache."
        )
        return

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = choose_device()

    print("=" * 80)
    print("LIBERO Mini-VLA Vision Feature Cache")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("[1/6] Loading LIBERO-10 dataset...")

    dataset = LeRobotDataset(REPO_ID)

    total_frames = len(dataset)

    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    print(f"Dataset: {REPO_ID}")
    print(f"Frames to process: {total_frames:,}")
    print("Dataset Loading Check: PASS ✅")
    print()

    # --------------------------------------------------------
    # First sample inspection
    # --------------------------------------------------------

    print("[2/6] Inspecting multimodal inputs...")

    sample = dataset[0]

    required_keys = [
        MAIN_CAMERA_KEY,
        WRIST_CAMERA_KEY,
        STATE_KEY,
        ACTION_KEY,
        TASK_KEY,
        EPISODE_KEY,
        FRAME_KEY,
    ]

    missing_keys = [
        key for key in required_keys
        if key not in sample
    ]

    if missing_keys:
        print("Available keys:")
        for key in sorted(sample.keys()):
            print(f"  {key}")

        raise KeyError(
            f"Missing required keys: {missing_keys}"
        )

    main_image = sample[MAIN_CAMERA_KEY]
    wrist_image = sample[WRIST_CAMERA_KEY]
    state = sample[STATE_KEY]
    action = sample[ACTION_KEY]

    print(
        f"Main Camera Shape:  {tuple(main_image.shape)} "
        f"dtype={main_image.dtype}"
    )
    print(
        f"Wrist Camera Shape: {tuple(wrist_image.shape)} "
        f"dtype={wrist_image.dtype}"
    )
    print(f"State Shape:  {tuple(state.shape)}")
    print(f"Action Shape: {tuple(action.shape)}")

    state_dim = int(np.prod(state.shape))
    action_dim = int(np.prod(action.shape))

    if state_dim != 8:
        raise ValueError(
            f"Expected State Dim=8, got {state_dim}"
        )

    if action_dim != 7:
        raise ValueError(
            f"Expected Action Dim=7, got {action_dim}"
        )

    print("Multimodal Input Shape Check: PASS ✅")
    print()

    # --------------------------------------------------------
    # Vision encoder
    # --------------------------------------------------------

    print("[3/6] Loading frozen MobileNetV3-Small...")

    encoder = MobileNetFeatureEncoder().to(device)
    encoder.eval()

    print("Encoder: MobileNetV3-Small")
    print("Weights: ImageNet pretrained")
    print("Encoder Status: Frozen")
    print(f"Feature Dim: {encoder.feature_dim}")
    print("Vision Encoder Check: PASS ✅")
    print()

    # --------------------------------------------------------
    # Create cache arrays
    # --------------------------------------------------------

    print("[4/6] Creating cache arrays...")

    n = total_frames
    feature_dim = encoder.feature_dim

    main_features = np.lib.format.open_memmap(
        MAIN_FEATURE_FILE,
        mode="w+",
        dtype=np.float16,
        shape=(n, feature_dim),
    )

    wrist_features = np.lib.format.open_memmap(
        WRIST_FEATURE_FILE,
        mode="w+",
        dtype=np.float16,
        shape=(n, feature_dim),
    )

    states = np.lib.format.open_memmap(
        STATE_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(n, state_dim),
    )

    actions = np.lib.format.open_memmap(
        ACTION_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(n, action_dim),
    )

    tasks = np.lib.format.open_memmap(
        TASK_FILE,
        mode="w+",
        dtype=np.int64,
        shape=(n,),
    )

    episodes = np.lib.format.open_memmap(
        EPISODE_FILE,
        mode="w+",
        dtype=np.int64,
        shape=(n,),
    )

    frames = np.lib.format.open_memmap(
        FRAME_FILE,
        mode="w+",
        dtype=np.int64,
        shape=(n,),
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("[5/6] Extracting vision features...")
    print()

    write_pos = 0

    with torch.inference_mode():

        for batch_idx, batch in enumerate(loader):

            if write_pos >= n:
                break

            main_batch = batch[MAIN_CAMERA_KEY]
            wrist_batch = batch[WRIST_CAMERA_KEY]

            current_batch_size = main_batch.shape[0]

            remaining = n - write_pos

            if current_batch_size > remaining:
                current_batch_size = remaining

                main_batch = main_batch[:remaining]
                wrist_batch = wrist_batch[:remaining]

            main_batch = main_batch.to(device)
            wrist_batch = wrist_batch.to(device)

            main_feat = encoder(main_batch)
            wrist_feat = encoder(wrist_batch)

            main_feat_np = (
                main_feat.detach()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            wrist_feat_np = (
                wrist_feat.detach()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            state_np = tensor_to_numpy(
                batch[STATE_KEY][:current_batch_size],
                np.float32,
            ).reshape(current_batch_size, -1)

            action_np = tensor_to_numpy(
                batch[ACTION_KEY][:current_batch_size],
                np.float32,
            ).reshape(current_batch_size, -1)

            task_np = get_1d_index(
                batch,
                TASK_KEY,
            )[:current_batch_size]

            episode_np = get_1d_index(
                batch,
                EPISODE_KEY,
            )[:current_batch_size]

            frame_np = get_1d_index(
                batch,
                FRAME_KEY,
            )[:current_batch_size]

            start = write_pos
            end = write_pos + current_batch_size

            main_features[start:end] = main_feat_np
            wrist_features[start:end] = wrist_feat_np

            states[start:end] = state_np
            actions[start:end] = action_np

            tasks[start:end] = task_np
            episodes[start:end] = episode_np
            frames[start:end] = frame_np

            write_pos = end

            if (
                batch_idx == 0
                or write_pos % 1000 < current_batch_size
                or write_pos == n
            ):
                progress = 100.0 * write_pos / n

                print(
                    f"Processed "
                    f"{write_pos:,}/{n:,} "
                    f"({progress:.2f}%)"
                )

    # --------------------------------------------------------
    # Flush
    # --------------------------------------------------------

    main_features.flush()
    wrist_features.flush()
    states.flush()
    actions.flush()
    tasks.flush()
    episodes.flush()
    frames.flush()

    del main_features
    del wrist_features
    del states
    del actions
    del tasks
    del episodes
    del frames

    print()
    print("Feature Extraction: PASS ✅")
    print()

    # --------------------------------------------------------
    # Reload and QC
    # --------------------------------------------------------

    print("[6/6] Running cache QC...")

    main_features = np.load(
        MAIN_FEATURE_FILE,
        mmap_mode="r",
    )

    wrist_features = np.load(
        WRIST_FEATURE_FILE,
        mmap_mode="r",
    )

    states = np.load(
        STATE_FILE,
        mmap_mode="r",
    )

    actions = np.load(
        ACTION_FILE,
        mmap_mode="r",
    )

    tasks = np.load(
        TASK_FILE,
        mmap_mode="r",
    )

    episodes = np.load(
        EPISODE_FILE,
        mmap_mode="r",
    )

    frames = np.load(
        FRAME_FILE,
        mmap_mode="r",
    )

    checks = {}

    checks["frame_count_check"] = (
        main_features.shape[0] == n
        and wrist_features.shape[0] == n
        and states.shape[0] == n
        and actions.shape[0] == n
    )

    checks["main_feature_shape_check"] = (
        main_features.shape == (n, feature_dim)
    )

    checks["wrist_feature_shape_check"] = (
        wrist_features.shape == (n, feature_dim)
    )

    checks["state_shape_check"] = (
        states.shape == (n, 8)
    )

    checks["action_shape_check"] = (
        actions.shape == (n, 7)
    )

    # Sample-based finite checks keep memory use low
    qc_count = min(n, 5000)

    qc_indices = np.linspace(
        0,
        n - 1,
        qc_count,
        dtype=np.int64,
    )

    main_qc = np.asarray(
        main_features[qc_indices],
        dtype=np.float32,
    )

    wrist_qc = np.asarray(
        wrist_features[qc_indices],
        dtype=np.float32,
    )

    state_qc = np.asarray(
        states[qc_indices],
        dtype=np.float32,
    )

    action_qc = np.asarray(
        actions[qc_indices],
        dtype=np.float32,
    )

    checks["main_nan_inf_check"] = (
        np.isfinite(main_qc).all()
    )

    checks["wrist_nan_inf_check"] = (
        np.isfinite(wrist_qc).all()
    )

    checks["state_nan_inf_check"] = (
        np.isfinite(state_qc).all()
    )

    checks["action_nan_inf_check"] = (
        np.isfinite(action_qc).all()
    )

    main_std = float(main_qc.std())
    wrist_std = float(wrist_qc.std())

    checks["main_feature_variation_check"] = (
        main_std > 1e-6
    )

    checks["wrist_feature_variation_check"] = (
        wrist_std > 1e-6
    )

    unique_tasks = np.unique(tasks)
    unique_episodes = np.unique(episodes)

    checks["all_tasks_present"] = (
        len(unique_tasks) == 10
    )

    all_pass = all(checks.values())

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": REPO_ID,
        "num_frames": int(n),
        "state_dim": 8,
        "action_dim": 7,
        "vision_encoder": "MobileNetV3-Small",
        "vision_encoder_pretrained": "ImageNet",
        "vision_encoder_frozen": True,
        "vision_feature_dim_per_camera": feature_dim,
        "main_camera_key": MAIN_CAMERA_KEY,
        "wrist_camera_key": WRIST_CAMERA_KEY,
        "feature_storage_dtype": "float16",
        "num_tasks": int(len(unique_tasks)),
        "num_episodes": int(len(unique_episodes)),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # CSV report
    # --------------------------------------------------------

    report_rows = [
        ["dataset", REPO_ID],
        ["frames", n],
        ["state_dim", 8],
        ["action_dim", 7],
        ["num_tasks", len(unique_tasks)],
        ["num_episodes", len(unique_episodes)],
        ["vision_encoder", "MobileNetV3-Small"],
        ["vision_feature_dim_per_camera", feature_dim],
        ["main_feature_std", main_std],
        ["wrist_feature_std", wrist_std],
    ]

    for key, value in checks.items():
        report_rows.append(
            [key, "PASS" if value else "FAIL"]
        )

    report_rows.append(
        ["overall_qc", "PASS" if all_pass else "FAIL"]
    )

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["metric", "value"]
        )

        writer.writerows(report_rows)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LIBERO Mini-VLA Feature Cache QC")
    print("=" * 80)

    print(f"Frames:          {n:,}")
    print(f"Tasks:           {len(unique_tasks)}")
    print(f"Episodes:        {len(unique_episodes)}")
    print()

    print(
        f"Main Feature:    {main_features.shape} "
        f"dtype={main_features.dtype}"
    )

    print(
        f"Wrist Feature:   {wrist_features.shape} "
        f"dtype={wrist_features.dtype}"
    )

    print(f"State:           {states.shape}")
    print(f"Action:          {actions.shape}")

    print()

    print(
        f"Main Feature STD:  {main_std:.6f}"
    )

    print(
        f"Wrist Feature STD: {wrist_std:.6f}"
    )

    print()

    for key, value in checks.items():

        status = "PASS ✅" if value else "FAIL ❌"

        print(
            f"{key:<35} {status}"
        )

    print()
    print("=" * 80)

    if all_pass:
        print("FINAL RESULT: PASS ✅")
        print(
            "Vision feature cache is ready for Mini-VLA training."
        )
    else:
        print("FINAL RESULT: FAIL ❌")
        print(
            "Do not continue to Mini-VLA training "
            "until failed checks are fixed."
        )

    print("=" * 80)

    print()
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"Metadata:        {METADATA_FILE}")
    print(f"QC Report:       {REPORT_FILE}")


if __name__ == "__main__":
    main()