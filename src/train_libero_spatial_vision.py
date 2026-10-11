from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from lerobot.datasets import LeRobotDataset


# ============================================================
# Configuration
# ============================================================

REPO_ID = "lerobot/libero_10"

MAIN_CAMERA_KEY = "observation.images.image"
WRIST_CAMERA_KEY = "observation.images.wrist_image"

CACHE_DIR = Path("cache") / "libero_minivla"
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

MAIN_SPATIAL_FILE = (
    CACHE_DIR / "main_spatial_features.npy"
)

WRIST_SPATIAL_FILE = (
    CACHE_DIR / "wrist_spatial_features.npy"
)

STATE_FILE = CACHE_DIR / "state.npy"
ACTION_FILE = CACHE_DIR / "action.npy"
TASK_FILE = CACHE_DIR / "task_index.npy"
EPISODE_FILE = CACHE_DIR / "episode_index.npy"

LANGUAGE_FILE = (
    Path("cache")
    / "libero_minilm_language_embeddings.npz"
)

OLD_NORMALIZATION_FILE = (
    CACHE_DIR / "minivla_normalization.npz"
)

SPATIAL_NORMALIZATION_FILE = (
    CACHE_DIR / "spatial_normalization.npz"
)

PROJECTED_COMPARISON_FILE = (
    REPORT_DIR
    / "libero_projected_fusion_comparison.csv"
)

SPATIAL_CACHE_REPORT = (
    REPORT_DIR
    / "libero_spatial_feature_cache_summary.csv"
)

SPATIAL_COMPARISON_REPORT = (
    REPORT_DIR
    / "libero_spatial_vision_comparison.csv"
)

SPATIAL_HISTORY_REPORT = (
    REPORT_DIR
    / "libero_spatial_vision_training_history.csv"
)

SPATIAL_PER_TASK_REPORT = (
    REPORT_DIR
    / "libero_spatial_vision_per_task_mae.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "libero_spatial_dual_state_language.pt"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def choose_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Image preprocessing
# ============================================================

def prepare_image_batch(images: torch.Tensor):

    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)

    if images.ndim != 4:
        raise ValueError(
            f"Expected 4D image batch, "
            f"got {tuple(images.shape)}"
        )

    # BCHW
    if images.shape[1] in (1, 3, 4):
        pass

    # BHWC -> BCHW
    elif images.shape[-1] in (1, 3, 4):

        images = images.permute(
            0,
            3,
            1,
            2,
        )

    else:
        raise ValueError(
            "Cannot determine image "
            f"channel dimension: "
            f"{tuple(images.shape)}"
        )

    if images.shape[1] == 4:
        images = images[:, :3]

    if images.shape[1] == 1:
        images = images.repeat(
            1,
            3,
            1,
            1,
        )

    if images.dtype == torch.uint8:

        images = (
            images.float()
            / 255.0
        )

    else:

        images = images.float()

        if float(
            images.max().item()
        ) > 1.5:

            images = (
                images
                / 255.0
            )

    return torch.clamp(
        images,
        0.0,
        1.0,
    )


# ============================================================
# Spatial MobileNet Encoder
# ============================================================

class MobileNetSpatialEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
        )

        backbone = mobilenet_v3_small(
            weights=weights
        )

        self.preprocess = (
            weights.transforms()
        )

        self.features = (
            backbone.features
        )

        # Instead of global average pooling,
        # keep a 3 x 3 spatial grid.
        self.spatial_pool = (
            nn.AdaptiveAvgPool2d(
                (3, 3)
            )
        )

        self.channels = 576
        self.grid_h = 3
        self.grid_w = 3

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    @torch.inference_mode()
    def forward(self, images):

        images = prepare_image_batch(
            images
        )

        images = self.preprocess(
            images
        )

        x = self.features(
            images
        )

        x = self.spatial_pool(
            x
        )

        return x


# ============================================================
# Build spatial cache
# ============================================================

def build_spatial_cache(
    device,
    batch_size,
    overwrite=False,
):

    if (
        MAIN_SPATIAL_FILE.exists()
        and
        WRIST_SPATIAL_FILE.exists()
        and
        not overwrite
    ):

        print(
            "Existing spatial feature cache "
            "detected."
        )

        print(
            "Spatial extraction skipped."
        )

        return

    print()
    print("=" * 90)
    print(
        "STEP A - BUILD SPATIAL VISION CACHE"
    )
    print("=" * 90)

    dataset = LeRobotDataset(
        REPO_ID
    )

    n = len(dataset)

    expected_n = len(
        np.load(
            STATE_FILE,
            mmap_mode="r",
        )
    )

    if n != expected_n:

        raise RuntimeError(
            f"Dataset frames {n} != "
            f"existing cache frames "
            f"{expected_n}"
        )

    print(
        f"Frames: {n:,}"
    )

    encoder = (
        MobileNetSpatialEncoder()
        .to(device)
    )

    print(
        "Encoder: MobileNetV3-Small"
    )

    print(
        "Weights: ImageNet pretrained"
    )

    print(
        "Encoder: Frozen"
    )

    print(
        "Spatial Grid: 576 x 3 x 3"
    )

    main_cache = (
        np.lib.format.open_memmap(
            MAIN_SPATIAL_FILE,
            mode="w+",
            dtype=np.float16,
            shape=(
                n,
                576,
                3,
                3,
            ),
        )
    )

    wrist_cache = (
        np.lib.format.open_memmap(
            WRIST_SPATIAL_FILE,
            mode="w+",
            dtype=np.float16,
            shape=(
                n,
                576,
                3,
                3,
            ),
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    write_pos = 0

    with torch.inference_mode():

        for batch_idx, batch in enumerate(
            loader
        ):

            main_images = (
                batch[
                    MAIN_CAMERA_KEY
                ].to(device)
            )

            wrist_images = (
                batch[
                    WRIST_CAMERA_KEY
                ].to(device)
            )

            main_feature = encoder(
                main_images
            )

            wrist_feature = encoder(
                wrist_images
            )

            main_np = (
                main_feature
                .detach()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            wrist_np = (
                wrist_feature
                .detach()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            batch_n = (
                main_np.shape[0]
            )

            start = write_pos
            end = start + batch_n

            main_cache[
                start:end
            ] = main_np

            wrist_cache[
                start:end
            ] = wrist_np

            write_pos = end

            if (
                batch_idx == 0
                or
                write_pos % 1000 < batch_n
                or
                write_pos == n
            ):

                pct = (
                    100.0
                    * write_pos
                    / n
                )

                print(
                    f"Processed "
                    f"{write_pos:,}/"
                    f"{n:,} "
                    f"({pct:.2f}%)"
                )

    main_cache.flush()
    wrist_cache.flush()

    del main_cache
    del wrist_cache

    print()
    print(
        "Spatial Feature Extraction: "
        "PASS ✅"
    )


# ============================================================
# Spatial cache QC
# ============================================================

def spatial_cache_qc():

    print()
    print("=" * 90)
    print(
        "STEP B - SPATIAL FEATURE CACHE QC"
    )
    print("=" * 90)

    main = np.load(
        MAIN_SPATIAL_FILE,
        mmap_mode="r",
    )

    wrist = np.load(
        WRIST_SPATIAL_FILE,
        mmap_mode="r",
    )

    n = len(
        np.load(
            STATE_FILE,
            mmap_mode="r",
        )
    )

    checks = {}

    checks[
        "main_shape_check"
    ] = (
        main.shape
        == (
            n,
            576,
            3,
            3,
        )
    )

    checks[
        "wrist_shape_check"
    ] = (
        wrist.shape
        == (
            n,
            576,
            3,
            3,
        )
    )

    qc_count = min(
        n,
        3000,
    )

    indices = np.linspace(
        0,
        n - 1,
        qc_count,
        dtype=np.int64,
    )

    main_sample = np.asarray(
        main[indices],
        dtype=np.float32,
    )

    wrist_sample = np.asarray(
        wrist[indices],
        dtype=np.float32,
    )

    checks[
        "main_finite_check"
    ] = np.isfinite(
        main_sample
    ).all()

    checks[
        "wrist_finite_check"
    ] = np.isfinite(
        wrist_sample
    ).all()

    main_std = float(
        main_sample.std()
    )

    wrist_std = float(
        wrist_sample.std()
    )

    checks[
        "main_variation_check"
    ] = (
        main_std > 1e-6
    )

    checks[
        "wrist_variation_check"
    ] = (
        wrist_std > 1e-6
    )

    all_pass = all(
        checks.values()
    )

    print(
        f"Main Spatial Shape:  "
        f"{main.shape}"
    )

    print(
        f"Wrist Spatial Shape: "
        f"{wrist.shape}"
    )

    print(
        f"Main Feature STD:     "
        f"{main_std:.6f}"
    )

    print(
        f"Wrist Feature STD:    "
        f"{wrist_std:.6f}"
    )

    print()

    for key, value in checks.items():

        status = (
            "PASS ✅"
            if value
            else "FAIL ❌"
        )

        print(
            f"{key:<30}"
            f"{status}"
        )

    with open(
        SPATIAL_CACHE_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        writer.writerow(
            [
                "frames",
                n,
            ]
        )

        writer.writerow(
            [
                "main_feature_std",
                main_std,
            ]
        )

        writer.writerow(
            [
                "wrist_feature_std",
                wrist_std,
            ]
        )

        for key, value in checks.items():

            writer.writerow(
                [
                    key,
                    (
                        "PASS"
                        if value
                        else "FAIL"
                    ),
                ]
            )

        writer.writerow(
            [
                "overall_qc",
                (
                    "PASS"
                    if all_pass
                    else "FAIL"
                ),
            ]
        )

    if not all_pass:

        raise RuntimeError(
            "Spatial cache QC failed."
        )

    print()
    print(
        "FINAL SPATIAL CACHE QC: "
        "PASS ✅"
    )


# ============================================================
# Language
# ============================================================

def load_language_embeddings(
    path: Path,
):

    data = np.load(
        path,
        allow_pickle=True,
    )

    preferred_keys = [
        "canonical_embeddings",
        "canonical_embedding",
        "correct_embeddings",
        "correct_language_embeddings",
        "language_embeddings",
        "embeddings",
    ]

    candidate = None
    selected_key = None

    for key in preferred_keys:

        if key not in data.files:
            continue

        arr = np.asarray(
            data[key]
        )

        if arr.shape == (
            10,
            384,
        ):

            candidate = arr
            selected_key = key
            break

    if candidate is None:

        for key in data.files:

            arr = np.asarray(
                data[key]
            )

            if arr.shape == (
                10,
                384,
            ):

                candidate = arr
                selected_key = key
                break

    if candidate is None:

        raise RuntimeError(
            "Could not find "
            "10 x 384 language embeddings."
        )

    candidate = np.asarray(
        candidate,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        candidate,
        axis=1,
        keepdims=True,
    )

    candidate = (
        candidate
        / np.maximum(
            norms,
            1e-8,
        )
    )

    print(
        f"Language key: "
        f"{selected_key}"
    )

    return candidate


# ============================================================
# Exact same episode split
# ============================================================

def build_episode_split(
    tasks,
    episodes,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    train_eps = []
    val_eps = []
    test_eps = []

    for task_id in np.sort(
        np.unique(tasks)
    ):

        task_episodes = np.unique(
            episodes[
                tasks == task_id
            ]
        ).copy()

        rng.shuffle(
            task_episodes
        )

        n = len(
            task_episodes
        )

        n_val = max(
            1,
            int(round(n * 0.10)),
        )

        n_test = max(
            1,
            int(round(n * 0.10)),
        )

        val = task_episodes[
            :n_val
        ]

        test = task_episodes[
            n_val:
            n_val + n_test
        ]

        train = task_episodes[
            n_val + n_test:
        ]

        train_eps.extend(
            train.tolist()
        )

        val_eps.extend(
            val.tolist()
        )

        test_eps.extend(
            test.tolist()
        )

    train_eps = np.asarray(
        train_eps,
        dtype=np.int64,
    )

    val_eps = np.asarray(
        val_eps,
        dtype=np.int64,
    )

    test_eps = np.asarray(
        test_eps,
        dtype=np.int64,
    )

    train_idx = np.where(
        np.isin(
            episodes,
            train_eps,
        )
    )[0].astype(np.int64)

    val_idx = np.where(
        np.isin(
            episodes,
            val_eps,
        )
    )[0].astype(np.int64)

    test_idx = np.where(
        np.isin(
            episodes,
            test_eps,
        )
    )[0].astype(np.int64)

    return (
        train_idx,
        val_idx,
        test_idx,
        train_eps,
        val_eps,
        test_eps,
    )


# ============================================================
# Streaming spatial normalization
# ============================================================

def spatial_stats(
    array,
    indices,
    chunk_size=512,
):

    shape = array.shape[1:]

    total_sum = np.zeros(
        shape,
        dtype=np.float64,
    )

    total_sq = np.zeros(
        shape,
        dtype=np.float64,
    )

    count = 0

    for start in range(
        0,
        len(indices),
        chunk_size,
    ):

        idx = indices[
            start:
            start + chunk_size
        ]

        x = np.asarray(
            array[idx],
            dtype=np.float32,
        )

        total_sum += x.sum(
            axis=0,
            dtype=np.float64,
        )

        total_sq += (
            x.astype(
                np.float64
            ) ** 2
        ).sum(
            axis=0
        )

        count += (
            x.shape[0]
        )

    mean = (
        total_sum
        / count
    )

    variance = (
        total_sq
        / count
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        1e-12,
    )

    std = np.sqrt(
        variance
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


# ============================================================
# Spatial Fusion Policy
# ============================================================

class SpatialProjectedPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.state_projector = (
            nn.Sequential(
                nn.Linear(
                    8,
                    64,
                ),
                nn.ReLU(),
            )
        )

        self.language_projector = (
            nn.Sequential(
                nn.Linear(
                    384,
                    128,
                ),
                nn.ReLU(),
            )
        )

        self.main_projector = (
            nn.Sequential(
                nn.Conv2d(
                    576,
                    64,
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(
                    64 * 3 * 3,
                    256,
                ),
                nn.ReLU(),
            )
        )

        self.wrist_projector = (
            nn.Sequential(
                nn.Conv2d(
                    576,
                    64,
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(
                    64 * 3 * 3,
                    256,
                ),
                nn.ReLU(),
            )
        )

        fusion_dim = (
            256
            + 256
            + 64
            + 128
        )

        self.fusion = (
            nn.Sequential(
                nn.Linear(
                    fusion_dim,
                    512,
                ),
                nn.ReLU(),

                nn.Linear(
                    512,
                    256,
                ),
                nn.ReLU(),

                nn.Linear(
                    256,
                    128,
                ),
                nn.ReLU(),

                nn.Linear(
                    128,
                    7,
                ),
            )
        )

    def forward(
        self,
        state,
        language,
        main,
        wrist,
    ):

        main_z = self.main_projector(
            main
        )

        wrist_z = self.wrist_projector(
            wrist
        )

        state_z = self.state_projector(
            state
        )

        language_z = (
            self.language_projector(
                language
            )
        )

        fused = torch.cat(
            [
                main_z,
                wrist_z,
                state_z,
                language_z,
            ],
            dim=1,
        )

        return self.fusion(
            fused
        )


# ============================================================
# Batch construction
# ============================================================

def build_batch(
    indices,
    states,
    actions,
    tasks,
    main_spatial,
    wrist_spatial,
    language_embeddings,
    base_stats,
    spatial_norm,
):

    state = np.asarray(
        states[indices],
        dtype=np.float32,
    )

    state = (
        state
        - base_stats[
            "state_mean"
        ]
    ) / base_stats[
        "state_std"
    ]

    action = np.asarray(
        actions[indices],
        dtype=np.float32,
    )

    action_norm = (
        action
        - base_stats[
            "action_mean"
        ]
    ) / base_stats[
        "action_std"
    ]

    task_ids = np.asarray(
        tasks[indices],
        dtype=np.int64,
    )

    language = (
        language_embeddings[
            task_ids
        ]
    )

    main = np.asarray(
        main_spatial[indices],
        dtype=np.float32,
    )

    wrist = np.asarray(
        wrist_spatial[indices],
        dtype=np.float32,
    )

    main = (
        main
        - spatial_norm[
            "main_mean"
        ]
    ) / spatial_norm[
        "main_std"
    ]

    wrist = (
        wrist
        - spatial_norm[
            "wrist_mean"
        ]
    ) / spatial_norm[
        "wrist_std"
    ]

    return {
        "state": state.astype(
            np.float32,
            copy=False,
        ),
        "language": (
            language.astype(
                np.float32,
                copy=False,
            )
        ),
        "main": main.astype(
            np.float32,
            copy=False,
        ),
        "wrist": wrist.astype(
            np.float32,
            copy=False,
        ),
        "target": (
            action_norm.astype(
                np.float32,
                copy=False,
            )
        ),
    }


# ============================================================
# Forward helper
# ============================================================

def forward_batch(
    model,
    batch,
    device,
):

    state = torch.from_numpy(
        batch["state"]
    ).to(device)

    language = torch.from_numpy(
        batch["language"]
    ).to(device)

    main = torch.from_numpy(
        batch["main"]
    ).to(device)

    wrist = torch.from_numpy(
        batch["wrist"]
    ).to(device)

    return model(
        state=state,
        language=language,
        main=main,
        wrist=wrist,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    indices,
    states,
    actions,
    tasks,
    main_spatial,
    wrist_spatial,
    language_embeddings,
    base_stats,
    spatial_norm,
    device,
    batch_size,
):

    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0

    task_abs_sum = {
        int(task): 0.0
        for task
        in np.unique(tasks)
    }

    task_elements = {
        int(task): 0
        for task
        in np.unique(tasks)
    }

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_idx = indices[
            start:
            start + batch_size
        ]

        batch = build_batch(
            indices=batch_idx,
            states=states,
            actions=actions,
            tasks=tasks,
            main_spatial=main_spatial,
            wrist_spatial=wrist_spatial,
            language_embeddings=language_embeddings,
            base_stats=base_stats,
            spatial_norm=spatial_norm,
        )

        pred_norm = forward_batch(
            model,
            batch,
            device,
        )

        pred_norm = (
            pred_norm
            .detach()
            .cpu()
            .numpy()
        )

        prediction = (
            pred_norm
            * base_stats[
                "action_std"
            ]
            + base_stats[
                "action_mean"
            ]
        )

        target = np.asarray(
            actions[batch_idx],
            dtype=np.float32,
        )

        error = (
            prediction
            - target
        )

        abs_error = np.abs(
            error
        )

        total_abs += float(
            abs_error.sum()
        )

        total_sq += float(
            (
                error ** 2
            ).sum()
        )

        total_elements += int(
            error.size
        )

        batch_tasks = np.asarray(
            tasks[batch_idx],
            dtype=np.int64,
        )

        for task_id in np.unique(
            batch_tasks
        ):

            mask = (
                batch_tasks
                == task_id
            )

            task_error = (
                abs_error[mask]
            )

            task_abs_sum[
                int(task_id)
            ] += float(
                task_error.sum()
            )

            task_elements[
                int(task_id)
            ] += int(
                task_error.size
            )

    mae = (
        total_abs
        / total_elements
    )

    mse = (
        total_sq
        / total_elements
    )

    per_task = {}

    for task_id in sorted(
        task_abs_sum
    ):

        per_task[
            task_id
        ] = (
            task_abs_sum[
                task_id
            ]
            / task_elements[
                task_id
            ]
        )

    return (
        mae,
        mse,
        per_task,
    )


# ============================================================
# Load previous projected results
# ============================================================

def load_projected_results():

    results = {}

    if not PROJECTED_COMPARISON_FILE.exists():

        raise FileNotFoundError(
            f"Missing previous report: "
            f"{PROJECTED_COMPARISON_FILE}"
        )

    with open(
        PROJECTED_COMPARISON_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            results[
                row["model"]
            ] = float(
                row["test_mae"]
            )

    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite-spatial",
        action="store_true",
    )

    args = parser.parse_args()

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        args.seed
    )

    device = choose_device()

    print("=" * 100)
    print(
        "LIBERO SPATIAL VISION REPRESENTATION EXPERIMENT"
    )
    print("=" * 100)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # A. Spatial extraction
    # ========================================================

    build_spatial_cache(
        device=device,
        batch_size=(
            args.extract_batch_size
        ),
        overwrite=(
            args.overwrite_spatial
        ),
    )

    # ========================================================
    # B. Cache QC
    # ========================================================

    spatial_cache_qc()

    # ========================================================
    # C. Load cached data
    # ========================================================

    print()
    print("=" * 90)
    print(
        "STEP C - TRAIN SPATIAL MINI-VLA"
    )
    print("=" * 90)

    main_spatial = np.load(
        MAIN_SPATIAL_FILE,
        mmap_mode="r",
    )

    wrist_spatial = np.load(
        WRIST_SPATIAL_FILE,
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

    language_embeddings = (
        load_language_embeddings(
            LANGUAGE_FILE
        )
    )

    base_norm = np.load(
        OLD_NORMALIZATION_FILE
    )

    base_stats = {
        key: np.asarray(
            base_norm[key],
            dtype=np.float32,
        )
        for key
        in base_norm.files
    }

    (
        train_idx,
        val_idx,
        test_idx,
        train_eps,
        val_eps,
        test_eps,
    ) = build_episode_split(
        tasks=tasks,
        episodes=episodes,
        seed=args.seed,
    )

    print(
        f"Train Episodes: "
        f"{len(train_eps)}"
    )

    print(
        f"Val Episodes:   "
        f"{len(val_eps)}"
    )

    print(
        f"Test Episodes:  "
        f"{len(test_eps)}"
    )

    print(
        f"Train Frames: "
        f"{len(train_idx):,}"
    )

    print(
        f"Val Frames:   "
        f"{len(val_idx):,}"
    )

    print(
        f"Test Frames:  "
        f"{len(test_idx):,}"
    )

    # ========================================================
    # D. Train-only spatial normalization
    # ========================================================

    print()
    print(
        "Computing train-only "
        "spatial normalization..."
    )

    main_mean, main_std = (
        spatial_stats(
            main_spatial,
            train_idx,
        )
    )

    wrist_mean, wrist_std = (
        spatial_stats(
            wrist_spatial,
            train_idx,
        )
    )

    np.savez(
        SPATIAL_NORMALIZATION_FILE,
        main_mean=main_mean,
        main_std=main_std,
        wrist_mean=wrist_mean,
        wrist_std=wrist_std,
    )

    spatial_norm = {
        "main_mean": main_mean,
        "main_std": main_std,
        "wrist_mean": wrist_mean,
        "wrist_std": wrist_std,
    }

    print(
        "Train-only Spatial "
        "Normalization: PASS ✅"
    )

    # ========================================================
    # E. Model
    # ========================================================

    model = (
        SpatialProjectedPolicy()
        .to(device)
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print()
    print(
        f"Spatial Model Parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    criterion = nn.MSELoss()

    best_val = float(
        "inf"
    )

    best_epoch = -1
    best_state = None

    no_improve = 0

    history = []

    rng = np.random.default_rng(
        args.seed
    )

    # ========================================================
    # F. Train
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        shuffled = (
            train_idx.copy()
        )

        rng.shuffle(
            shuffled
        )

        total_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            len(shuffled),
            args.train_batch_size,
        ):

            batch_idx = shuffled[
                start:
                start
                + args.train_batch_size
            ]

            batch = build_batch(
                indices=batch_idx,
                states=states,
                actions=actions,
                tasks=tasks,
                main_spatial=main_spatial,
                wrist_spatial=wrist_spatial,
                language_embeddings=(
                    language_embeddings
                ),
                base_stats=base_stats,
                spatial_norm=spatial_norm,
            )

            target = torch.from_numpy(
                batch["target"]
            ).to(device)

            optimizer.zero_grad()

            prediction = (
                forward_batch(
                    model,
                    batch,
                    device,
                )
            )

            loss = criterion(
                prediction,
                target,
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            batch_count += 1

        train_loss = (
            total_loss
            / max(
                batch_count,
                1,
            )
        )

        val_mae, val_mse, _ = (
            evaluate(
                model=model,
                indices=val_idx,
                states=states,
                actions=actions,
                tasks=tasks,
                main_spatial=main_spatial,
                wrist_spatial=wrist_spatial,
                language_embeddings=(
                    language_embeddings
                ),
                base_stats=base_stats,
                spatial_norm=spatial_norm,
                device=device,
                batch_size=(
                    args.train_batch_size
                ),
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": (
                    train_loss
                ),
                "val_mae": val_mae,
                "val_mse": val_mse,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Norm MSE "
            f"{train_loss:.6f} | "
            f"Val MAE "
            f"{val_mae:.6f}"
        )

        if (
            val_mae
            < best_val
            - 1e-7
        ):

            best_val = val_mae
            best_epoch = epoch

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            no_improve = 0

        else:

            no_improve += 1

        if (
            no_improve
            >= args.patience
        ):

            print(
                f"Early Stopping "
                f"at Epoch {epoch}"
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No best model saved."
        )

    model.load_state_dict(
        best_state
    )

    # ========================================================
    # G. Test
    # ========================================================

    (
        test_mae,
        test_mse,
        per_task,
    ) = evaluate(
        model=model,
        indices=test_idx,
        states=states,
        actions=actions,
        tasks=tasks,
        main_spatial=main_spatial,
        wrist_spatial=wrist_spatial,
        language_embeddings=(
            language_embeddings
        ),
        base_stats=base_stats,
        spatial_norm=spatial_norm,
        device=device,
        batch_size=(
            args.train_batch_size
        ),
    )

    print()
    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Val MAE: "
        f"{best_val:.6f}"
    )

    print(
        f"Spatial Dual Test MAE: "
        f"{test_mae:.6f}"
    )

    print(
        f"Spatial Dual Test MSE: "
        f"{test_mse:.6f}"
    )

    # ========================================================
    # H. Save model
    # ========================================================

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "parameter_count": (
                parameter_count
            ),
            "spatial_grid": [
                3,
                3,
            ],
        },
        MODEL_FILE,
    )

    # ========================================================
    # I. Reports
    # ========================================================

    with open(
        SPATIAL_HISTORY_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_normalized_mse",
                "val_mae",
                "val_mse",
            ],
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    with open(
        SPATIAL_PER_TASK_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_index",
                "test_mae",
            ],
        )

        writer.writeheader()

        for task_id, mae in (
            per_task.items()
        ):

            writer.writerow(
                {
                    "task_index": (
                        task_id
                    ),
                    "test_mae": mae,
                }
            )

    previous = (
        load_projected_results()
    )

    state_mae = previous[
        "projected_state_language"
    ]

    global_dual_mae = previous[
        "projected_dual_state_language"
    ]

    spatial_vs_state = (
        (
            test_mae
            - state_mae
        )
        / state_mae
        * 100.0
    )

    spatial_vs_global = (
        (
            test_mae
            - global_dual_mae
        )
        / global_dual_mae
        * 100.0
    )

    comparison_rows = [
        {
            "model": (
                "projected_state_language"
            ),
            "test_mae": (
                state_mae
            ),
        },
        {
            "model": (
                "global_projected_dual"
            ),
            "test_mae": (
                global_dual_mae
            ),
        },
        {
            "model": (
                "spatial_projected_dual"
            ),
            "test_mae": (
                test_mae
            ),
        },
    ]

    with open(
        SPATIAL_COMPARISON_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "test_mae",
            ],
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 105)
    print(
        "FINAL SPATIAL VISION REPRESENTATION EXPERIMENT"
    )
    print("=" * 105)

    print(
        f"Projected State + Language: "
        f"{state_mae:.6f}"
    )

    print(
        f"Global-Pooled Projected Dual: "
        f"{global_dual_mae:.6f}"
    )

    print(
        f"Spatial Projected Dual:       "
        f"{test_mae:.6f}"
    )

    print()

    print(
        "Spatial vs State+Language: "
        f"{spatial_vs_state:+.2f}%"
    )

    print(
        "Spatial vs Global Vision: "
        f"{spatial_vs_global:+.2f}%"
    )

    print()

    print("=" * 105)

    if (
        test_mae
        < global_dual_mae
    ):

        print(
            "SPATIAL REPRESENTATION "
            "IMPROVEMENT: PASS ✅"
        )

    else:

        print(
            "SPATIAL REPRESENTATION "
            "IMPROVEMENT: "
            "NOT CONFIRMED ⚠️"
        )

    if (
        test_mae
        < state_mae
    ):

        print(
            "VISION INFORMATION VALUE "
            "WITH SPATIAL FEATURES: "
            "PASS ✅"
        )

    else:

        print(
            "VISION INFORMATION VALUE "
            "WITH SPATIAL FEATURES: "
            "NOT CONFIRMED ⚠️"
        )

    print("=" * 105)

    print()
    print(
        f"Model: "
        f"{MODEL_FILE}"
    )

    print(
        f"Cache QC Report: "
        f"{SPATIAL_CACHE_REPORT}"
    )

    print(
        f"Comparison Report: "
        f"{SPATIAL_COMPARISON_REPORT}"
    )

    print(
        f"History Report: "
        f"{SPATIAL_HISTORY_REPORT}"
    )

    print(
        f"Per-task Report: "
        f"{SPATIAL_PER_TASK_REPORT}"
    )


if __name__ == "__main__":
    main()