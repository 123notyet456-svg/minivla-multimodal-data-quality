from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

CACHE_DIR = Path("cache") / "libero_minivla"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MAIN_FILE = CACHE_DIR / "main_vision_features.npy"
WRIST_FILE = CACHE_DIR / "wrist_vision_features.npy"
STATE_FILE = CACHE_DIR / "state.npy"
ACTION_FILE = CACHE_DIR / "action.npy"
TASK_FILE = CACHE_DIR / "task_index.npy"
EPISODE_FILE = CACHE_DIR / "episode_index.npy"
FRAME_FILE = CACHE_DIR / "frame_index.npy"

NORMALIZATION_FILE = (
    CACHE_DIR / "minivla_normalization.npz"
)

LANGUAGE_FILE = (
    Path("cache")
    / "libero_minilm_language_embeddings.npz"
)

MODEL_FILE = (
    MODEL_DIR
    / "libero_minivla_dual_state_language.pt"
)

REPORT_FILE = (
    REPORT_DIR
    / "libero_minivla_temporal_vision_intervention.csv"
)


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
# Model
# ============================================================

class MiniVLAPolicy(nn.Module):

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


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

    for key in preferred_keys:

        if key not in data.files:
            continue

        arr = np.asarray(
            data[key]
        )

        if arr.shape == (10, 384):
            candidate = arr
            print(
                f"Language key: {key}"
            )
            break

    if candidate is None:

        for key in data.files:

            arr = np.asarray(
                data[key]
            )

            if arr.shape == (10, 384):
                candidate = arr
                print(
                    f"Language key: {key}"
                )
                break

    if candidate is None:
        raise RuntimeError(
            "No 10x384 language embeddings found."
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

    return candidate


# ============================================================
# Exact test split reconstruction
# ============================================================

def build_test_split(
    tasks,
    episodes,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    test_episodes = []

    for task_id in np.sort(
        np.unique(tasks)
    ):

        task_eps = np.unique(
            episodes[
                tasks == task_id
            ]
        ).copy()

        rng.shuffle(
            task_eps
        )

        n = len(
            task_eps
        )

        n_val = max(
            1,
            int(round(n * 0.10)),
        )

        n_test = max(
            1,
            int(round(n * 0.10)),
        )

        selected = task_eps[
            n_val:
            n_val + n_test
        ]

        test_episodes.extend(
            selected.tolist()
        )

    test_episodes = np.asarray(
        test_episodes,
        dtype=np.int64,
    )

    test_idx = np.where(
        np.isin(
            episodes,
            test_episodes,
        )
    )[0].astype(
        np.int64
    )

    return test_idx


# ============================================================
# Same-episode temporal source
# ============================================================

def build_temporal_sources(
    test_idx,
    episodes,
    frames,
    offset,
):

    source_idx = np.empty_like(
        test_idx
    )

    actual_distances = []

    test_lookup = {
        int(idx): pos
        for pos, idx
        in enumerate(test_idx)
    }

    test_episodes = np.asarray(
        episodes[test_idx],
        dtype=np.int64,
    )

    for episode_id in np.unique(
        test_episodes
    ):

        episode_positions = np.where(
            test_episodes == episode_id
        )[0]

        episode_global_idx = test_idx[
            episode_positions
        ]

        episode_frames = np.asarray(
            frames[
                episode_global_idx
            ],
            dtype=np.int64,
        )

        order = np.argsort(
            episode_frames
        )

        sorted_global_idx = (
            episode_global_idx[
                order
            ]
        )

        sorted_frames = (
            episode_frames[
                order
            ]
        )

        n = len(
            sorted_global_idx
        )

        if n <= 1:
            raise RuntimeError(
                f"Episode {episode_id} "
                "has <= 1 test frame."
            )

        for local_pos in range(n):

            target_global = int(
                sorted_global_idx[
                    local_pos
                ]
            )

            target_frame = int(
                sorted_frames[
                    local_pos
                ]
            )

            # Prefer future frame
            source_pos = (
                local_pos
                + offset
            )

            # If future frame is unavailable,
            # use past frame
            if source_pos >= n:

                source_pos = (
                    local_pos
                    - offset
                )

            # If requested offset is too large,
            # select the farthest available frame
            if source_pos < 0:

                if local_pos < n // 2:
                    source_pos = n - 1
                else:
                    source_pos = 0

            if source_pos == local_pos:

                source_pos = (
                    local_pos + 1
                ) % n

            source_global = int(
                sorted_global_idx[
                    source_pos
                ]
            )

            source_frame = int(
                sorted_frames[
                    source_pos
                ]
            )

            output_position = (
                test_lookup[
                    target_global
                ]
            )

            source_idx[
                output_position
            ] = source_global

            actual_distances.append(
                abs(
                    source_frame
                    - target_frame
                )
            )

    if np.any(
        source_idx == test_idx
    ):
        raise RuntimeError(
            f"Temporal offset {offset} "
            "contains exact matches."
        )

    return (
        source_idx,
        np.asarray(
            actual_distances,
            dtype=np.float32,
        ),
    )


# ============================================================
# Input
# ============================================================

def build_input(
    target_idx,
    vision_source_idx,
    states,
    tasks,
    main_features,
    wrist_features,
    language_embeddings,
    stats,
):

    state = np.asarray(
        states[target_idx],
        dtype=np.float32,
    )

    state = (
        state
        - stats["state_mean"]
    ) / stats["state_std"]

    task_ids = np.asarray(
        tasks[target_idx],
        dtype=np.int64,
    )

    language = (
        language_embeddings[
            task_ids
        ]
    )

    main = np.asarray(
        main_features[
            vision_source_idx
        ],
        dtype=np.float32,
    )

    main = (
        main
        - stats["main_mean"]
    ) / stats["main_std"]

    wrist = np.asarray(
        wrist_features[
            vision_source_idx
        ],
        dtype=np.float32,
    )

    wrist = (
        wrist
        - stats["wrist_mean"]
    ) / stats["wrist_std"]

    x = np.concatenate(
        [
            main,
            wrist,
            state,
            language,
        ],
        axis=1,
    )

    return x.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Prediction
# ============================================================

@torch.inference_mode()
def predict(
    model,
    test_idx,
    source_idx,
    states,
    tasks,
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    device,
    batch_size=512,
):

    predictions = []

    for start in range(
        0,
        len(test_idx),
        batch_size,
    ):

        end = start + batch_size

        target_batch = (
            test_idx[start:end]
        )

        source_batch = (
            source_idx[start:end]
        )

        x_np = build_input(
            target_idx=target_batch,
            vision_source_idx=source_batch,
            states=states,
            tasks=tasks,
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
        )

        x = torch.from_numpy(
            x_np
        ).to(device)

        pred_norm = model(
            x
        )

        pred_norm = (
            pred_norm
            .cpu()
            .numpy()
        )

        pred = (
            pred_norm
            * stats["action_std"]
            + stats["action_mean"]
        )

        predictions.append(
            pred.astype(
                np.float32
            )
        )

    return np.concatenate(
        predictions,
        axis=0,
    )


# ============================================================
# Metrics
# ============================================================

def metrics(
    prediction,
    target,
):

    error = (
        prediction
        - target
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    return mae, mse


# ============================================================
# Main
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device()

    print("=" * 100)
    print(
        "LIBERO Mini-VLA "
        "Temporal Vision Intervention"
    )
    print("=" * 100)

    print(
        f"Device: {device}"
    )

    print()

    # --------------------------------------------------------
    # Load cache
    # --------------------------------------------------------

    main_features = np.load(
        MAIN_FILE,
        mmap_mode="r",
    )

    wrist_features = np.load(
        WRIST_FILE,
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

    print(
        f"Frames: {len(states):,}"
    )

    print(
        "Cache Loading: PASS ✅"
    )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    norm = np.load(
        NORMALIZATION_FILE
    )

    stats = {
        key: np.asarray(
            norm[key],
            dtype=np.float32,
        )
        for key
        in norm.files
    }

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language_embeddings = (
        load_language_embeddings(
            LANGUAGE_FILE
        )
    )

    print(
        "Language Loading: PASS ✅"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
    )

    seed = int(
        checkpoint.get(
            "seed",
            42,
        )
    )

    input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    model = MiniVLAPolicy(
        input_dim=input_dim
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print(
        "Frozen Dual Mini-VLA: PASS ✅"
    )

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    test_idx = build_test_split(
        tasks=tasks,
        episodes=episodes,
        seed=seed,
    )

    target = np.asarray(
        actions[test_idx],
        dtype=np.float32,
    )

    print(
        f"Test Frames: "
        f"{len(test_idx):,}"
    )

    # --------------------------------------------------------
    # Correct prediction
    # --------------------------------------------------------

    correct_prediction = predict(
        model=model,
        test_idx=test_idx,
        source_idx=test_idx,
        states=states,
        tasks=tasks,
        main_features=main_features,
        wrist_features=wrist_features,
        language_embeddings=language_embeddings,
        stats=stats,
        device=device,
    )

    correct_mae, correct_mse = (
        metrics(
            correct_prediction,
            target,
        )
    )

    results = [
        {
            "condition": "correct",
            "requested_offset_frames": 0,
            "mean_actual_offset_frames": 0.0,
            "mean_actual_offset_seconds": 0.0,
            "mae": correct_mae,
            "mse": correct_mse,
            "mae_change_vs_correct_pct": 0.0,
            "prediction_shift_mae": 0.0,
        }
    ]

    # --------------------------------------------------------
    # Temporal interventions
    # --------------------------------------------------------

    offsets = [
        1,
        5,
        20,
        50,
    ]

    for offset in offsets:

        print()
        print(
            f"Testing temporal offset "
            f"{offset} frames..."
        )

        source_idx, distances = (
            build_temporal_sources(
                test_idx=test_idx,
                episodes=episodes,
                frames=frames,
                offset=offset,
            )
        )

        prediction = predict(
            model=model,
            test_idx=test_idx,
            source_idx=source_idx,
            states=states,
            tasks=tasks,
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
            device=device,
        )

        mae, mse = metrics(
            prediction,
            target,
        )

        change_pct = (
            (mae - correct_mae)
            / correct_mae
            * 100.0
        )

        prediction_shift = float(
            np.mean(
                np.abs(
                    prediction
                    - correct_prediction
                )
            )
        )

        mean_distance = float(
            distances.mean()
        )

        mean_seconds = (
            mean_distance / 10.0
        )

        results.append(
            {
                "condition": (
                    f"offset_{offset}"
                ),
                "requested_offset_frames": (
                    offset
                ),
                "mean_actual_offset_frames": (
                    mean_distance
                ),
                "mean_actual_offset_seconds": (
                    mean_seconds
                ),
                "mae": mae,
                "mse": mse,
                "mae_change_vs_correct_pct": (
                    change_pct
                ),
                "prediction_shift_mae": (
                    prediction_shift
                ),
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "requested_offset_frames",
                "mean_actual_offset_frames",
                "mean_actual_offset_seconds",
                "mae",
                "mse",
                "mae_change_vs_correct_pct",
                "prediction_shift_mae",
            ],
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "FINAL TEMPORAL VISION INTERVENTION"
    )
    print("=" * 110)

    print(
        f"{'Condition':<16}"
        f"{'Mean Offset(s)':>18}"
        f"{'MAE':>14}"
        f"{'vs Correct':>16}"
        f"{'Prediction Shift':>20}"
    )

    print("-" * 110)

    for row in results:

        print(
            f"{row['condition']:<16}"
            f"{row['mean_actual_offset_seconds']:>18.3f}"
            f"{row['mae']:>14.6f}"
            f"{row['mae_change_vs_correct_pct']:>15.2f}%"
            f"{row['prediction_shift_mae']:>20.6f}"
        )

    print("=" * 110)

    print()
    print(
        f"Correct MAE: "
        f"{correct_mae:.6f}"
    )

    last_mae = (
        results[-1]["mae"]
    )

    if (
        last_mae
        > correct_mae
    ):

        print(
            "Temporal Vision Sensitivity: "
            "PRESENT ✅"
        )

    else:

        print(
            "Temporal Vision Sensitivity: "
            "NOT CONFIRMED ⚠️"
        )

    print()
    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()