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
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

MAIN_FILE = CACHE_DIR / "main_vision_features.npy"
WRIST_FILE = CACHE_DIR / "wrist_vision_features.npy"
STATE_FILE = CACHE_DIR / "state.npy"
ACTION_FILE = CACHE_DIR / "action.npy"
TASK_FILE = CACHE_DIR / "task_index.npy"
EPISODE_FILE = CACHE_DIR / "episode_index.npy"

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

COMPARISON_FILE = (
    REPORT_DIR
    / "libero_minivla_comparison.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "libero_minivla_vision_intervention.csv"
)

PER_TASK_FILE = (
    REPORT_DIR
    / "libero_minivla_vision_intervention_per_task.csv"
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
# Must match train_libero_minivla.py exactly
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
# Language embeddings
# ============================================================

def load_language_embeddings(
    path: Path,
    num_tasks: int = 10,
    embedding_dim: int = 384,
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
            num_tasks,
            embedding_dim,
        ):
            candidate = arr
            selected_key = key
            break

    if candidate is None:

        candidates = []

        for key in data.files:

            arr = np.asarray(
                data[key]
            )

            if arr.shape == (
                num_tasks,
                embedding_dim,
            ):
                candidates.append(
                    (key, arr)
                )

        if not candidates:
            raise RuntimeError(
                "No 10 x 384 language "
                "embedding matrix found."
            )

        preferred = [
            item
            for item in candidates
            if (
                "canonical"
                in item[0].lower()
                or
                "correct"
                in item[0].lower()
            )
        ]

        if preferred:
            selected_key, candidate = (
                preferred[0]
            )
        else:
            selected_key, candidate = (
                candidates[0]
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

    candidate = candidate / np.maximum(
        norms,
        1e-8,
    )

    print(
        f"Language key: {selected_key}"
    )

    print(
        f"Language shape: "
        f"{candidate.shape}"
    )

    return candidate


# ============================================================
# Same episode split as training
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

    unique_tasks = np.sort(
        np.unique(tasks)
    )

    for task_id in unique_tasks:

        mask = (
            tasks == task_id
        )

        task_eps = np.unique(
            episodes[mask]
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

        test_eps = task_eps[
            n_val:
            n_val + n_test
        ]

        test_episodes.extend(
            test_eps.tolist()
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
    )[0].astype(np.int64)

    return (
        test_idx,
        test_episodes,
    )


# ============================================================
# Shuffle construction
# ============================================================

def global_shuffle_sources(
    test_idx: np.ndarray,
):

    n = len(
        test_idx
    )

    if n < 2:
        raise RuntimeError(
            "Not enough test frames."
        )

    shift = max(
        1,
        n // 2,
    )

    source_idx = np.roll(
        test_idx,
        shift,
    )

    if np.any(
        source_idx == test_idx
    ):
        raise RuntimeError(
            "Global shuffle contains "
            "fixed matches."
        )

    return source_idx


def derangement(
    n: int,
    rng,
):

    if n <= 1:
        return np.arange(
            n
        )

    original = np.arange(
        n
    )

    for _ in range(1000):

        perm = rng.permutation(
            n
        )

        if np.all(
            perm != original
        ):
            return perm

    # Deterministic fallback
    return np.roll(
        original,
        1,
    )


def within_task_shuffle_sources(
    test_idx,
    tasks,
    seed,
):

    rng = np.random.default_rng(
        seed + 1001
    )

    source_idx = (
        test_idx.copy()
    )

    test_tasks = np.asarray(
        tasks[test_idx],
        dtype=np.int64,
    )

    for task_id in np.unique(
        test_tasks
    ):

        positions = np.where(
            test_tasks == task_id
        )[0]

        if len(positions) <= 1:
            continue

        perm = derangement(
            len(positions),
            rng,
        )

        source_idx[
            positions
        ] = test_idx[
            positions[perm]
        ]

    return source_idx


# ============================================================
# Build model input
# ============================================================

def build_input(
    target_indices,
    main_source_indices,
    wrist_source_indices,
    states,
    tasks,
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    zero_main=False,
    zero_wrist=False,
):

    state = np.asarray(
        states[target_indices],
        dtype=np.float32,
    )

    state = (
        state
        - stats["state_mean"]
    ) / stats["state_std"]

    task_ids = np.asarray(
        tasks[target_indices],
        dtype=np.int64,
    )

    language = (
        language_embeddings[
            task_ids
        ]
    )

    # --------------------------------------------
    # Main camera
    # --------------------------------------------

    if zero_main:

        # Normalized zero corresponds to
        # train mean visual feature.
        main = np.zeros(
            (
                len(target_indices),
                576,
            ),
            dtype=np.float32,
        )

    else:

        main = np.asarray(
            main_features[
                main_source_indices
            ],
            dtype=np.float32,
        )

        main = (
            main
            - stats["main_mean"]
        ) / stats["main_std"]

    # --------------------------------------------
    # Wrist camera
    # --------------------------------------------

    if zero_wrist:

        wrist = np.zeros(
            (
                len(target_indices),
                576,
            ),
            dtype=np.float32,
        )

    else:

        wrist = np.asarray(
            wrist_features[
                wrist_source_indices
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
def predict_condition(
    model,
    condition,
    test_idx,
    global_sources,
    within_task_sources,
    states,
    tasks,
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    device,
    batch_size=512,
):

    model.eval()

    predictions = []

    for start in range(
        0,
        len(test_idx),
        batch_size,
    ):

        end = (
            start
            + batch_size
        )

        target_idx = test_idx[
            start:end
        ]

        correct_source = (
            target_idx
        )

        global_source = (
            global_sources[
                start:end
            ]
        )

        within_source = (
            within_task_sources[
                start:end
            ]
        )

        zero_main = False
        zero_wrist = False

        if condition == "correct":

            main_source = (
                correct_source
            )

            wrist_source = (
                correct_source
            )

        elif condition == "zero_main":

            main_source = (
                correct_source
            )

            wrist_source = (
                correct_source
            )

            zero_main = True

        elif condition == "zero_wrist":

            main_source = (
                correct_source
            )

            wrist_source = (
                correct_source
            )

            zero_wrist = True

        elif condition == "zero_both":

            main_source = (
                correct_source
            )

            wrist_source = (
                correct_source
            )

            zero_main = True
            zero_wrist = True

        elif (
            condition
            == "global_shuffle_both"
        ):

            main_source = (
                global_source
            )

            wrist_source = (
                global_source
            )

        elif (
            condition
            == "within_task_shuffle_both"
        ):

            main_source = (
                within_source
            )

            wrist_source = (
                within_source
            )

        else:

            raise ValueError(
                f"Unknown condition: "
                f"{condition}"
            )

        x_np = build_input(
            target_indices=target_idx,
            main_source_indices=main_source,
            wrist_source_indices=wrist_source,
            states=states,
            tasks=tasks,
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
            zero_main=zero_main,
            zero_wrist=zero_wrist,
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

def calculate_metrics(
    prediction,
    target,
):

    error = (
        prediction
        - target
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    return mae, mse


def load_baseline_mae():

    if not COMPARISON_FILE.exists():
        return None

    with open(
        COMPARISON_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            if (
                row["model"]
                == "state_language"
            ):
                return float(
                    row[
                        "test_mae"
                    ]
                )

    return None


# ============================================================
# Main
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device()

    print("=" * 90)
    print(
        "LIBERO Mini-VLA "
        "Test-time Vision Intervention"
    )
    print("=" * 90)

    print(
        f"Device: {device}"
    )

    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "[1/6] Loading cached data..."
    )

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

    print(
        f"Frames: "
        f"{len(states):,}"
    )

    print(
        "Cache Loading Check: "
        "PASS ✅"
    )

    print()

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    print(
        "[2/6] Loading normalization..."
    )

    norm_data = np.load(
        NORMALIZATION_FILE
    )

    stats = {
        key: np.asarray(
            norm_data[key],
            dtype=np.float32,
        )
        for key
        in norm_data.files
    }

    print(
        "Normalization Check: "
        "PASS ✅"
    )

    print()

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    print(
        "[3/6] Loading language..."
    )

    language_embeddings = (
        load_language_embeddings(
            LANGUAGE_FILE
        )
    )

    print(
        "Language Check: "
        "PASS ✅"
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "[4/6] Loading frozen "
        "Dual Mini-VLA model..."
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
    )

    input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    seed = int(
        checkpoint.get(
            "seed",
            42,
        )
    )

    if input_dim != (
        576
        + 576
        + 8
        + 384
    ):
        raise RuntimeError(
            f"Unexpected input dim: "
            f"{input_dim}"
        )

    model = MiniVLAPolicy(
        input_dim=input_dim,
        action_dim=7,
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

    for param in model.parameters():
        param.requires_grad = False

    print(
        f"Seed: {seed}"
    )

    print(
        "Model Frozen: PASS ✅"
    )

    print()

    # --------------------------------------------------------
    # Rebuild exact test split
    # --------------------------------------------------------

    print(
        "[5/6] Reconstructing "
        "test split..."
    )

    test_idx, test_eps = (
        build_test_split(
            tasks=tasks,
            episodes=episodes,
            seed=seed,
        )
    )

    print(
        f"Test Episodes: "
        f"{len(test_eps)}"
    )

    print(
        f"Test Frames: "
        f"{len(test_idx):,}"
    )

    print()

    global_sources = (
        global_shuffle_sources(
            test_idx
        )
    )

    within_task_sources = (
        within_task_shuffle_sources(
            test_idx=test_idx,
            tasks=tasks,
            seed=seed,
        )
    )

    global_matches = int(
        np.sum(
            global_sources
            == test_idx
        )
    )

    within_matches = int(
        np.sum(
            within_task_sources
            == test_idx
        )
    )

    print(
        f"Global Shuffle "
        f"Exact Matches: "
        f"{global_matches}"
    )

    print(
        f"Within-task Shuffle "
        f"Exact Matches: "
        f"{within_matches}"
    )

    if global_matches != 0:
        raise RuntimeError(
            "Global shuffle "
            "has exact matches."
        )

    print(
        "Intervention Construction: "
        "PASS ✅"
    )

    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "[6/6] Running "
        "test-time interventions..."
    )

    conditions = [
        "correct",
        "zero_main",
        "zero_wrist",
        "zero_both",
        "global_shuffle_both",
        "within_task_shuffle_both",
    ]

    target = np.asarray(
        actions[test_idx],
        dtype=np.float32,
    )

    test_tasks = np.asarray(
        tasks[test_idx],
        dtype=np.int64,
    )

    predictions = {}

    results = {}

    for condition in conditions:

        prediction = (
            predict_condition(
                model=model,
                condition=condition,
                test_idx=test_idx,
                global_sources=global_sources,
                within_task_sources=within_task_sources,
                states=states,
                tasks=tasks,
                main_features=main_features,
                wrist_features=wrist_features,
                language_embeddings=language_embeddings,
                stats=stats,
                device=device,
            )
        )

        predictions[
            condition
        ] = prediction

        mae, mse = (
            calculate_metrics(
                prediction,
                target,
            )
        )

        results[
            condition
        ] = {
            "mae": mae,
            "mse": mse,
        }

    correct_prediction = (
        predictions[
            "correct"
        ]
    )

    correct_mae = (
        results[
            "correct"
        ]["mae"]
    )

    # --------------------------------------------------------
    # Prediction sensitivity
    # --------------------------------------------------------

    for condition in conditions:

        prediction_shift = float(
            np.mean(
                np.abs(
                    predictions[
                        condition
                    ]
                    - correct_prediction
                )
            )
        )

        mae = results[
            condition
        ]["mae"]

        change_pct = (
            (
                mae
                - correct_mae
            )
            / correct_mae
            * 100.0
        )

        results[
            condition
        ][
            "prediction_shift"
        ] = prediction_shift

        results[
            condition
        ][
            "mae_change_vs_correct_pct"
        ] = change_pct

    # --------------------------------------------------------
    # Per-task
    # --------------------------------------------------------

    per_task_rows = []

    for condition in conditions:

        pred = predictions[
            condition
        ]

        for task_id in np.unique(
            test_tasks
        ):

            mask = (
                test_tasks
                == task_id
            )

            mae = float(
                np.mean(
                    np.abs(
                        pred[mask]
                        - target[mask]
                    )
                )
            )

            per_task_rows.append(
                {
                    "condition": (
                        condition
                    ),
                    "task_index": int(
                        task_id
                    ),
                    "mae": mae,
                }
            )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    rows = []

    for condition in conditions:

        r = results[
            condition
        ]

        rows.append(
            {
                "condition": condition,
                "mae": r["mae"],
                "mse": r["mse"],
                "mae_change_vs_correct_pct": (
                    r[
                        "mae_change_vs_correct_pct"
                    ]
                ),
                "prediction_shift_mae": (
                    r[
                        "prediction_shift"
                    ]
                ),
            }
        )

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
                "mae",
                "mse",
                "mae_change_vs_correct_pct",
                "prediction_shift_mae",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    with open(
        PER_TASK_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "task_index",
                "mae",
            ],
        )

        writer.writeheader()

        writer.writerows(
            per_task_rows
        )

    # --------------------------------------------------------
    # Verify original Dual result
    # --------------------------------------------------------

    expected_dual = None

    if COMPARISON_FILE.exists():

        with open(
            COMPARISON_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                if (
                    row["model"]
                    == "dual_state_language"
                ):
                    expected_dual = float(
                        row[
                            "test_mae"
                        ]
                    )
                    break

    baseline_mae = (
        load_baseline_mae()
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "FINAL TEST-TIME "
        "VISION INTERVENTION"
    )
    print("=" * 100)

    print(
        f"{'Condition':<30}"
        f"{'MAE':>12}"
        f"{'vs Correct':>16}"
        f"{'Prediction Shift':>20}"
    )

    print("-" * 100)

    for condition in conditions:

        r = results[
            condition
        ]

        print(
            f"{condition:<30}"
            f"{r['mae']:>12.6f}"
            f"{r['mae_change_vs_correct_pct']:>15.2f}%"
            f"{r['prediction_shift']:>20.6f}"
        )

    print("=" * 100)

    print()

    if expected_dual is not None:

        difference = (
            correct_mae
            - expected_dual
        )

        print(
            f"Original Dual Test MAE: "
            f"{expected_dual:.6f}"
        )

        print(
            f"Reproduced Correct MAE: "
            f"{correct_mae:.6f}"
        )

        print(
            f"Difference: "
            f"{difference:+.8f}"
        )

        if abs(
            difference
        ) < 1e-5:

            print(
                "Test Reproduction Check: "
                "PASS ✅"
            )

        else:

            print(
                "Test Reproduction Check: "
                "WARNING ⚠️"
            )

    if baseline_mae is not None:

        print()

        print(
            f"State + Language Baseline: "
            f"{baseline_mae:.6f}"
        )

        dual_vs_baseline = (
            (
                correct_mae
                - baseline_mae
            )
            / baseline_mae
            * 100.0
        )

        print(
            f"Correct Dual vs Baseline: "
            f"{dual_vs_baseline:+.2f}%"
        )

    print()

    zero_both_change = (
        results[
            "zero_both"
        ][
            "mae_change_vs_correct_pct"
        ]
    )

    global_change = (
        results[
            "global_shuffle_both"
        ][
            "mae_change_vs_correct_pct"
        ]
    )

    within_change = (
        results[
            "within_task_shuffle_both"
        ][
            "mae_change_vs_correct_pct"
        ]
    )

    correct_shift_threshold = 0.001

    global_shift = (
        results[
            "global_shuffle_both"
        ][
            "prediction_shift"
        ]
    )

    print("=" * 100)
    print("INTERPRETATION SIGNALS")
    print("=" * 100)

    if (
        abs(global_change) < 2.0
        and
        global_shift
        < correct_shift_threshold
    ):

        print(
            "Vision Dependency: WEAK ⚠️"
        )

        print(
            "The policy appears to use "
            "little visual information."
        )

    else:

        print(
            "Vision Dependency: PRESENT ✅"
        )

        print(
            "Changing vision materially "
            "changes policy predictions "
            "and/or prediction error."
        )

    print()

    print(
        "Zero Both vs Correct: "
        f"{zero_both_change:+.2f}%"
    )

    print(
        "Global Shuffle vs Correct: "
        f"{global_change:+.2f}%"
    )

    print(
        "Within-task Shuffle vs Correct: "
        f"{within_change:+.2f}%"
    )

    print()

    if (
        global_change
        > within_change + 5.0
    ):

        print(
            "Possible Task-Codebook Signal ⚠️"
        )

        print(
            "Global visual mismatch hurts "
            "more than within-task mismatch."
        )

        print(
            "Vision may be encoding "
            "task identity more than "
            "frame-specific control state."
        )

    elif within_change > 5.0:

        print(
            "Frame-specific Vision Signal ✅"
        )

        print(
            "Even same-task wrong frames "
            "hurt performance."
        )

        print(
            "The policy is sensitive to "
            "frame-level visual state."
        )

    else:

        print(
            "Frame-specific Vision Signal: "
            "WEAK / UNCLEAR ⚠️"
        )

    print("=" * 100)

    print()
    print(
        f"Report: "
        f"{REPORT_FILE}"
    )

    print(
        f"Per-task Report: "
        f"{PER_TASK_FILE}"
    )


if __name__ == "__main__":
    main()