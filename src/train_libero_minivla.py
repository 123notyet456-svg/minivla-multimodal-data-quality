from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

CACHE_DIR = Path("cache") / "libero_minivla"

MAIN_FEATURE_FILE = CACHE_DIR / "main_vision_features.npy"
WRIST_FEATURE_FILE = CACHE_DIR / "wrist_vision_features.npy"
STATE_FILE = CACHE_DIR / "state.npy"
ACTION_FILE = CACHE_DIR / "action.npy"
TASK_FILE = CACHE_DIR / "task_index.npy"
EPISODE_FILE = CACHE_DIR / "episode_index.npy"

LANGUAGE_CACHE_FILE = (
    Path("cache") / "libero_minilm_language_embeddings.npz"
)

REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

SPLIT_REPORT_FILE = (
    REPORT_DIR / "libero_minivla_split_summary.csv"
)

COMPARISON_REPORT_FILE = (
    REPORT_DIR / "libero_minivla_comparison.csv"
)

PER_TASK_REPORT_FILE = (
    REPORT_DIR / "libero_minivla_per_task_mae.csv"
)

HISTORY_REPORT_FILE = (
    REPORT_DIR / "libero_minivla_training_history.csv"
)

NORMALIZATION_FILE = (
    CACHE_DIR / "minivla_normalization.npz"
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

def choose_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Load language embedding
# ============================================================

def load_language_embeddings(
    path: Path,
    num_tasks: int = 10,
    embedding_dim: int = 384,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"Language cache not found:\n{path}"
        )

    data = np.load(path, allow_pickle=True)

    print("Language cache keys:")

    for key in data.files:
        arr = np.asarray(data[key])
        print(
            f"  {key:<35} shape={arr.shape}"
        )

    # Preferred key names
    preferred_keys = [
        "canonical_embeddings",
        "canonical_embedding",
        "correct_embeddings",
        "correct_language_embeddings",
        "language_embeddings",
        "embeddings",
    ]

    candidate = None
    candidate_key = None

    for key in preferred_keys:

        if key not in data.files:
            continue

        arr = np.asarray(data[key])

        if arr.shape == (
            num_tasks,
            embedding_dim,
        ):
            candidate = arr
            candidate_key = key
            break

    # Fallback:
    # search for a 10 x 384 array
    if candidate is None:

        candidates = []

        for key in data.files:

            arr = np.asarray(data[key])

            if arr.shape == (
                num_tasks,
                embedding_dim,
            ):
                candidates.append(
                    (key, arr)
                )

        if len(candidates) == 0:

            raise RuntimeError(
                "Could not find a "
                "10 x 384 canonical language "
                "embedding matrix."
            )

        # Prefer keys that look canonical
        canonical_candidates = [
            item
            for item in candidates
            if (
                "canonical" in item[0].lower()
                or "correct" in item[0].lower()
            )
        ]

        if canonical_candidates:
            candidate_key, candidate = (
                canonical_candidates[0]
            )
        else:
            candidate_key, candidate = (
                candidates[0]
            )

    candidate = (
        np.asarray(candidate)
        .astype(np.float32)
    )

    # L2 normalization
    norms = np.linalg.norm(
        candidate,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-8,
    )

    candidate = candidate / norms

    print()
    print(
        "Selected language embedding key: "
        f"{candidate_key}"
    )
    print(
        f"Language embedding shape: "
        f"{candidate.shape}"
    )

    return candidate


# ============================================================
# Episode-level stratified split
# ============================================================

def build_episode_stratified_split(
    tasks: np.ndarray,
    episodes: np.ndarray,
    seed: int,
):

    rng = np.random.default_rng(seed)

    unique_tasks = np.sort(
        np.unique(tasks)
    )

    train_episodes = []
    val_episodes = []
    test_episodes = []

    task_rows = []

    for task_id in unique_tasks:

        mask = tasks == task_id

        task_eps = np.unique(
            episodes[mask]
        )

        task_eps = task_eps.copy()

        rng.shuffle(task_eps)

        n = len(task_eps)

        n_val = max(
            1,
            int(round(n * 0.10)),
        )

        n_test = max(
            1,
            int(round(n * 0.10)),
        )

        if (
            n_val + n_test
            >= n
        ):
            raise RuntimeError(
                f"Task {task_id} "
                "does not have enough episodes."
            )

        val_eps = task_eps[:n_val]

        test_eps = task_eps[
            n_val:n_val + n_test
        ]

        train_eps = task_eps[
            n_val + n_test:
        ]

        train_episodes.extend(
            train_eps.tolist()
        )

        val_episodes.extend(
            val_eps.tolist()
        )

        test_episodes.extend(
            test_eps.tolist()
        )

        task_rows.append(
            {
                "task_index": int(task_id),
                "total_episodes": int(n),
                "train_episodes": int(
                    len(train_eps)
                ),
                "val_episodes": int(
                    len(val_eps)
                ),
                "test_episodes": int(
                    len(test_eps)
                ),
            }
        )

    train_episodes = np.asarray(
        train_episodes,
        dtype=np.int64,
    )

    val_episodes = np.asarray(
        val_episodes,
        dtype=np.int64,
    )

    test_episodes = np.asarray(
        test_episodes,
        dtype=np.int64,
    )

    train_mask = np.isin(
        episodes,
        train_episodes,
    )

    val_mask = np.isin(
        episodes,
        val_episodes,
    )

    test_mask = np.isin(
        episodes,
        test_episodes,
    )

    train_idx = np.where(
        train_mask
    )[0].astype(np.int64)

    val_idx = np.where(
        val_mask
    )[0].astype(np.int64)

    test_idx = np.where(
        test_mask
    )[0].astype(np.int64)

    # Leakage checks
    train_set = set(
        train_episodes.tolist()
    )

    val_set = set(
        val_episodes.tolist()
    )

    test_set = set(
        test_episodes.tolist()
    )

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    return (
        train_idx,
        val_idx,
        test_idx,
        train_episodes,
        val_episodes,
        test_episodes,
        task_rows,
    )


# ============================================================
# Statistics
# ============================================================

def basic_stats(
    array: np.ndarray,
    indices: np.ndarray,
):

    x = np.asarray(
        array[indices],
        dtype=np.float32,
    )

    mean = x.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = x.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = np.maximum(
        std,
        1e-6,
    )

    return mean, std


def streaming_feature_stats(
    array: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 4096,
):

    feature_dim = array.shape[1]

    total_sum = np.zeros(
        feature_dim,
        dtype=np.float64,
    )

    total_sq = np.zeros(
        feature_dim,
        dtype=np.float64,
    )

    count = 0

    for start in range(
        0,
        len(indices),
        chunk_size,
    ):

        idx = indices[
            start:start + chunk_size
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
            x.astype(np.float64) ** 2
        ).sum(axis=0)

        count += x.shape[0]

    mean = total_sum / count

    variance = (
        total_sq / count
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        1e-12,
    )

    std = np.sqrt(variance)

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


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
            nn.Linear(
                input_dim,
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
                action_dim,
            ),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Input construction
# ============================================================

def build_input_batch(
    mode: str,
    indices: np.ndarray,
    states: np.ndarray,
    tasks: np.ndarray,
    main_features: np.ndarray,
    wrist_features: np.ndarray,
    language_embeddings: np.ndarray,
    stats: dict,
):

    state = np.asarray(
        states[indices],
        dtype=np.float32,
    )

    state = (
        state
        - stats["state_mean"]
    ) / stats["state_std"]

    task_ids = np.asarray(
        tasks[indices],
        dtype=np.int64,
    )

    language = (
        language_embeddings[
            task_ids
        ]
    )

    pieces = []

    if mode in (
        "main_state_language",
        "dual_state_language",
    ):

        main = np.asarray(
            main_features[indices],
            dtype=np.float32,
        )

        main = (
            main
            - stats["main_mean"]
        ) / stats["main_std"]

        pieces.append(main)

    if mode in (
        "wrist_state_language",
        "dual_state_language",
    ):

        wrist = np.asarray(
            wrist_features[indices],
            dtype=np.float32,
        )

        wrist = (
            wrist
            - stats["wrist_mean"]
        ) / stats["wrist_std"]

        pieces.append(wrist)

    pieces.append(state)
    pieces.append(language)

    x = np.concatenate(
        pieces,
        axis=1,
    )

    return x.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_model(
    model,
    mode,
    indices,
    states,
    actions,
    tasks,
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    device,
    batch_size,
):

    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0

    task_abs_sum = {
        int(task): 0.0
        for task in np.unique(tasks)
    }

    task_elements = {
        int(task): 0
        for task in np.unique(tasks)
    }

    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_idx = indices[
            start:start + batch_size
        ]

        x_np = build_input_batch(
            mode=mode,
            indices=batch_idx,
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

        pred_norm = model(x)

        pred_norm = (
            pred_norm
            .detach()
            .cpu()
            .numpy()
        )

        pred = (
            pred_norm
            * action_std
            + action_mean
        )

        target = np.asarray(
            actions[batch_idx],
            dtype=np.float32,
        )

        error = pred - target

        abs_error = np.abs(error)

        total_abs += float(
            abs_error.sum()
        )

        total_sq += float(
            (error ** 2).sum()
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
                batch_tasks == task_id
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
        task_abs_sum.keys()
    ):

        per_task[task_id] = (
            task_abs_sum[task_id]
            / task_elements[task_id]
        )

    return mae, mse, per_task


# ============================================================
# Train one model
# ============================================================

def train_one_model(
    mode,
    input_dim,
    train_idx,
    val_idx,
    test_idx,
    states,
    actions,
    tasks,
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    device,
    epochs,
    batch_size,
    learning_rate,
    patience,
    seed,
):

    print()
    print("=" * 80)
    print(f"Training: {mode}")
    print("=" * 80)
    print(f"Input Dim: {input_dim}")
    print()

    set_seed(seed)

    model = MiniVLAPolicy(
        input_dim=input_dim,
        action_dim=7,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.MSELoss()

    action_mean = stats[
        "action_mean"
    ]

    action_std = stats[
        "action_std"
    ]

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None

    epochs_without_improvement = 0

    history = []

    rng = np.random.default_rng(
        seed
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        shuffled = (
            train_idx.copy()
        )

        rng.shuffle(shuffled)

        total_loss = 0.0
        total_batches = 0

        for start in range(
            0,
            len(shuffled),
            batch_size,
        ):

            batch_idx = shuffled[
                start:start + batch_size
            ]

            x_np = build_input_batch(
                mode=mode,
                indices=batch_idx,
                states=states,
                tasks=tasks,
                main_features=main_features,
                wrist_features=wrist_features,
                language_embeddings=language_embeddings,
                stats=stats,
            )

            target = np.asarray(
                actions[batch_idx],
                dtype=np.float32,
            )

            target_norm = (
                target
                - action_mean
            ) / action_std

            x = torch.from_numpy(
                x_np
            ).to(device)

            y = torch.from_numpy(
                target_norm
            ).to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(
                pred,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            total_batches += 1

        train_loss = (
            total_loss
            / max(
                total_batches,
                1,
            )
        )

        val_mae, val_mse, _ = (
            evaluate_model(
                model=model,
                mode=mode,
                indices=val_idx,
                states=states,
                actions=actions,
                tasks=tasks,
                main_features=main_features,
                wrist_features=wrist_features,
                language_embeddings=language_embeddings,
                stats=stats,
                device=device,
                batch_size=batch_size,
            )
        )

        history.append(
            {
                "model": mode,
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
            f"{val_mae:.6f} | "
            f"Val MSE "
            f"{val_mse:.6f}"
        )

        if (
            val_mae
            < best_val_mae
            - 1e-7
        ):

            best_val_mae = (
                val_mae
            )

            best_epoch = epoch

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "Early Stopping "
                f"at Epoch {epoch}"
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No best model state saved."
        )

    model.load_state_dict(
        best_state
    )

    test_mae, test_mse, per_task = (
        evaluate_model(
            model=model,
            mode=mode,
            indices=test_idx,
            states=states,
            actions=actions,
            tasks=tasks,
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
            device=device,
            batch_size=batch_size,
        )
    )

    print()
    print(
        f"Best Epoch: {best_epoch}"
    )
    print(
        f"Best Val MAE: "
        f"{best_val_mae:.6f}"
    )
    print(
        f"Test MAE: "
        f"{test_mae:.6f}"
    )
    print(
        f"Test MSE: "
        f"{test_mse:.6f}"
    )

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "per_task": per_task,
        "history": history,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
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

    args = parser.parse_args()

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

    print("=" * 80)
    print("LIBERO Mini-VLA Controlled Experiment")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # --------------------------------------------------------
    # Load cache
    # --------------------------------------------------------

    print("[1/6] Loading cached multimodal data...")

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

    n = len(states)

    assert main_features.shape == (
        n,
        576,
    )

    assert wrist_features.shape == (
        n,
        576,
    )

    assert states.shape == (
        n,
        8,
    )

    assert actions.shape == (
        n,
        7,
    )

    assert tasks.shape == (
        n,
    )

    assert episodes.shape == (
        n,
    )

    print(
        f"Frames: {n:,}"
    )
    print(
        "Multimodal Cache Loading Check: "
        "PASS ✅"
    )
    print()

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    print("[2/6] Loading language embeddings...")

    language_embeddings = (
        load_language_embeddings(
            LANGUAGE_CACHE_FILE
        )
    )

    print(
        "Language Embedding Check: "
        "PASS ✅"
    )
    print()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print("[3/6] Building episode-level stratified split...")

    (
        train_idx,
        val_idx,
        test_idx,
        train_eps,
        val_eps,
        test_eps,
        task_rows,
    ) = build_episode_stratified_split(
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

    print()

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

    print()
    print(
        "Episode Leakage Check: "
        "PASS ✅"
    )
    print(
        "Task Stratification Check: "
        "PASS ✅"
    )
    print()

    # Save split report
    with open(
        SPLIT_REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_index",
                "total_episodes",
                "train_episodes",
                "val_episodes",
                "test_episodes",
            ],
        )

        writer.writeheader()
        writer.writerows(
            task_rows
        )

    # --------------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------------

    print("[4/6] Computing train-only normalization statistics...")

    state_mean, state_std = (
        basic_stats(
            states,
            train_idx,
        )
    )

    action_mean, action_std = (
        basic_stats(
            actions,
            train_idx,
        )
    )

    main_mean, main_std = (
        streaming_feature_stats(
            main_features,
            train_idx,
        )
    )

    wrist_mean, wrist_std = (
        streaming_feature_stats(
            wrist_features,
            train_idx,
        )
    )

    stats = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "main_mean": main_mean,
        "main_std": main_std,
        "wrist_mean": wrist_mean,
        "wrist_std": wrist_std,
    }

    np.savez(
        NORMALIZATION_FILE,
        **stats,
    )

    print(
        "Normalization uses Train only: "
        "PASS ✅"
    )
    print()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("[5/6] Training controlled Mini-VLA models...")

    experiment_configs = [
        (
            "state_language",
            8 + 384,
        ),
        (
            "main_state_language",
            576 + 8 + 384,
        ),
        (
            "wrist_state_language",
            576 + 8 + 384,
        ),
        (
            "dual_state_language",
            576 + 576 + 8 + 384,
        ),
    ]

    results = {}

    all_history = []

    for mode, input_dim in experiment_configs:

        result = train_one_model(
            mode=mode,
            input_dim=input_dim,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            states=states,
            actions=actions,
            tasks=tasks,
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )

        results[mode] = result

        all_history.extend(
            result["history"]
        )

        model_path = (
            MODEL_DIR
            / f"libero_minivla_{mode}.pt"
        )

        torch.save(
            {
                "mode": mode,
                "input_dim": input_dim,
                "model_state_dict": (
                    result["model"]
                    .state_dict()
                ),
                "best_epoch": (
                    result[
                        "best_epoch"
                    ]
                ),
                "seed": args.seed,
            },
            model_path,
        )

        print(
            f"Saved: {model_path}"
        )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    print()
    print("[6/6] Building comparison reports...")

    baseline_mae = (
        results[
            "state_language"
        ]["test_mae"]
    )

    comparison_rows = []

    for mode, _ in experiment_configs:

        result = results[mode]

        delta_pct = (
            (
                result["test_mae"]
                - baseline_mae
            )
            / baseline_mae
            * 100.0
        )

        comparison_rows.append(
            {
                "model": mode,
                "best_epoch": (
                    result[
                        "best_epoch"
                    ]
                ),
                "best_val_mae": (
                    result[
                        "best_val_mae"
                    ]
                ),
                "test_mae": (
                    result[
                        "test_mae"
                    ]
                ),
                "test_mse": (
                    result[
                        "test_mse"
                    ]
                ),
                "mae_change_vs_state_language_pct": (
                    delta_pct
                ),
            }
        )

    with open(
        COMPARISON_REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "best_epoch",
                "best_val_mae",
                "test_mae",
                "test_mse",
                "mae_change_vs_state_language_pct",
            ],
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )

    per_task_rows = []

    for mode, _ in experiment_configs:

        per_task = (
            results[
                mode
            ]["per_task"]
        )

        for task_id, mae in per_task.items():

            per_task_rows.append(
                {
                    "model": mode,
                    "task_index": (
                        task_id
                    ),
                    "test_mae": (
                        mae
                    ),
                }
            )

    with open(
        PER_TASK_REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "task_index",
                "test_mae",
            ],
        )

        writer.writeheader()
        writer.writerows(
            per_task_rows
        )

    with open(
        HISTORY_REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "epoch",
                "train_normalized_mse",
                "val_mae",
                "val_mse",
            ],
        )

        writer.writeheader()
        writer.writerows(
            all_history
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL MINI-VLA CONTROLLED EXPERIMENT")
    print("=" * 90)

    print(
        f"{'Model':<30}"
        f"{'Test MAE':>14}"
        f"{'vs Baseline':>18}"
    )

    print("-" * 90)

    for row in comparison_rows:

        print(
            f"{row['model']:<30}"
            f"{row['test_mae']:>14.6f}"
            f"{row['mae_change_vs_state_language_pct']:>17.2f}%"
        )

    print("=" * 90)

    main_mae = (
        results[
            "main_state_language"
        ]["test_mae"]
    )

    wrist_mae = (
        results[
            "wrist_state_language"
        ]["test_mae"]
    )

    dual_mae = (
        results[
            "dual_state_language"
        ]["test_mae"]
    )

    main_change = (
        (main_mae - baseline_mae)
        / baseline_mae
        * 100
    )

    wrist_change = (
        (wrist_mae - baseline_mae)
        / baseline_mae
        * 100
    )

    dual_change = (
        (dual_mae - baseline_mae)
        / baseline_mae
        * 100
    )

    dual_vs_main = (
        (dual_mae - main_mae)
        / main_mae
        * 100
    )

    dual_vs_wrist = (
        (dual_mae - wrist_mae)
        / wrist_mae
        * 100
    )

    print()
    print(
        "Main Vision vs State+Language: "
        f"{main_change:+.2f}%"
    )

    print(
        "Wrist Vision vs State+Language: "
        f"{wrist_change:+.2f}%"
    )

    print(
        "Dual Vision vs State+Language: "
        f"{dual_change:+.2f}%"
    )

    print()
    print(
        "Dual Vision vs Main Vision: "
        f"{dual_vs_main:+.2f}%"
    )

    print(
        "Dual Vision vs Wrist Vision: "
        f"{dual_vs_wrist:+.2f}%"
    )

    print()
    print("=" * 90)

    if dual_mae < baseline_mae:

        print(
            "VISION INFORMATION VALUE: "
            "PASS ✅"
        )

        print(
            "Dual-view visual information "
            "improves action prediction "
            "over State + Language."
        )

    else:

        print(
            "VISION INFORMATION VALUE: "
            "NOT CONFIRMED ⚠️"
        )

        print(
            "Adding dual-view visual features "
            "did not improve the "
            "State + Language baseline."
        )

    print("=" * 90)

    print()
    print(
        f"Comparison Report: "
        f"{COMPARISON_REPORT_FILE}"
    )

    print(
        f"Per-task Report:  "
        f"{PER_TASK_REPORT_FILE}"
    )

    print(
        f"Training History: "
        f"{HISTORY_REPORT_FILE}"
    )

    print(
        f"Split Report:     "
        f"{SPLIT_REPORT_FILE}"
    )


if __name__ == "__main__":
    main()