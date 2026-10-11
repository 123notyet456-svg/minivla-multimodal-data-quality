from __future__ import annotations

import argparse
import csv
import random
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

NORMALIZATION_FILE = (
    CACHE_DIR / "minivla_normalization.npz"
)

LANGUAGE_FILE = (
    Path("cache")
    / "libero_minilm_language_embeddings.npz"
)

OLD_COMPARISON_FILE = (
    REPORT_DIR
    / "libero_minivla_comparison.csv"
)

NEW_COMPARISON_FILE = (
    REPORT_DIR
    / "libero_projected_fusion_comparison.csv"
)

HISTORY_FILE = (
    REPORT_DIR
    / "libero_projected_fusion_training_history.csv"
)

PER_TASK_FILE = (
    REPORT_DIR
    / "libero_projected_fusion_per_task_mae.csv"
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
# Language
# ============================================================

def load_language_embeddings(path: Path):

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

        if arr.shape == (10, 384):

            candidate = arr
            selected_key = key
            break

    if candidate is None:

        for key in data.files:

            arr = np.asarray(
                data[key]
            )

            if arr.shape == (10, 384):

                candidate = arr
                selected_key = key
                break

    if candidate is None:

        raise RuntimeError(
            "Could not find 10 x 384 "
            "language embeddings."
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
        f"Language key: {selected_key}"
    )

    print(
        f"Language shape: "
        f"{candidate.shape}"
    )

    return candidate


# ============================================================
# Exact same split
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

    assert set(
        train_eps.tolist()
    ).isdisjoint(
        set(val_eps.tolist())
    )

    assert set(
        train_eps.tolist()
    ).isdisjoint(
        set(test_eps.tolist())
    )

    assert set(
        val_eps.tolist()
    ).isdisjoint(
        set(test_eps.tolist())
    )

    return (
        train_idx,
        val_idx,
        test_idx,
        train_eps,
        val_eps,
        test_eps,
    )


# ============================================================
# Model
# ============================================================

class ProjectedFusionPolicy(nn.Module):

    def __init__(
        self,
        use_vision: bool,
    ):
        super().__init__()

        self.use_vision = (
            use_vision
        )

        # ----------------------------------------
        # Modality-specific projection
        # ----------------------------------------

        self.state_projector = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
        )

        self.language_projector = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
        )

        if self.use_vision:

            self.main_projector = nn.Sequential(
                nn.Linear(576, 256),
                nn.ReLU(),
            )

            self.wrist_projector = nn.Sequential(
                nn.Linear(576, 256),
                nn.ReLU(),
            )

            fusion_dim = (
                256
                + 256
                + 64
                + 128
            )

        else:

            fusion_dim = (
                64
                + 128
            )

        # ----------------------------------------
        # Fusion network
        # ----------------------------------------

        self.fusion = nn.Sequential(
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

    def forward(
        self,
        state,
        language,
        main=None,
        wrist=None,
    ):

        state_z = (
            self.state_projector(
                state
            )
        )

        language_z = (
            self.language_projector(
                language
            )
        )

        pieces = []

        if self.use_vision:

            if main is None:
                raise RuntimeError(
                    "Main vision missing."
                )

            if wrist is None:
                raise RuntimeError(
                    "Wrist vision missing."
                )

            main_z = (
                self.main_projector(
                    main
                )
            )

            wrist_z = (
                self.wrist_projector(
                    wrist
                )
            )

            pieces.extend(
                [
                    main_z,
                    wrist_z,
                ]
            )

        pieces.extend(
            [
                state_z,
                language_z,
            ]
        )

        fused = torch.cat(
            pieces,
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
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    use_vision,
):

    state = np.asarray(
        states[indices],
        dtype=np.float32,
    )

    state = (
        state
        - stats["state_mean"]
    ) / stats["state_std"]

    action = np.asarray(
        actions[indices],
        dtype=np.float32,
    )

    action_norm = (
        action
        - stats["action_mean"]
    ) / stats["action_std"]

    task_ids = np.asarray(
        tasks[indices],
        dtype=np.int64,
    )

    language = (
        language_embeddings[
            task_ids
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    batch = {
        "state": state.astype(
            np.float32,
            copy=False,
        ),
        "language": language,
        "target": action_norm.astype(
            np.float32,
            copy=False,
        ),
    }

    if use_vision:

        main = np.asarray(
            main_features[indices],
            dtype=np.float32,
        )

        main = (
            main
            - stats["main_mean"]
        ) / stats["main_std"]

        wrist = np.asarray(
            wrist_features[indices],
            dtype=np.float32,
        )

        wrist = (
            wrist
            - stats["wrist_mean"]
        ) / stats["wrist_std"]

        batch["main"] = (
            main.astype(
                np.float32,
                copy=False,
            )
        )

        batch["wrist"] = (
            wrist.astype(
                np.float32,
                copy=False,
            )
        )

    return batch


# ============================================================
# Forward helper
# ============================================================

def model_forward(
    model,
    batch,
    device,
    use_vision,
):

    state = torch.from_numpy(
        batch["state"]
    ).to(device)

    language = torch.from_numpy(
        batch["language"]
    ).to(device)

    if use_vision:

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

    return model(
        state=state,
        language=language,
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
    main_features,
    wrist_features,
    language_embeddings,
    stats,
    device,
    use_vision,
    batch_size,
):

    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0

    task_abs_sum = {
        int(x): 0.0
        for x in np.unique(tasks)
    }

    task_elements = {
        int(x): 0
        for x in np.unique(tasks)
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
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
            use_vision=use_vision,
        )

        pred_norm = model_forward(
            model=model,
            batch=batch,
            device=device,
            use_vision=use_vision,
        )

        pred_norm = (
            pred_norm
            .cpu()
            .numpy()
        )

        prediction = (
            pred_norm
            * stats["action_std"]
            + stats["action_mean"]
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
# Train one model
# ============================================================

def train_model(
    name,
    use_vision,
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
    lr,
    patience,
    seed,
):

    print()
    print("=" * 90)
    print(
        f"Training {name}"
    )
    print("=" * 90)

    set_seed(
        seed
    )

    model = ProjectedFusionPolicy(
        use_vision=use_vision
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
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

        rng.shuffle(
            shuffled
        )

        total_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            len(shuffled),
            batch_size,
        ):

            batch_idx = shuffled[
                start:
                start + batch_size
            ]

            batch = build_batch(
                indices=batch_idx,
                states=states,
                actions=actions,
                tasks=tasks,
                main_features=main_features,
                wrist_features=wrist_features,
                language_embeddings=language_embeddings,
                stats=stats,
                use_vision=use_vision,
            )

            target = torch.from_numpy(
                batch["target"]
            ).to(device)

            optimizer.zero_grad()

            prediction = model_forward(
                model=model,
                batch=batch,
                device=device,
                use_vision=use_vision,
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
                main_features=main_features,
                wrist_features=wrist_features,
                language_embeddings=language_embeddings,
                stats=stats,
                device=device,
                use_vision=use_vision,
                batch_size=batch_size,
            )
        )

        history.append(
            {
                "model": name,
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

            best_val = (
                val_mae
            )

            best_epoch = (
                epoch
            )

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            no_improve = 0

        else:

            no_improve += 1

        if (
            no_improve
            >= patience
        ):

            print(
                f"Early Stopping "
                f"at Epoch {epoch}"
            )

            break

    if best_state is None:

        raise RuntimeError(
            "Best model was not saved."
        )

    model.load_state_dict(
        best_state
    )

    test_mae, test_mse, per_task = (
        evaluate(
            model=model,
            indices=test_idx,
            states=states,
            actions=actions,
            tasks=tasks,
            main_features=main_features,
            wrist_features=wrist_features,
            language_embeddings=language_embeddings,
            stats=stats,
            device=device,
            use_vision=use_vision,
            batch_size=batch_size,
        )
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
        f"Test MAE: "
        f"{test_mae:.6f}"
    )

    print(
        f"Test MSE: "
        f"{test_mse:.6f}"
    )

    return {
        "model": model,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "per_task": per_task,
        "history": history,
    }


# ============================================================
# Old results
# ============================================================

def load_old_results():

    results = {}

    if not OLD_COMPARISON_FILE.exists():

        return results

    with open(
        OLD_COMPARISON_FILE,
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

    device = choose_device()

    print("=" * 90)
    print(
        "LIBERO Projected Multimodal Fusion Experiment"
    )
    print("=" * 90)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load cached data
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

    print(
        f"Frames: "
        f"{len(states):,}"
    )

    print(
        "Cache Loading Check: "
        "PASS ✅"
    )

    # --------------------------------------------------------
    # Normalization
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

    print(
        "Normalization Loading: "
        "PASS ✅"
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language_embeddings = (
        load_language_embeddings(
            LANGUAGE_FILE
        )
    )

    print(
        "Language Loading: "
        "PASS ✅"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

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

    print()

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

    print(
        "Split Reconstruction: "
        "PASS ✅"
    )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    configurations = [
        (
            "projected_state_language",
            False,
        ),
        (
            "projected_dual_state_language",
            True,
        ),
    ]

    results = {}
    history_rows = []

    for name, use_vision in configurations:

        result = train_model(
            name=name,
            use_vision=use_vision,
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
            lr=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )

        results[
            name
        ] = result

        history_rows.extend(
            result["history"]
        )

        model_path = (
            MODEL_DIR
            / f"libero_{name}.pt"
        )

        torch.save(
            {
                "model_state_dict": (
                    result[
                        "model"
                    ].state_dict()
                ),
                "model_name": name,
                "use_vision": use_vision,
                "seed": args.seed,
                "best_epoch": (
                    result[
                        "best_epoch"
                    ]
                ),
            },
            model_path,
        )

        print(
            f"Saved: "
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Old results
    # --------------------------------------------------------

    old = load_old_results()

    old_state = old.get(
        "state_language"
    )

    old_dual = old.get(
        "dual_state_language"
    )

    new_state = results[
        "projected_state_language"
    ]["test_mae"]

    new_dual = results[
        "projected_dual_state_language"
    ]["test_mae"]

    # --------------------------------------------------------
    # New Vision information value
    # --------------------------------------------------------

    projected_vision_change = (
        (
            new_dual
            - new_state
        )
        / new_state
        * 100.0
    )

    if old_dual is not None:

        fusion_change = (
            (
                new_dual
                - old_dual
            )
            / old_dual
            * 100.0
        )

    else:

        fusion_change = None

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    rows = []

    for name, _ in configurations:

        r = results[name]

        rows.append(
            {
                "model": name,
                "parameter_count": (
                    r[
                        "parameter_count"
                    ]
                ),
                "best_epoch": (
                    r[
                        "best_epoch"
                    ]
                ),
                "best_val_mae": (
                    r[
                        "best_val_mae"
                    ]
                ),
                "test_mae": (
                    r[
                        "test_mae"
                    ]
                ),
                "test_mse": (
                    r[
                        "test_mse"
                    ]
                ),
            }
        )

    with open(
        NEW_COMPARISON_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "parameter_count",
                "best_epoch",
                "best_val_mae",
                "test_mae",
                "test_mse",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
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
            history_rows
        )

    # --------------------------------------------------------
    # Per-task
    # --------------------------------------------------------

    per_task_rows = []

    for name, _ in configurations:

        for task_id, mae in (
            results[
                name
            ][
                "per_task"
            ].items()
        ):

            per_task_rows.append(
                {
                    "model": name,
                    "task_index": (
                        task_id
                    ),
                    "test_mae": (
                        mae
                    ),
                }
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
                "model",
                "task_index",
                "test_mae",
            ],
        )

        writer.writeheader()
        writer.writerows(
            per_task_rows
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "FINAL PROJECTED FUSION EXPERIMENT"
    )
    print("=" * 100)

    if old_state is not None:

        print(
            f"Old Naive State + Language: "
            f"{old_state:.6f}"
        )

    if old_dual is not None:

        print(
            f"Old Naive Dual Mini-VLA:    "
            f"{old_dual:.6f}"
        )

    print()

    print(
        f"Projected State + Language: "
        f"{new_state:.6f}"
    )

    print(
        f"Projected Dual Mini-VLA:    "
        f"{new_dual:.6f}"
    )

    print()

    print(
        "Projected Dual vs "
        "Projected State+Language: "
        f"{projected_vision_change:+.2f}%"
    )

    if fusion_change is not None:

        print(
            "Projected Dual vs "
            "Old Naive Dual: "
            f"{fusion_change:+.2f}%"
        )

    print()

    print("=" * 100)

    if new_dual < new_state:

        print(
            "VISION INFORMATION VALUE "
            "UNDER PROJECTED FUSION: "
            "PASS ✅"
        )

    else:

        print(
            "VISION INFORMATION VALUE "
            "UNDER PROJECTED FUSION: "
            "NOT CONFIRMED ⚠️"
        )

    if (
        old_dual is not None
        and
        new_dual < old_dual
    ):

        print(
            "PROJECTED FUSION IMPROVEMENT: "
            "PASS ✅"
        )

    elif old_dual is not None:

        print(
            "PROJECTED FUSION IMPROVEMENT: "
            "NOT CONFIRMED ⚠️"
        )

    print("=" * 100)

    print()

    print(
        f"Comparison Report: "
        f"{NEW_COMPARISON_FILE}"
    )

    print(
        f"History Report:    "
        f"{HISTORY_FILE}"
    )

    print(
        f"Per-task Report:   "
        f"{PER_TASK_FILE}"
    )


if __name__ == "__main__":
    main()