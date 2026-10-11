from pathlib import Path
import copy
import csv
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from huggingface_hub import snapshot_download


# ============================================================
# Configuration
# ============================================================

REPO_ID = "lerobot/libero_10"

STATE_DIM = 8
ACTION_DIM = 7
NUM_TASKS = 10

HIDDEN_DIM = 128

BATCH_SIZE = 256
MAX_EPOCHS = 50

LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 8

SEED = 42


REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


SPLIT_REPORT_PATH = (
    REPORT_DIR
    / "libero_task_conditioning_split_summary.csv"
)

COMPARISON_REPORT_PATH = (
    REPORT_DIR
    / "libero_task_conditioning_comparison.csv"
)

PER_TASK_REPORT_PATH = (
    REPORT_DIR
    / "libero_task_conditioning_per_task_mae.csv"
)

HISTORY_REPORT_PATH = (
    REPORT_DIR
    / "libero_task_conditioning_training_history.csv"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 90)
print("LIBERO TASK CONDITIONING CONTROLLED EXPERIMENT")
print("LIBERO 任务条件控制变量实验")
print("=" * 90)

print(
    f"Device（计算设备）: {DEVICE}"
)

print(
    f"Seed（随机种子）: {SEED}"
)


# ============================================================
# Locate Cached Tabular Dataset
# ============================================================
#
# 你上一轮已经下载过这些 parquet。
#
# snapshot_download 会优先使用 Hugging Face Cache，
# 正常情况下不会重新下载 7.76 MB。
#
# 不下载 Video（视频）。
# ============================================================

print("\n" + "=" * 90)
print("LOADING TABULAR DATA")
print("读取机器人表格数据")
print("=" * 90)


snapshot_path = snapshot_download(

    repo_id=REPO_ID,

    repo_type="dataset",

    allow_patterns=[
        "data/*/*.parquet",
    ],
)


snapshot_path = Path(
    snapshot_path
)


parquet_files = sorted(

    snapshot_path.glob(
        "data/*/*.parquet"
    )
)


print(
    f"Parquet Files（表格文件）: "
    f"{len(parquet_files)}"
)


if len(parquet_files) == 0:

    raise RuntimeError(
        "No LIBERO parquet files found."
    )


# ============================================================
# Load Required Columns
# ============================================================

required_columns = [

    "observation.state",
    "action",

    "episode_index",
    "task_index",
    "frame_index",
]


tables = []


for file_number, path in enumerate(
    parquet_files,
    start=1,
):

    table = pd.read_parquet(

        path,

        columns=required_columns,
    )


    tables.append(
        table
    )


    if (
        file_number == 1
        or file_number % 25 == 0
        or file_number == len(parquet_files)
    ):

        print(
            f"Loaded "
            f"{file_number}/"
            f"{len(parquet_files)} files"
        )


data_df = pd.concat(

    tables,

    ignore_index=True,
)


print(
    f"\nTotal Frames（总帧数）: "
    f"{len(data_df)}"
)


if len(data_df) != 101469:

    raise RuntimeError(
        f"Expected 101469 frames, "
        f"got {len(data_df)}."
    )


# ============================================================
# Convert State / Action
# ============================================================

print(
    "\nConverting State / Action arrays..."
)


states = np.stack(

    data_df[
        "observation.state"
    ].to_numpy()

).astype(
    np.float32
)


actions = np.stack(

    data_df[
        "action"
    ].to_numpy()

).astype(
    np.float32
)


episodes = (

    data_df[
        "episode_index"
    ]

    .to_numpy()

    .astype(
        np.int64
    )
)


tasks = (

    data_df[
        "task_index"
    ]

    .to_numpy()

    .astype(
        np.int64
    )
)


frames = (

    data_df[
        "frame_index"
    ]

    .to_numpy()

    .astype(
        np.int64
    )
)


print(
    f"State Shape（状态形状） : "
    f"{states.shape}"
)

print(
    f"Action Shape（动作形状）: "
    f"{actions.shape}"
)


if states.shape != (
    101469,
    STATE_DIM,
):

    raise RuntimeError(
        "State shape incorrect."
    )


if actions.shape != (
    101469,
    ACTION_DIM,
):

    raise RuntimeError(
        "Action shape incorrect."
    )


print(
    "State / Action Shape Check: PASS ✅"
)


# ============================================================
# Episode-level Table
# ============================================================
#
# Split 必须按 Episode，而不能随机拆 Frame。
#
# 防止同一条轨迹同时出现在
# Train / Val / Test。
# ============================================================

episode_df = (

    data_df

    .groupby(
        "episode_index",
        as_index=False,
    )

    .agg(
        task_index=(
            "task_index",
            "first",
        ),

        frame_count=(
            "frame_index",
            "count",
        ),
    )
)


episode_df[
    "episode_index"
] = episode_df[
    "episode_index"
].astype(int)


episode_df[
    "task_index"
] = episode_df[
    "task_index"
].astype(int)


print("\n" + "=" * 90)
print("EPISODE-LEVEL STRATIFIED SPLIT")
print("按任务分层的 Episode 数据划分")
print("=" * 90)


# ============================================================
# Stratified Episode Split
# ============================================================
#
# Stratified Split（分层划分）：
#
# 每个 Task 内独立随机打乱 Episode，
# 再按约 80% / 10% / 10%
# 分配到 Train / Val / Test。
#
# 这样十个任务在三个 Split 中都会存在。
# ============================================================

rng = np.random.default_rng(
    SEED
)


train_episode_ids = []

val_episode_ids = []

test_episode_ids = []


split_summary_rows = []


for task_index in range(
    NUM_TASKS
):

    task_episodes = (

        episode_df[
            episode_df[
                "task_index"
            ]
            == task_index
        ][
            "episode_index"
        ]

        .to_numpy()

        .astype(
            np.int64
        )
    )


    task_episodes = rng.permutation(
        task_episodes
    )


    n_total = len(
        task_episodes
    )


    n_val = max(
        1,
        int(
            round(
                n_total
                * 0.10
            )
        ),
    )


    n_test = max(
        1,
        int(
            round(
                n_total
                * 0.10
            )
        ),
    )


    n_train = (
        n_total
        - n_val
        - n_test
    )


    if n_train <= 0:

        raise RuntimeError(
            f"Task {task_index} "
            f"does not have enough episodes."
        )


    train_ids = task_episodes[
        :n_train
    ]

    val_ids = task_episodes[
        n_train:
        n_train + n_val
    ]

    test_ids = task_episodes[
        n_train + n_val:
    ]


    train_episode_ids.extend(
        train_ids.tolist()
    )

    val_episode_ids.extend(
        val_ids.tolist()
    )

    test_episode_ids.extend(
        test_ids.tolist()
    )


    split_summary_rows.append(
        {
            "task_index":
                task_index,

            "total_episodes":
                n_total,

            "train_episodes":
                len(train_ids),

            "val_episodes":
                len(val_ids),

            "test_episodes":
                len(test_ids),
        }
    )


train_episode_ids = np.array(
    train_episode_ids,
    dtype=np.int64,
)

val_episode_ids = np.array(
    val_episode_ids,
    dtype=np.int64,
)

test_episode_ids = np.array(
    test_episode_ids,
    dtype=np.int64,
)


# ============================================================
# Leakage Check
# ============================================================

train_set = set(
    train_episode_ids.tolist()
)

val_set = set(
    val_episode_ids.tolist()
)

test_set = set(
    test_episode_ids.tolist()
)


leakage_check = (

    train_set.isdisjoint(
        val_set
    )

    and train_set.isdisjoint(
        test_set
    )

    and val_set.isdisjoint(
        test_set
    )
)


coverage_check = (

    len(
        train_set
        | val_set
        | test_set
    )

    == 379
)


print(
    "Episode Leakage Check"
    "（轨迹泄漏检查）: "
    + (
        "PASS ✅"
        if leakage_check
        else "FAIL ❌"
    )
)


print(
    "Episode Coverage Check"
    "（轨迹覆盖检查）: "
    + (
        "PASS ✅"
        if coverage_check
        else "FAIL ❌"
    )
)


if not (
    leakage_check
    and coverage_check
):

    raise RuntimeError(
        "Episode split QC failed."
    )


# ============================================================
# Frame Masks
# ============================================================

train_mask = np.isin(
    episodes,
    train_episode_ids,
)

val_mask = np.isin(
    episodes,
    val_episode_ids,
)

test_mask = np.isin(
    episodes,
    test_episode_ids,
)


print(
    f"\nTrain Episodes: "
    f"{len(train_episode_ids)}"
)

print(
    f"Val Episodes  : "
    f"{len(val_episode_ids)}"
)

print(
    f"Test Episodes : "
    f"{len(test_episode_ids)}"
)


print(
    f"\nTrain Frames: "
    f"{train_mask.sum()}"
)

print(
    f"Val Frames  : "
    f"{val_mask.sum()}"
)

print(
    f"Test Frames : "
    f"{test_mask.sum()}"
)


# ============================================================
# Check All Tasks in Every Split
# ============================================================

def detected_tasks(
    mask,
):

    return sorted(
        np.unique(
            tasks[
                mask
            ]
        ).tolist()
    )


expected_tasks = list(
    range(
        NUM_TASKS
    )
)


train_task_check = (
    detected_tasks(
        train_mask
    )
    == expected_tasks
)


val_task_check = (
    detected_tasks(
        val_mask
    )
    == expected_tasks
)


test_task_check = (
    detected_tasks(
        test_mask
    )
    == expected_tasks
)


print(
    "\nAll Tasks in Train: "
    + (
        "PASS ✅"
        if train_task_check
        else "FAIL ❌"
    )
)


print(
    "All Tasks in Val  : "
    + (
        "PASS ✅"
        if val_task_check
        else "FAIL ❌"
    )
)


print(
    "All Tasks in Test : "
    + (
        "PASS ✅"
        if test_task_check
        else "FAIL ❌"
    )
)


if not (
    train_task_check
    and val_task_check
    and test_task_check
):

    raise RuntimeError(
        "Task coverage across splits failed."
    )


# ============================================================
# Save Split Report
# ============================================================

split_summary_df = pd.DataFrame(
    split_summary_rows
)


split_summary_df.to_csv(

    SPLIT_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Build Split Arrays
# ============================================================

train_state = states[
    train_mask
]

val_state = states[
    val_mask
]

test_state = states[
    test_mask
]


train_action = actions[
    train_mask
]

val_action = actions[
    val_mask
]

test_action = actions[
    test_mask
]


train_task = tasks[
    train_mask
]

val_task = tasks[
    val_mask
]

test_task = tasks[
    test_mask
]


train_episode = episodes[
    train_mask
]

val_episode = episodes[
    val_mask
]

test_episode = episodes[
    test_mask
]


# ============================================================
# State Normalization
# ============================================================
#
# Normalization（标准化）
# 只使用 Train 统计量。
#
# 避免 Data Leakage（数据泄漏）。
# ============================================================

EPS = 1e-6


state_mean = train_state.mean(
    axis=0,
    keepdims=True,
)


state_std = train_state.std(
    axis=0,
    keepdims=True,
)


state_std = np.maximum(
    state_std,
    EPS,
)


def normalize_state(
    x,
):

    return (
        (
            x
            - state_mean
        )
        / state_std
    ).astype(
        np.float32
    )


train_state_n = normalize_state(
    train_state
)

val_state_n = normalize_state(
    val_state
)

test_state_n = normalize_state(
    test_state
)


print("\n" + "=" * 90)
print("STATE NORMALIZATION")
print("状态标准化")
print("=" * 90)


print(
    f"Train Mean: "
    f"{train_state_n.mean():.6f}"
)

print(
    f"Train Std : "
    f"{train_state_n.std():.6f}"
)


# ============================================================
# Task One-Hot
# ============================================================
#
# Task ID:
#
# 例如 Task 3
#
# [0,0,0,1,0,0,0,0,0,0]
#
# 这不是 Language。
#
# 只是明确告诉模型：
# “现在是哪一个任务。”
# ============================================================

def one_hot_task(
    task_ids,
):

    output = np.zeros(
        (
            len(task_ids),
            NUM_TASKS,
        ),
        dtype=np.float32,
    )


    output[
        np.arange(
            len(task_ids)
        ),
        task_ids,
    ] = 1.0


    return output


train_task_correct = one_hot_task(
    train_task
)

val_task_correct = one_hot_task(
    val_task
)

test_task_correct = one_hot_task(
    test_task
)


train_task_zero = np.zeros_like(
    train_task_correct
)

val_task_zero = np.zeros_like(
    val_task_correct
)

test_task_zero = np.zeros_like(
    test_task_correct
)


# ============================================================
# Shuffled Task IDs
# ============================================================
#
# Control Experiment（控制实验）
#
# 在 Episode 级别重新分配 Task ID。
#
# 要求：
#
# 1. Task 总体分布保持不变
# 2. 每条 Episode 仍只有一个 Task
# 3. 尽可能保证错配
# 4. Train / Val / Test 分别独立打乱
#
# 这样可以破坏：
#
# Correct Task → Action
#
# 的真实对应关系。
# ============================================================

def build_episode_shuffled_tasks(
    episode_array,
    task_array,
    seed,
):

    unique_episodes = np.unique(
        episode_array
    )


    episode_task = {}


    for episode_id in unique_episodes:

        episode_indices = np.where(
            episode_array
            == episode_id
        )[0]


        original_task = int(
            task_array[
                episode_indices[0]
            ]
        )


        episode_task[
            int(
                episode_id
            )
        ] = original_task


    original_labels = np.array(
        [
            episode_task[
                int(
                    episode_id
                )
            ]

            for episode_id
            in unique_episodes
        ],
        dtype=np.int64,
    )


    rng_local = np.random.default_rng(
        seed
    )


    best_labels = None
    best_matches = len(
        original_labels
    )


    # 多次寻找接近完全错配的 Episode permutation
    for _ in range(
        5000
    ):

        candidate = rng_local.permutation(
            original_labels
        )


        matches = int(
            (
                candidate
                == original_labels
            ).sum()
        )


        if matches < best_matches:

            best_matches = matches
            best_labels = candidate.copy()


        if matches == 0:

            break


    if best_labels is None:

        raise RuntimeError(
            "Failed to build shuffled task labels."
        )


    shuffled_lookup = {

        int(
            episode_id
        ):
            int(
                task_id
            )

        for episode_id, task_id
        in zip(
            unique_episodes,
            best_labels,
        )
    }


    shuffled_task_array = np.array(
        [
            shuffled_lookup[
                int(
                    episode_id
                )
            ]

            for episode_id
            in episode_array
        ],
        dtype=np.int64,
    )


    return (
        shuffled_task_array,
        best_matches,
    )


(
    train_task_shuffled_ids,
    train_matches,
) = build_episode_shuffled_tasks(

    train_episode,
    train_task,
    SEED + 100,
)


(
    val_task_shuffled_ids,
    val_matches,
) = build_episode_shuffled_tasks(

    val_episode,
    val_task,
    SEED + 200,
)


(
    test_task_shuffled_ids,
    test_matches,
) = build_episode_shuffled_tasks(

    test_episode,
    test_task,
    SEED + 300,
)


train_task_shuffled = one_hot_task(
    train_task_shuffled_ids
)

val_task_shuffled = one_hot_task(
    val_task_shuffled_ids
)

test_task_shuffled = one_hot_task(
    test_task_shuffled_ids
)


print("\n" + "=" * 90)
print("SHUFFLED TASK CONTROL")
print("错配任务控制组")
print("=" * 90)


print(
    f"Train Episode Label Matches"
    f"（仍然碰巧正确）: "
    f"{train_matches}"
)

print(
    f"Val Episode Label Matches  : "
    f"{val_matches}"
)

print(
    f"Test Episode Label Matches : "
    f"{test_matches}"
)


# ============================================================
# Build Model Inputs
# ============================================================

train_state_only = (
    train_state_n
)

val_state_only = (
    val_state_n
)

test_state_only = (
    test_state_n
)


train_zero = np.concatenate(
    [
        train_state_n,
        train_task_zero,
    ],
    axis=1,
)


val_zero = np.concatenate(
    [
        val_state_n,
        val_task_zero,
    ],
    axis=1,
)


test_zero = np.concatenate(
    [
        test_state_n,
        test_task_zero,
    ],
    axis=1,
)


train_correct = np.concatenate(
    [
        train_state_n,
        train_task_correct,
    ],
    axis=1,
)


val_correct = np.concatenate(
    [
        val_state_n,
        val_task_correct,
    ],
    axis=1,
)


test_correct = np.concatenate(
    [
        test_state_n,
        test_task_correct,
    ],
    axis=1,
)


train_shuffled = np.concatenate(
    [
        train_state_n,
        train_task_shuffled,
    ],
    axis=1,
)


val_shuffled = np.concatenate(
    [
        val_state_n,
        val_task_shuffled,
    ],
    axis=1,
)


test_shuffled = np.concatenate(
    [
        test_state_n,
        test_task_shuffled,
    ],
    axis=1,
)


print("\n" + "=" * 90)
print("MODEL INPUT STRUCTURE")
print("模型输入结构")
print("=" * 90)


print(
    f"State-only Input: "
    f"{train_state_only.shape[1]}D"
)

print(
    f"State + Task Input: "
    f"{train_correct.shape[1]}D"
)


# ============================================================
# Policy Model
# ============================================================

class MLPPolicy(nn.Module):

    def __init__(
        self,
        input_dim,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                ACTION_DIM,
            ),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    x,
    y,
    shuffle,
):

    x_tensor = torch.from_numpy(
        x
    ).float()


    y_tensor = torch.from_numpy(
        y
    ).float()


    dataset = TensorDataset(
        x_tensor,
        y_tensor,
    )


    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )


    return DataLoader(

        dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            shuffle,

        num_workers=0,

        generator=(
            generator
            if shuffle
            else None
        ),
    )


# ============================================================
# Train Model
# ============================================================

def train_model(
    model_name,
    train_x,
    val_x,
):

    print("\n" + "=" * 90)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 90)


    set_seed(SEED)


    model = MLPPolicy(
        input_dim=
            train_x.shape[1]
    ).to(
        DEVICE
    )


    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        "Trainable Parameters"
        "（可训练参数）: "
        f"{parameter_count:,}"
    )


    train_loader = make_loader(
        train_x,
        train_action,
        shuffle=True,
    )


    val_loader = make_loader(
        val_x,
        val_action,
        shuffle=False,
    )


    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,
    )


    best_val_mse = float(
        "inf"
    )

    best_epoch = -1

    best_weights = None

    epochs_without_improvement = 0

    history = []


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        total_loss = 0.0
        total_count = 0


        for (
            x_batch,
            y_batch,
        ) in train_loader:

            x_batch = x_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            optimizer.zero_grad()


            prediction = model(
                x_batch
            )


            loss = criterion(
                prediction,
                y_batch,
            )


            loss.backward()

            optimizer.step()


            batch_size_now = len(
                x_batch
            )


            total_loss += (
                loss.item()
                * batch_size_now
            )

            total_count += (
                batch_size_now
            )


        train_mse = (
            total_loss
            / total_count
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        total_loss = 0.0
        total_count = 0


        with torch.inference_mode():

            for (
                x_batch,
                y_batch,
            ) in val_loader:

                x_batch = x_batch.to(
                    DEVICE
                )

                y_batch = y_batch.to(
                    DEVICE
                )


                prediction = model(
                    x_batch
                )


                loss = criterion(
                    prediction,
                    y_batch,
                )


                batch_size_now = len(
                    x_batch
                )


                total_loss += (
                    loss.item()
                    * batch_size_now
                )

                total_count += (
                    batch_size_now
                )


        val_mse = (
            total_loss
            / total_count
        )


        history.append(
            {
                "model":
                    model_name,

                "epoch":
                    epoch,

                "train_mse":
                    train_mse,

                "val_mse":
                    val_mse,
            }
        )


        # ----------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------

        if val_mse < best_val_mse:

            best_val_mse = val_mse

            best_epoch = epoch

            best_weights = copy.deepcopy(
                model.state_dict()
            )


            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == MAX_EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:02d}/"
                f"{MAX_EPOCHS} | "
                f"Train MSE: "
                f"{train_mse:.8f} | "
                f"Val MSE: "
                f"{val_mse:.8f}"
            )


        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly Stopping"
                f"（提前停止） at "
                f"Epoch {epoch}"
            )

            break


    model.load_state_dict(
        best_weights
    )


    print(
        f"\nBest Epoch（最佳轮次）: "
        f"{best_epoch}"
    )


    print(
        f"Best Val MSE（最佳验证MSE）: "
        f"{best_val_mse:.8f}"
    )


    return (
        model,
        history,
        best_epoch,
        best_val_mse,
    )


# ============================================================
# Prediction
# ============================================================

def predict(
    model,
    x,
):

    tensor = torch.from_numpy(
        x
    ).float()


    loader = DataLoader(

        TensorDataset(
            tensor
        ),

        batch_size=
            BATCH_SIZE,

        shuffle=False,

        num_workers=0,
    )


    outputs = []


    model.eval()


    with torch.inference_mode():

        for (
            x_batch,
        ) in loader:

            x_batch = x_batch.to(
                DEVICE
            )


            prediction = model(
                x_batch
            )


            outputs.append(
                prediction
                .cpu()
                .numpy()
            )


    return np.concatenate(
        outputs,
        axis=0,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    prediction,
):

    error = np.abs(
        prediction
        - test_action
    )


    squared_error = (
        prediction
        - test_action
    ) ** 2


    overall_mae = float(
        error.mean()
    )


    overall_mse = float(
        squared_error.mean()
    )


    per_task_mae = {}


    for task_index in range(
        NUM_TASKS
    ):

        task_mask = (
            test_task
            == task_index
        )


        per_task_mae[
            task_index
        ] = float(
            error[
                task_mask
            ].mean()
        )


    macro_task_mae = float(
        np.mean(
            list(
                per_task_mae.values()
            )
        )
    )


    return {

        "overall_mae":
            overall_mae,

        "overall_mse":
            overall_mse,

        "macro_task_mae":
            macro_task_mae,

        "per_task_mae":
            per_task_mae,
    }


# ============================================================
# Run 4 Models
# ============================================================

experiment_configs = [

    (
        "State-only",
        train_state_only,
        val_state_only,
        test_state_only,
    ),

    (
        "State + Zero Task",
        train_zero,
        val_zero,
        test_zero,
    ),

    (
        "State + Shuffled Task",
        train_shuffled,
        val_shuffled,
        test_shuffled,
    ),

    (
        "State + Correct Task",
        train_correct,
        val_correct,
        test_correct,
    ),
]


results = []

all_history = []

trained_models = {}


for (
    model_name,
    train_x,
    val_x,
    test_x,
) in experiment_configs:

    (
        model,
        history,
        best_epoch,
        best_val_mse,
    ) = train_model(

        model_name,

        train_x,

        val_x,
    )


    prediction = predict(
        model,
        test_x,
    )


    metrics = calculate_metrics(
        prediction
    )


    trained_models[
        model_name
    ] = model


    all_history.extend(
        history
    )


    results.append(
        {
            "model":
                model_name,

            "overall_mae":
                metrics[
                    "overall_mae"
                ],

            "overall_mse":
                metrics[
                    "overall_mse"
                ],

            "macro_task_mae":
                metrics[
                    "macro_task_mae"
                ],

            "best_epoch":
                best_epoch,

            "best_val_mse":
                best_val_mse,

            "per_task_mae":
                metrics[
                    "per_task_mae"
                ],
        }
    )


# ============================================================
# Key Comparisons
# ============================================================

result_lookup = {

    row[
        "model"
    ]:
        row

    for row in results
}


state_mae = result_lookup[
    "State-only"
][
    "overall_mae"
]


zero_mae = result_lookup[
    "State + Zero Task"
][
    "overall_mae"
]


shuffled_mae = result_lookup[
    "State + Shuffled Task"
][
    "overall_mae"
]


correct_mae = result_lookup[
    "State + Correct Task"
][
    "overall_mae"
]


def percent_change(
    baseline,
    new,
):

    return (
        (
            new
            - baseline
        )
        / baseline
        * 100.0
    )


correct_vs_state = percent_change(
    state_mae,
    correct_mae,
)


correct_vs_zero = percent_change(
    zero_mae,
    correct_mae,
)


correct_vs_shuffled = percent_change(
    shuffled_mae,
    correct_mae,
)


# ============================================================
# Final Results
# ============================================================

print("\n" + "=" * 90)
print("FINAL TASK CONDITIONING EXPERIMENT")
print("最终任务条件控制实验")
print("=" * 90)


for row in results:

    print("\n" + "-" * 90)


    print(
        row[
            "model"
        ]
    )


    print(
        f"Overall MAE"
        f"（整体平均绝对误差）: "
        f"{row['overall_mae']:.8f}"
    )


    print(
        f"Overall MSE"
        f"（整体均方误差）: "
        f"{row['overall_mse']:.8f}"
    )


    print(
        f"Macro Task MAE"
        f"（任务等权平均MAE）: "
        f"{row['macro_task_mae']:.8f}"
    )


    print(
        f"Best Epoch"
        f"（最佳轮次）: "
        f"{row['best_epoch']}"
    )


# ============================================================
# Key Comparison
# ============================================================

print("\n" + "=" * 90)
print("KEY COMPARISON")
print("关键对比")
print("=" * 90)


print(
    "\nCorrect Task vs State-only:"
)

print(
    f"{correct_vs_state:+.2f}%"
)


print(
    "\nCorrect Task vs Zero Task:"
)

print(
    f"{correct_vs_zero:+.2f}%"
)


print(
    "\nCorrect Task vs Shuffled Task:"
)

print(
    f"{correct_vs_shuffled:+.2f}%"
)


# ============================================================
# Automatic Interpretation
# ============================================================

print("\n" + "=" * 90)
print("AUTOMATIC INTERPRETATION")
print("自动实验解释")
print("=" * 90)


if (
    correct_mae < state_mae

    and correct_mae < zero_mae

    and correct_mae < shuffled_mae
):

    print(
        "\nRESULT A ✅"
    )


    print(
        "Correct Task Conditioning"
        "（正确任务条件）"
        "明显改善了 Action Prediction。"
    )


    print(
        "\n结论：Task Identity"
        "（任务身份）提供了 State 中缺失的"
        "增量动作信息。"
    )


    print(
        "\n下一步可以进入真正的 "
        "Language Conditioning"
        "（语言条件控制）。"
    )


elif (
    abs(
        correct_mae
        - shuffled_mae
    )
    / shuffled_mae
    < 0.02
):

    print(
        "\nRESULT B ⚠️"
    )


    print(
        "Correct Task 与 Shuffled Task "
        "表现非常接近。"
    )


    print(
        "\n没有证据证明模型真正利用了"
        "任务身份。"
    )


else:

    print(
        "\nRESULT C"
    )


    print(
        "结果呈混合模式。"
    )


    print(
        "需要结合 Macro Task MAE "
        "与 Per-task MAE 一起判断。"
    )


# ============================================================
# Save Comparison Report
# ============================================================

comparison_rows = []


for row in results:

    comparison_rows.append(
        {
            "model":
                row[
                    "model"
                ],

            "overall_mae":
                row[
                    "overall_mae"
                ],

            "overall_mse":
                row[
                    "overall_mse"
                ],

            "macro_task_mae":
                row[
                    "macro_task_mae"
                ],

            "best_epoch":
                row[
                    "best_epoch"
                ],

            "best_val_mse":
                row[
                    "best_val_mse"
                ],
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(

    COMPARISON_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Per-task Report
# ============================================================

per_task_rows = []


for task_index in range(
    NUM_TASKS
):

    row = {
        "task_index":
            task_index
    }


    for result in results:

        row[
            result[
                "model"
            ]
        ] = (
            result[
                "per_task_mae"
            ][
                task_index
            ]
        )


    per_task_rows.append(
        row
    )


per_task_df = pd.DataFrame(
    per_task_rows
)


per_task_df.to_csv(

    PER_TASK_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Training History
# ============================================================

history_df = pd.DataFrame(
    all_history
)


history_df.to_csv(

    HISTORY_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Models
# ============================================================

for model_name, model in trained_models.items():

    safe_name = (

        model_name

        .lower()

        .replace(
            " ",
            "_",
        )

        .replace(
            "+",
            "plus",
        )
    )


    model_path = (
        MODEL_DIR
        / f"libero_{safe_name}.pt"
    )


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "state_mean":
                state_mean,

            "state_std":
                state_std,

            "seed":
                SEED,

            "state_dim":
                STATE_DIM,

            "action_dim":
                ACTION_DIM,

            "num_tasks":
                NUM_TASKS,
        },

        model_path,
    )


# ============================================================
# Artifact Summary
# ============================================================

print("\n" + "=" * 90)
print("ARTIFACT SUMMARY")
print("实验资产汇总")
print("=" * 90)


print(
    f"\nSplit Report（数据划分报告）:\n"
    f"{SPLIT_REPORT_PATH}"
)


print(
    f"\nComparison Report（实验对比报告）:\n"
    f"{COMPARISON_REPORT_PATH}"
)


print(
    f"\nPer-task Report（逐任务报告）:\n"
    f"{PER_TASK_REPORT_PATH}"
)


print(
    f"\nTraining History（训练历史）:\n"
    f"{HISTORY_REPORT_PATH}"
)


print(
    "\nLIBERO Task Conditioning "
    "Experiment Complete ✅"
)