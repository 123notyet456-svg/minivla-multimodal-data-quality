from pathlib import Path
import copy
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
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

REPO_ID = "lerobot/libero_10"

LANGUAGE_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

STATE_DIM = 8
ACTION_DIM = 7
NUM_TASKS = 10

LANGUAGE_DIM = 384

STATE_BRANCH_DIM = 32
LANGUAGE_BRANCH_DIM = 32

HIDDEN_DIM = 128

BATCH_SIZE = 256
MAX_EPOCHS = 50

LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 8

SEED = 42


REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
CACHE_DIR = Path("cache")

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


COMPARISON_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_conditioning_comparison.csv"
)

PER_TASK_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_conditioning_per_task.csv"
)

HISTORY_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_conditioning_history.csv"
)

EMBEDDING_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_embedding_similarity.csv"
)

LANGUAGE_CACHE_PATH = (
    CACHE_DIR
    / "libero_minilm_language_embeddings.npz"
)


# ============================================================
# Canonical Language Instructions
# 原始自然语言任务指令
# ============================================================

TASK_TEXT_MAP = {

    0:
        "put the white mug on the left plate "
        "and put the yellow and white mug on the right plate",

    1:
        "put the white mug on the plate "
        "and put the chocolate pudding to the right of the plate",

    2:
        "put the yellow and white mug in the microwave "
        "and close it",

    3:
        "turn on the stove and put the moka pot on it",

    4:
        "put both the alphabet soup "
        "and the cream cheese box in the basket",

    5:
        "put both the alphabet soup "
        "and the tomato sauce in the basket",

    6:
        "put both moka pots on the stove",

    7:
        "put both the cream cheese box "
        "and the butter in the basket",

    8:
        "put the black bowl in the bottom drawer "
        "of the cabinet and close it",

    9:
        "pick up the book and place it "
        "in the back compartment of the caddy",
}


# ============================================================
# Paraphrased Language Instructions
# 同义改写语言任务
#
# 语义保持一致，
# 但文字表达发生变化。
# ============================================================

PARAPHRASE_TEXT_MAP = {

    0:
        "place the white mug on the left plate, "
        "then place the yellow and white mug "
        "on the right plate",

    1:
        "place the white mug on the plate, "
        "then position the chocolate pudding "
        "to the right side of the plate",

    2:
        "place the yellow and white mug "
        "inside the microwave, "
        "then close the microwave",

    3:
        "switch on the stove "
        "and then place the moka pot on the stove",

    4:
        "put the alphabet soup "
        "and the cream cheese box into the basket",

    5:
        "place the alphabet soup "
        "and the tomato sauce together in the basket",

    6:
        "place both moka pots on top of the stove",

    7:
        "put the cream cheese box "
        "and the butter into the basket",

    8:
        "place the black bowl "
        "in the cabinet's bottom drawer "
        "and then close it",

    9:
        "pick up the book "
        "and put it into the rear compartment "
        "of the caddy",
}


# ============================================================
# Reproducibility
# 可复现性
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
print("LIBERO LANGUAGE CONDITIONING EXPERIMENT")
print("LIBERO 自然语言条件控制实验")
print("=" * 90)

print(
    f"Device（计算设备）: {DEVICE}"
)

print(
    f"Language Encoder（语言编码器）: "
    f"{LANGUAGE_MODEL_NAME}"
)


# ============================================================
# Load Robot Data
# 读取机器人表格数据
# ============================================================

print("\n" + "=" * 90)
print("LOADING LIBERO ROBOT DATA")
print("读取 LIBERO 机器人数据")
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
        f"Unexpected LIBERO frame count: "
        f"{len(data_df)}"
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


print(
    f"State Shape（状态） : "
    f"{states.shape}"
)

print(
    f"Action Shape（动作）: "
    f"{actions.shape}"
)


if states.shape != (
    101469,
    STATE_DIM,
):

    raise RuntimeError(
        f"State shape incorrect: "
        f"{states.shape}"
    )


if actions.shape != (
    101469,
    ACTION_DIM,
):

    raise RuntimeError(
        f"Action shape incorrect: "
        f"{actions.shape}"
    )


print(
    "State / Action Shape Check: PASS ✅"
)


# ============================================================
# Episode-level Table
# Episode 级数据表
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


# ============================================================
# Reproduce Previous Split
# 复现上一轮 Task Conditioning 数据划分
#
# Seed = 42
#
# 每个 Task 内：
# 约 80% Train
# 10% Val
# 10% Test
# ============================================================

print("\n" + "=" * 90)
print("REPRODUCING PREVIOUS DATA SPLIT")
print("复现上一轮数据划分")
print("=" * 90)


rng = np.random.default_rng(
    SEED
)


train_episode_ids = []
val_episode_ids = []
test_episode_ids = []


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


    train_episode_ids.extend(

        task_episodes[
            :n_train
        ].tolist()
    )


    val_episode_ids.extend(

        task_episodes[
            n_train:
            n_train + n_val
        ].tolist()
    )


    test_episode_ids.extend(

        task_episodes[
            n_train + n_val:
        ].tolist()
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
    f"Train Episodes: "
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
# Leakage Check
# 数据泄漏检查
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
    "\nEpisode Leakage Check"
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
# Split Arrays
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
# Task Coverage Check
# ============================================================

expected_tasks = list(
    range(
        NUM_TASKS
    )
)


def split_tasks(
    task_array,
):

    return sorted(
        np.unique(
            task_array
        ).tolist()
    )


train_task_check = (
    split_tasks(
        train_task
    )
    == expected_tasks
)


val_task_check = (
    split_tasks(
        val_task
    )
    == expected_tasks
)


test_task_check = (
    split_tasks(
        test_task
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
        "Task coverage check failed."
    )


# ============================================================
# State Normalization
# 状态标准化
#
# 只能使用 Train 统计量，
# 防止 Data Leakage（数据泄漏）。
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
# Language Encoder
# 自然语言编码器
#
# Sentence Transformer：
#
# Language Instruction（语言指令）
# ↓
# MiniLM
# ↓
# 384D Language Embedding（语言嵌入）
#
# Language Encoder 本身冻结。
# ============================================================

print("\n" + "=" * 90)
print("LANGUAGE ENCODER")
print("自然语言编码器")
print("=" * 90)


if LANGUAGE_CACHE_PATH.exists():

    print(
        "Language Embedding Cache"
        "（语言嵌入缓存）已存在 ✅"
    )


    with np.load(
        LANGUAGE_CACHE_PATH
    ) as cache:

        canonical_embeddings = (

            cache[
                "canonical_embeddings"
            ]

            .copy()

            .astype(
                np.float32
            )
        )


        paraphrase_embeddings = (

            cache[
                "paraphrase_embeddings"
            ]

            .copy()

            .astype(
                np.float32
            )
        )


else:

    print(
        "Loading MiniLM Language Encoder..."
    )


    language_encoder = SentenceTransformer(
        LANGUAGE_MODEL_NAME,
        device="cpu",
    )


    canonical_sentences = [

        TASK_TEXT_MAP[
            task_index
        ]

        for task_index
        in range(
            NUM_TASKS
        )
    ]


    paraphrase_sentences = [

        PARAPHRASE_TEXT_MAP[
            task_index
        ]

        for task_index
        in range(
            NUM_TASKS
        )
    ]


    canonical_embeddings = (

        language_encoder.encode(

            canonical_sentences,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=True,
        )

        .astype(
            np.float32
        )
    )


    paraphrase_embeddings = (

        language_encoder.encode(

            paraphrase_sentences,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=True,
        )

        .astype(
            np.float32
        )
    )


    np.savez_compressed(

        LANGUAGE_CACHE_PATH,

        canonical_embeddings=
            canonical_embeddings,

        paraphrase_embeddings=
            paraphrase_embeddings,
    )


print(
    f"Canonical Embeddings"
    f"（原始语言嵌入）: "
    f"{canonical_embeddings.shape}"
)

print(
    f"Paraphrase Embeddings"
    f"（改写语言嵌入）: "
    f"{paraphrase_embeddings.shape}"
)


if canonical_embeddings.shape != (
    NUM_TASKS,
    LANGUAGE_DIM,
):

    raise RuntimeError(
        "Unexpected Canonical Language "
        "Embedding dimension."
    )


if paraphrase_embeddings.shape != (
    NUM_TASKS,
    LANGUAGE_DIM,
):

    raise RuntimeError(
        "Unexpected Paraphrase Language "
        "Embedding dimension."
    )


print(
    "Language Embedding Shape Check: PASS ✅"
)


# ============================================================
# Language Similarity Check
# 语言语义相似度检查
#
# Embedding 已经 L2 Normalization（L2归一化）
#
# 所以 Dot Product（点积）
# 等价于 Cosine Similarity（余弦相似度）。
# ============================================================

print("\n" + "=" * 90)
print("LANGUAGE SEMANTIC SIMILARITY")
print("语言语义相似度检查")
print("=" * 90)


similarity_rows = []


for task_index in range(
    NUM_TASKS
):

    similarity = float(

        np.dot(

            canonical_embeddings[
                task_index
            ],

            paraphrase_embeddings[
                task_index
            ],
        )
    )


    similarity_rows.append(
        {
            "task_index":
                task_index,

            "canonical_instruction":
                TASK_TEXT_MAP[
                    task_index
                ],

            "paraphrase_instruction":
                PARAPHRASE_TEXT_MAP[
                    task_index
                ],

            "cosine_similarity":
                similarity,
        }
    )


    print(
        f"Task {task_index}: "
        f"{similarity:.4f}"
    )


similarity_df = pd.DataFrame(
    similarity_rows
)


similarity_df.to_csv(

    EMBEDDING_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Language Embedding Mapping
# Task ID → Language Embedding
# ============================================================

def build_language_array(
    task_ids,
    embedding_table,
):

    return embedding_table[
        task_ids
    ].astype(
        np.float32
    )


train_language_correct = build_language_array(

    train_task,

    canonical_embeddings,
)


val_language_correct = build_language_array(

    val_task,

    canonical_embeddings,
)


test_language_correct = build_language_array(

    test_task,

    canonical_embeddings,
)


test_language_paraphrase = build_language_array(

    test_task,

    paraphrase_embeddings,
)


train_language_zero = np.zeros_like(
    train_language_correct
)

val_language_zero = np.zeros_like(
    val_language_correct
)

test_language_zero = np.zeros_like(
    test_language_correct
)


# ============================================================
# Strict Shuffled Language
# 严格错配语言
#
# 目标：
#
# 1. 每个 Episode 都获得错误 Task / Language
# 2. Correct Episode Matches = 0
# 3. Correct Frame Matches = 0
# 4. Task 标签总体分布完全不变
#
# 使用 Deterministic Derangement
# （确定性错排）
#
# 不再使用随机搜索碰运气。
# ============================================================

def build_strict_shuffled_task_ids(
    episode_array,
    task_array,
    seed,
):

    # ========================================================
    # Step 1
    # 获取每个 Episode 对应的唯一 Task
    # ========================================================

    unique_episodes = np.unique(
        episode_array
    )


    episode_task_pairs = []


    for episode_id in unique_episodes:

        mask = (
            episode_array
            == episode_id
        )


        labels = np.unique(
            task_array[
                mask
            ]
        )


        if len(labels) != 1:

            raise RuntimeError(
                f"Episode {episode_id} "
                f"contains multiple task labels."
            )


        episode_task_pairs.append(
            (
                int(
                    episode_id
                ),

                int(
                    labels[0]
                ),
            )
        )


    # ========================================================
    # Step 2
    # 按 Task 排序
    #
    # 同一个 Task 的 Episode
    # 会形成连续的 block（块）。
    # ========================================================

    episode_task_pairs = sorted(

        episode_task_pairs,

        key=lambda x: (
            x[1],
            x[0],
        ),
    )


    sorted_episodes = np.array(

        [
            pair[0]

            for pair
            in episode_task_pairs
        ],

        dtype=np.int64,
    )


    original_labels = np.array(

        [
            pair[1]

            for pair
            in episode_task_pairs
        ],

        dtype=np.int64,
    )


    # ========================================================
    # Step 3
    # Task Distribution（任务分布）
    # ========================================================

    task_counts = np.bincount(

        original_labels,

        minlength=NUM_TASKS,
    )


    max_task_count = int(
        task_counts.max()
    )


    total_episode_count = len(
        original_labels
    )


    print(
        "\nTask Episode Counts"
        "（各任务 Episode 数）:"
    )


    print(
        task_counts.tolist()
    )


    print(
        "Largest Task Count"
        "（最大单任务 Episode 数）: "
        f"{max_task_count}"
    )


    print(
        "Total Episodes"
        "（当前 Split Episode 总数）: "
        f"{total_episode_count}"
    )


    # ========================================================
    # 严格错排是否可能
    #
    # 最大 Task 数量不能超过总 Episode 的一半。
    #
    # 如果超过一半，
    # 那么该 Task 的 Episode 太多，
    # 必然有部分 Episode 无法获得不同 Task。
    # ========================================================

    if (
        max_task_count
        * 2
        > total_episode_count
    ):

        raise RuntimeError(
            "Strict zero-match task derangement "
            "is mathematically impossible because "
            "one task contains more than half "
            "of all episodes."
        )


    # ========================================================
    # Step 4
    # Deterministic Derangement
    # 确定性错排
    #
    # 已经按照 Task 排序，
    # 再循环平移最大类别大小。
    #
    # 这样：
    #
    # Task 标签数量保持不变
    # 但每个 Episode 获得不同 Task。
    # ========================================================

    shuffled_labels = np.roll(

        original_labels,

        -max_task_count,
    )


    # ========================================================
    # Step 5
    # Episode-level Match Check
    # ========================================================

    episode_matches = int(
        (
            shuffled_labels
            == original_labels
        ).sum()
    )


    if episode_matches != 0:

        raise RuntimeError(
            f"Deterministic derangement failed: "
            f"{episode_matches} Episode matches remain."
        )


    # ========================================================
    # Step 6
    # Task Distribution Preservation
    # 检查错排前后 Task 数量是否完全一致
    # ========================================================

    original_counts = np.bincount(

        original_labels,

        minlength=NUM_TASKS,
    )


    shuffled_counts = np.bincount(

        shuffled_labels,

        minlength=NUM_TASKS,
    )


    distribution_preserved = np.array_equal(

        original_counts,

        shuffled_counts,
    )


    if not distribution_preserved:

        raise RuntimeError(
            "Shuffled Task Distribution changed."
        )


    # ========================================================
    # Step 7
    # Episode → Shuffled Task Mapping
    # ========================================================

    episode_to_task = {

        int(
            episode_id
        ):
            int(
                shuffled_task
            )

        for episode_id, shuffled_task
        in zip(
            sorted_episodes,
            shuffled_labels,
        )
    }


    # ========================================================
    # Step 8
    # 展开回 Frame-level Task Array
    # ========================================================

    output = np.array(

        [

            episode_to_task[
                int(
                    episode_id
                )
            ]

            for episode_id
            in episode_array
        ],

        dtype=np.int64,
    )


    # ========================================================
    # Step 9
    # Frame-level Match Check
    # ========================================================

    frame_matches = int(
        (
            output
            == task_array
        ).sum()
    )


    if frame_matches != 0:

        raise RuntimeError(
            f"Frame-level shuffled language "
            f"still has {frame_matches} "
            f"correct task matches."
        )


    print(
        "Task Distribution Preserved"
        "（任务分布保持）: PASS ✅"
    )


    return (
        output,
        episode_matches,
        frame_matches,
    )


# ============================================================
# Build Strict Shuffled Tasks
# ============================================================

print("\n" + "=" * 90)
print("BUILDING STRICT SHUFFLED LANGUAGE")
print("构建严格错配语言")
print("=" * 90)


(
    train_shuffled_task,
    train_episode_matches,
    train_frame_matches,
) = build_strict_shuffled_task_ids(

    train_episode,

    train_task,

    SEED + 100,
)


(
    val_shuffled_task,
    val_episode_matches,
    val_frame_matches,
) = build_strict_shuffled_task_ids(

    val_episode,

    val_task,

    SEED + 200,
)


(
    test_shuffled_task,
    test_episode_matches,
    test_frame_matches,
) = build_strict_shuffled_task_ids(

    test_episode,

    test_task,

    SEED + 300,
)


train_language_shuffled = build_language_array(

    train_shuffled_task,

    canonical_embeddings,
)


val_language_shuffled = build_language_array(

    val_shuffled_task,

    canonical_embeddings,
)


test_language_shuffled = build_language_array(

    test_shuffled_task,

    canonical_embeddings,
)


print("\n" + "=" * 90)
print("STRICT SHUFFLED LANGUAGE CHECK")
print("严格错配语言检查")
print("=" * 90)


print(
    f"Train Episode Matches: "
    f"{train_episode_matches}"
)

print(
    f"Val Episode Matches  : "
    f"{val_episode_matches}"
)

print(
    f"Test Episode Matches : "
    f"{test_episode_matches}"
)


print(
    f"\nTrain Frame Matches: "
    f"{train_frame_matches}"
)

print(
    f"Val Frame Matches  : "
    f"{val_frame_matches}"
)

print(
    f"Test Frame Matches : "
    f"{test_frame_matches}"
)


if (
    train_episode_matches != 0
    or val_episode_matches != 0
    or test_episode_matches != 0
    or train_frame_matches != 0
    or val_frame_matches != 0
    or test_frame_matches != 0
):

    raise RuntimeError(
        "Strict Language Shuffle failed."
    )


print(
    "\nStrict Language Shuffle: PASS ✅"
)


# ============================================================
# State-only Policy
# 纯状态策略
# ============================================================

class StateOnlyPolicy(nn.Module):

    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                STATE_DIM,
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
        state,
    ):

        return self.network(
            state
        )


# ============================================================
# Language-conditioned Policy
# 语言条件策略
#
# State Branch（状态分支）
# 8D → 32D
#
# Language Branch（语言分支）
# 384D → 32D
#
# Fusion（融合）
# 32 + 32 = 64D
#
# ↓
#
# 128
# ↓
# 128
# ↓
# Action 7D
#
# Zero / Shuffled / Correct
# 使用完全相同结构。
# ============================================================

class LanguageConditionedPolicy(nn.Module):

    def __init__(self):

        super().__init__()


        self.state_branch = nn.Sequential(

            nn.Linear(
                STATE_DIM,
                STATE_BRANCH_DIM,
            ),

            nn.ReLU(),
        )


        self.language_branch = nn.Sequential(

            nn.Linear(
                LANGUAGE_DIM,
                LANGUAGE_BRANCH_DIM,
            ),

            nn.ReLU(),
        )


        fusion_dim = (
            STATE_BRANCH_DIM
            + LANGUAGE_BRANCH_DIM
        )


        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_dim,
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
        state,
        language,
    ):

        state_feature = self.state_branch(
            state
        )


        language_feature = self.language_branch(
            language
        )


        fused = torch.cat(
            [
                state_feature,
                language_feature,
            ],
            dim=1,
        )


        return self.fusion(
            fused
        )


# ============================================================
# DataLoader
# ============================================================

def make_state_loader(
    state,
    action,
    shuffle,
):

    dataset = TensorDataset(

        torch.from_numpy(
            state
        ).float(),

        torch.from_numpy(
            action
        ).float(),
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


def make_language_loader(
    state,
    language,
    action,
    shuffle,
):

    dataset = TensorDataset(

        torch.from_numpy(
            state
        ).float(),

        torch.from_numpy(
            language
        ).float(),

        torch.from_numpy(
            action
        ).float(),
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
# Train State-only
# ============================================================

def train_state_only():

    print("\n" + "=" * 90)
    print("TRAINING: State-only")
    print("训练：纯状态模型")
    print("=" * 90)


    set_seed(SEED)


    model = StateOnlyPolicy().to(
        DEVICE
    )


    parameter_count = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


    print(
        f"Trainable Parameters"
        f"（可训练参数）: "
        f"{parameter_count:,}"
    )


    train_loader = make_state_loader(

        train_state_n,

        train_action,

        True,
    )


    val_loader = make_state_loader(

        val_state_n,

        val_action,

        False,
    )


    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,
    )


    best_val = float(
        "inf"
    )

    best_epoch = -1

    best_weights = None

    no_improvement = 0

    history = []


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        train_sum = 0.0
        train_count = 0


        for (
            state_batch,
            action_batch,
        ) in train_loader:

            state_batch = state_batch.to(
                DEVICE
            )

            action_batch = action_batch.to(
                DEVICE
            )


            optimizer.zero_grad()


            prediction = model(
                state_batch
            )


            loss = criterion(
                prediction,
                action_batch,
            )


            loss.backward()

            optimizer.step()


            n = len(
                state_batch
            )


            train_sum += (
                loss.item()
                * n
            )

            train_count += n


        train_mse = (
            train_sum
            / train_count
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_sum = 0.0
        val_count = 0


        with torch.inference_mode():

            for (
                state_batch,
                action_batch,
            ) in val_loader:

                state_batch = state_batch.to(
                    DEVICE
                )

                action_batch = action_batch.to(
                    DEVICE
                )


                prediction = model(
                    state_batch
                )


                loss = criterion(
                    prediction,
                    action_batch,
                )


                n = len(
                    state_batch
                )


                val_sum += (
                    loss.item()
                    * n
                )

                val_count += n


        val_mse = (
            val_sum
            / val_count
        )


        history.append(
            {
                "model":
                    "State-only",

                "epoch":
                    epoch,

                "train_mse":
                    train_mse,

                "val_mse":
                    val_mse,
            }
        )


        # ----------------------------------------------------
        # Best Model / Early Stopping
        # ----------------------------------------------------

        if val_mse < best_val:

            best_val = val_mse

            best_epoch = epoch

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            no_improvement = 0


        else:

            no_improvement += 1


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
            no_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly Stopping"
                f"（提前停止） at "
                f"Epoch {epoch}"
            )

            break


    if best_weights is None:

        raise RuntimeError(
            "State-only training failed "
            "to produce best weights."
        )


    model.load_state_dict(
        best_weights
    )


    print(
        f"\nBest Epoch（最佳轮次）: "
        f"{best_epoch}"
    )


    print(
        f"Best Val MSE（最佳验证MSE）: "
        f"{best_val:.8f}"
    )


    return (
        model,
        history,
        best_epoch,
        best_val,
    )


# ============================================================
# Train Language Policy
# 训练语言条件模型
# ============================================================

def train_language_policy(
    model_name,
    train_language,
    val_language,
):

    print("\n" + "=" * 90)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 90)


    set_seed(SEED)


    model = (
        LanguageConditionedPolicy()
        .to(DEVICE)
    )


    parameter_count = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


    print(
        f"Trainable Parameters"
        f"（可训练参数）: "
        f"{parameter_count:,}"
    )


    train_loader = make_language_loader(

        train_state_n,

        train_language,

        train_action,

        True,
    )


    val_loader = make_language_loader(

        val_state_n,

        val_language,

        val_action,

        False,
    )


    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,
    )


    best_val = float(
        "inf"
    )

    best_epoch = -1

    best_weights = None

    no_improvement = 0

    history = []


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        train_sum = 0.0
        train_count = 0


        for (
            state_batch,
            language_batch,
            action_batch,
        ) in train_loader:

            state_batch = state_batch.to(
                DEVICE
            )

            language_batch = language_batch.to(
                DEVICE
            )

            action_batch = action_batch.to(
                DEVICE
            )


            optimizer.zero_grad()


            prediction = model(
                state_batch,
                language_batch,
            )


            loss = criterion(
                prediction,
                action_batch,
            )


            loss.backward()

            optimizer.step()


            n = len(
                state_batch
            )


            train_sum += (
                loss.item()
                * n
            )

            train_count += n


        train_mse = (
            train_sum
            / train_count
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_sum = 0.0
        val_count = 0


        with torch.inference_mode():

            for (
                state_batch,
                language_batch,
                action_batch,
            ) in val_loader:

                state_batch = state_batch.to(
                    DEVICE
                )

                language_batch = language_batch.to(
                    DEVICE
                )

                action_batch = action_batch.to(
                    DEVICE
                )


                prediction = model(
                    state_batch,
                    language_batch,
                )


                loss = criterion(
                    prediction,
                    action_batch,
                )


                n = len(
                    state_batch
                )


                val_sum += (
                    loss.item()
                    * n
                )

                val_count += n


        val_mse = (
            val_sum
            / val_count
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
        # Best Model / Early Stopping
        # ----------------------------------------------------

        if val_mse < best_val:

            best_val = val_mse

            best_epoch = epoch

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            no_improvement = 0


        else:

            no_improvement += 1


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
            no_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly Stopping"
                f"（提前停止） at "
                f"Epoch {epoch}"
            )

            break


    if best_weights is None:

        raise RuntimeError(
            f"{model_name} failed "
            f"to produce best weights."
        )


    model.load_state_dict(
        best_weights
    )


    print(
        f"\nBest Epoch（最佳轮次）: "
        f"{best_epoch}"
    )


    print(
        f"Best Val MSE（最佳验证MSE）: "
        f"{best_val:.8f}"
    )


    return (
        model,
        history,
        best_epoch,
        best_val,
    )


# ============================================================
# Prediction Functions
# 推理函数
# ============================================================

def predict_state_only(
    model,
):

    loader = make_state_loader(

        test_state_n,

        test_action,

        False,
    )


    outputs = []


    model.eval()


    with torch.inference_mode():

        for (
            state_batch,
            _,
        ) in loader:

            state_batch = state_batch.to(
                DEVICE
            )


            prediction = model(
                state_batch
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


def predict_language(
    model,
    language,
):

    loader = make_language_loader(

        test_state_n,

        language,

        test_action,

        False,
    )


    outputs = []


    model.eval()


    with torch.inference_mode():

        for (
            state_batch,
            language_batch,
            _,
        ) in loader:

            state_batch = state_batch.to(
                DEVICE
            )

            language_batch = language_batch.to(
                DEVICE
            )


            prediction = model(

                state_batch,

                language_batch,
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
# 评测指标
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


    per_task = {}


    for task_index in range(
        NUM_TASKS
    ):

        task_mask = (
            test_task
            == task_index
        )


        per_task[
            task_index
        ] = float(

            error[
                task_mask
            ].mean()
        )


    return {

        "overall_mae":
            float(
                error.mean()
            ),

        "overall_mse":
            float(
                squared_error.mean()
            ),

        "macro_task_mae":
            float(
                np.mean(
                    list(
                        per_task.values()
                    )
                )
            ),

        "per_task":
            per_task,
    }


# ============================================================
# Train All Models
# ============================================================

(
    state_model,
    state_history,
    state_best_epoch,
    state_best_val,
) = train_state_only()


(
    zero_model,
    zero_history,
    zero_best_epoch,
    zero_best_val,
) = train_language_policy(

    "State + Zero Language",

    train_language_zero,

    val_language_zero,
)


(
    shuffled_model,
    shuffled_history,
    shuffled_best_epoch,
    shuffled_best_val,
) = train_language_policy(

    "State + Shuffled Language",

    train_language_shuffled,

    val_language_shuffled,
)


(
    correct_model,
    correct_history,
    correct_best_epoch,
    correct_best_val,
) = train_language_policy(

    "State + Correct Language",

    train_language_correct,

    val_language_correct,
)


# ============================================================
# Test Predictions
# ============================================================

state_prediction = predict_state_only(
    state_model
)


zero_prediction = predict_language(

    zero_model,

    test_language_zero,
)


shuffled_prediction = predict_language(

    shuffled_model,

    test_language_shuffled,
)


correct_prediction = predict_language(

    correct_model,

    test_language_correct,
)


# ============================================================
# Semantic Generalization Test
# 语义泛化测试
#
# 这里不重新训练模型。
#
# 使用：
#
# Correct Canonical Language
# （正确原始语言）
#
# 训练好的模型
#
# 直接测试：
#
# Paraphrased Language
# （从未训练过的同义改写语言）
#
# 用来测试：
#
# 模型是否能够利用
# Language Semantics（语言语义）
# ============================================================

paraphrase_prediction = predict_language(

    correct_model,

    test_language_paraphrase,
)


# ============================================================
# Calculate Metrics
# ============================================================

results = {

    "State-only":
        calculate_metrics(
            state_prediction
        ),

    "State + Zero Language":
        calculate_metrics(
            zero_prediction
        ),

    "State + Shuffled Language":
        calculate_metrics(
            shuffled_prediction
        ),

    "State + Correct Language":
        calculate_metrics(
            correct_prediction
        ),

    "Correct Model + Paraphrased Language":
        calculate_metrics(
            paraphrase_prediction
        ),
}


# ============================================================
# Percentage Change
# ============================================================

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


state_mae = results[
    "State-only"
][
    "overall_mae"
]


zero_mae = results[
    "State + Zero Language"
][
    "overall_mae"
]


shuffled_mae = results[
    "State + Shuffled Language"
][
    "overall_mae"
]


correct_mae = results[
    "State + Correct Language"
][
    "overall_mae"
]


paraphrase_mae = results[
    "Correct Model + Paraphrased Language"
][
    "overall_mae"
]


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


paraphrase_vs_correct = percent_change(

    correct_mae,

    paraphrase_mae,
)


# ============================================================
# Previous Task-ID Baseline
# 读取上一轮 Task Conditioning 结果
# ============================================================

previous_report_path = (

    REPORT_DIR
    / "libero_task_conditioning_comparison.csv"
)


previous_task_mae = None


if previous_report_path.exists():

    previous_df = pd.read_csv(
        previous_report_path
    )


    row = previous_df[
        previous_df[
            "model"
        ]
        == "State + Correct Task"
    ]


    if len(row) == 1:

        previous_task_mae = float(
            row.iloc[0][
                "overall_mae"
            ]
        )


# ============================================================
# Final Results
# ============================================================

print("\n" + "=" * 90)
print("FINAL LANGUAGE CONDITIONING EXPERIMENT")
print("最终自然语言条件控制实验")
print("=" * 90)


for model_name, metrics in results.items():

    print("\n" + "-" * 90)

    print(
        model_name
    )


    print(
        f"Overall MAE"
        f"（整体平均绝对误差）: "
        f"{metrics['overall_mae']:.8f}"
    )


    print(
        f"Overall MSE"
        f"（整体均方误差）: "
        f"{metrics['overall_mse']:.8f}"
    )


    print(
        f"Macro Task MAE"
        f"（任务等权MAE）: "
        f"{metrics['macro_task_mae']:.8f}"
    )


# ============================================================
# Key Comparison
# ============================================================

print("\n" + "=" * 90)
print("KEY COMPARISON")
print("关键对比")
print("=" * 90)


print(
    "\nCorrect Language vs State-only:"
)

print(
    f"{correct_vs_state:+.2f}%"
)


print(
    "\nCorrect Language vs Zero Language:"
)

print(
    f"{correct_vs_zero:+.2f}%"
)


print(
    "\nCorrect Language vs Shuffled Language:"
)

print(
    f"{correct_vs_shuffled:+.2f}%"
)


print(
    "\nParaphrased Language vs Correct Language:"
)

print(
    f"{paraphrase_vs_correct:+.2f}%"
)


if previous_task_mae is not None:

    print(
        "\nPrevious Correct Task-ID MAE"
        "（上一轮正确任务ID MAE）:"
    )

    print(
        f"{previous_task_mae:.8f}"
    )


    language_vs_task_id = percent_change(

        previous_task_mae,

        correct_mae,
    )


    print(
        "\nCorrect Language vs Correct Task-ID:"
    )

    print(
        f"{language_vs_task_id:+.2f}%"
    )


# ============================================================
# Automatic Interpretation
# ============================================================

print("\n" + "=" * 90)
print("AUTOMATIC INTERPRETATION")
print("自动实验解释")
print("=" * 90)


language_useful = (

    correct_mae < state_mae

    and correct_mae < zero_mae

    and correct_mae < shuffled_mae
)


paraphrase_gap = abs(
    paraphrase_vs_correct
)


if (
    language_useful
    and paraphrase_gap <= 5.0
):

    print(
        "\nRESULT A ✅"
    )


    print(
        "Correct Language"
        "（正确自然语言）"
        "明显改善 Action Prediction"
        "（动作预测）。"
    )


    print(
        "\n而且 Paraphrased Language"
        "（同义改写语言）"
        "与原始语言表现接近。"
    )


    print(
        "\n结论：不仅 Task Identity"
        "（任务身份）有价值，"
        "自然语言的 Semantic Representation"
        "（语义表示）也能够作为机器人策略的"
        "有效任务条件。"
    )


elif language_useful:

    print(
        "\nRESULT B ✅/⚠️"
    )


    print(
        "Correct Language 明显有效，"
        "但同义改写语言出现较大性能下降。"
    )


    print(
        "\n结论：Language Embedding"
        "（语言嵌入）能够充当 "
        "Task Condition（任务条件），"
        "但 Semantic Generalization"
        "（语义泛化）仍然有限。"
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
        "\nRESULT C ⚠️"
    )


    print(
        "Correct Language 与 "
        "Shuffled Language 表现接近。"
    )


    print(
        "\n没有证据证明策略真正利用了"
        "语言条件。"
    )


else:

    print(
        "\nRESULT D"
    )


    print(
        "结果呈混合模式。"
    )


    print(
        "需要结合 Per-task MAE"
        "（逐任务误差）进一步判断。"
    )


# ============================================================
# Save Comparison Report
# ============================================================

comparison_rows = []


for model_name, metrics in results.items():

    comparison_rows.append(
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
        }
    )


pd.DataFrame(
    comparison_rows
).to_csv(

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
            task_index,

        "instruction":
            TASK_TEXT_MAP[
                task_index
            ],
    }


    for model_name, metrics in results.items():

        row[
            model_name
        ] = metrics[
            "per_task"
        ][
            task_index
        ]


    per_task_rows.append(
        row
    )


pd.DataFrame(
    per_task_rows
).to_csv(

    PER_TASK_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Training History
# ============================================================

all_history = (

    state_history

    + zero_history

    + shuffled_history

    + correct_history
)


pd.DataFrame(
    all_history
).to_csv(

    HISTORY_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Correct Language Model
# ============================================================

torch.save(

    {

        "model_state_dict":
            correct_model.state_dict(),

        "state_mean":
            state_mean,

        "state_std":
            state_std,

        "canonical_embeddings":
            canonical_embeddings,

        "paraphrase_embeddings":
            paraphrase_embeddings,

        "language_model":
            LANGUAGE_MODEL_NAME,

        "language_dim":
            LANGUAGE_DIM,

        "state_dim":
            STATE_DIM,

        "action_dim":
            ACTION_DIM,

        "state_branch_dim":
            STATE_BRANCH_DIM,

        "language_branch_dim":
            LANGUAGE_BRANCH_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "seed":
            SEED,

        "best_epoch":
            correct_best_epoch,

        "best_val_mse":
            correct_best_val,
    },

    MODEL_DIR
    / "libero_state_correct_language.pt",
)


# ============================================================
# Artifact Summary
# ============================================================

print("\n" + "=" * 90)
print("ARTIFACT SUMMARY")
print("实验资产汇总")
print("=" * 90)


print(
    f"\nLanguage Embedding Cache"
    f"（语言嵌入缓存）:\n"
    f"{LANGUAGE_CACHE_PATH}"
)


print(
    f"\nComparison Report"
    f"（语言条件对比报告）:\n"
    f"{COMPARISON_REPORT_PATH}"
)


print(
    f"\nPer-task Report"
    f"（逐任务语言报告）:\n"
    f"{PER_TASK_REPORT_PATH}"
)


print(
    f"\nLanguage Similarity Report"
    f"（语言相似度报告）:\n"
    f"{EMBEDDING_REPORT_PATH}"
)


print(
    f"\nTraining History"
    f"（训练历史）:\n"
    f"{HISTORY_REPORT_PATH}"
)


print(
    "\nCorrect Language Model"
    "（正确语言模型）:\n"
    "models\\libero_state_correct_language.pt"
)


print(
    "\nLIBERO Language Conditioning "
    "Experiment Complete ✅"
)