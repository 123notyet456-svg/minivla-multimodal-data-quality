from pathlib import Path
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

LANGUAGE_DIM = 384

STATE_BRANCH_DIM = 32
LANGUAGE_BRANCH_DIM = 32

HIDDEN_DIM = 128

BATCH_SIZE = 256

SEED = 42


MODEL_PATH = Path(
    "models/libero_state_correct_language.pt"
)

REPORT_DIR = Path(
    "reports"
)

REPORT_DIR.mkdir(
    exist_ok=True
)


COMPARISON_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_test_time_intervention.csv"
)

WRONG_SHIFT_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_wrong_shift_results.csv"
)

PER_TASK_REPORT_PATH = (
    REPORT_DIR
    / "libero_language_intervention_per_task.csv"
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
print("LANGUAGE TEST-TIME INTERVENTION EXPERIMENT")
print("语言测试时干预实验")
print("=" * 90)

print(
    f"Device（计算设备）: "
    f"{DEVICE}"
)

print(
    f"Model（模型）: "
    f"{MODEL_PATH}"
)


# ============================================================
# Check Model
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: "
        f"{MODEL_PATH}"
    )


# ============================================================
# Language-conditioned Policy
# 必须与上一轮训练结构完全一致
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

        state_feature = (
            self.state_branch(
                state
            )
        )


        language_feature = (
            self.language_branch(
                language
            )
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
# Load Trained Correct Language Model
# ============================================================

print("\n" + "=" * 90)
print("LOADING TRAINED MODEL")
print("加载已经训练好的正确语言模型")
print("=" * 90)


checkpoint = torch.load(

    MODEL_PATH,

    map_location="cpu",
    weights_only=False,
)


model = (
    LanguageConditionedPolicy()
    .to(DEVICE)
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


state_mean = (

    checkpoint[
        "state_mean"
    ]

    .astype(
        np.float32
    )
)


state_std = (

    checkpoint[
        "state_std"
    ]

    .astype(
        np.float32
    )
)


canonical_embeddings = (

    checkpoint[
        "canonical_embeddings"
    ]

    .astype(
        np.float32
    )
)


paraphrase_embeddings = (

    checkpoint[
        "paraphrase_embeddings"
    ]

    .astype(
        np.float32
    )
)


print(
    "Model Loading Check"
    "（模型加载检查）: PASS ✅"
)


print(
    f"Canonical Language Shape"
    f"（原始语言嵌入）: "
    f"{canonical_embeddings.shape}"
)


print(
    f"Paraphrase Language Shape"
    f"（同义改写语言嵌入）: "
    f"{paraphrase_embeddings.shape}"
)


# ============================================================
# Shape Check
# ============================================================

if canonical_embeddings.shape != (
    NUM_TASKS,
    LANGUAGE_DIM,
):

    raise RuntimeError(
        "Canonical embedding shape incorrect."
    )


if paraphrase_embeddings.shape != (
    NUM_TASKS,
    LANGUAGE_DIM,
):

    raise RuntimeError(
        "Paraphrase embedding shape incorrect."
    )


print(
    "Language Embedding Shape Check: PASS ✅"
)


# ============================================================
# Load LIBERO Tabular Data
# ============================================================

print("\n" + "=" * 90)
print("LOADING LIBERO TEST DATA")
print("读取 LIBERO 测试数据")
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


# ============================================================
# Convert Arrays
# ============================================================

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


# ============================================================
# Reproduce EXACT Same Episode Split
# 与之前实验完全相同
# ============================================================

print("\n" + "=" * 90)
print("REPRODUCING TEST SPLIT")
print("复现完全相同的测试集")
print("=" * 90)


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
    )
)


episode_df[
    "episode_index"
] = (

    episode_df[
        "episode_index"
    ]
    .astype(int)
)


episode_df[
    "task_index"
] = (

    episode_df[
        "task_index"
    ]
    .astype(int)
)


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


test_episode_ids = np.array(

    test_episode_ids,

    dtype=np.int64,
)


test_mask = np.isin(

    episodes,

    test_episode_ids,
)


test_state = states[
    test_mask
]


test_action = actions[
    test_mask
]


test_task = tasks[
    test_mask
]


test_episode = episodes[
    test_mask
]


print(
    f"Test Episodes"
    f"（测试轨迹）: "
    f"{len(test_episode_ids)}"
)


print(
    f"Test Frames"
    f"（测试帧）: "
    f"{len(test_state)}"
)


if len(test_episode_ids) != 38:

    raise RuntimeError(
        "Unexpected Test Episode count."
    )


if len(test_state) != 9971:

    raise RuntimeError(
        "Unexpected Test Frame count."
    )


print(
    "Test Split Reproduction: PASS ✅"
)


# ============================================================
# State Normalization
# 使用训练模型保存下来的统计量
# ============================================================

test_state_n = (

    (
        test_state
        - state_mean
    )

    / state_std

).astype(
    np.float32
)


# ============================================================
# Build Language Inputs
# ============================================================

# Correct Language
test_language_correct = (

    canonical_embeddings[
        test_task
    ]

    .astype(
        np.float32
    )
)


# Paraphrased Correct Language
test_language_paraphrase = (

    paraphrase_embeddings[
        test_task
    ]

    .astype(
        np.float32
    )
)


# Zero Language
test_language_zero = np.zeros_like(

    test_language_correct,

    dtype=np.float32,
)


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    state_array,
    language_array,
):

    dataset = TensorDataset(

        torch.from_numpy(
            state_array
        ).float(),

        torch.from_numpy(
            language_array
        ).float(),
    )


    return DataLoader(

        dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=False,

        num_workers=0,
    )


# ============================================================
# Prediction
# ============================================================

def predict(
    language_array,
):

    loader = make_loader(

        test_state_n,

        language_array,
    )


    outputs = []


    model.eval()


    with torch.inference_mode():

        for (
            state_batch,
            language_batch,
        ) in loader:

            state_batch = state_batch.to(
                DEVICE
            )


            language_batch = (
                language_batch.to(
                    DEVICE
                )
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
# ============================================================

def calculate_metrics(
    prediction,
):

    absolute_error = np.abs(

        prediction
        - test_action
    )


    squared_error = (

        prediction
        - test_action

    ) ** 2


    overall_mae = float(
        absolute_error.mean()
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

            absolute_error[
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
# Correct Language
# ============================================================

print("\n" + "=" * 90)
print("TESTING CORRECT LANGUAGE")
print("测试正确语言")
print("=" * 90)


correct_prediction = predict(

    test_language_correct
)


correct_metrics = calculate_metrics(

    correct_prediction
)


print(
    f"Overall MAE: "
    f"{correct_metrics['overall_mae']:.8f}"
)


# ============================================================
# Paraphrased Language
# ============================================================

print("\n" + "=" * 90)
print("TESTING PARAPHRASED LANGUAGE")
print("测试同义改写语言")
print("=" * 90)


paraphrase_prediction = predict(

    test_language_paraphrase
)


paraphrase_metrics = calculate_metrics(

    paraphrase_prediction
)


print(
    f"Overall MAE: "
    f"{paraphrase_metrics['overall_mae']:.8f}"
)


# ============================================================
# Zero Language
# ============================================================

print("\n" + "=" * 90)
print("TESTING ZERO LANGUAGE")
print("测试全零语言")
print("=" * 90)


zero_prediction = predict(

    test_language_zero
)


zero_metrics = calculate_metrics(

    zero_prediction
)


print(
    f"Overall MAE: "
    f"{zero_metrics['overall_mae']:.8f}"
)


# ============================================================
# Wrong Language Interventions
#
# 10 个任务共有 9 个非零循环平移。
#
# shift = 1:
# Task 0 → Language 1
# Task 1 → Language 2
# ...
#
# shift = 9:
# Task 0 → Language 9
# ...
#
# 每一种：
# Correct Match = 0
#
# 不训练。
# 只做 Test-time Intervention（测试时干预）。
# ============================================================

print("\n" + "=" * 90)
print("TESTING 9 STRICT WRONG LANGUAGE INTERVENTIONS")
print("测试9种严格错误语言干预")
print("=" * 90)


wrong_shift_rows = []

wrong_shift_metrics = []


for shift in range(
    1,
    NUM_TASKS
):

    wrong_task_ids = (

        (
            test_task
            + shift
        )

        % NUM_TASKS
    )


    correct_matches = int(

        (
            wrong_task_ids
            == test_task
        ).sum()
    )


    if correct_matches != 0:

        raise RuntimeError(
            f"Shift {shift} contains "
            f"{correct_matches} correct matches."
        )


    wrong_language = (

        canonical_embeddings[
            wrong_task_ids
        ]

        .astype(
            np.float32
        )
    )


    prediction = predict(
        wrong_language
    )


    metrics = calculate_metrics(
        prediction
    )


    wrong_shift_metrics.append(
        metrics
    )


    wrong_shift_rows.append(
        {
            "shift":
                shift,

            "correct_frame_matches":
                correct_matches,

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


    print(
        f"Shift {shift}: "
        f"MAE = "
        f"{metrics['overall_mae']:.8f}"
    )


# ============================================================
# Wrong Language Aggregate
# ============================================================

wrong_maes = np.array(

    [
        metrics[
            "overall_mae"
        ]

        for metrics
        in wrong_shift_metrics
    ],

    dtype=np.float64,
)


wrong_mses = np.array(

    [
        metrics[
            "overall_mse"
        ]

        for metrics
        in wrong_shift_metrics
    ],

    dtype=np.float64,
)


wrong_macro_maes = np.array(

    [
        metrics[
            "macro_task_mae"
        ]

        for metrics
        in wrong_shift_metrics
    ],

    dtype=np.float64,
)


wrong_mean_mae = float(
    wrong_maes.mean()
)


wrong_std_mae = float(
    wrong_maes.std()
)


wrong_best_mae = float(
    wrong_maes.min()
)


wrong_worst_mae = float(
    wrong_maes.max()
)


wrong_mean_mse = float(
    wrong_mses.mean()
)


wrong_mean_macro_mae = float(
    wrong_macro_maes.mean()
)


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


paraphrase_vs_correct = percent_change(

    correct_metrics[
        "overall_mae"
    ],

    paraphrase_metrics[
        "overall_mae"
    ],
)


zero_vs_correct = percent_change(

    correct_metrics[
        "overall_mae"
    ],

    zero_metrics[
        "overall_mae"
    ],
)


wrong_vs_correct = percent_change(

    correct_metrics[
        "overall_mae"
    ],

    wrong_mean_mae,
)


best_wrong_vs_correct = percent_change(

    correct_metrics[
        "overall_mae"
    ],

    wrong_best_mae,
)


# ============================================================
# Final Results
# ============================================================

print("\n" + "=" * 90)
print("FINAL LANGUAGE TEST-TIME INTERVENTION")
print("最终语言测试时干预实验")
print("=" * 90)


print(
    "\nCorrect Language"
    "（正确语言）:"
)

print(
    f"MAE = "
    f"{correct_metrics['overall_mae']:.8f}"
)


print(
    "\nParaphrased Language"
    "（同义改写语言）:"
)

print(
    f"MAE = "
    f"{paraphrase_metrics['overall_mae']:.8f}"
)


print(
    "\nZero Language"
    "（全零语言）:"
)

print(
    f"MAE = "
    f"{zero_metrics['overall_mae']:.8f}"
)


print(
    "\nWrong Language Average"
    "（9种错误语言平均）:"
)

print(
    f"MAE Mean = "
    f"{wrong_mean_mae:.8f}"
)

print(
    f"MAE Std  = "
    f"{wrong_std_mae:.8f}"
)

print(
    f"Best Wrong MAE  = "
    f"{wrong_best_mae:.8f}"
)

print(
    f"Worst Wrong MAE = "
    f"{wrong_worst_mae:.8f}"
)


# ============================================================
# Key Comparison
# ============================================================

print("\n" + "=" * 90)
print("KEY COMPARISON")
print("关键对比")
print("=" * 90)


print(
    "\nParaphrased vs Correct:"
)

print(
    f"{paraphrase_vs_correct:+.2f}%"
)


print(
    "\nZero vs Correct:"
)

print(
    f"{zero_vs_correct:+.2f}%"
)


print(
    "\nWrong Language Average vs Correct:"
)

print(
    f"{wrong_vs_correct:+.2f}%"
)


print(
    "\nBest Wrong Language vs Correct:"
)

print(
    f"{best_wrong_vs_correct:+.2f}%"
)


# ============================================================
# Automatic Interpretation
# ============================================================

print("\n" + "=" * 90)
print("AUTOMATIC INTERPRETATION")
print("自动实验解释")
print("=" * 90)


correct_mae = (
    correct_metrics[
        "overall_mae"
    ]
)


paraphrase_mae = (
    paraphrase_metrics[
        "overall_mae"
    ]
)


zero_mae = (
    zero_metrics[
        "overall_mae"
    ]
)


# ------------------------------------------------------------
# Strongest Evidence
#
# Correct 应明显优于：
#
# Zero
# Wrong Average
#
# 同时 Paraphrase 与 Correct 接近。
# ------------------------------------------------------------

if (
    wrong_vs_correct >= 5.0
    and zero_vs_correct >= 5.0
    and abs(
        paraphrase_vs_correct
    ) <= 5.0
):

    print(
        "\nRESULT A ✅"
    )


    print(
        "固定同一个 Correct Language Model"
        "（正确语言模型）后，"
        "错误语言和全零语言都使动作预测明显变差。"
    )


    print(
        "\n同时 Paraphrased Language"
        "（同义改写语言）"
        "仍与正确原始语言表现接近。"
    )


    print(
        "\n结论：有较强证据表明该策略"
        "真正依赖正确的 Language Condition"
        "（语言条件），"
        "并具有一定 Semantic Robustness"
        "（语义鲁棒性）。"
    )


elif (
    wrong_vs_correct >= 2.0
    and abs(
        paraphrase_vs_correct
    ) <= 5.0
):

    print(
        "\nRESULT B ✅/⚠️"
    )


    print(
        "错误语言会导致性能下降，"
        "同义改写语言仍较稳定。"
    )


    print(
        "\n结论：存在 Language Semantic Effect"
        "（语言语义效应）的证据，"
        "但效应强度中等。"
    )


elif abs(
    wrong_vs_correct
) < 2.0:

    print(
        "\nRESULT C ⚠️"
    )


    print(
        "Wrong Language 与 Correct Language "
        "整体表现非常接近。"
    )


    print(
        "\n没有充分证据表明模型"
        "在测试时真正依赖正确语言语义。"
    )


else:

    print(
        "\nRESULT D"
    )


    print(
        "结果呈混合模式。"
    )


    print(
        "需要结合 Zero Language、"
        "Wrong Language 分布以及 Per-task MAE"
        "进一步分析。"
    )


# ============================================================
# Save Wrong Shift Report
# ============================================================

wrong_shift_df = pd.DataFrame(
    wrong_shift_rows
)


wrong_shift_df.to_csv(

    WRONG_SHIFT_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Main Comparison Report
# ============================================================

comparison_rows = [

    {
        "condition":
            "Correct Language",

        "overall_mae":
            correct_metrics[
                "overall_mae"
            ],

        "overall_mse":
            correct_metrics[
                "overall_mse"
            ],

        "macro_task_mae":
            correct_metrics[
                "macro_task_mae"
            ],
    },

    {
        "condition":
            "Paraphrased Language",

        "overall_mae":
            paraphrase_metrics[
                "overall_mae"
            ],

        "overall_mse":
            paraphrase_metrics[
                "overall_mse"
            ],

        "macro_task_mae":
            paraphrase_metrics[
                "macro_task_mae"
            ],
    },

    {
        "condition":
            "Zero Language",

        "overall_mae":
            zero_metrics[
                "overall_mae"
            ],

        "overall_mse":
            zero_metrics[
                "overall_mse"
            ],

        "macro_task_mae":
            zero_metrics[
                "macro_task_mae"
            ],
    },

    {
        "condition":
            "Wrong Language Average",

        "overall_mae":
            wrong_mean_mae,

        "overall_mse":
            wrong_mean_mse,

        "macro_task_mae":
            wrong_mean_macro_mae,
    },
]


pd.DataFrame(
    comparison_rows
).to_csv(

    COMPARISON_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Save Per-task Results
# ============================================================

per_task_rows = []


for task_index in range(
    NUM_TASKS
):

    wrong_task_values = [

        metrics[
            "per_task_mae"
        ][
            task_index
        ]

        for metrics
        in wrong_shift_metrics
    ]


    per_task_rows.append(
        {
            "task_index":
                task_index,

            "correct_language_mae":
                correct_metrics[
                    "per_task_mae"
                ][
                    task_index
                ],

            "paraphrase_language_mae":
                paraphrase_metrics[
                    "per_task_mae"
                ][
                    task_index
                ],

            "zero_language_mae":
                zero_metrics[
                    "per_task_mae"
                ][
                    task_index
                ],

            "wrong_language_mean_mae":
                float(
                    np.mean(
                        wrong_task_values
                    )
                ),

            "wrong_language_std_mae":
                float(
                    np.std(
                        wrong_task_values
                    )
                ),
        }
    )


pd.DataFrame(
    per_task_rows
).to_csv(

    PER_TASK_REPORT_PATH,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# Artifact Summary
# ============================================================

print("\n" + "=" * 90)
print("ARTIFACT SUMMARY")
print("实验资产汇总")
print("=" * 90)


print(
    f"\nMain Comparison Report"
    f"（主要对比报告）:\n"
    f"{COMPARISON_REPORT_PATH}"
)


print(
    f"\nWrong Shift Report"
    f"（9种错配语言报告）:\n"
    f"{WRONG_SHIFT_REPORT_PATH}"
)


print(
    f"\nPer-task Report"
    f"（逐任务报告）:\n"
    f"{PER_TASK_REPORT_PATH}"
)


print(
    "\nLanguage Test-time Intervention "
    "Experiment Complete ✅"
)