# 可公开的实验结果表

本目录保存从原项目 `reports/` 目录筛选出的 15 份小型聚合级 CSV，用于支撑 README、实验报告和面试沟通中的关键结论。所有误差指标均为离线动作预测结果；MAE（Mean Absolute Error，平均绝对误差）越低越好。

## 文件说明

| 主题 | 文件 | 内容 |
|---|---|---|
| 任务条件 | `libero_task_conditioning_comparison.csv` | State、零任务、错配任务与正确任务的对照 |
| 语言条件 | `libero_language_conditioning_comparison.csv`、`libero_language_embedding_similarity.csv` | 语言模型对照及原始指令/同义改写的余弦相似度 |
| 语言干预 | `libero_language_test_time_intervention.csv`、`libero_language_wrong_shift_results.csv` | 正确、移除、改写/错配语言的测试时结果 |
| 切分与缓存质检 | `libero_minivla_split_summary.csv`、`libero_minivla_feature_cache_summary.csv`、`libero_spatial_feature_cache_summary.csv` | 轨迹级数据切分与特征缓存检查汇总 |
| 视觉基线与干预 | `libero_minivla_comparison.csv`、`libero_minivla_vision_intervention.csv`、`libero_minivla_temporal_vision_intervention.csv` | 视觉输入基线、置零/错配与时间错位结果 |
| 融合与空间视觉 | `libero_projected_fusion_comparison.csv`、`libero_spatial_vision_comparison.csv`、`libero_spatial_vision_per_task_mae.csv`、`libero_spatial_vision_training_history.csv` | 投影融合、最终空间视觉、逐任务 MAE 与训练历史 |

## 未纳入的材料

为避免泄露数据、超出上游许可或造成仓库膨胀，本目录不含原始 RGB 帧、数据集副本、特征缓存、样本级预测、权重、训练检查点、运行日志或代码。CSV 是结果核对材料，不足以端到端复现；完整边界见 [证据范围与复现边界](../docs/evidence-scope.md)。
