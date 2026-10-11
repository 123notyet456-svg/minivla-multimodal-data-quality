# 核心代码与运行边界

本目录只保留直接支撑公开实验结论的 9 个 Python 脚本，使用相对路径读写 `cache/`、`models/` 和 `reports/`。这些目录均在 `.gitignore` 中，不会被提交。

## 代码导航

| 阶段 | 脚本 |
|---|---|
| RGB 转视觉特征与缓存质检 | `build_libero_minivla_feature_cache.py` |
| Task ID、语言条件实验 | `train_libero_task_conditioning.py`、`train_libero_language_conditioning.py` |
| 语言反事实干预 | `test_language_intervention.py` |
| 双视觉基线与视觉干预 | `train_libero_minivla.py`、`test_minivla_vision_intervention.py`、`test_minivla_temporal_vision_intervention.py` |
| 投影融合与空间视觉 | `train_libero_projected_fusion.py`、`train_libero_spatial_vision.py` |

## 运行前提

1. 使用 Python 环境安装 `../requirements.txt` 中列出的依赖。该清单未锁定版本，因为历史环境的完整版本快照未保留。
2. 从上游获取 `lerobot/libero_10`，并按[数据与许可说明](../docs/sources-and-licensing.md)确认适用条款；不要将数据副本提交到仓库。
3. 在仓库根目录运行脚本。先运行特征缓存构建脚本，再运行训练与干预脚本；后续阶段依赖前序的本地缓存和模型检查点。
4. 公开仓库没有权重、缓存和完整原始输出，因此这是一份可读、可审查的实现，而不是开箱即用的可复现实验包。

## 已知边界

脚本保留了原始实验的相对目录结构和结果产出逻辑。由于依赖版本未锁定、上游数据可能更新且最终结论基于 Seed=42，重新运行的数值不应被默认视为与已发布结果完全一致。
