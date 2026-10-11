# 数据、模型来源与许可

## 数据集

- 实验报告记录的数据源：[`lerobot/libero_10` 数据集页](https://huggingface.co/datasets/lerobot/libero_10)。该页面当前标注 Apache-2.0，记录 379 条轨迹、10 FPS、2 个相机视角和 101,469 行数据。
- [LIBERO 上游项目](https://github.com/Lifelong-Robot-Learning/LIBERO) 的许可表将代码标为 MIT、数据集标为 CC BY 4.0。它与 Hugging Face 镜像当前显示的 Apache-2.0 并不一致。
- 由于当前项目资料没有记录实际下载的修订号、文件清单或来源确认，无法判断具体使用版本应适用哪一项声明。仓库不包含或再分发数据；使用者应先核对上游条款及数据版本。

## 视觉模型

- 项目分析文档记录使用冻结的 MobileNetV3-Small ImageNet 预训练特征；模型结构与预训练权重选项见 [Torchvision 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)。
- 当前项目资料没有记录具体权重枚举、下载文件哈希或额外权重许可说明。本仓库不包含模型权重；如需重新下载或分发，需先确认对应版本的条款。

## 本仓库许可

本仓库当前没有自有 `LICENSE` 文件，因此仓库内容的公开访问不自动授予复制、修改或再分发许可。新增许可证前需由项目所有者确认授权意图；第三方数据、模型及代码仍各自受其上游条款约束。

## 来源证据缺口

当前项目镜像中的 `sources/` 目录为空，引用对话没有附件。公开仓库只包含 Markdown 项目摘要，没有原始数据、训练代码、运行日志、指标导出或数据修订号。本文的结果数字因此只能称为“项目分析文档记录”，不能视为已从原始实验独立复核。
