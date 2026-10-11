# 数据、模型来源与许可

## 数据集

- 实验数据源为 [`lerobot/libero_10`](https://huggingface.co/datasets/lerobot/libero_10)。其数据集页当前标注 Apache-2.0，并记录 379 条轨迹、10 FPS、2 个相机与 101,469 行数据。
- [LIBERO 上游项目](https://github.com/Lifelong-Robot-Learning/LIBERO) 的许可表将代码列为 MIT、数据集列为 CC BY 4.0；这与 Hugging Face 镜像展示的许可存在差异。
- 项目公开页不分发数据、帧、缓存或标注。实际复现或再分发前，应依据所使用数据修订版核对上游条款。

## 模型

- 视觉编码器：冻结的 ImageNet 预训练 MobileNetV3-Small，见 [Torchvision 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)。
- 语言编码器：冻结的 `sentence-transformers/all-MiniLM-L6-v2`，用于生成 384 维语言嵌入。
- 本仓库不含模型权重、下载文件哈希或检查点；重新下载或分发前应确认对应版本条款。

## 本仓库许可

本仓库当前没有自有 `LICENSE` 文件。公开访问不自动授予复制、修改或再分发许可；如要添加许可证，应由项目所有者确认仓库文字、图表和衍生内容的授权意图。第三方数据、模型和代码分别受其上游条款约束。
