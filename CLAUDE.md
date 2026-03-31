# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 行为约束

1. **语言**：与用户的所有对话及代码注释均使用中文。
2. **代码质量**：每次生成代码或输出结果前，必须反复检查逻辑、验证边界条件、确认正确性后，再向用户输出，不允许输出未经验证的代码。

## 项目概述

这是一个**图像信号处理（ISP）流水线**研究与学习项目，实现了从原始 Bayer 传感器数据到最终输出的完整 ISP 处理链，并包含基于 AI 的增强模块。文件按编号顺序对应流水线的各个阶段。

## 环境配置

- Python 3，使用 conda（VS Code 已配置为 conda 环境管理器）
- 核心依赖：`opencv-python`、`numpy`、`torch`、`matplotlib`
- 可选：`open3d`（用于 `open3d-1.py` 中的三维点云处理）

直接运行脚本：
```bash
python 5.demosaic.py
python 6.ai_demosaic.py  # 训练 + 推理
```

## ISP 流水线架构

文件编号对应处理顺序：

```
RAW (Bayer) → DPC → BLC → LSC → Demosaic → AWB → CCM → AE → Gamma → 锐化 → NR → YUV
   7.py（预处理）          5.py / 6.py    8.py         10.py    9.py（后处理）
```

| 文件 | 阶段 | 核心算法 |
|------|------|----------|
| `7.DPC_BLC_LSC.py` | 传感器预处理 | 坏点校正（DPC）、黑电平校正（BLC）、镜头阴影校正（LSC） |
| `5.demosaic.py` | 去马赛克 | 最近邻、双线性插值、边缘感知（Hamilton-Adams） |
| `6.ai_demosaic.py` | AI 去马赛克 | 基于残差块的 PyTorch CNN（`AdvancedISP` 模型） |
| `8.Demosaic-AWB-CCM.py` | 颜色处理 | 自动白平衡（灰世界法）、色彩校正矩阵（sRGB） |
| `9.Gamma-sharp-NR-YUV.py` | 后处理 | Gamma 校正、USM 锐化、双边滤波去噪、YUV420 |
| `10.AE.py` | 曝光控制 | PID 自动曝光、中心加权测光 |

基础学习文件：`1.py`（基础）、`2histogram.py`、`3.wavelengths.py`、`4.brightness_contrast.py`、`5.1mosaic.py`。

## RAW 数据格式

主要测试数据：`test_2724x1848_rggb_8bit.raw` — 8-bit RGGB Bayer 格式，分辨率 2724×1848。

各脚本通用读取方式：
```python
raw = np.fromfile('test_2724x1848_rggb_8bit.raw', dtype=np.uint8).reshape(1848, 2724)
```

## AI 模型（`6.ai_demosaic.py`）

- 两种网络结构：`SimpleISP`（3层 CNN）和 `AdvancedISP`（残差块）
- 训练数据使用 DIV2K 数据集（`data/DIV2K_train_LR_bicubic/`），从 RGB 图像合成 Bayer 图案
- 最优模型保存至 `best_advanced_isp.pth`，推理结果输出至 `ai_final_output.png`
- 在 `__main__` 块中切换训练/推理模式

## 输出文件

脚本结果输出到工作目录：
- `demosaic_*.png` — 各去马赛克方法对比结果
- `isp_preprocess_result.png` — DPC/BLC/LSC 处理结果
- `brightness_contrast_result.png` — 亮度对比度调整结果
- `ai_final_output.png` — AI 去马赛克输出
- `*.pth` — PyTorch 模型权重
