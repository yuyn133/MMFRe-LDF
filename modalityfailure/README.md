# M2FRe-LDF

**Multiscale Intra-Modality Feature Refinement and Local Deformable Convolution Fusion for Multispectral Object Detection**

![Python](https://img.shields.io/badge/Python-3.8-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10.1%2Bcu113-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-11.3-76B900)
![License](https://img.shields.io/badge/License-AGPL_v3-blue)

> 面向多光谱（可见光 RGB + 红外 IR）目标检测。在 CFT 双流 YOLOv5 框架基础上，引入
> **M2FRe**（Multiscale intra-Modality Feature Refinement，多尺度模态内特征细化）与
> **LDF**（Local Deformable convolution Fusion，局部可变形卷积融合，核心算子为 InternImage 的 DCNv3）。

论文：[Journal of Visual Communication and Image Representation, 2026](https://doi.org/10.1016/j.jvcir.2026.104936)　·　DOI: `10.1016/j.jvcir.2026.104936`

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 方法概览](#2-方法概览)
- [3. 与上游工作的关系](#3-与上游工作的关系)
- [4. 环境要求](#4-环境要求)
- [5. 快速开始](#5-快速开始)
- [6. 引用](#6-引用)
- [7. 致谢](#7-致谢)

---

## 1. 项目简介

多光谱目标检测利用可见光与红外图像的互补性（红外对光照变化、夜间场景更鲁棒，可见光提供纹理与颜色细节），在自动驾驶、安防监控、夜间行人检测等场景中具有重要价值。本文提出 **M2FRe-LDF**：

- **M2FRe（多尺度模态内特征细化）**：在骨干网络的 P3/P4/P5 多个尺度上，先对每个模态自身的特征进行细化，抑制模态内噪声、增强目标相关响应，为后续融合提供更"干净"的特征；
- **LDF（局部可变形卷积融合）**：以 InternImage 的 **DCNv3** 为核心算子，在跨模态融合阶段引入可学习的采样偏移与调制权重，使融合过程能够在局部感受野内自适应地对齐两个模态的特征，从而缓解配准残差带来的错位问题。

整体框架沿用 CFT 的双流 YOLOv5 结构，最终在 FLIR-aligned、LLVIP、M3FD 三个公开多光谱检测数据集上验证有效性。

> 说明：本仓库是论文的官方实现。

---

## 2. 方法概览

<!-- 框架图请放到 docs/figures/fig2_overview.png -->
<img src="docs/figures/fig2_overview.png" alt="M2FRe-LDF framework" width="85%" />

三个尺度（P3/8、P4/16、P5/32）分别执行"模态内细化 → 跨模态可变形融合"，融合后的特征再送入 YOLOv5 的 FPN/PAN 与检测头。

---

## 3. 与上游工作的关系

本项目的代码由两个开源工程组合而来，**复现时必须先完成 DCNv3 的本地编译**，这是整个流程中最容易出错的环节。

| 上游工程 | 在本项目中的用途 | 是否必须编译 |
| :--- | :--- | :---: |
| [InternImage](https://github.com/OpenGVLab/InternImage) | DCNv3 核心算子（`ops_dcnv3`：CUDA/C++ 扩展 + PyTorch 封装），用于 LDF 中的可变形卷积 | ✅ 必须 |
| [CFT / multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) | 双流 YOLOv5 骨干、跨模态 Transformer 融合范式、训练/测试/推理脚本、数据组织方式 | ❌ 无需编译 |

> **注意**：本项目**不需要**安装 `mmcv` / `mmdet` / `mmsegmentation`。InternImage 仓库中的 `detection/` 是 MMDetection 训练框架，本仓库并未使用，**只需要其中的 `ops_dcnv3` 算子**。请勿按 InternImage detection 的 README 去安装 MMDetection 全家桶，否则容易引发版本冲突。

---

## 4. 环境要求

### 4.1 硬件

| 项目 | 最低要求 | 推荐 |
| :--- | :--- | :--- |
| GPU | 显存 ≥ 12 GB | RTX 3090 / 4090 / A100（24 GB 及以上） |
| 系统 | Linux（Ubuntu 18.04+） | Ubuntu 20.04 / 22.04 |
| 内存 | 16 GB | 32 GB 及以上 |

> ⚠️ **仅支持 Linux**。DCNv3 的 `setup.py` 依赖 `CUDAExtension` 与 `nvcc`，Windows 下编译需要额外改造，不在本仓库支持范围内。

### 4.2 软件版本

| 组件 | 版本 | 说明 |
| :--- | :--- | :--- |
| Python | 3.8（3.7–3.9 均可） | 与预编译 whl 的 cp38 标签对应 |
| CUDA Toolkit | 11.3 | 11.1 – 11.7 亦可，**必须与 `torch.version.cuda` 一致** |
| PyTorch | 1.10.1+cu113 | ≥ 1.10（DCNv3 要求） |
| torchvision | 0.11.2+cu113 | 与 PyTorch 配套 |
| GCC / G++ | 7.5 或 9.3 | CUDA 11.3 要求 gcc ≤ 10，gcc 11 会编译失败 |
| cuDNN | ≥ 7 | 随 CUDA 安装 |

### 4.3 一键环境自检

编译 DCNv3 之前，先确认下面三个 CUDA 版本彼此匹配（这是 90% 编译失败的根源）：

```bash
nvcc -V                                             # 系统 CUDA 版本
python -c "import torch; print(torch.__version__, torch.version.cuda)"
gcc --version                                       # 主版本号 ≤ 10
```

预期输出示例：

```text
Cuda compilation tools, release 11.3, V11.3.109
1.10.1+cu113 11.3
gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
```

---

## 5. 快速开始

```bash
# 1) 拉取代码
git clone https://github.com/yuyn133/MMFRe-LDF.git
cd MMFRe-LDF

# 2) 创建并激活环境
conda create -n m2fre python=3.8 -y && conda activate m2fre
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt && pip install "numpy<2"

# 3) 编译 DCNv3 算子（仓库已随附源码，必须执行）
cd models/ops_dcnv3
sh make.sh && python test.py                     # 全部检查应为 True
python -c "import DCNv3; print(DCNv3.__file__)"  # 能打印路径即安装成功
cd ../..

# 4) 准备数据并修改数据配置（本仓库自带示例见 data/multispectral-yyn/）

# 5) 训练
python train.py --cfg <你的模型配置>.yaml --data data/multispectral-yyn/FLIR_aligned.yaml \
    --weights yolov5l.pt --epochs 100 --batch-size 4 --img-size 640 640 --device 0

# 6) 测试
python test.py --weights runs/train/exp/weights/best.pt \
    --data data/multispectral-yyn/FLIR_aligned.yaml --batch-size 4 --img-size 640
```

---

## 6. 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@article{YU2026104936,
title = {M2FRe-LDF: Multiscale intra-Modality Feature Refinement and Local Deformable convolution Fusion For multispectral object detection},
journal = {Journal of Visual Communication and Image Representation},
volume = {121},
pages = {104936},
year = {2026},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2026.104936},
author = {Yaonan Yu and Zheng Zhang and Zhiming Song and Xinyu Ou}
}
```

同时请引用本项目所依赖的两个工作：

```bibtex
@inproceedings{wang2023internimage,
  title     = {InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author    = {Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi
               and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {14408--14419},
  year      = {2023}
}

@article{fang2021cross,
  title   = {Cross-Modality Fusion Transformer for Multispectral Object Detection},
  author  = {Fang, Qingyun and Han, Dapeng and Wang, Zhaokui},
  journal = {arXiv preprint arXiv:2111.00273},
  year    = {2021}
}
```

---

## 7. 致谢

本项目的实现得益于以下开源工作，在此表示感谢：

- [InternImage](https://github.com/OpenGVLab/InternImage)：提供 DCNv3 核心算子（CUDA 实现与 PyTorch 封装）；
- [multispectral-object-detection (CFT)](https://github.com/DocF/multispectral-object-detection)：提供双流 YOLOv5 框架、跨模态融合范式与训练/评估脚本；
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)：提供检测框架基础；

## 许可证

本项目基于 **AGPL-3.0** 发布，详见 [LICENSE](LICENSE)。本项目基于以 AGPL-3.0 发布的 [multispectral-object-detection (CFT)](https://github.com/DocF/multispectral-object-detection) 修改而来，作为衍生作品需遵循同一许可证。

`models/ops_dcnv3/` 下的 DCNv3 算子来自 [InternImage](https://github.com/OpenGVLab/InternImage)，遵循 MIT License，版权归 OpenGVLab 所有。
