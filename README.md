# APRA

Adaptive Progressive Robust Aggregation for Backdoor-Resilient Federated Learning

## 项目简介

本项目实现了一个面向联邦学习后门攻击的防御方法：**APRA（Adaptive Progressive Robust Aggregation，自适应渐进鲁棒聚合）**。它的目标不是单纯追求最高的干净样本精度，而是在 A3FL、DOBA、ReBA、Neurotoxin、ModelReplace 等后门攻击下，尽可能保留全局模型的正常分类能力，同时显著降低后门攻击成功率（ASR）。

仓库中不仅包含 APRA 的核心实现，还包含多种基线防御方法、攻击实现、700 轮实验结果、过程审计 CSV、可视化脚本，以及课程设计报告与演示图片，适合用于：

- 联邦学习后门防御实验
- APRA 方法复现与代码阅读
- 不同聚合策略的对比研究
- 课程设计、论文复盘与结果展示

更完整的设计说明见 [report/main.typ](report/main.typ) 和 [report/main.pdf](report/main.pdf)。

## APRA 方法概览

APRA 的核心思想是：恶意客户端即使伪装得较为隐蔽，仍然可能在**更新范数、输出层偏置、随机噪声输入响应、与其他客户端的方向关系**上留下痕迹。与单一步骤的硬过滤不同，APRA 将防御拆成一个渐进式流程：

1. **多维特征提取**
   使用指定层的展平更新作为参数特征，并计算整模型更新的 L2 范数；可选地拼接 NBD/NDIF 行为特征，再通过 PCA 降维。
2. **自适应 MAD 预过滤**
   基于更新范数的中位数与中位绝对偏差（MAD）做鲁棒异常值筛选，阈值会随训练轮数逐步收紧。
3. **层次聚类与可信簇选择**
   对通过 MAD 的客户端做凝聚层次聚类，并通过轮廓系数自动选择簇数，再优先保留“规模较大且内部方向一致”的可信簇。
4. **信任加权与自适应裁剪**
   对最终保留的客户端计算信任权重；权重越低，裁剪越严格，在最终聚合中的贡献也越小。

对应实现位于 [fl_utils/apra.py](fl_utils/apra.py)。

## 主要特点

- 支持 **APRA、FedAvg、Clip、DeepSight、FoolsGold、RFLBAT** 等多种聚合方法对比
- 支持 **A3FL、ModelReplace、DOBA（`sin-adv`）、Neurotoxin、ReBA** 等攻击场景
- 内置 APRA 过程记录：
  - `apra_client_trace.csv`：记录单客户端在每轮中的 MAD 分数、聚类标签、最终是否入选、信任权重、裁剪因子等
  - `apra_round_summary.csv`：记录每轮采样客户端、恶意客户端、各阶段筛选结果与最终统计
- 已包含一批 `main/re_result` 与 `main/re_result_pre` 下的实验结果，便于直接复盘
- 提供 `main/apra_dashboard.py` 用于浏览 APRA 审计结果

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `fl_utils/apra.py` | APRA 聚合器实现，包含特征提取、MAD 过滤、层次聚类、信任加权与裁剪 |
| `fl_utils/aggregator.py` | 聚合方法统一入口，负责在 `avg`、`apra`、`clip`、`deepsight`、`foolsgold`、`rflbat` 等方法之间分发 |
| `fl_utils/fler.py` | 联邦训练主循环，负责客户端训练、攻击训练、聚合调用、测试与轨迹记录 |
| `fl_utils/attacker.py` | 后门攻击相关逻辑 |
| `main/main.py` | 训练入口 |
| `main/yamls/cifar10_apra.yaml` | 主要实验配置模板 |
| `main/run.sh` | 批量对比实验脚本，会遍历多种防御和攻击组合 |
| `main/apra_dashboard.py` | APRA 审计结果查看脚本 |
| `main/re_result*` | 训练结果、日志、准确率曲线、轨迹记录、APRA 过程记录 |
| `report/main.typ` | 课程设计报告源文件 |
| `report/main.pdf` | 课程设计报告 PDF |
| `APRA_Presentation_meaningful_images/` | PPT 中提取出的关键实验图 |

## 环境依赖

仓库当前没有单独维护 `requirements.txt`，按代码依赖，至少需要以下组件：

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- scipy
- pyyaml

如果需要重新编译报告，还需要安装 Typst。

一个最小安装示例：

```bash
pip install torch torchvision numpy pandas scikit-learn scipy pyyaml
```

## 快速开始

### 1. 运行单次实验

当前默认训练入口是 [main/main.py](main/main.py)。建议始终**显式传入配置文件**：

```bash
cd main
python main.py --params ./yamls/cifar10_apra.yaml --gpu 0
```

注意：

- `main/main.py` 中默认的 `--params` 指向 `./yamls/cifar100_DOBA_nomia.yaml`，但该文件当前并不在仓库中，因此不要依赖默认值。
- `main/yamls/cifar10_apra.yaml` 目前默认的 `agg_method` 是 `clip`，如果要运行 APRA，请先改成 `apra`。
- 当前训练代码大量使用 `.cuda()` 和 `torch.cuda.set_device(...)`，默认按 GPU 环境设计。

### 2. 关键配置项

以 `main/yamls/cifar10_apra.yaml` 为例，常用配置项包括：

```yaml
agg_method: apra
attacker_method: a3fl
dataset: cifar10
model: resnet18
epochs: 700
num_total_participants: 100
num_sampled_participants: 10
num_adversaries: 5
dirichlet_alpha: 0.7

apra_pca_components: 3
apra_base_clip: 1.0
apra_k_init: 5.0
apra_k_decay: 0.1
```

字段含义：

- `agg_method`：聚合方法，可选 `apra`、`avg`、`clip`、`deepsight`、`foolsgold`、`rflbat`
- `attacker_method`：攻击方法，可选 `a3fl`、`modelreplace`、`sin-adv`、`neurotoxin`、`reba`
- `dirichlet_alpha`：Non-IID 数据划分强度
- `apra_pca_components`：PCA 降维维度
- `apra_base_clip`：APRA 中裁剪基值
- `apra_k_init` / `apra_k_decay`：MAD 阈值的初始值与衰减速度

### 3. 数据集准备

按当前代码逻辑：

- CIFAR-10 / CIFAR-100 / MNIST 会通过 `torchvision` 自动下载到 `../data/` 或对应目录
- Tiny-ImageNet 需要按 `../data/tiny-imagenet-200/{train,val}` 的目录结构准备

## 批量实验

仓库提供了 [main/run.sh](main/run.sh) 和 [main/run_end.sh](main/run_end.sh) 用于批量实验。

示例：

```bash
cd main
bash run.sh
```

脚本行为说明：

- 会通过 `sed` **直接改写** `yamls/cifar10_apra.yaml`
- 会遍历多种 `agg_method` 与 `attacker_method`
- 会调用 `send_mail.py` 发送完成通知；如果本地没有配置邮件环境，建议先注释相关代码

## 输出结果说明

实验结果默认保存在 `main/re_result/` 下，目录名通常编码了：

- 数据集
- 训练轮数
- 时间戳
- 聚合方法
- 是否投毒
- 是否噪声
- 攻击方法

常见输出文件包括：

- `params.yaml.txt`：该次实验的完整参数快照
- `log.txt`：训练日志
- `*_accuracy.csv`：主任务准确率、后门准确率、学习率等曲线数据
- `*_trajectory.csv`：逐轮轨迹数据
- `apra_client_trace.csv`：APRA 客户端级审计记录
- `apra_round_summary.csv`：APRA 轮级摘要记录

## APRA 审计面板

如果某次实验产出了 `apra_client_trace.csv` 和 `apra_round_summary.csv`，可以用内置脚本快速查看：

```bash
cd main
python apra_dashboard.py --host 127.0.0.1 --port 8925
```

然后在浏览器中访问 `http://127.0.0.1:8925`。

## 实验结论摘要

根据 [report/main.typ](report/main.typ) 中汇总的 CIFAR-10、700 轮训练结果，APRA 在 A3FL、DOBA、ReBA、Neurotoxin 四类攻击下表现出“**略牺牲主任务精度，显著降低 ASR**”的特征。

### 平均主任务准确率

| 方法 | 平均 Accuracy (%) |
| --- | ---: |
| APRA | 90.6895 |
| FedAvg | 91.2867 |
| Clip | 92.0233 |
| DeepSight | 90.7311 |
| FoolsGold | 90.9321 |
| RFLBAT | 91.6191 |

### 平均后门攻击成功率（ASR）

| 方法 | 平均 ASR (%) |
| --- | ---: |
| APRA | 14.1549 |
| FedAvg | 89.2924 |
| Clip | 88.4903 |
| DeepSight | 85.6135 |
| FoolsGold | 87.4537 |
| RFLBAT | 79.1497 |

从这些结果可以看出，APRA 的优势主要不在于把主任务准确率推到最高，而在于把后门攻击成功率压到远低于基线的水平。特别是在 Neurotoxin 场景下，APRA 的 ASR 可降至约 `9.43%`。


## 参考与说明

- 项目报告： [report/main.pdf](report/main.pdf)
- 报告源文件： [report/main.typ](report/main.typ)
- APRA 实现： [fl_utils/apra.py](fl_utils/apra.py)
- 审计结果面板： [main/apra_dashboard.py](main/apra_dashboard.py)

