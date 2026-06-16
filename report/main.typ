#let primary = rgb("#24476f")
#let accent = rgb("#2f7d68")
#let muted = rgb("#687385")
#let light = rgb("#f4f7fb")
#let border = rgb("#d7dee8")
#let code-bg = rgb("#eef3f8")
#let code-font = ("SimHei", "Microsoft YaHei", "Source Han Sans", "Arial")


#set document(
  title: "APRA 自适应渐进鲁棒聚合防御方法课程设计报告",
  author: "黄康",
)

#set page(
  paper: "a4",
  margin: (top: 2.4cm, bottom: 2.2cm, left: 2.35cm, right: 2.35cm),
)
#set text(font: ("Source Han Serif SC", "Times New Roman"), size: 10.5pt, lang: "zh")
#set par(justify: true, first-line-indent: (amount: 2em, all: true), leading: 0.86em)

#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  set par(first-line-indent: 0pt)
  v(1.35em)
  text(fill: primary, weight: "bold", size: 16pt, it.body)
  v(0.35em)
  line(length: 100%, stroke: 0.8pt + primary)
  v(0.5em)
}
#show heading.where(level: 2): it => {
  set par(first-line-indent: 0pt)
  v(0.9em)
  text(fill: primary, weight: "bold", size: 13pt, it.body)
  v(0.15em)
}
#show heading.where(level: 3): it => {
  set par(first-line-indent: 0pt)
  v(0.7em)
  text(fill: accent, weight: "bold", size: 11pt, it.body)
}
#show raw.where(block: true): it => block(
  width: 100%,
  fill: code-bg,
  stroke: 0.65pt + border,
  radius: 5pt,
  inset: (x: 9pt, y: 8pt),
  breakable: true,
)[
  #set text(font: code-font, size: 10.6pt)
  #set par(first-line-indent: 0pt, justify: false, leading: 1.00em)
  #it
]
#show raw.where(block: false): it => box(
  fill: code-bg,
  radius: 2pt,
  inset: (x: 2pt, y: 0.5pt),
)[
  #set text(font: code-font, size: 10.8pt)
  #it
]

#let box(title, body) = block(
  width: 100%,
  fill: light,
  stroke: 0.7pt + border,
  radius: 4pt,
  inset: 9pt,
)[
  #text(fill: primary, weight: "bold")[#title]
  #v(0.35em)
  #body
]

#let figcell(path, title) = [
  #image(path, width: 100%)
  #v(0.25em)
  #align(center)[#text(size: 8.6pt, fill: muted)[#title]]
]

#let source-note(body) = align(right)[#text(size: 8pt, fill: muted)[#body]]

#align(center)[
  #v(1.2cm)
  #rect(width: 100%, height: 8pt, fill: primary, radius: 2pt)
  #v(1.1cm)
  #text(size: 24pt, weight: "bold", fill: primary)[课程设计报告]
  #v(1.0cm)
  #text(size: 24pt, weight: "bold")[APRA 自适应渐进鲁棒聚合防御方法]
  #v(0.35cm)
  #text(size: 10pt, fill: muted)[Adaptive Progressive Robust Aggregation for Backdoor-Resilient Federated Learning]
  #v(2.0cm)
  #block(width: 78%, inset: 14pt, fill: light, stroke: 0.7pt + border, radius: 5pt)[
    #set text(size: 12.2pt)
    #set par(first-line-indent: 0pt, justify: false)
    #grid(
      columns: (1fr, 2fr),
      row-gutter: 12pt,
      [课题名称], [APRA 自适应渐进鲁棒聚合防御方法],
      [班级], [1623201],
      [学号], [*162320127* 162320115 162320129],
      [组员], [*黄康* 黄智辉 何明迅],
      [成绩], [\_\_\_\_\_\_\_\_],
      [指导教师], [\_\_\_\_\_\_\_\_],
    )
  ]
  #v(1.2cm)
  #text(fill: muted)[源代码仓库：https://github.com/GreenInsect/APRA]

  #v(1fr)
  #text(fill: muted)[#datetime.today().display("[year] 年 [month repr:numerical] 月 [day] 日")]
]

#pagebreak()

#set page(
  paper: "a4",
  margin: (top: 2.35cm, bottom: 2.2cm, left: 2.35cm, right: 2.35cm),
  header: align(right)[#text(size: 8.5pt, fill: muted)[APRA 课程设计报告]],
  footer: context align(center)[#text(size: 8.5pt, fill: muted)[#counter(page).display("1")]],
)

#outline(title: [目录])

= 课题概述

联邦学习允许多个客户端在不上传本地原始数据的前提下共同训练模型。它解决了数据集中存储带来的隐私与合规压力，但也把模型训练的信任边界推到了客户端侧。服务端只能看到客户端模型更新，难以直接确认这些更新是否来自正常数据和正常训练过程。一旦部分客户端被攻击者控制，恶意更新就可能在全局模型中植入后门，使模型在普通样本上保持较高准确率，却在带触发器的输入上输出攻击者指定类别。

本课程设计围绕 #strong[APRA（#emph[Adaptive Progressive Robust Aggregation]，自适应渐进鲁棒聚合）] 展开。项目目标不是单纯提高 CIFAR-10 主任务准确率，而是在 A3FL、DOBA、ReBA、Neurotoxin 等后门攻击下，让全局模型尽量保留正常分类能力，同时 #strong[显著降低攻击成功率]。当前仓库已经包含 APRA 聚合器实现、若干基线防御方法、700 轮训练结果、演示 PPT 及实验曲线图片，本报告在这些材料基础上整理问题分析、设计方法、实现细节、实验结果和总结。

#box[仓库材料使用情况][
本报告主要依据 `Requirements.md`、`fl_utils/apra.py`、`fl_utils/aggregator.py`、`main/yamls/cifar10_apra.yaml`、`main/re_result`、`main/re_result_pre`、`APRA_Presentation_AI.md` 与 `APRA_Presentation_meaningful_images` 编写。报告中的主任务准确率与 #emph[ASR] 汇总表采用演示稿中的 700 轮均值口径；APRA 过程筛选分析采用 `main/re_result` 中实际保存的 `apra_round_summary.csv`。
]

= 问题分析

== 联邦学习中的后门威胁

在标准 FedAvg 中，服务端通常对每轮采样客户端的更新做平均：

$ w_(t+1) = w_t + 1 / n sum_(i=1)^n Delta w_i $

这个公式默认每个客户端都可信，或者至少恶意更新在平均后会被稀释。但在后门攻击中，攻击者会刻意构造方向一致、幅度异常或触发器相关的模型更新，让全局模型在后门样本上形成稳定记忆。更麻烦的是，攻击者往往同时约束主任务损失，使模型在干净测试集上的准确率看起来仍然正常。

本项目关注的攻击具有几个共同点：第一，攻击不一定表现为简单的大范数异常；第二，持久性攻击会让后门在攻击停止后继续残留；第三，自适应攻击会根据训练动态调整触发器或参数位置，从而绕过单一规则防御。由此可见，只靠固定阈值裁剪或只看客户端更新相似度，都难以覆盖全部攻击情形。

== 现有方法的不足

FedAvg 缺少异常检测机制，对任意被采样的恶意客户端都会给予同等权重。Clip 可以限制更新幅度，但如果攻击者把恶意目标压进较小范数更新，单纯裁剪无法判断方向是否有问题。FoolsGold 通过客户端历史更新相似度降低协同攻击者权重，对 #emph[Sybil] 式攻击有效，但在 #emph[Non-IID] 数据下容易把真实分布差异误判为异常。DeepSight 和 RFLBAT 引入了模型检查、聚类或降维，但仍可能在持久化后门、低幅度后门或客户端分布差异较强时出现漏检。

因此，APRA 的设计重点是把 #strong[多维特征、鲁棒统计过滤、聚类判断和加权裁剪] 串成一个渐进流程。前面的步骤先尽量排除明显异常，后面的步骤再对剩余客户端做细粒度选择与权重控制，避免某一步判断失误就完全决定聚合结果。

= 研究现状与存在问题

联邦学习安全研究通常从两条路线展开。一条路线研究攻击，例如模型替换、分布式后门、持久性后门和自适应触发器优化；另一条路线研究防御，例如范数裁剪、鲁棒聚合、相似度降权、模型输出检查和聚类过滤。已有工作说明，后门攻击可以在主任务准确率几乎不下降的情况下成功植入目标行为，而防御方法需要同时面对 #emph[Non-IID] 数据、客户端采样随机性、恶意客户端数量未知和攻击策略变化。

本项目中的 APRA 不是完全替代已有防御，而是把几类有效信号合并到一个实现中：更新范数用于发现显著异常，输出层和模型响应特征用于观察类别偏置，#emph[PCA] 与层次聚类用于在低维空间划分客户端群体，信任权重与裁剪用于降低可疑客户端的实际贡献。这样的组合更符合课程设计目标：既能解释每一步的作用，也能通过实验结果验证整体防御效果。

= 基本解决思路和设计思想

APRA 的核心假设是：恶意客户端为了植入后门，最终会在更新幅度、输出层偏置、噪声输入响应或与其他客户端的方向关系上留下痕迹。单个痕迹可能不稳定，但 #strong[多维信号合并后，恶意更新与多数正常更新之间会形成可利用的距离]。

整体流程分为四步：

1. 提取客户端更新特征。对选定层的模型更新进行压平，并计算每个客户端的整体 L2 范数；在配置开启时拼接 #emph[NBD/NDIF] 一类模型响应特征，再通过 #emph[PCA] 降维。
2. 做自适应 MAD 预过滤。以更新范数的中位数和中位绝对偏差为基准，使用随训练轮次衰减的阈值筛除明显异常值。
3. 做层次聚类与簇选择。对通过预过滤的客户端使用凝聚层次聚类，并用轮廓系数自动选择簇数，再根据簇规模和内部相似度选择可信簇。
4. 做信任加权聚合。对最终保留客户端计算相似度权重，并按权重自适应裁剪更新幅度，最后加权写回全局模型。

#box[设计取舍][
APRA 没有把“检测出攻击者”作为唯一目标，而是把 #strong[“降低恶意更新对聚合结果的贡献”] 作为直接目标。这样即使某些恶意客户端没有被完全剔除，也会在后续权重和裁剪阶段被限制影响。
]

= 设计方法与实现细节

== 模块结构

仓库中的核心代码分布如下：

#table(
  columns: (1.35fr, 3.7fr),
  inset: 6pt,
  stroke: 0.45pt + border,
  align: (left, left),
  table.header([文件], [作用]),
  [`fl_utils/apra.py`], [实现 APRA 聚合器，包括特征提取、MAD 过滤、聚类、信任加权聚合和过程记录。],
  [`fl_utils/aggregator.py`], [统一调度 FedAvg、APRA、Clip、DeepSight、FoolsGold、RFLBAT 等聚合方法。],
  [`fl_utils/fler.py`], [联邦训练主循环，负责客户端训练、攻击训练、聚合调用和测试。],
  [`fl_utils/attacker.py`], [实现 A3FL、ReBA、Neurotoxin 等攻击相关训练逻辑。],
  [`main/yamls/cifar10_apra.yaml`], [CIFAR-10 实验配置模板，包含数据集、模型、攻击、防御和 APRA 参数。],
  [`main/re_result*`], [保存训练日志、准确率/#emph[ASR] 曲线、APRA 客户端筛选记录。],
)

聚合器入口位于 `fl_utils/aggregator.py`。当配置项 `agg_method` 为 `apra` 时，训练循环会调用 `APRAAggregator.aggregate`，否则进入相应基线方法：

```python
if self.helper.config['agg_method'] == 'avg':
    return self.average_models(...)
elif self.helper.config['agg_method'] == 'apra':
    apraAggregator = APRAAggregator(self.helper)
    return apraAggregator.aggregate(...)
elif self.helper.config['agg_method'] == 'clip':
    self.clip_updates(...)
    return self.average_shrink_models(...)
```

这种结构让 APRA 可以和其他防御在同一训练框架下对比，减少了因训练流程不同造成的实验偏差。

== APRA 四阶段算法

APRA 主流程可以概括为以下伪代码：

```text
输入：全局模型 w_t，客户端更新 {Δw_i}，客户端模型 {m_i}，采样客户端集合 S
输出：更新后的全局模型 w_{t+1}

1. 对每个客户端 i 提取特征 f_i 和更新范数 n_i
2. 根据 median(n) 与 MAD(n) 计算 modified z-score
3. 使用 k_t = max(2, k_0 exp(-λt)) 过滤明显异常客户端
4. 对剩余客户端特征做层次聚类，并以轮廓系数选择簇数
5. 选择“规模较大且内部方向一致”的可信簇
6. 计算可信客户端之间的相似度权重
7. 按信任权重进行自适应裁剪和加权聚合
8. 记录本轮 MAD、聚类、最终选择和裁剪信息
```

在实现中，MAD 过滤使用的核心公式为：

$ z_i = 0.6745 (n_i - "median"(n)) / "MAD"(n) $

$ k_t = "max"(2.0, k_"init" "exp"(-k_"decay" t)) $

其中 `k_init` 和 `k_decay` 来自配置文件。训练早期阈值较宽松，有利于模型正常收敛；随着轮次增加，阈值逐渐收紧，有利于抑制后期持续注入的异常更新。如果过滤后剩余客户端过少，代码会触发安全保留机制，保留最接近中位数的一部分客户端，避免聚合阶段无客户端可用。

== 特征提取

`_extract_features` 首先对指定层的更新做压平，然后计算每个客户端更新的 L2 范数。对 CIFAR-10，`get_layer_name_for_dataset` 会指向分类器相关层，使特征更集中地反映类别决策边界的变化。配置开启时，代码还会拼接 NBD/NDIF 特征：NBD 观察输出层偏置变化，NDIF 使用随机噪声输入探测模型输出分布是否异常偏向某类。

```python
flat = flatten_update(update, layer_names=[layer_name])
flat_updates.append(flat)

l2 = 0.0
for name, data in update.items():
    if 'num_batches_tracked' in name:
        continue
    l2 += torch.norm(data, p=2).item() ** 2
update_norms.append(np.sqrt(l2))
```

这里的实现重点是把 #emph[“参数空间的变化”] 和 #emph[“模型行为的变化”] 都转化为可比较的数值特征。后续 #emph[PCA] 降维能降低聚类噪声，也能控制计算开销。

== 层次聚类与可信簇选择

MAD 过滤后，APRA 使用凝聚层次聚类对剩余客户端分组。簇数不是固定写死，而是在可行范围内搜索，并用轮廓系数评价聚类质量。最终选择簇时，代码综合簇内样本数和余弦相似度，避免选择只有单个异常点的小簇。

```python
for k in range(2, max_k + 1):
    clustering = AgglomerativeClustering(
        n_clusters=k, metric='euclidean', linkage='ward'
    )
    labels = clustering.fit_predict(features)
    score = silhouette_score(features, labels)

for c in np.unique(labels):
    indices = np.where(labels == c)[0]
    internal_sim = np.mean(sk_cosine_similarity(features[indices]))
    cluster_scores[c] = len(indices) * internal_sim
```

这样的策略适合课程实验中的设置：每轮只采样 10 个客户端，样本规模不大，使用层次聚类比训练额外检测模型更轻量，也更便于解释。

== 信任权重与裁剪

通过聚类的客户端并不被完全等同对待。APRA 会继续计算保留客户端之间的余弦相似度，并将其转化为信任权重。#strong[权重越低的客户端，裁剪因子越小；权重越高的客户端，对全局模型更新的贡献越大。]

```python
ratio = (w_i + 1e-8) / (max_trust + 1e-8)
clip_factors[i] = base_clip * ratio

update = client_weight[name].float() * trust_weights[idx_in_list]
averaged_weights[name] += update
```

这一步使 APRA 具有容错性：即使某些可疑客户端没有在 MAD 或聚类阶段被剔除，它们的更新仍然会被更强裁剪，并在加权平均中占较小比例。

== 过程记录

APRA 还会将每轮筛选信息写入 CSV。`apra_client_trace.csv` 保存单个客户端的角色、更新范数、MAD 分数、聚类标签、最终是否入选、信任权重和裁剪因子；`apra_round_summary.csv` 保存本轮采样客户端、恶意客户端、各阶段通过/剔除列表和最终统计。这些记录让实验不仅能看最终准确率，也能追溯每轮防御行为。

= 实验设置

实验使用 CIFAR-10 数据集，模型为 ResNet18，联邦训练轮数为 700。当前配置文件与结果参数记录中，客户端总数为 100，每轮采样 10 个客户端，恶意客户端数为 5，数据划分为 #emph[Non-IID]，Dirichlet 参数为 0.7。主要评价指标包括：

1. 主任务准确率（#emph[Main Task Accuracy]）：模型在干净测试集上的分类准确率，越高越好。
2. 攻击成功率（#emph[Attack Success Rate]，#emph[ASR]）：带触发器样本被攻击目标误分类的比例，越低表示防御越有效。
3. APRA 筛选记录：恶意客户端被剔除比例、恶意客户端入选比例、良性客户端误剔除比例等，用于解释防御过程。

对比方法包括 FedAvg、Clip、DeepSight、FoolsGold、RFLBAT 和 APRA。攻击类型包括 A3FL、DOBA、ReBA 和 Neurotoxin。报告中的汇总数值以 700 轮均值为主，这比单轮最终值更能反映训练过程稳定性。

= 实验结果与分析

== 主任务准确率

#text(size: 8.2pt)[
#table(
  columns: (1.05fr, 0.86fr, 0.86fr, 0.86fr, 0.96fr, 0.96fr, 0.86fr),
  inset: 4.3pt,
  stroke: 0.38pt + border,
  align: center,
  table.header([攻击], [APRA], [FedAvg], [Clip], [DeepSight], [FoolsGold], [RFLBAT]),
  [A3FL], [90.3862], [92.2713], [92.2840], [91.8031], [92.3456], [92.2495],
  [DOBA], [90.7134], [92.1527], [92.1648], [91.7948], [92.3430], [92.2688],
  [Neurotoxin], [89.8676], [88.2799], [91.1706], [87.3673], [87.1985], [89.5529],
  [ReBA], [91.7909], [92.4428], [92.4738], [91.9591], [91.8411], [92.4053],
  [平均], [90.6895], [91.2867], [92.0233], [90.7311], [90.9321], [91.6191],
)
]
#source-note[数据来源：`APRA_Presentation_AI.md` 中主任务准确率汇总表，单位为百分比。]

从主任务准确率看，APRA 的平均准确率为 #strong[90.6895%]，略低于 Clip 和 RFLBAT 等基线，但仍保持在 90% 左右。这个结果说明 APRA 为了压制后门牺牲了一小部分干净精度，但没有造成模型崩溃。尤其在 Neurotoxin 场景下，APRA 达到 #strong[89.8676%]，高于 FedAvg、DeepSight 和 FoolsGold，说明它对持久性毒化攻击更稳。

#figure(
  grid(
    columns: 1,
    row-gutter: 11pt,
    figcell("../APRA_Presentation_meaningful_images/slide06_main_accuracy_a3fl_curve.png", "A3FL"),
    figcell("../APRA_Presentation_meaningful_images/slide06_main_accuracy_doba_curve.png", "DOBA"),
    figcell("../APRA_Presentation_meaningful_images/slide06_main_accuracy_reba_curve.png", "ReBA"),
    figcell("../APRA_Presentation_meaningful_images/slide06_main_accuracy_neurotoxin_curve.png", "Neurotoxin"),
  ),
  caption: [四类攻击下主任务准确率曲线],
)

曲线显示 APRA 的主任务准确率整体波动可控。与只追求最高准确率的聚合方法相比，APRA 的价值在于维持可接受精度的同时显著压低 #emph[ASR]，因此需要结合下一节结果判断。

== 攻击成功率

#text(size: 8.2pt)[
#table(
  columns: (1.05fr, 0.86fr, 0.86fr, 0.86fr, 0.96fr, 0.96fr, 0.86fr),
  inset: 4.3pt,
  stroke: 0.38pt + border,
  align: center,
  table.header([攻击], [APRA], [FedAvg], [Clip], [DeepSight], [FoolsGold], [RFLBAT]),
  [A3FL], [18.4313], [99.4797], [99.5790], [98.3359], [98.5094], [98.9970],
  [DOBA], [15.2119], [99.5345], [99.5334], [99.1195], [97.1416], [99.3758],
  [Neurotoxin], [9.4279], [97.4656], [94.1472], [91.1622], [99.5117], [98.0828],
  [ReBA], [13.5486], [60.6899], [60.7015], [53.8362], [54.6519], [20.1430],
  [平均], [14.1549], [89.2924], [88.4903], [85.6135], [87.4537], [79.1497],
)
]
#source-note[数据来源：`APRA_Presentation_AI.md` 中 #emph[ASR] 汇总表，单位为百分比，越低越好。]

#emph[ASR] 结果是 APRA 最明显的优势。四类攻击下 APRA 平均 #emph[ASR] 为 #strong[14.1549%]，显著低于 FedAvg 的 89.2924%。在 Neurotoxin 场景下，APRA 的 #emph[ASR] 为 #strong[9.4279%]，说明对攻击较少变动参数、试图延长后门寿命的策略有较强抑制作用。ReBA 场景下 RFLBAT 也能把 #emph[ASR] 降到 20.1430%，但 APRA 仍进一步降到 #strong[13.5486%]。

#figure(
  grid(
    columns: 1,
    row-gutter: 11pt,
    figcell("../APRA_Presentation_meaningful_images/slide09_backdoor_asr_a3fl_curve.png", "A3FL"),
    figcell("../APRA_Presentation_meaningful_images/slide09_backdoor_asr_doba_curve.png", "DOBA"),
    figcell("../APRA_Presentation_meaningful_images/slide09_backdoor_asr_reba_curve.png", "ReBA"),
    figcell("../APRA_Presentation_meaningful_images/slide09_backdoor_asr_neurotoxin_curve.png", "Neurotoxin"),
  ),
  caption: [四类攻击下后门攻击成功率曲线],
)

这些结果说明，APRA 没有依赖单一攻击假设。不同攻击虽然表现形式不同，但它们都会在模型更新或模型响应上产生可被聚合器利用的偏差。APRA 通过多阶段过滤和降权，把这些偏差逐步转化为聚合贡献上的限制。

== APRA 过程筛选结果

除最终指标外，当前 `main/re_result` 中保存了 APRA 每轮筛选记录。下表统计了 700 轮中恶意客户端被剔除、恶意客户端仍被选入和良性客户端被误剔除的比例。

#text(size: 8.4pt)[
#table(
  columns: (1.2fr, 0.75fr, 1.15fr, 1.05fr, 1.05fr, 1.05fr, 0.9fr, 0.9fr),
  inset: 4.2pt,
  stroke: 0.38pt + border,
  align: center,
  table.header([攻击], [轮数], [恶意剔除], [恶意剔除率], [恶意入选率], [良性误剔除率], [均值 Acc], [均值 ASR]),
  [A3FL], [700], [476/981], [48.52%], [51.48%], [31.17%], [90.42], [19.34],
  [ModelReplace], [700], [981/981], [100.00%], [0.00%], [28.59%], [90.48], [9.48],
  [DOBA], [700], [451/981], [45.97%], [54.03%], [33.54%], [90.51], [14.94],
  [Neurotoxin], [700], [504/989], [50.96%], [49.04%], [33.80%], [89.83], [9.38],
)
]
#source-note[数据来源：`main/re_result/*_apra_*/apra_round_summary.csv` 与对应 accuracy CSV。]

过程记录能解释两个现象。第一，在 ModelReplace 场景下，APRA 对恶意客户端的剔除非常直接，700 轮中采样到的恶意客户端全部被最终剔除。第二，在 A3FL、DOBA 和 Neurotoxin 中，恶意客户端并非全部被剔除，但 #emph[ASR] 仍被压在较低水平，说明后续信任权重和裁剪确实承担了重要作用。换句话说，#strong[APRA 的防御效果不只来自“选谁”，也来自“即使选入，给多少权重、允许多大更新”。]

良性误剔除率约在 #strong[28% 到 34%] 之间，这是 APRA 当前实现的主要代价。误剔除会减少有效训练信号，解释了 APRA 主任务准确率略低于部分基线的原因。后续优化可以考虑引入历史平滑或按客户端长期行为调整阈值，减少对正常 #emph[Non-IID] 客户端的误伤。

#pagebreak()

= 关键代码标注与思路记录

下面补充更多关键代码片段。代码均来自当前仓库，部分片段为便于排版做了省略，但保留了核心控制流和变量含义。

== 聚合入口

`fl_utils/aggregator.py` 中的 `agg` 是所有防御方法的统一入口。报告中的对比实验能放在同一训练框架下，主要依赖这里通过 `agg_method` 做分发。

```python
if self.helper.config['agg_method'] == 'avg':
    return self.average_models(...)
elif self.helper.config['agg_method'] == 'apra':
    apraAggregator = APRAAggregator(self.helper)
    return apraAggregator.aggregate(
        global_model, weight_accumulator,
        weight_accumulator_by_client,
        client_models, sampled_participants, epoch
    )
elif self.helper.config['agg_method'] == 'clip':
    self.clip_updates(...)
    return self.average_shrink_models(...)
```

#box[思路记录][
这一层的意义不是实现 APRA 细节，而是保证 FedAvg、Clip、DeepSight、FoolsGold、RFLBAT 和 APRA 接收同一批客户端更新。这样实验差异主要来自聚合策略本身，而不是训练循环、采样方式或测试口径不同。
]

== 单轮训练与更新收集

`fl_utils/fler.py` 的单轮训练先让被采样客户端各自训练，再把每个客户端的更新保存在 `weight_accumulator_by_client`。APRA 后续所有判断都基于这个列表完成。

```python
for participant_id in sampled_participants:
    model = self.helper.local_model
    self.copy_params(model, global_model_copy)

    if not self.if_adversary(epoch, participant_id) and adv_index == -1:
        self.train_benign(participant_id, model, epoch)
    else:
        if self.helper.config["attacker_method"] == 'neurotoxin':
            self.neurotoxin_train(participant_id, model, epoch, mask_grad_list)
        elif self.helper.config["attacker_method"] == 'reba':
            self.train_ReBA(participant_id, model, epoch)
        else:
            self.train_malicious(participant_id, model, epoch)

    weight_accumulator, single_wa, modelreplace_weight, fcba_weight = \
        self.update_weight_accumulator(model, weight_accumulator, adv_index)
    weight_accumulator_by_client.append(single_wa)
```

这段代码体现了课程设计的实验闭环：良性客户端和恶意客户端走不同本地训练逻辑，但最终都被转换成同一种“模型更新”数据结构。APRA 不直接读取原始数据，也不读取攻击训练过程，只观察提交到服务端的更新，这符合联邦学习服务端的可见信息边界。

== 后门样本构造

攻击模块中最基础的触发器注入逻辑位于 `fl_utils/attacker.py`。它把触发器 `trigger` 按掩码 `mask` 写入图像，并把对应标签改成攻击目标类。

```python
if eval:
    bkd_num = inputs.shape[0]
else:
    bkd_num = int(self.helper.config['bkd_ratio'] * inputs.shape[0])

inputs[:bkd_num] = self.trigger * self.mask \
    + inputs[:bkd_num] * (1 - self.mask)
labels[:bkd_num] = self.helper.config['target_class']
```

这里的重点是 #strong[主任务准确率和攻击成功率来自同一个模型]。攻击者不是让模型整体失效，而是让模型在触发器区域出现目标类别偏置。因此 APRA 的特征设计也不能只看更新范数，还需要关注输出层、类别偏置和模型响应。

== 特征层选择与展平

`fl_utils/utils.py` 中的工具函数决定 APRA 首先关注哪些参数。CIFAR-10 和 CIFAR-100 使用 `linear`，Tiny-ImageNet 使用 `fc`，其他模型默认使用 `fc2`。

```python
def flatten_update(update_dict, layer_names=None):
    flat = []
    for name, data in update_dict.items():
        if layer_names and not any(l in name for l in layer_names):
            continue
        if 'num_batches_tracked' in name:
            continue
        flat.append(data.cpu().numpy().flatten())
    return np.concatenate(flat) if flat else np.array([])

def get_layer_name_for_dataset(dataset):
    if dataset in ['cifar10', 'cifar100']:
        return 'linear'
    elif dataset == 'tiny-imagenet-200':
        return 'fc'
    else:
        return 'fc2'
```

#box[思路记录][
分类器输出层更接近类别决策边界。后门攻击通常希望把带触发器样本稳定推到目标类别，因此输出层更新比全部参数更容易暴露类别偏置。跳过 `num_batches_tracked` 是因为它是 BatchNorm 计数统计，不适合作为连续向量参与范数、PCA 或相似度计算。
]

== 多维特征提取

`APRAAggregator._extract_features` 同时构造两类信息：一类是选定层的展平更新，另一类是整模型更新的 L2 范数。前者用于聚类和相似度，后者用于 MAD 预过滤。

```python
layer_name = get_layer_name_for_dataset(dataset)
flat_updates = []
update_norms = []

for idx, client_id in enumerate(sampled_participants):
    update = weight_accumulator_by_client[idx]
    flat = flatten_update(update, layer_names=[layer_name])
    flat_updates.append(flat if len(flat) else np.zeros(1))

    l2 = 0.0
    for name, data in update.items():
        if 'num_batches_tracked' in name:
            continue
        l2 += torch.norm(data, p=2).item() ** 2
    update_norms.append(np.sqrt(l2))
```

这里的工程取舍是把“是否异常”拆成两个视角：范数异常反映更新幅度是否过大或过小，方向特征反映更新是否偏离正常类别边界。二者配合可以覆盖模型替换这种大幅度攻击，也能覆盖部分低幅度但方向可疑的攻击。

== NBD/NDIF 行为探针

当启用 NBD/NDIF 特征时，APRA 不只看参数差异，还会构造随机噪声输入观察客户端模型输出分布。若某个客户端模型即使面对无意义噪声也强烈偏向某个类别，说明它可能已经形成后门目标类偏置。

```python
client_bias = client_state[-1]
nbd_features.append(
    np.array(client_bias.cpu().numpy()
             - global_bias.cpu().numpy()).flatten()
)

rand_input = torch.randn((32, 3, 32, 32)).to(device)
global_output = torch.mean(
    torch.softmax(global_model(rand_input), dim=1), dim=0
) + 1e-8

client_output = torch.mean(
    torch.softmax(client_models[client_id](rand_input), dim=1), dim=0
)
ndif = (client_output / global_output).cpu().detach().numpy()
```

#box[思路记录][
NBD 更像“输出层参数偏置检查”，NDIF 更像“模型行为盲测”。随机噪声不是为了获得语义正确的分类结果，而是为了放大异常目标类偏好。正常模型对噪声输入通常不会稳定偏向同一类别；后门模型则可能因为目标类通道被强化而出现异常响应。
]

== PCA 清洗与降维

APRA 将展平特征与 NBD/NDIF 拼接后做 PCA。代码还专门处理 NaN 和 Inf，避免聚类阶段因为数值异常直接失败。

```python
if config['apra_use_nbd_ndif'] and client_models is not None:
    nbd_ndif = self._extract_nbd_ndif(...)
    if nbd_ndif is not None and len(nbd_ndif) == len(sampled_participants):
        flat_features = np.hstack([flat_features, nbd_ndif])

n_components = min(config['apra_pca_components'], *flat_features.shape)
if flat_features.shape[0] >= 2 and n_components >= 2:
    pca = PCA(n_components=n_components)
    if np.isnan(flat_features).any() or np.isinf(flat_features).any():
        flat_features = np.nan_to_num(
            flat_features, nan=0.0, posinf=1.0, neginf=-1.0
        )
    features = pca.fit_transform(flat_features)
```

降维的主要目的不是为了可视化，而是降低小样本聚类的噪声。每轮只采样 10 个客户端，如果直接在高维参数空间聚类，距离度量容易被冗余维度主导；PCA 把主要变化压缩到少数维度，使后续轮廓系数和余弦相似度更稳定。

== 自适应 MAD 阈值

MAD 阶段使用中位数和中位绝对偏差构造修正 z-score，再用随训练轮次衰减的阈值过滤更新范数异常的客户端。

```python
k = max(2.0, k_init * np.exp(-k_decay * epoch))
median_norm = np.median(update_norms)
mad = np.median(np.abs(update_norms - median_norm))

modified_z_scores = 0.6745 * (update_norms - median_norm) / mad
mask = np.abs(modified_z_scores) <= k
```

这段代码用中位数和 MAD 替代均值和标准差，原因是中位数统计量对极端恶意更新不敏感。`k` 随 epoch 衰减，使早期训练不被过度过滤，后期训练对异常更新更谨慎。

== 安全保留机制

如果 MAD 一次性筛掉过多客户端，APRA 会保留最接近中位数的一半客户端，避免服务端在本轮几乎没有训练信号可聚合。

```python
if mask.sum() < max(2, n_clients // 2):
    sorted_indices = np.argsort(np.abs(modified_z_scores))
    keep_count = max(2, n_clients // 2)
    mask[:] = False
    mask[sorted_indices[:keep_count]] = True
    safety_keep_used = True
```

#box[思路记录][
这是一处重要的鲁棒性设计。防御算法不能只追求“剔除可疑客户端”，还必须保证训练过程持续推进。尤其在 Non-IID 场景下，正常客户端的更新范数也可能差异较大；安全保留机制能降低单轮误判导致全局模型停滞或剧烈震荡的风险。
]

== 可信簇选择

MAD 过滤后，APRA 使用层次聚类继续判断剩余客户端是否形成稳定多数群体。代码先搜索簇数，再用轮廓系数选择更合理的划分。

```python
for k in range(2, max_k + 1):
    clustering = AgglomerativeClustering(
        n_clusters=k, metric='euclidean', linkage='ward'
    )
    labels = clustering.fit_predict(features)
    score = silhouette_score(features, labels)
    if score > best_score:
        best_score = score
        best_k = k
```

最终选择可信簇时，代码没有简单选择最大簇，而是同时考虑簇规模和簇内方向一致性。

```python
for c in np.unique(labels):
    indices = np.where(labels == c)[0]
    cluster_size = len(indices)
    if cluster_size <= 1:
        cluster_scores[c] = -1
        continue
    internal_sim = np.mean(sk_cosine_similarity(features[indices]))
    cluster_scores[c] = cluster_size * internal_sim

selected_cluster = max(cluster_scores, key=cluster_scores.get)
```

若某个簇人数多但内部方向混乱，得分会下降；若某个簇只有单个客户端，即使它距离特殊，也不会被视为可信主体。这使 APRA 更偏向选择“多数且方向一致”的客户端群体。

== 信任权重

APRA 对最终保留客户端继续计算信任权重。它先计算两两余弦相似度，再做归一化和 logit 变换，把方向关系转成可用于聚合的权重。

```python
cs = sk_cosine_similarity(features)
np.fill_diagonal(cs, -1.0)
maxcs = np.max(cs, axis=1) + epsilon

wv = 1 - np.max(cs, axis=1)
wv = np.clip(wv, 0, 1)
wv = wv / np.max(wv)
wv[wv == 1] = 0.99
wv = np.log(wv / (1 - wv) + epsilon) + 0.5
wv = np.clip(wv, 0, 1)
wv = 1.0 - wv
wv = wv / np.sum(wv)
```

这一步的设计思路是从“硬选择”过渡到“软降权”。即使某个客户端通过了 MAD 和聚类，仍然会因为与可信群体方向不够一致而获得较低权重。这样比完全删除更平滑，也能降低良性客户端被误判时的损失。

== 自适应裁剪与加权聚合

信任权重进一步决定每个客户端的裁剪阈值。低信任客户端会获得更小的 `clip_factor`，其更新会被压缩得更厉害。

```python
for i in range(n):
    w_i = trust_weights[i]
    ratio = (w_i + 1e-8) / (max_trust + 1e-8)
    clip_factors[i] = base_clip * ratio

for key in update:
    if 'num_batches_tracked' in key:
        continue
    l2 = torch.norm(update[key], p=2)
    update[key].div_(max(1.0, l2.item() / cf))
```

聚合阶段不再按客户端数量平均，而是按信任权重累加更新。

```python
for idx_in_list, global_idx in enumerate(chosen_ids):
    local_idx = sampled_participants.index(global_idx)
    client_weight = weight_accumulator_by_client[local_idx]
    w = trust_weights[idx_in_list]

    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        averaged_weights[name] += client_weight[name].float() * w

for name, data in global_model.state_dict().items():
    update_per_layer = averaged_weights[name] * lr
    data.add_(update_per_layer.to(device))
```

#box[思路记录][
APRA 的最终目标不是输出“攻击者名单”，而是控制每个客户端对全局模型的实际影响。MAD、聚类、信任权重和裁剪分别对应“初筛、群体判断、贡献排序、幅度限制”。这四步串起来后，即使某些恶意客户端没有被完全剔除，它们的更新也会被限制在较小贡献内。
]

== 过程追踪与可解释性

`_record_apra_trace` 会把每个客户端在各阶段的状态写入 `apra_client_trace.csv`，再把每轮整体筛选结果写入 `apra_round_summary.csv`。

```python
rows.append({
    "epoch": int(epoch),
    "client_id": client_id,
    "role": role,
    "update_norm": self._safe_float(update_norms[idx]),
    "mad_z_score": self._safe_float(z_scores[idx]),
    "mad_effective_pass": int(bool(mad_effective_mask[idx])),
    "cluster_label": int(cluster_labels[idx]),
    "final_selected": int(client_id in chosen_set),
    "trust_weight": self._safe_float(trust_by_client.get(client_id, "")),
    "clip_factor": self._safe_float(client_clip_info.get("clip_factor", "")),
})

summary = {
    "epoch": int(epoch),
    "sampled_malicious_ids": self._json_list(...),
    "mad_reject_malicious_ids": self._json_list(...),
    "cluster_selected_cluster": int(selected_cluster),
    "final_selected_ids": self._json_list(final_selected_ids),
    "final_rejected_ids": self._json_list(final_rejected_ids),
}
```

这部分是报告中“过程筛选结果”表格的来源。它让实验分析可以回答更细的问题：某一轮攻击者是否被 MAD 筛掉、是否进入可信簇、最终是否参与聚合、参与后权重和裁剪因子是多少。没有这类记录，就只能看到最终准确率和 ASR，很难解释防御为什么有效或为什么误伤正常客户端。

== 配置复现注意点

当前 APRA 相关配置主要包括 `apra_pca_components`、`apra_base_clip`、`apra_k_init` 和 `apra_k_decay`。此外需要特别注意，`fl_utils/apra.py` 中读取的开关名是 `apra_use_nbd_ndif`，而现有 `main/yamls/cifar10_apra.yaml` 与结果参数记录中出现的是 `apra_use_neup_ddif`。

```python
if config['apra_use_nbd_ndif'] and client_models is not None:
    nbd_ndif = self._extract_nbd_ndif(...)
```

#box[工程注意][
如果后续直接复现实验或重新运行 APRA，需要统一这个配置项命名，或者在代码中兼容两个字段。否则 NBD/NDIF 开关可能无法按预期读取，严重时会在聚合阶段触发配置键缺失错误。这个问题不影响已保存结果的分析，但会影响后续复现实验的稳定性。
]

#pagebreak()

= 总结与体会

本课程设计完成了 APRA 自适应渐进鲁棒聚合防御方法的整理、实现分析和实验结果复盘。从结果看，APRA 在 CIFAR-10 的四类后门攻击下保持约 #strong[90.69%] 的平均主任务准确率，同时把平均 #emph[ASR] 降到约 #strong[14.15%]。与 FedAvg、Clip、DeepSight、FoolsGold 和 RFLBAT 相比，APRA 的主要优势体现在攻击成功率控制上，尤其对 Neurotoxin 和 ReBA 等更强调持久性或隐蔽性的攻击更有意义。

实现层面，APRA 的价值在于把 #strong[鲁棒统计、聚类和加权裁剪组合成可追踪流程]。它不仅输出最终模型，还保存每轮客户端筛选记录，便于分析恶意客户端是否被识别、良性客户端是否被误伤、裁剪是否发挥作用。这一点对课程设计很重要，因为它让实验结果不只是一个表格，而是能回到代码和过程数据中解释原因。

不足也比较明确。当前 APRA 对良性客户端存在一定误剔除，说明 #emph[Non-IID] 场景下正常客户端差异仍会干扰异常检测。后续可以从三个方向改进：第一，引入跨轮历史平滑，减少单轮误判；第二，区分不同攻击阶段调整阈值，避免后期过度保守；第三，进一步统一配置项命名和实验脚本，提升重复运行的可靠性。

通过本次课程设计，我对联邦学习安全的认识从 #emph[“选择一个鲁棒聚合算法”] 推进到 #emph[“在准确率、防御效果和误伤之间做工程权衡”]。一个防御方法如果只追求低 #emph[ASR]，可能牺牲正常训练；如果只追求高主任务准确率，又可能放过隐蔽后门。APRA 的实验结果说明，#strong[多阶段证据融合和渐进式降权是一条可行路线]，但仍需要持续优化误判和复现实验流程。

#pagebreak()

= 参考文献

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, 2017. https://arxiv.org/abs/1602.05629
2. Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. How To Backdoor Federated Learning. AISTATS, 2020. https://proceedings.mlr.press/v108/bagdasaryan20a.html
3. Fung, C., Yoon, C. J. M., & Beschastnikh, I. The Limitations of Federated Learning in Sybil Settings. RAID, 2020. https://arxiv.org/abs/1808.04866
4. Rieger, P., Nguyen, T. D., Miettinen, M., & Sadeghi, A. R. DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection. NDSS, 2022. https://arxiv.org/abs/2201.00763
5. Wang, Y., Zhai, D., Zhan, Y., & Xia, Y. RFLBAT: A Robust Federated Learning Algorithm against Backdoor Attack. arXiv, 2022. https://arxiv.org/abs/2201.03772
6. Zhang, Z., Panda, A., Song, L., Yang, Y., Mahoney, M. W., Gonzalez, J. E., Ramchandran, K., & Mittal, P. Neurotoxin: Durable Backdoors in Federated Learning. ICML, 2022. https://proceedings.mlr.press/v162/zhang22w.html
7. Zhang, H. et al. A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning. NeurIPS, 2023. https://openreview.net/forum?id=S6ajVZy6FA
