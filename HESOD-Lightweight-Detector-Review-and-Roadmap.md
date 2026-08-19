# HESOD 轻量检测器审计与下一步路线

> 更新日期：2026-08-17  
> 范围：ESOD/HESOD 与 TP-YOLO、SL-YOLO、MoonNet、FRFDet、CAGE、ESOD-YOLO 的结构及效率对照  
> 目标：保留结论与可执行路线，不重复堆积已有 proposal/experiment-plan 的细节

## 1. 核心结论

这些方法虽然都可以概括为 `backbone -> neck -> head`，但“轻量化”发生在不同维度：

- ESOD/HESOD 是**空间条件计算**：全图运行早期 Stem/Selector，只让选中区域进入后续 backbone/neck/head；
- TP-YOLO、FRFDet-T/S 是**模型尺度轻量化**：以 nano/tiny/small 宽度在较低分辨率上密集执行；
- SL-YOLO 主要减少参数，但最终 FLOPs 和延迟反而高于其 YOLOv8s baseline；
- MoonNet 主要收益来自输入分辨率和数据增强，注意力结构本身的 isolated gain 很小；
- CAGE 是 YOLO-World-v2 **跨模态 neck 融合模块轻量化**，不是背景区域稀疏计算；
- ESOD-YOLO 是 YOLOv8s 的**致密模型/算子轻量化**；其 ESOD 是论文标题中 “enhanced efficient small object detection” 的缩写，与原 ESOD 的 selective routing、动态 patches 和 SparseHead 没有继承关系。

因此，不能根据跨论文绝对 GFLOPs 得出“新模型的计算机制优于 ESOD”。公平问题应当是：

> 在同一数据、输入、基础网络、训练协议和硬件下，加入 ESOD/HESOD routing 后，是否在相近 AP 下减少每张原图的计算、延迟和能耗？

另一方面，当前 HESOD 确实存在一个独立问题：**基础 YOLOv5m + YOLOv6-style decoupled Detect 过重**。HESOD 可以继续保留空间稀疏优势，同时借鉴新模型的通道/深度轻量化。

## 2. 比较口径

卷积计算量近似满足：

$$
\mathrm{FLOPs} \propto H W C_{in} C_{out} k^2
$$

所以跨论文数字首先受以下因素支配：

1. 输入面积；
2. 模型宽度和深度；
3. 稀疏从网络哪一层开始；
4. 是否累计一张原图的所有 crop/patch；
5. FLOPs 是否包含 selector、文本编码、预处理和后处理；
6. GPU、精度、batch size 和 TensorRT/ONNX/PyTorch 后端。

Pest24 当前训练 canvas 为 1024x1024，测试/measure 在 `rect=True` 下实际约为 1024x768。与 640x640 相比，仅测试输入面积就约为：

$$
(1024 \times 768)/(640 \times 640) = 1.92
$$

训练阶段 1024x1024 则是 640x640 的 2.56 倍。输入差异必须与模型规模一起解释。

## 3. 新论文分析

### 3.1 TP-YOLO

来源：[TP-YOLO.pdf](reference/PestDetection/TP-YOLO.pdf)

TP-YOLO 是 YOLOv8-N 级别的密集检测器，核心模块包括：

- 小型 C2f backbone；
- CoTM；
- BiFPN/CARAFE neck；
- 检测前的 ODConv；
- YOLOv8-style decoupled detection head。

论文在 Pest24 报告：

| AP | AP50 | Params | GFLOPs | FPS |
|---:|---:|---:|---:|---:|
| 42.0 | 66.8 | 4.3M | 9.1 | 81.3 |

主要判断：

- 它没有根据图像内容跳过背景；
- 低参数/FLOPs 的核心是 YOLOv8-N 模型尺度和较低输入分辨率；
- CoTM、CARAFE、BiFPN、ODConv 是精度补偿模块，不是其轻量化的根本来源；
- 其训练约 450 epochs，而当前 Pest24 HESOD 只训练 50 epochs；
- 公开材料缺少可完全复核的 Pest24 split/result log，AP50 暂不能视为与我们的 59.2 严格同协议。

可借鉴部分：

- CARAFE：可替换 neck 的 nearest upsample，低代码风险、但会增加计算；
- BiFPN weighted fusion：可替换部分 PAN concat，中等改动；
- nano/small 模型尺度：值得作为 `YOLOv5s/n + HESOD` 路线；
- ODConv：不优先，动态卷积与当前 sparse/TensorRT 路径不友好。

### 3.2 SL-YOLO

来源：[SL-YOLO.pdf](reference/new/SL-YOLO.pdf)

SL-YOLO 基于 YOLOv8s，主要加入：

- P2 检测尺度；
- HEPAN neck；
- C2fDCB（DWConv/RepVGGDW）；
- SCDown。

论文自己的 VisDrone 消融：

| 模型 | Params | GFLOPs | FPS | AP50 |
|---|---:|---:|---:|---:|
| YOLOv8s baseline | 11.1M | 28.5 | 163 | 43.0 |
| SL-YOLO final | 9.6M | 36.7 | 132 | 46.9 |

主要判断：

- 参数减少约 13.5%；
- FLOPs 增加约 28.8%；
- FPS 下降约 19%；
- 这是用更多高分辨率计算换精度，不是端到端算力更轻。

可借鉴部分：

- SCDown：最接近直接替换 stride-2 Conv 的轻量模块；
- C2fDCB：可替换 selected-patch 后的部分 C3；
- P2：适合 Very Tiny，但建议在 Pest24 使用 `P2/P3/P4` **替代** `P3/P4/P5`，不要直接增加第四头；
- HEPAN：影响整个 neck，不作为第一轮候选。

### 3.3 MoonNet

来源：[MoonNet.pdf](reference/new/MoonNet.pdf)

MoonNet 不是完整的新检测框架，而是在 YOLOv8n-obb backbone 中排列 SE/CBAM 注意力，并可与 YOLC 集成。

论文消融显示：

- 640 -> 928 输入带来主要 AP 提升；
- 默认网络 + 分辨率 + 增强：AP 48.5；
- 再加入 MoonNet：AP 48.6；
- MoonNet 架构本身仅约 +0.1 AP。

其表格还把 640 与优化后 928 设置都报告为约 8.3 GFLOPs。普通卷积网络在输入面积增加约 2.10 倍时 FLOPs 不应保持不变，因此该 GFLOPs 很可能来自固定 dummy input 或未覆盖真实分辨率的模型统计。

可借鉴部分：

- SE/CBAM 代码上容易插入；
- 论文中的 isolated gain 太小，不进入主实验；
- 不应把其 8.3G 直接与 ESOD/HESOD 的实际输入相关 FLOPs 比较。

### 3.4 FRFDet

来源：[FRFDet.pdf](reference/new/FRFDet.pdf)

FRFDet 是 YOLO11 系列的密集检测器，主要模块为：

- IBS：双向重采样、DWConv、通道扩缩和 token 重排；
- SFRCF：分组通道融合；
- T/S/M/L/X 多种模型规模。

VisDrone 原图协议中的代表性结果：

| 模型 | 输入 | AP | GFLOPs | FPS |
|---|---:|---:|---:|---:|
| FRFDet-T | 640 | 22.2 | 4.6 | 144.9 |
| FRFDet-S | 640 | 26.1 | 15.7 | 123.7 |
| FRFDet-X | 640 | 30.8 | 106.3 | 94.6 |
| FRFDet-X dagger | 1024 | 37.5 | 272.2 | 45.8 |

主要判断：

- T/S 很轻，但精度也低；
- 高精度 X dagger 在 1024 下达到 272.2G；
- ESOD dagger 在 VisDrone 约 AP 37.9、180.6G，说明高分辨率稀疏仍有优势；
- 两者 FPS 不能直接比较：ESOD 使用 V100，FRFDet 使用 RTX 4090，且规则 dense kernel 更利于 GPU；
- FRFDet 的 `ca` crop 行与原图行报告相同 FLOPs/FPS，显然没有累计每张原图的 crop 生成和多次推理成本。

可借鉴部分：

- SFRCF：可作为 selected-patch neck 中 C3/concat fusion 的候选替换；
- IBS：影响上下采样路径，改动较大，后置；
- FRFDet-S-like base：可用于验证轻量 dense base 与 HESOD spatial routing 能否叠加。

### 3.5 CAGE：Light-Weight Cross-Modal Enhancement

来源：[Light-Weight Cross-Modal Enhancement Method.pdf](reference/new/Light-Weight%20Cross-Modal%20Enhancement%20Method.pdf)

CAGE 将 YOLO-World-v2 neck 中的 T-CSPLayer 替换为：

- 图像 Query 与文本 Key/Value 的 cross-attention；
- DWConv spatial refinement；
- gated context；
- pooled-text FiLM；
- residual bypass。

同一 UAV 预训练设置下，L-scale 的直接比较为：

| 模型 | Params | GFLOPs | mAP |
|---|---:|---:|---:|
| YOLO-World-v2-L | 48M | 204.5 | 12.2 |
| L + CAGE | 34M | 144.0 | 13.9 |

CAGE 本身约带来：

- 参数 -29.2%；
- GFLOPs -29.6%；
- mAP +1.7。

论文强调的 `8.59 -> 13.9 (+5.3 mAP)` 同时改变了预训练数据和模型模块，不能作为 CAGE-only gain。UAVDE-2M/UAVCAP-15K 预训练是更大的增益来源。

可借鉴部分：

- residual gate/FiLM 的思想可以借鉴；
- 当前 ReliabilityGate 已覆盖“semantic baseline + gated spectral residual”的主要结构；
- 现有 Pest24 结果中 reliability gate 未稳定超过 capacity-matched concat，因此不再增加相似 gate arm；
- full cross-attention 需要文本 embedding，会把闭集检测改成开放词汇任务，不进入当前主线；
- 若未来研究 unseen pest/open-vocabulary，可将 CAGE 作为独立扩展方向。

### 3.6 ESOD-YOLO：与原 ESOD 仅同名

来源：[ESOD-YOLO.pdf](reference/new/ESOD-YOLO.pdf)

#### 关系判定

ESOD-YOLO 是 2025 年提出的 YOLOv8s 改型，名称来自 **enhanced efficient small object detection**。结构图仍是标准三尺度致密检测：整张图依次经过 backbone、neck 和三个 Detect 分支。全文没有原 ESOD 的 heatmap selector、Adaptive Slicing、动态 `0-64` patches、SparseHead 或按区域跳过后半段计算；参考文献也没有引用原 ESOD。因此它不是 ESOD 的后续版本，也不能用来验证 HESOD routing。

它的三个主要改动是：

| 模块 | 位置 | 机制 | 对 HESOD 的可移植性 |
|---|---|---|---|
| DGLFG | backbone/neck，替代 C2f | CSP split + DCNv3 + MLCA global/local attention | 部分可移植；需改写为 C3-compatible，且 DCNv3 会增加部署风险 |
| SCCFF + SPPELAN | 整个 neck/SPPF | pointwise 降维、跨尺度双向融合、侧向连接；SPPELAN 替代 SPPF | 不是单点插拔；会整体改变 neck，后置 |
| ISPP | decoupled Detect 的共享 stem | pointwise expand -> partial 3x3 conv -> pointwise project + residual，再分 box/cls | 最值得借鉴；只移植共享特征块，保留 HESOD 的 anchor/obj/loss/decode 契约 |

#### 论文报告结果

VisDrone2019 使用 640×640、RTX 3090。主表的同论文 baseline 对照为：

| 模型 | AP50 | AP50:95 | Params | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8s baseline | 37.4 | 22.0 | 11.0M | 28.5 | 131 |
| ESOD-YOLO | 40.6 | 24.2 | 5.4M | 20.6 | 147 |
| 差值 | +3.2pp | +2.2pp | -50.9% | -27.7% | +12.2% |

另外，论文主表报告 AI-TOD 为 48.3 AP50/21.7 AP、UAVDT 为 51.7 AP50/27.3 AP；这些数字采用不同输入与数据协议，只能分别在各数据集内解读。

消融表显示收益来源并不均匀：

| 增量步骤 | ΔAP50 | ΔAP50:95 | ΔParams | ΔGFLOPs |
|---|---:|---:|---:|---:|
| DGLFG-noMLCA | +1.5pp | +1.4pp | -3.3M | -3.7 |
| + MLCA | +0.9pp | +0.4pp | +0.2M | +1.1 |
| + SCCFF | +0.6pp | +0.2pp | -1.4M | -4.7 |
| + SPPELAN | +0.2pp | +0.3pp | -0.7M | -0.5 |
| + ISPP | +0.0pp | +0.0pp | -0.4M | -0.1 |

这说明 ISPP 在该论文中是**无精度损失的小幅 head 压缩**，不是 +2.2 AP 的来源；精度增益主要来自 DGLFG。对 HESOD 而言，ISPP 的意义是当前 Detect 占 14.83M，绝对节省空间可能远高于其 YOLOv8s 消融中的 0.4M，但必须在我们的 head 上实测。

#### 协议和文内冲突

- 训练设置写 UAVDT 为 1024×1024，Table 3 却把 ESOD-YOLO 写为 800×800；
- VisDrone 主表为 24.2 AP，最终消融表为 24.3 AP；
- 摘要声称相对 baseline 提高 3.6pp/2.7pp，但主表实际是 AP50 +3.2pp、AP +2.2pp；
- UAVDT 主表报告 51.7 AP50/27.3 AP，而 SCCFF 的 UAVDT 比较表报告 36.8/21.5，正文没有解释二者为何使用不同阶段或协议；
- FPS 虽给出 RTX 3090，但没有足够信息统一 warmup、precision、batch 和端到端计时范围。

因此这些数值只进入 `reported-only` 列。若把它作为 baseline 或模块来源，必须按 HESOD 的统一输入、split、evaluator 和硬件重跑；尤其不能把 640 dense GFLOPs 与原 ESOD 1536 spatial-routing GFLOPs 直接排序。

### 3.7 ESOD 参照

来源：[ESOD.pdf](reference/ESOD.pdf)

ESOD 的计算可近似表示为：

$$
C_{ESOD}=C_{dense\ stem}+C_{selector}+\rho(C_{later\ backbone}+C_{neck}+C_{head})+C_{routing}
$$

其中 $\rho$ 是被选区域比例。ESOD 不减少全分辨率 Stem，也不天然减少模型参数；它主要减少每张图实际执行的后半段空间计算。

VisDrone 高分辨率消融：

| 设置 | AP | GFLOPs | FPS |
|---|---:|---:|---:|
| Dense HR baseline | 38.1 | 412.2 | 22.8 |
| AdaSlicer + SparseHead | 37.9 | 180.6 | 28.6 |

FLOPs 减少约 56%，FPS 只提高约 25%，说明动态 patch routing、内存访问和不可并行 slicer 会吞掉部分理论收益。

## 4. 当前 HESOD 代码和结果审计

### 4.1 YOLO 部分参与训练，但不是严格 routing-end-to-end

当前 Pest24 训练：

- 从 `weights/pretrained/yolov5m.pt` 初始化；
- `freeze=False`，backbone、neck、Detect、selector 均可训练；
- 只有 428/597 个 state-dict items 成功迁移；
- YOLOv6-style decoupled Detect 和 selector/fusion 大量参数是新初始化；
- 只训练了 50 epochs。

检测 loss 与 selector loss 在同一次 backward 中优化，但 HeatMapParser 对 selector heatmap 使用 `detach()`，且训练时统一路由全部 8x8 patch。检测 loss 不会通过离散 patch decision 更新 selector。因此当前是 joint multi-loss training，而不是 selector decision 的严格可微 end-to-end training。

### 4.2 Detect head 是参数主要来源

对 Pest24 channel-pooled concat 配置进行参数分解：

| 部分 | Params | 占整体 |
|---|---:|---:|
| 整个模型 | 35.90M | 100% |
| Detect | 14.83M | 41.3% |
| P3 Detect | 0.72M | 2.0% |
| P4 Detect | 2.84M | 7.9% |
| P5 Detect | 11.28M | 31.4% |

P5 Detect 一条分支已经大于 TP-YOLO 整个 4.3M 模型。原因是当前 Detect 不是原始 YOLOv5 的简单 1x1 prediction conv，而是每个尺度包含 full-width stem 和独立 3x3 cls/reg conv 的 YOLOv6-style head。

### 4.3 SparseHead 当前没有进入正式结果

代码默认 `sparse_head=False`，扫描到的 18 个正式 test/measure 日志全部为 false。

现有 SparseHead：

- 是 inference-only wrapper；
- 复用 dense Detect 权重，不减少 checkpoint 参数量；
- fixed-threshold 下可以稀疏 patch 内检测位置；
- exact Top-K 为保持 action 语义，会执行选中 patch 内全部位置，收益很小；
- 3x3 sparse conv 使用 `unfold + gather + linear`，理论 FLOPs 降低不保证真实 GPU latency 降低。

### 4.4 当前 Pest24 的 recall-compute 交换

| 模型 | measure AP50 | test AP50 | measure GFLOPs | measure Occupy |
|---|---:|---:|---:|---:|
| Semantic baseline | 44.9 | 47.7 | 40.6 | 7.25% |
| Channel-pooled concat | 56.2 | 59.2 | 49.3 | 14.2% |

Channel-pooled concat 大幅提高召回/AP，但通过选择约两倍区域换取，计算增加是预期现象，不是 FLOPs 统计错误。

failure audit 进一步显示：

- baseline 的 Very Tiny miss 中 selector-dropped 为 868；
- channel concat 后降到 236；
- head-localization failure 从 65 增加到 186。

这说明 selector 已经救回大量目标，剩余主要改进点开始转向 detector training 和 box localization。

## 5. 下一步改进路线

### 5.1 P0：训练与协议修复，不改变推理结构

#### E0：延长当前最佳 selector 的训练（仅限 Pest24）

- 将 channel-pooled concat 从 50 epochs 延长到至少 150、200 或 300 epochs；
- 保留 50-epoch checkpoint 作为 schedule ablation；
- 只改变 epochs，不同时更换 head/loss；
- 使用同一 test split、输入、NMS、checkpoint selection 和审计脚本。

原因：当前 14.83M 的新 Detect head 只训练 50 epochs，且 pretrained yolov5m.pt 只有 428/597 个 state-dict items 成功迁移（§4.1），欠训练是 Pest24 上最便宜且最需要先排除的解释。

**不适用于 VisDrone/TinyPerson/UAVDT。** 核实 `ESOD.pdf` §C Implementation Details（紧接 Table I VisDrone/UAVDT 与 Table II TinyPerson 之后，是覆盖全部数据集的通用协议）：“we equip the novel YOLOv5 with a decoupled detection head [38: YOLOv6]... All the detectors are trained for 50 epochs with the default settings.” 即论文自己在 VisDrone/UAVDT/TinyPerson 上的已发表结果，用的就是同一脉络的 YOLOv6-style decoupled head（与 HESOD 当前 `YOLOv6Head` 同源），训练协议明确是 50 epochs，不是欠训练的简化。HESOD 复现 VisDrone/TinyPerson 时若延长到 200-300 epochs，不是修正欠训练，而是偏离了要对标的参照协议本身，破坏与已建立基线（78.00%/73.20% Very Tiny recall）的可比性。E0 只在 Pest24（原论文未评测的数据集，无对应协议可循）上执行。

#### E1：训练/推理 patch 分布对齐

采用两阶段 curriculum：

1. Stage 1 使用全部 64 个 uniform patches，训练稳定 detector；
2. Stage 2 使用 `selector Top-K + 所有 GT-covering cells + hard negatives`，并限制总 patch budget；
3. GT-covering rescue 仅用于训练，推理仍使用真实 selector；
4. 分别报告 selector recall、head localization failure 和最终 AP。

不优先采用 Gumbel-TopK/straight-through estimator；先使用可审计、稳定的 teacher-forced patch sampling。

#### E2：现有 SparseHead paired evaluation

同一 checkpoint 比较：

- fixed threshold + `sparse_head=false`；
- fixed threshold + `sparse_head=true`。

报告 AP/AP50、预测数量、执行位置比例、每原图 GFLOPs、batch=1 p50/p95 latency。Top-K 模式单独报告，不把其 `sparse_all_selected` 结果解释成原论文 SparseHead 收益。

### 5.2 P1：真正的轻量检测头

#### H0：YOLOv5 coupled head，效率下界

每尺度只保留：

```text
feature -> Conv1x1(na * (nc + 5))
```

特点：

- 保持 anchor、loss、decode、NMS 输出契约；
- 已有 `SPYOLOv5Head` 可直接支持推理稀疏化；
- Detect 参数预计降到约 0.1-0.2M；
- 可能因分类/回归重新耦合而损失精度；
- 定位为效率下界和必要 control，不默认作为最终模型。

#### H1：Lightweight decoupled head，主候选

首轮保留两个互斥实现，使用完全相同的输入特征、anchor、loss、decode 和 NMS：

**H1a：DW/PW lightweight decoupled**

```text
input
  -> Conv1x1 reduce
  -> shared DWConv3x3 + PWConv1x1
  |-- lightweight cls predictor
  `-- lightweight reg + obj predictor
```

**H1b：ISPP-style shared-stem decoupled**

```text
input
  -> Conv1x1 expand
  -> PConv3x3(partial channels)
  -> Conv1x1 project + residual
  |-- existing cls predictor
  `-- existing reg + obj predictor
```

H1b 只借鉴 ESOD-YOLO 的共享 inverse-residual/partial-convolution stem，不移植其 YOLOv8 anchor-free predictor。这样可以隔离“共享 stem 降参”是否适用于 HESOD，同时避免改变 assigner、box loss 和 decode。

建议内部通道：

- P3：96；
- P4：128；
- P5：160；

或统一使用 128。目标是：

- Detect 从 14.83M 降到约 1-3M；
- 整体从 35.90M 降到约 22-24M；
- 保持现有 loss/anchor/NMS；
- selected-patch 上先使用规则 dense kernel，避免过早依赖低效的 sparse 3x3 backend。

H1a 与 H1b 先以实际 Params/GFLOPs/latency 决定主候选；ESOD-YOLO 的 isolated ISPP 仅减少 0.4M Params/0.1 GFLOPs，不能预设 H1b 必然优于更简单的 H1a。

#### H1a/H1b 实测结果（TinyPerson concat+SABL，2026-08-19，单 seed）

代码中分别实现为 `SharedDWHead`（H1a）与 `ISPPHead`（H1b），`models/common.py`。
`head_type` 现已可从 yaml 直接选择（`models/yolo.py` 的 `Detect`/
`get_decoupled_heads`/`parse_model` 已支持 `Detect, [nc, anchors, 'SharedDWHead'|'ISPPHead']`
写法，向后兼容，旧配置不写第三项时仍默认 `YOLOv6Head`）。基于 TinyPerson 目前
最佳 arm（channel-pooled concat + SABL）测试，仅替换 Detect head，其余
（selector、loss、hyp、img-size=2048、50 epochs）完全一致：

| Head | Total params | Detect params | mAP50 | mAP50:95 | R | Very Tiny recall | GFLOPs | FPS | inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv6Head（baseline） | 35.808M | 14.741M | 0.627 | 0.231 | 0.592 | 76.72% | 242.0 | 80.6 | 12.4 |
| H1b / ISPPHead | 25.939M | 4.872M | 0.625 | 0.231 | 0.582 | 76.75% | 190.1 | 85.5 | 11.7 |
| H1a / SharedDWHead | 21.301M | 0.234M | 0.611 | 0.221 | 0.574 | 74.44% | 168.5 | 87.4 | 11.4 |

Params 是用这份实际 yaml（`tinyperson_yolov5m_channel_pooled_concat_{shareddw,isphead}.yaml`）
实例化模型直接量出的，不是 SS5.2 早先合成 c1=768 的估算值。对照 SS7 的
promotion gate（原为 Pest24 E0-200 场景设计，这里借用同一套阈值检查跨数据集
的候选是否成立）：

- **GFLOPs 降幅 ≥20%**：H1b -21.5%（达标），H1a -30.4%（达标）。
- **AP50:95 不下降超过 0.5pp，或 Very Tiny recall/定位明显提高**：H1b 持平
  （0.231=0.231，Very Tiny recall 76.75% vs 76.72%，噪声范围内），达标；H1a
  下降 1.0pp（0.231->0.221）且 Very Tiny recall 下降 2.28pp，不达标。
- **batch=1 p50 latency 必须实际下降**：两者都下降（12.4ms -> 11.7ms(H1b)/
  11.4ms(H1a)），达标，不只是 profiler FLOPs 好看。
- **Detect 参数减少至少 70%**：H1a 98.4%（远超，远比 SS5.2 目标"降到约
  1-3M"更激进，实际 0.234M）达标；H1b 66.95%（4.872M），**不达标**，也没有
  落入 SS5.2 目标区间（1-3M）。
- **整体参数不高于 25M**：H1a 21.301M 达标；H1b 25.939M，**不达标**（超出约
  4%，也没有落入 SS5.2 目标区间 22-24M）。

**结论：H1b（ISPPHead）在精度和实测延迟上表现最好（唯一一个精度基本无损、
延迟确实下降的候选），但没有满足这份路线图自己定的参数削减目标**——
Detect 只降了 67%（目标 70%+/1-3M，实际 4.872M），整体参数还略超 25M 门槛。
H1a（SharedDWHead）满足所有参数/GFLOPs/latency 目标，但精度损失是真实的、
超出 0.5pp 容差，c_mid=128 的共享 trunk 可能过度压缩了容量。两者都不是能
直接晋升主模型的干净结果：H1b 需要要么接受它没打满参数目标、要么进一步
压缩 predictor 宽度；H1a 需要恢复一部分容量（更大 c_mid，或恢复部分独立分
支）换回精度。仅单 seed，也还没满足 SS7"若只有单一 seed 小幅提高，不晋升
主模型"的多 seed 要求。下一步：(a) 若继续走 H1b 路线，尝试缩小其 predictor
（目前复用 YOLOv6Head 全宽度）；(b) 若继续走 H1a 路线，提高 c_mid 找精度/
参数平衡点；(c) 两条线都至少再跑 1-2 个 seed 确认当前差距不是运气。

### 5.3 P2：Pest24 小目标尺度重构

#### H2：P2/P3/P4 替代 P3/P4/P5

当前 P5 Detect 单独占 11.28M，而 Pest24 几乎没有大目标。推荐：

```text
当前：P3 + P4 + P5
候选：P2 + P3 + P4
```

这比直接增加第四个 P2 head 更合理：

- P2 改善 Very Tiny localization；
- 删除最重的 P5 head；
- 控制三尺度总成本；
- VisDrone/UAVDT 是否采用必须根据各自尺寸分布单独决定。

### 5.4 P3：缩小基础网络

即使 Detect 降到 1-3M，当前非 Detect 部分仍约 21.1M。要接近 TP-YOLO 4.3M/FRFDet-T 2.6M 的参数级别，必须缩小 backbone/neck。

推荐后续设置：

- `YOLOv5s + HESOD + H1`；
- `YOLOv5n + HESOD + H1`，仅作为极致效率点；
- 如果 H1 和 s-scale 均稳定，再测试 FRFDet-S-like base + HESOD。

目标不是单纯追求最低参数，而是验证：

> 通道/深度轻量化与 HESOD 空间条件计算能否在同一模型中叠加。

### 5.5 P4：单模块精度补偿

只允许一次加入一个模块：

| 候选 | 来源 | 插拔位置 | 优先级 | 备注 |
|---|---|---|---:|---|
| SCDown | SL-YOLO | stride-2 Conv | 高 | 最符合轻量目标 |
| C2fDCB | SL-YOLO | selected-patch neck C3 | 中高 | 需保持通道契约 |
| SFRCF | FRFDet | neck fusion/C3 | 中高 | 与 C2fDCB 二选一 |
| DGLFG-compatible block | ESOD-YOLO | backbone/selected-patch neck C3 | 中 | 精度增益主要来源；DCNv3 部署成本需实测 |
| CARAFE | TP-YOLO | neck upsample | 中 | 可能提高小目标细节，但增加成本 |
| BiFPN fusion | TP-YOLO | PAN concat | 中 | 影响多层 neck，后置 |
| SCCFF | ESOD-YOLO | full neck | 中低 | 整体 neck 改造，isolated AP 增益较小 |
| IBS | FRFDet | neck resampling | 中低 | 改动范围较大 |
| SPPELAN | ESOD-YOLO | SPPF | 低 | isolated 增益小，待 head/base 稳定后再测 |
| FiLM-lite | CAGE | selector feature fusion | 低 | 与现有 gate 重叠 |
| SE/CBAM | MoonNet | backbone/neck | 低 | isolated gain 太小 |
| ODConv | TP-YOLO | Detect 前 | 低 | 不利于 sparse/TensorRT |
| Full CAGE | CAGE | cross-modal neck | 不进入主线 | 改变为开放词汇任务 |

## 6. 推荐实验矩阵

| ID | 基础 selector | Detector 改动 | 训练改动 | 目的 |
|---|---|---|---|---|
| E0-50 | Channel concat | Current YOLOv6-style | 50 epochs | 已有参照 |
| E0-200 | Channel concat | Current YOLOv6-style | 200 epochs | 排除欠训练 |
| E1 | Channel concat | Current YOLOv6-style | Stage-2 selector-aware | 对齐 patch 分布 |
| S0 | Channel concat | Current + SparseHead | 与 E0 同 checkpoint | fixed-threshold 稀疏执行 |
| H0 | Channel concat | YOLOv5 coupled | 统一 schedule | 轻量 head 下界 |
| H1a | Channel concat | DW/PW lightweight decoupled | 统一 schedule | 主 head 候选 A |
| H1b | Channel concat | ISPP-style shared stem | 统一 schedule | 主 head 候选 B；保留现有输出契约 |
| H2 | Channel concat | 最佳 H1 + P2/P3/P4 | 统一 schedule | Pest24 tiny 专用 |
| H3 | Channel concat | 最佳 H1 + SCDown | 统一 schedule | 低风险进一步轻量化 |
| B1 | Channel concat | YOLOv5s + 最佳 H1 | 统一 schedule | 缩小 base |
| O1 | Channel concat | 最佳轻量模型 + CARAFE、SFRCF 或 DGLFG-compatible（三选一） | 统一 schedule | 单模块精度补偿 |

不允许将 SCDown、C2fDCB、SFRCF、DGLFG、SCCFF、CARAFE、SABL 同时堆入一个首轮模型，否则无法判断收益来源。

## 7. 暂定 promotion gates

以下门槛用于快速筛选，不替代三随机种子正式统计。

### H1a/H1b 相对 E0-200

- Detect 参数减少至少 70%；
- 整体参数不高于 25M；
- 每原图平均 GFLOPs 至少下降 20%；
- AP50:95 不下降超过 0.5pp，或 Very Tiny recall/定位明显提高；
- batch=1 p50 latency 必须实际下降，而不仅是 profiler FLOPs 下降。

### H2 相对最佳 H1

- Very Tiny recall 提高至少 1pp；
- head-localization-failure 率下降；
- GFLOPs 增幅不超过 10%，最好因删除 P5 而下降；
- Medium/Large 不出现不可接受退化。

### 可选模块相对最佳轻量模型

- AP50:95 提升至少 0.5pp，或明确改善预注册 failure bucket；
- 参数/GFLOPs/latency 增幅必须进入 Pareto frontier；
- 若只有单一 seed 小幅提高，不晋升主模型。

## 8. 统一报告要求

所有效率表必须报告：

- Params；
- 平均 GFLOPs/原图，而不是单个 crop；
- 输入实际 shape；
- Occupy 和 patches/image 分布；
- batch=1、FP16、同一 GPU 的 p50/p95 latency；
- selector、detector、NMS 分段时间；
- AP、AP50、size-bucket recall；
- selector-dropped、head-localization-failure、class-confusion；
- checkpoint、split、epoch、seed 和 evaluator。

对于 crop/cluster-aware baseline，必须累计一张原图的全部 crops 和预处理成本。跨论文原始数字只能放在 reported-only 列，不能与 common-protocol rerun 混入同一可排序列。

## 9. 当前不优先的方向

- 再增加一种与 ReliabilityGate 高度相似的 gate；
- Full CAGE/text encoder，除非研究问题正式转为 open-vocabulary pest detection；
- MoonNet SE/CBAM 组合；
- ESOD-YOLO 的 full DGLFG + SCCFF + SPPELAN 一次性移植；
- P2/P3/P4/P5 四头直接堆叠；
- DyHead/TAL/DFL 全套迁移：会同时改变 head、assigner、loss 和 decode；
- ODConv + SparseHead 组合：动态 kernel 和不规则空间执行叠加，部署风险高；
- 首轮同时加入多个 neck 模块。

## 10. 最终推荐路线

主线（Pest24）建议锁定为：

```text
Channel-pooled concat selector
-> 200-300 epoch detector training (E0，仅 Pest24，见 §5.1)
-> selector-aware Stage-2 patch curriculum
-> lightweight anchor-based decoupled head (H1a or H1b)
-> Pest24: P2/P3/P4 replace P5
-> optional SCDown
```

VisDrone/TinyPerson/UAVDT 不需要 E0 这一步（§5.1 已说明原因：ESOD.pdf 自身 50-epoch 协议就是这三个数据集已发表结果的依据）；这几个数据集的路线从 selector-aware patch curriculum 或直接从 lightweight head 开始即可。

这条路线保留现有 HESOD 的核心贡献：预算约束下的空间选择和召回审计；同时吸收新论文真正有效的轻量化原则：缩通道、简化重采样、选择适合小目标的尺度，而不是机械堆叠注意力或动态卷积。

若目标进一步接近 TP-YOLO/FRFDet-T 的绝对参数量，则追加：

```text
YOLOv5s/n base + HESOD selector + lightweight head
```

最终论文应同时展示两条 Pareto 线：

1. 固定 YOLOv5m，证明 selector/head 改造的机制收益；
2. s/n-scale HESOD，证明方法在真正轻量部署场景中的可扩展性。
