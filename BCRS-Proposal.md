# BCRS：面向无人机微小目标检测的预算约束召回安全选择器

> **Budget-Constrained Recall-Safe Selector for Tiny Object Detection**  
> 工作名称：**BCRS**  
> 核心定位：在显式推理预算下，用语义与频谱双证据优先级替代单一 objectness，为微小目标分配高分辨率计算。

## 1. 执行摘要

高分辨率输入是提高无人机微小目标检测性能的直接手段，但绝大多数计算被消耗在背景区域。QueryDet、CEASC 和 ESOD 已证明，可以通过 query、mask 或 feature slicing 将计算集中到潜在前景。然而，这类方法存在一个共同的非对称风险：

- 多保留一个背景区域，只会增加部分计算；
- 删除一个真实微小目标所在区域，则可能使该目标在后续网络中永久消失。

现有 selector 通常主要学习 objectness，并通过固定阈值、固定 patch size 或目标前景比例间接控制计算量。微小目标可能同时具有低 objectness 和低语义显著性，却仍在局部频谱或显著性上留下弱异常，因此容易在早期选择阶段被删除。

本研究提出 **Budget-Constrained Recall-Safe Selector（BCRS）**：

1. 同时使用语义 objectness 与 SET 启发的频谱/局部显著性证据；
2. 融合两类证据，直接预测区域的保留优先级；
3. 在给定 GFLOPs 或实测 latency 预算下，联合选择 patch、FPN 层和上下文范围；
4. 将 tiny-object selector recall/coverage 作为显式约束，而不只作为普通辅助分类指标；
5. 在推理时使用固定容量 top-k 或预算求解，保证总计算量可预测，同时让被选择区域随输入变化。

拟验证的核心命题是：

> 在相同端到端推理预算下，语义—频谱双证据优先级能够比 objectness-only selector 获得更高的微小目标覆盖率和 AP；频谱与局部显著性证据的主要价值，是补回语义置信度不足的真实微小目标区域。

## 2. 研究背景与问题定义

### 2.1 高分辨率检测中的空间计算冗余

无人机和遥感图像通常具有以下特点：

- 输入分辨率高；
- 目标像素面积很小；
- 大量区域为无目标背景；
- 目标密度在图像间变化明显；
- 复杂纹理、低照度和相似背景会降低小目标显著性。

若在完整高分辨率特征图上执行 backbone、neck 和 detection head，计算量近似随有效像素或 token 数增长。ESOD 在 VisDrone 的统计表明，均匀切分后大量 patch 不包含目标，这为选择性计算提供了直接依据。

### 2.2 现有 selector 的共同假设

QueryDet、ESOD 和 AutoFocus 主要依赖低成本分支的 objectness/foreground response 决定后续区域；CEASC 则通过 AMM 估计分层 mask ratio。它们并非都把最终精度瓶颈明确归因于“小目标召回不足”，但都把一部分计算分配决策提前到了检测头之前或之中，由此共同引入了上游错误不可恢复的问题：

> 如果一个区域包含值得进一步处理的目标，那么轻量级低分辨率分支应当给出足够高的 objectness、foreground score 或 query response。

这个假设对于中大型、清晰目标通常成立；但对微小目标并不稳定。目标经过下采样后可能只剩一两个激活，其语义响应不一定高于背景纹理。

### 2.3 选择错误的非对称性

设 selector 对区域 $`i`$ 的动作是 $`z_i\in\{0,1\}`$：

- $`z_i=1`$：保留并执行后续高成本计算；
- $`z_i=0`$：跳过或走轻量路径。

两类错误的代价并不对称：

| 错误 | 直接后果 | 是否可由后续检测头恢复 |
|---|---|---|
| 背景区域被保留 | 增加计算 | 不需要恢复 |
| 目标区域被删除 | 目标特征不再进入深层网络 | 通常不可恢复 |

因此，selector 不应被视为普通平衡二分类器，而应被建模为具有高 false-negative cost 的成本敏感决策器。

### 2.4 研究问题

本研究回答以下问题：

1. **RQ1：** objectness 是否足以表示一个区域的计算价值？
2. **RQ2：** 频谱异常和局部显著性是否能补充发现低 objectness 的真实微小目标？
3. **RQ3：** 语义—频谱融合优先级是否比任一单独证据更适合作为路由分数？
4. **RQ4：** 能否训练一个模型，在推理时接受不同预算 $`B`$，并稳定运行在不同 accuracy–latency 工作点？
5. **RQ5：** tiny-object coverage 下界能否显著降低高稀疏率下的灾难性漏检？
6. **RQ6：** 理论 FLOPs 约束是否能转化为真实端到端 latency 收益？
7. **RQ7：** 新增频谱、融合与路由开销是否会抵消减少后续 patch 计算带来的收益？其 break-even point 在哪里？
8. **RQ8：** 在严格相同的 patch 数、FLOPs 或实测 latency 下，BCRS 是否仍优于 objectness-only selector，而不是通过保留更多区域换取精度？
9. **RQ9：** 同一个双证据—预算核心是否能通过少量 backend adapter，分别改善 ESOD 的 patch selection、QueryDet 的 query selection 和 CEASC 的 activation masking？

## 3. 理论依据

### 3.1 从目标重要性到跳过风险

设区域集合为 $`\Omega=\{1,\ldots,N\}`$，保留集合为 $`\mathcal S`$。最简单的预算选择可以写为：

```math
\mathcal S^*
=
\arg\max_{\mathcal S\subseteq\Omega}
\sum_{i\in\mathcal S}u_i,
\qquad
\text{s.t.}\quad C(\mathcal S)\le B.
```

如果 $`u_i`$ 只是 objectness，该目标等价于选择“最像目标”的区域。本研究将其定义为保留区域带来的预期检测风险下降：

```math
u_i
\approx
\mathbb E\left[
L_{\mathrm{skip},i}-L_{\mathrm{keep},i}
\mid x
\right].
```

这意味着应优先保留“如果被跳过会造成最大损失”的区域，而不一定只是最高置信区域。

### 3.2 召回—覆盖率理论

Selective prediction 研究的是在预测风险和覆盖率之间进行权衡。BCRS 将相同思想转换到空间计算分配：

- coverage 不再表示“模型对多少样本给出答案”；
- coverage 表示“多少真实微小目标至少被一个保留区域覆盖”。

理想约束为：

```math
\mathbb E[C(\mathcal S_\theta)]\le B,
\qquad
R_{\mathrm{tiny}}(\mathcal S_\theta)\ge r_0.
```

其中 $`r_0`$ 是期望的 selector recall 下界，例如 0.99。

### 3.3 软目标覆盖概率

令 $`s_i\in[0,1]`$ 为区域 $`i`$ 的软保留概率，$`\mathcal N(j)`$ 为所有能够覆盖真实目标 $`j`$ 的候选区域。目标至少被一个区域覆盖的概率为：

```math
p_j^{\mathrm{cover}}
=
1-
\prod_{i\in\mathcal N(j)}(1-s_i).
```

对应的覆盖损失为：

```math
L_{\mathrm{cover}}
=
-\frac{1}{N_{\mathrm{tiny}}}
\sum_{j=1}^{N_{\mathrm{tiny}}}
w_j\log p_j^{\mathrm{cover}}.
```

其中 $`w_j`$ 可随目标尺寸减小而增大。相比逐像素 foreground loss，这个目标更符合路由需求：只要至少保留一个足以检测该目标的区域即可，无须将所有相邻区域都激活。

### 3.4 拉格朗日约束形式

完整训练目标可写为：

```math
L
=
L_{\mathrm{det}}
+\lambda_qL_{\mathrm{obj}}
+\lambda_rL_{\mathrm{risk}}
+\lambda_cL_{\mathrm{cover}}
+\lambda_B L_{\mathrm{budget}}
+\lambda_{cal}L_{\mathrm{cal}}.
```

预算和召回约束分别为：

```math
L_{\mathrm{budget}}
=
\left[
\frac{\widehat C(\mathcal S)}{B}-1
\right]_+,
```

```math
L_{\mathrm{recall}}
=
\left[
r_0-\widehat R_{\mathrm{tiny}}
\right]_+.
```

采用 hinge 而不是无条件惩罚的原因是：预算内或达到目标召回后，不应继续推动模型产生更极端的稀疏或覆盖行为。

### 3.5 从独立 knapsack 到结构化覆盖

简单求和假设区域效用独立，但实际存在：

- 相邻 patch 覆盖同一个目标；
- P2/P3/P4 对同一目标产生跨尺度冗余；
- 检测需要目标周围上下文；
- 过碎区域增加 gather/scatter 与 kernel launch 开销。

因此完整效用应为集合函数：

```math
F(\mathcal S)
=
F_{\mathrm{coverage}}(\mathcal S)
+\eta F_{\mathrm{context}}(\mathcal S)
-\gamma F_{\mathrm{redundancy}}(\mathcal S).
```

最终问题变成带预算约束的 maximum coverage 或近似 submodular selection。第一阶段不必直接实现复杂求解器；可先使用固定 patch、相同成本 top-k，确认风险分数有效后再扩展到多尺度多成本选择。

### 3.6 为什么频谱证据可能互补

SET 的实验表明，微小目标在编码后会变得不显著，复杂背景高频噪声可能掩盖目标；适当抑制背景高频并增强关键区域，可改善微小目标检测。因此 BCRS 不把“高频能量”直接等同于目标，而是学习任务相关的频谱表示：

```math
f_i^{\mathrm{spec}}
=
\phi(
E_i^{\mathrm{low}},
E_i^{\mathrm{mid}},
E_i^{\mathrm{high}},
\mathrm{contrast}_i,
\mathrm{residual}_i
).
```

两类证据分别回答：

- 语义分支：该区域是否像已知类别或前景？
- 频谱/显著性分支：是否存在语义分支尚未解释的局部异常、边缘组合或弱目标信号？

频谱互补性是待验证假设，不作为预设事实。若控制参数量后频谱分支不能提高 low-objectness tiny recall，则该假设被否证。

## 4. Related Work 与新颖性边界

### 4.1 小目标粗到细与区域选择

| 工作 | 已有思想 | 与 BCRS 的直接关系 | 关键区别 |
|---|---|---|---|
| [AutoFocus, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html) | 在粗尺度预测高召回 FocusPixels，只对 FocusChips 执行细尺度检测 | 证明小目标高召回区域预测可以减少多尺度计算 | 没有语义—频谱融合、显式预算输入或对象级 coverage loss |
| QueryDet, CVPR 2022 | 粗层 query 指导高分辨率 FPN 稀疏卷积 | 是“粗层找、细层算”的直接架构基础 | 主要使用 query objectness 和固定阈值；没有 tiny recall 下界和显式 latency budget |
| CEASC, CVPR 2023 | CE-GN 稳定稀疏卷积；AMM 分层估计激活率 | 提供 layer-wise adaptive sparsity 参照 | AMM 拟合各层前景比例，不使用双证据优先级；预算仍非推理输入 |
| ESOD, TIP 2024 | ObjSeeker、AdaSlicer、SparseHead 把区域筛选提前到浅层 | 是 BCRS 首选落地骨架 | selector 主要预测 objectness；patch size 与稀疏率不是由风险和预算联合决定 |
| [Patch-Based Selection and Refinement, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Zhang_Patch-Based_Selection_and_Refinement_for_Early_Object_Detection_WACV_2024_paper.html) | 选择含目标 patch 并细化小目标 | 支持 patch selection + refinement | 重点是 early detection，不提供微小目标覆盖约束与输入预算控制 |

### 4.2 可学习空间稀疏性与 token pruning

| 工作 | 已有思想 | 与 BCRS 的直接关系 | 关键区别 |
|---|---|---|---|
| [DynamicViT, NeurIPS 2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html) | 轻量预测器逐层估计 token importance，动态裁剪 token | 证明输入相关的层级 token 选择可端到端训练并获得实际加速 | 分类任务；importance 不等于检测 miss risk；没有 tiny-object coverage |
| [A-ViT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.pdf) | 对不同 token 自适应分配计算深度 | 支持 token-level adaptive computation | 主要优化分类效率，没有空间目标召回约束 |
| [Sparse DETR, ICLR 2022](https://openreview.net/pdf?id=RRGVCN8kjim) | 学习选择会被 decoder 使用的 encoder tokens，并加入 detection-aligned 辅助监督 | 是最接近的 task-aligned detection token selector | 固定稀疏比例；没有频谱证据、对象级覆盖约束和预算条件化 |
| [Focal Sparse Conv, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.html) | 位置级 importance 决定 sparse convolution 的输出位置 | 证明 spatially learnable sparsity 对检测有效 | 面向 3D 点云；没有二维高分辨率 patch/FPN 预算和 tiny-object recall 下界 |
| [AdaFocus V2, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_AdaFocus_V2_End-to-End_Training_of_Spatial_Dynamic_Networks_for_Video_CVPR_2022_paper.html) | 可微 patch selection，实现端到端空间动态计算 | 可借鉴可微区域选择与训练稳定策略 | 视频分类；没有检测覆盖和漏检成本的不对称建模 |
| [MSViT, ICCV-W 2023](https://openaccess.thecvf.com/content/ICCV2023W/NIVT/html/Havtorn_MSViT_Dynamic_Mixed-Scale_Tokenization_for_Vision_Transformers_ICCVW_2023_paper.html) | 根据局部内容动态选择 token 尺度 | 支持区域级多尺度成本选择 | 面向分类/分割；没有对象级 risk–coverage 约束 |

### 4.3 显式计算预算与条件计算

| 工作 | 已有思想 | 与 BCRS 的直接关系 | 关键区别 |
|---|---|---|---|
| [Mixture-of-Depths, 2024](https://arxiv.org/abs/2404.02258) | 每层限制 top-k token，保证总 FLOPs 可预测，token 身份随输入变化 | 是“固定容量、动态身份”最直接的预算实现参照 | 语言模型；未处理空间覆盖、目标检测和不同 patch/context 成本 |
| [HeatViT, 2022](https://arxiv.org/abs/2211.08110) | latency-aware token pruning，并面向硬件优化目标剪枝率 | 支持用真实 latency 而非纯 FLOPs 训练 | 面向 ViT 分类和 FPGA；不是风险约束检测 selector |
| [Fully Dynamic Inference, 2020](https://arxiv.org/abs/2007.15151) | 按输入动态跳过层与通道 | 支持 instance-dependent compute allocation | 动态维度是层/通道，不是目标区域覆盖 |

### 4.4 覆盖约束选择

| 工作 | 已有思想 | 与 BCRS 的直接关系 | 关键区别 |
|---|---|---|---|
| [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) | 端到端联合优化预测与 reject option，建立 risk–coverage trade-off | 为“在覆盖率约束下最小化错误风险”提供理论基础 | coverage 是样本级是否输出预测；BCRS 是对象/区域级是否分配计算 |
| [Selective Classification via One-Sided Prediction, AISTATS 2021](https://proceedings.mlr.press/v130/gangrade21a.html) | 以 one-sided risk 控制选择性预测错误 | 支持不对称错误建模 | 分类场景；未涉及检测区域覆盖与计算预算 |

### 4.5 频谱增强与频率引导 token 选择

| 工作 | 已有思想 | 与 BCRS 的直接关系 | 关键区别 |
|---|---|---|---|
| [SET, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html) | 频域遮挡分析；HBS 抑制背景高频噪声；API 增强关键区域显著性 | 为 tiny-object 频谱证据提供直接实验证据 | SET 增强检测特征，不用频谱估计区域 miss risk 或分配计算预算 |
| [Frequency-Aware Token Reduction, NeurIPS 2025](https://openreview.net/forum?id=Dr06Wjh45k) | 保留高频 token、聚合低频 token，改善 token reduction | 证明频率特性可参与 token 压缩决策 | 目标是缓解 ViT rank collapse；不是微小目标检测，频率定义和 SET 的背景去噪机制也不同 |
| [Dyna-ViT, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Rubab_Dyna-ViT_Parameter-Free_Pre-Encoder_Token_Pruning_for_Efficient_Vision_Transformers_CVPRF_2026_paper.html) | 以能量、Sobel、entropy 等无监督 saliency proxy 排序 patch | 说明低级显著性可以作为早期 pruning 基线 | 分类任务；没有语义—频谱联合监督，也不保证目标覆盖 |

### 4.6 与已有工作的重合与非重合部分

**已经被充分验证的组件：**

- 粗层选择高分辨率区域：AutoFocus、QueryDet、ESOD；
- 可学习的空间/token 稀疏性：DynamicViT、Sparse DETR、Focal Sparse Conv；
- 固定总容量、动态选择身份：Mixture-of-Depths；
- coverage-constrained selection：SelectiveNet 等；
- tiny-object 频谱增强：SET。

**BCRS 不应声称的新颖性：**

- 首次动态选择 patch；
- 首次使用稀疏卷积；
- 首次用频域处理小目标；
- 首次在预算下做条件计算；

**BCRS 可验证的新颖性边界：**

1. 将语义与频谱证据联合用于预算约束的区域保留优先级，而不是单一 objectness；
2. 在 tiny-object selector 中显式优化对象级 coverage/recall 下界；
3. 将 SET 启发的频谱证据用于 selector 排序，而非全图特征增强；
4. 让同一检测器接受预算 $`B`$，联合控制 patch、FPN 层和上下文范围；
5. 同时评估 selector quality、tiny coverage 和真实端到端 latency。

这里的“backend-agnostic”仅指共享问题定义、风险估计接口、预算条件化和约束目标；它不表示同一张 mask 可以直接用于三种方法，也不默认三个 backend 共享同一组训练权重。若只完成 ESOD 实现，则只能声明 patch-level BCRS；至少完成 QueryDet 的第二种粒度验证后，才能支持跨 selector framework 的通用性表述。CEASC 的 activation-level 适配作为更强扩展验证。

截至本 proposal 整理时，尚未在上述代表工作中发现完全相同的二维无人机微小目标检测设定。但该结论必须在正式投稿前通过系统文献检索继续更新。

## 5. 方法设计

### 5.1 总体架构

首选基线为 ESOD 风格的浅层选择框架：

1. 输入经过低成本 stem，得到高分辨率浅层特征；
2. Dual-Evidence Risk Estimator 预测区域语义、频谱证据与跳过风险；
3. Budget Router 根据预算 $`B`$ 选择 patch、FPN 深度和上下文动作；
4. 被选择区域进入后续 backbone/neck/SparseHead；
5. 未被选择区域被跳过或进入极轻量 fallback；
6. 输出在原图坐标中合并。

QueryDet 作为第二架构验证：保留其 CSQ 路径，用 BCRS 替换 query score 和固定阈值策略，验证方法是否跨 selector framework 有效。

### 5.2 BCRS-Core 与 Backend Adapters

BCRS 采用“共享核心接口 + 后端动作适配器”的结构：

```text
Semantic Evidence ─┐                    ┌─ Patch Adapter → ESOD
                   ├→ BCRS-Core ───────┼─ Query Adapter → QueryDet
Spectral Evidence ─┘                    └─ Mask Adapter  → CEASC
```

共享核心输出：

```math
U_b
=
R_{\theta_b}
(F_b^{\mathrm{sem}},F_b^{\mathrm{spec}},B)
=
\{q_i,u_i\}_{i\in\Omega_b},
```

后端适配器执行：

```math
\mathcal S_b
=
A_b(U_b,B,C_b),
```

其中 $`b\in\{\mathrm{ESOD},\mathrm{QueryDet},\mathrm{CEASC}\}`$。共享的是 $`R`$ 的接口与训练原则；$`\theta_b`$、候选集合 $`\Omega_b`$、coverage neighborhood 和成本模型 $`C_b`$ 默认按 backend 独立训练和标定。

| Backend | 候选单位 | Adapter 输出 | 对象级 coverage | 主要成本单位 |
|---|---|---|---|---|
| ESOD | feature patch | patch indices/boxes | GT 被至少一个有效 patch 完整覆盖 | patch 后续 backbone/neck/head 成本 |
| QueryDet | FPN query point + context | sparse query coordinates | query context 到达 GT 在对应 FPN 层的位置 | query 数、context size、级联层数 |
| CEASC | 各 FPN 层 activation position | layer-wise sparse mask | GT 的至少一个正样本位置保持激活 | 各层 sparse-conv activation 与运行时成本 |

#### A. ESOD Patch Adapter：主实现

保留 ESOD 的 AdaSlicer 与 SparseHead，用 BCRS-Core 替换或扩展 ObjSeeker 的评分输出。给定等成本 patch，执行：

```math
\mathcal S_B^{\mathrm{patch}}
=
\operatorname{TopK}(u_i,k(B)).
```

对目标 $`j`$，$`\mathcal N_{\mathrm{patch}}(j)`$ 表示能够达到规定 box coverage、且不严重截断目标的 patch 集合：

```math
p_{j,\mathrm{ESOD}}^{\mathrm{cover}}
=
1-
\prod_{i\in\mathcal N_{\mathrm{patch}}(j)}(1-s_i).
```

这是 MVP，因为 patch 成本规则、容易形成 dense batch，也最容易把计算节省扩展到后续 backbone、neck 和 head。

#### B. QueryDet Query Adapter：跨框架验证

保留 QueryDet 的 Query Head/CSQ 框架，但用 BCRS utility 替换仅依赖 query score 阈值的排序。若第 $`l`$ 层目标投影为 $`\pi_l(GT_j)`$，query $`i`$ 的上下文半径为 $`r_i`$，则：

```math
\mathcal N_{\mathrm{query}}(j)
=
\left\{
i:\left\|i-\pi_l(GT_j)\right\|\le r_i
\right\},
```

```math
p_{j,\mathrm{QueryDet}}^{\mathrm{cover}}
=
1-
\prod_{i\in\mathcal N_{\mathrm{query}}(j)}(1-s_i).
```

第一版固定 CSQ 起始层和 context size，只比较相同 query 数量下的 priority ranking；之后才让预算同时控制 query 数和 1×1/3×3/5×5 context。

QueryDet 的低分辨率语义特征与更细粒度频谱证据可能不在同一 FPN 层，因此需要轻量跨层对齐。该 adapter 的新增对齐成本必须单独计入，不能假定与 ESOD 相同。

#### C. CEASC Mask Adapter：分层稀疏扩展

CEASC 的 CESC/CE-GN 用于补偿稀疏卷积的上下文与统计缺失，应予保留。BCRS 主要替换或扩展 AMM 的 mask utility 与 layer-wise budget allocation：

```math
\{\mathcal M_l^*\}_{l=1}^{L}
=
\arg\max_{\{\mathcal M_l\}}
\sum_lF_l(\mathcal M_l),
\qquad
\text{s.t.}\quad
\sum_l C_l(\mathcal M_l)\le B.
```

令 $`\mathcal P_l(j)`$ 为标签分配后 GT $`j`$ 在第 $`l`$ 个 FPN 层的正样本位置，则：

```math
p_{j,\mathrm{CEASC}}^{\mathrm{cover}}
=
1-
\prod_l
\prod_{i\in\mathcal P_l(j)}(1-s_{l,i}).
```

CEASC 的粒度最细、稀疏 runtime 最敏感，因此必须为每个 FPN 层建立独立成本表。该 adapter 不能直接沿用 ESOD 的 patch latency，也不应在 CE-GN 之前删除其所需的全局上下文统计路径。

#### D. 通用性边界

三个 backend 可以共享：

- 双证据优先级的输入/输出定义；
- budget embedding；
- object-level coverage/recall constraint；
- selector quality 与 object coverage 评价协议。

三个 backend 必须分别适配：

- 候选单位和 GT 映射；
- coverage neighborhood；
- 动作集合；
- 单位动作成本与 latency lookup；
- gather/scatter、sparse kernel 和 batching；
- 输入特征层与必要的对齐模块。

默认实验验证的是**算法原则和接口通用性**，不是跨 backend 权重共享。将 ESOD 训练好的 spectral encoder 或 fusion head 初始化到 QueryDet/CEASC，可作为额外 transfer experiment，但不是 BCRS 成立的必要条件。

### 5.3 双证据优先级选择器

#### A. Semantic/Objectness Branch

输入浅层语义特征，预测：

```math
q_i=P(y_i=1\mid f_i^{\mathrm{sem}}).
```

监督可沿用 ESOD 的 Gaussian/SAM hybrid pseudo mask，但必须额外报告这些 pseudo labels 对 tiny objects 的覆盖偏差。

#### B. Spectral/Saliency Branch

不直接对原图 FFT 后以高频阈值选择，而采用局部、多频带、可学习表示，例如：

- depthwise Laplacian/Sobel residual；
- DCT 或小波分解后的低/中/高频能量；
- 局部对比度和频谱残差；
- HBS 前后的响应差；
- 浅层特征的局部熵。

分支必须轻量，且其计算计入 selector overhead。

#### C. Evidence Fusion and Priority Head

语义与频谱特征经过轻量融合头，直接输出目标概率和路由优先级：

```math
(q_i,u_i) = g(f_i^{\mathrm{sem}},f_i^{\mathrm{spec}},e_B)
```

其中 $`q_i`$ 是辅助 objectness 输出，$`u_i`$ 是在给定预算下用于 top-k 的最终优先级，$`e_B`$ 是可选预算编码。

### 5.4 直接监督路由优先级

优先级监督直接由训练集 GT、目标尺寸和当前检测难度构造：

```math
r_i^{\mathrm{proxy}}
=
\max_{j\in\mathcal G(i)}
w_{\mathrm{size}}(j)
\cdot
w_{\mathrm{hard}}(j).
```

其中 $`\mathcal G(i)`$ 是区域覆盖的 GT 集合，$`w_{\mathrm{size}}`$ 提高微小目标权重，$`w_{\mathrm{hard}}`$ 可由当前检测损失或低 objectness 程度计算并停止梯度。训练同时使用 objectness、对象级 coverage 和 budget loss，直接学习“哪些区域在有限预算下应优先保留”。

### 5.5 预算条件化

训练时从预算分布采样：

```math
B\sim\mathcal U(B_{\min},B_{\max})
```

或从离散预算集合中采样：

```math
B\in\{B_{25},B_{50},B_{75},B_{100}\}.
```

预算编码通过小型 MLP 或 FiLM 注入 risk/router，使同一组主干权重适应多个工作点。必须与“每个预算单独训练一个模型”比较，判断统一预算模型是否存在明显性能折损。

### 5.6 路由动作空间

#### Action Space A：固定成本 patch top-k

所有候选 patch 尺寸相同，保留数 $`k(B)`$ 固定。推理采用 hard top-k：

```math
\mathcal S_B=\operatorname{TopK}(u,k(B)).
```

这是最稳定、最易获得真实加速的版本。

#### Action Space B：FPN 层与上下文联合动作

每个区域选择动作：

```math
a_i\in
\{\text{skip},
P4\text{-only},
P3{:}P4,
P2{:}P4\text{ with }3\!\times\!3,
P2{:}P4\text{ with }5\!\times\!5\}.
```

每个动作具有不同成本和精度效用，形成 multiple-choice knapsack。可使用离线 latency lookup + 动态规划/贪心近似；训练时采用 Gumbel-Softmax、straight-through estimator 或 differentiable top-k。

### 5.7 实际 latency 成本模型

不能简单令成本等于理论 FLOPs。端到端成本至少包括：

```math
C(\mathcal S)
=
C_{\mathrm{stem}}
+C_{\mathrm{selector}}
+C_{\mathrm{slice/gather}}
+C_{\mathrm{backbone}}
+C_{\mathrm{neck}}
+C_{\mathrm{head}}
+C_{\mathrm{merge/NMS}}.
```

实施时建立设备特定的 latency lookup table：

- patch 数量；
- patch 尺寸；
- FPN 层组合；
- batch packing 方式；
- 上下文范围；
- sparse/dense kernel 类型。

对非加性 latency，可训练轻量成本预测器 $`\widehat C_\psi(\mathcal S)`$。最终仍必须在目标硬件上端到端计时。

### 5.8 效率优先的部署约束

BCRS 的成立条件不是 selector 本身“有用”，而是它的新增成本小于后续计算节省：

```math
C_{\mathrm{BCRS}}
=
C_{\mathrm{base}}
-\Delta C_{\mathrm{saved}}
+C_{\mathrm{extra}},
```

其中：

```math
C_{\mathrm{extra}}
=
C_{\mathrm{spectral}}
+C_{\mathrm{fusion}}
+C_{\mathrm{topk}}
+C_{\mathrm{dispatch}}.
```

必须满足：

```math
C_{\mathrm{extra}}<\Delta C_{\mathrm{saved}}.
```

第一版实现遵循以下约束：

1. 复用 ESOD/QueryDet 已有浅层 selector 特征，不引入第二个 backbone；
2. 不在每个 patch 上单独执行 FFT；优先使用共享特征图上的 Sobel/Laplacian、depthwise 多频带滤波或轻量小波近似；
3. 频谱特征先压缩到 8–16 个通道，再进入融合优先级头；
4. 推理只进行一次 selector 前向，不引入额外模型；
5. MVP 使用固定 patch size 和 GPU hard top-k，不在线运行通用动态规划；
6. 对等成本 patch 通过固定 $`k(B)`$ 保证静态 tensor 容量，降低动态 shape 与 batching 损失；
7. 在方法获得固定预算精度增益前，不扩展复杂的 patch–FPN–context 联合动作空间。

训练成本和推理成本必须分开披露：

| 组件 | 训练时 | 推理时 | 处理原则 |
|---|---:|---:|---|
| Objectness/coverage/budget loss | 有 | 无 | 直接使用 GT |
| Spectral proxy branch | 有 | 有 | 必须轻量并单独计时 |
| Evidence fusion/priority head | 有 | 有 | 只执行一次 |
| top-k + gather/scatter | 有 | 有 | 必须计入端到端 latency |

### 5.9 训练策略

建议分阶段训练，避免 selector 与检测器共同坍缩：

1. 训练或加载 dense high-resolution detector；
2. 冻结主体，训练 semantic/spectral fusion selector；
3. 加入软路由和 coverage/budget loss；
4. 联合微调 detector + router；
5. 切换 hard top-k，进行短期 latency-aware fine-tuning；
6. 固定部署预算与 top-k 映射。

## 6. 核心科学假设

### H1：Objectness 不充分假设

在 low-objectness 区域中存在一组可观的真实微小目标；这些目标构成 objectness-only selector 在高稀疏率下的主要漏检来源。

### H2：双证据互补假设

在相同 selector 参数量与计算量下，语义 + 频谱/显著性证据比语义单分支具有更高的 low-objectness tiny recall。

### H3：双证据优先级假设

语义—频谱融合优先级比 objectness-only、spectral-only 或随机 top-k 产生更优的 APtiny–budget Pareto 曲线。

### H4：召回安全约束假设

加入对象级 coverage/recall 下界后，高稀疏率下的 selector catastrophic misses 显著减少，且计算增量可控。

### H5：单模型多预算假设

预算条件化模型可覆盖多个推理工作点，其精度接近分别训练的单预算 oracle，同时减少模型维护与切换成本。

### H6：真实加速假设

在计入 selector、切片、数据搬运和 NMS 后，BCRS 仍能在目标 GPU/边缘设备获得与理论计算下降方向一致的端到端 latency 改善。

### H7：轻量选择器假设

共享浅层特征上的 depthwise 多频带代理与单次融合优先级头能够改善微小目标选择，同时将新增推理 latency 控制在总延迟的 5% 以内，并小于其所节省后续计算的 10%。

### H8：跨选择粒度通用性假设

在分别适配候选单位、coverage 与成本模型后，双证据优先级在 patch-level ESOD 和 query-level QueryDet 上均优于各自的 objectness-only selector；进一步在 activation-level CEASC 上保持同方向收益。该假设不要求三者共享同一组权重。

## 7. 实验设计

### 7.1 数据集角色

| 数据集 | 角色 | 原因 |
|---|---|---|
| AI-TOD | 机制主数据集 | 平均目标尺寸小，适合检验 APvt/APt 与频谱证据 |
| VisDrone | 主航拍验证集 | 场景、密度、类别和背景多样，是选择性计算的核心 UAV benchmark |
| UAVDT | 跨数据/视频帧分布验证 | 类别较少、分辨率与采集条件不同，可验证迁移和预算稳定性 |
| TinyPerson（可选） | 极小行人外部验证 | 检验方法是否超出 UAV 类别体系 |

### 7.2 基础检测器

**主线：**

- YOLOv5/RTMDet + ESOD-style ObjSeeker/AdaSlicer/SparseHead；
- 保持输入尺寸、backbone、训练 schedule 一致。

**跨框架验证：**

- RetinaNet + QueryDet/CSQ；
- 用 BCRS utility 替换原 query threshold。

**分层稀疏扩展：**

- GFL V1 + CEASC；
- 保留 CESC/CE-GN，用 Mask Adapter 替换或扩展 AMM；
- 只在 ESOD 与 QueryDet 已验证风险排序后实施。

**可选实时验证：**

- FBRT-YOLO-S 作为 dense fallback/实时对照，而不是直接作为主结构。

### 7.3 必须包含的基线

1. Dense detector，原分辨率；
2. Dense detector，高分辨率；
3. QueryDet；
4. CEASC；
5. ESOD；
6. Objectness-only fixed threshold；
7. Objectness-only fixed top-k；
9. Spectral-only top-k；
10. Oracle GT coverage selector；
11. Semantic + spectral GT-coverage oracle。

两个 GT oracle 的作用不是作为可部署方法，而是回答：

- 选择策略理论上还有多少上限？
- 若 learned selector 不成功，是风险估计失败，还是路由空间本身没有价值？

### 7.4 逐步消融

#### Evidence Ablation

- semantic only；
- spectral only；
- semantic + raw high-frequency energy；
- semantic + learned multi-band spectral features；
- semantic + spectral + local saliency。

#### Fusion Ablation

- objectness-only priority；
- spectral-only priority；
- semantic + spectral concatenation；
- semantic + spectral gated fusion；
- 完整融合 + object-level coverage。

#### Constraint Ablation

- 无 coverage loss；
- pixel-wise mask loss；
- object-level soft coverage；
- soft coverage + recall hinge；
- 不同目标下界 $`r_0`$。

#### Budget Ablation

- 固定阈值；
- 固定 top-k；
- 单预算独立模型；
- 预算条件化统一模型；
- FLOPs budget；
- latency-aware budget。

#### Action-Space Ablation

- patch only；
- patch + context；
- patch + FPN depth；
- patch + FPN depth + context。

### 7.5 难例分桶

除全局平均外，必须按以下维度报告：

- very tiny/tiny/small；
- low/mid/high object density；
- low/mid/high objectness；
- low light；
- 高背景纹理；
- 遮挡；
- 图像边缘目标；
- 类别频率；
- selector priority 分位数。

### 7.6 选择质量诊断

除 AP 外，直接检查 selector 是否把有限预算分配给了正确区域：

- selector recall 与 object-level coverage；
- 不同预算下的 tiny miss rate；
- low-objectness tiny 的保留率；
- priority 与 GT coverage target 的排序相关性；
- AI-TOD 训练、VisDrone/UAVDT 测试时的排序迁移。

### 7.7 模块级开销与 break-even 实验

首先对新增模块做独立 microbenchmark：

| 模块 | 必测变量 |
|---|---|
| Semantic/Objectness head | 特征分辨率、通道数、batch size |
| Spectral proxy | FFT、固定滤波、learned depthwise、小波近似 |
| Fusion priority head | 融合方式、通道宽度、输出分辨率 |
| Router | threshold、hard top-k、lookup-based multi-action |
| Dispatch | patch 数、patch 尺寸、gather/scatter、packing 策略 |

每个模块分别报告 FLOPs、GPU latency、峰值显存和 kernel 数量。随后计算：

```math
T_{\mathrm{net\ saving}}
=
T_{\mathrm{baseline}}
-T_{\mathrm{BCRS}},
```

```math
\mathrm{OverheadRatio}
=
\frac{T_{\mathrm{spectral}}+T_{\mathrm{fusion}}+T_{\mathrm{routing}}}
{T_{\mathrm{BCRS}}}.
```

绘制“保留 patch 比例—新增 selector 成本—后续节省—净 latency”曲线，并找到：

- 最低需要删除多少 patch 才能达到 break-even；
- 不同输入分辨率下的 break-even point；
- 不同目标密度下的净收益；
- batch size 1 与批处理场景是否一致。

### 7.8 严格等预算比较

必须分别进行三组公平比较：

1. **Fixed patch count：** 所有 selector 保留完全相同数量、相同尺寸的 patch；
2. **Fixed theoretical compute：** 对齐总 FLOPs/MACs，且包含新增 spectral/fusion head；
3. **Fixed measured latency：** 在同一设备上调节各自预算，使端到端 median 或 P90 latency 对齐。

比较方法至少包括：

- objectness threshold；
- objectness top-k；
- spectral-only top-k；
- semantic + spectral top-k；
- semantic + spectral + coverage；
- GT coverage oracle；

这组实验用于排除一个关键混淆：BCRS 的精度提升是否只是因为 coverage loss 迫使其保留更多区域。主结果应优先报告 fixed measured latency，而不是只报告各方法默认阈值。

### 7.9 低开销实现选择消融

#### Spectral Implementation

- 无频谱分支；
- 原图或 patch FFT（accuracy oracle，不作为首选部署）；
- 固定 Sobel/Laplacian depthwise filters；
- learned multi-kernel depthwise filters；
- lightweight DCT/wavelet approximation。

#### Routing Implementation

- fixed threshold；
- GPU hard top-k；
- soft/differentiable top-k；
- latency lookup + 多动作近似求解。

除了 AP 和 recall，还必须比较新增 latency 和 P95 抖动。如果复杂频谱实现仅带来极小增益，则应选择更简单实现。

### 7.10 端到端效率报告

同时报告：

- 理论 FLOPs/MACs；
- 实际激活 patch/token 数；
- selector 各子模块 overhead；
- end-to-end latency：mean、median、P90、P95；
- throughput；
- 峰值显存；
- 能耗（若设备支持）；
- batch size 1 和实际部署 batch；
- warm-up、同步和计时方式。

### 7.11 分辨率与密度缩放实验

在至少三个输入分辨率、三个目标密度桶上测量：

- selector overhead 是否随像素数线性增长；
- 被选择 patch 数与目标数的关系；
- 相同预算下 coverage 是否随密度下降；
- 低密度图像的节省是否足以补偿高密度图像的 fallback；
- 每图 latency 分布是否出现长尾。

如果高密度场景中 BCRS 无法节省计算，应允许 dense fallback，并分别报告路由到 dense 路径的比例及总成本。

### 7.12 跨 Backend 通用性实验

通用性验证按顺序进行：

1. **ESOD/Patch：** 完整 BCRS，包括双证据融合、coverage、预算和 latency；
2. **QueryDet/Query：** 固定 CSQ 层与 context，先只替换排序分数；若成立，再加入预算控制 context；
3. **CEASC/Mask：** 保留 CE-GN/CESC，先做固定 layer-wise activation budget，再探索跨层预算分配。

每个 backend 内部必须分别比较：

- 原始 selector；
- objectness-only top-k；
- BCRS expected risk；
- BCRS + coverage；
- 完整 BCRS。

跨 backend 不直接比较未经统一硬件复测的原论文 FPS。主要通用性判据是：在各自 backend 内固定候选数量/FLOPs/latency 后，risk ranking 是否产生一致方向的 coverage 和 APtiny 改善。

可选的权重迁移实验：

- ESOD risk core 初始化 QueryDet；
- 冻结 spectral encoder，只微调 feature adapter/fusion head；
- 复用 spectral encoder 初始化 QueryDet，再单独训练其 fusion priority head。

若迁移失败但三个 backend 独立训练后均有效，则仍支持“方法原则通用”，但不支持“权重可直接迁移”。

### 7.13 公平性要求

- 所有主要方法使用相同输入尺寸、backbone、训练 schedule 和增强；
- FPS 必须在同一硬件、软件环境重新测量；
- 将 selector、切片、gather/scatter、merge 和 NMS 纳入计时；
- 不用不同论文原表中的 FPS 直接排序；
- 统一 sparse operator 实现或明确其差异；
- 同时给出固定 AP 比 latency 和固定 latency 比 AP。

## 8. 关键评价指标

### 8.1 检测性能

- AP、AP50、AP75；
- APvt、APt、APs；
- per-class AP；
- false negatives per image。

### 8.2 Selector 性能

- object-level selector recall；
- BPRbox/BPRcenter；
- low-objectness tiny recall；
- retained background ratio；
- foreground ratio；
- coverage per GFLOP / per millisecond。

不同 backend 还需分别报告：

- ESOD：box coverage、目标截断率、每图 patch 数；
- QueryDet：query-center coverage、不同 context 半径覆盖率、各 FPN 层 query 数；
- CEASC：GT positive activation coverage、各层 mask ratio、CE-GN 全局路径成本。

### 8.3 路由优先级质量

- priority–coverage curve；
- top-priority 区域中的真实微小目标集中度；
- priority 与 GT coverage target 的排序相关性；
- low-objectness tiny 的保留率。

### 8.4 效率

- FLOPs、参数量；
- latency 分布；
- selector overhead ratio；
- memory；
- Pareto hypervolume 或 dominated/non-dominated 工作点数量。

## 9. 成功标准与可证伪条件

### 9.1 最小成功标准

在同一检测器、相同端到端 latency 或 FLOPs 下，相对 objectness-only selector：

- tiny selector recall 提升至少 1 个百分点，或 miss rate 相对下降至少 15%；
- APt/APvt 有稳定提升；
- 总 AP 不下降超过统计波动；
- 至少在 AI-TOD 与 VisDrone 两个数据集成立；
- 在 fixed patch count、fixed FLOPs 或 fixed measured latency 中至少两种对齐方式下仍有增益；
- 新增 selector latency 原则上不超过端到端延迟的 5%，且不超过所节省后续 latency 的 10%；若目标设备无法达到该阈值，必须报告 break-even 分析而不能只展示 FLOPs。

具体阈值需在 baseline variance 测量后锁定，不应事后调整。

### 9.2 强成功标准

- ESOD 主实现在多个预算点形成优于其 objectness-only 版本的 Pareto frontier；
- QueryDet adapter 在相同 query 数或相同实测 latency 下获得同方向收益，从而支持跨 selector framework 通用性；
- CEASC adapter 若完成，应在相同 layer-wise activation budget 下改善 coverage/APtiny，作为更强而非 conference 必需证据；
- 在未见数据集上仍能维持召回约束，或通过预算/阈值重标定后恢复；
- latency-aware 路由在真实设备上带来可复现加速；
- 频谱分支的增益集中在 low-objectness/high-texture tiny objects，机制与假设一致。

### 9.3 否证条件

以下任一结果都应被视为重要否证，而不是通过更换指标掩盖：

1. 控制参数量和训练量后，频谱分支不优于普通卷积分支；
2. 双证据 priority 与 GT coverage target 无正相关；
3. coverage loss 只通过保留更多背景获得增益，固定预算下无改善；
4. learned selector 与 objectness top-k 差异很小，而 GT coverage oracle 也没有明显上限；
5. FLOPs 下降但端到端 latency 不降或更慢；
6. 预算条件化模型显著弱于每预算独立模型；
7. 方法只在单数据集单密度区间有效。

## 10. 风险与应对

### 风险 1：频谱响应退化为背景纹理检测器

**症状：** 树叶、建筑边缘、道路纹理占满预算。  
**应对：** 频谱证据必须与语义特征联合；加入 background-hard negatives，并用对象级 coverage 监督抑制纯纹理高响应。

### 风险 2：频谱分支与语义分支高度冗余

**症状：** 双分支增益完全来自参数量。  
**应对：** 参数匹配卷积基线；报告特征互信息/相关性；在 low-objectness tiny subset 单独比较。

### 风险 3：coverage loss 导致全部保留

**症状：** selector recall 提高，但稀疏率消失。  
**应对：** 使用预算硬容量 top-k；采用约束优化而不是无限提高 coverage 权重；报告固定预算结果。

### 风险 4：稀疏算子没有真实加速

**应对：** MVP 阶段使用规则 patch packing 与密集 batch kernel；先做设备 latency lookup；避免只依赖不规则逐点 sparse convolution。

### 风险 5：统一多预算模型训练不稳定

**应对：** 先训练离散预算点；使用 budget curriculum；对每个预算维护独立拉格朗日乘子或归一化 budget loss。

### 风险 6：召回约束无法跨域保持

**应对：** 单独做 calibration set；temperature scaling/conformal-style threshold calibration；保留 dense fallback 或最小安全预算。

### 风险 7：把接口通用性误写成零适配或权重通用性

**症状：** 只在 ESOD 上完成实验，却声称可直接植入所有 selector；或不同 backend 独立训练有效，却声称同一组权重可无缝迁移。  
**应对：** 按证据强度分级表述：仅 ESOD 支持 patch-level 方法；ESOD + QueryDet 支持风险原则跨选择粒度；加入 CEASC 才支持更完整的 patch/query/mask framework。权重迁移单独作为 transfer experiment，不能与接口通用性混为一谈。

## 11. 分阶段研究计划

### 11.1 研究问题—假设—实验追踪表

| 研究问题 | 对应假设 | 主要方法组件 | 决定性实验 | 通过标准 |
|---|---|---|---|---|
| RQ1：objectness 是否充分 | H1 | semantic baseline、GT coverage oracle | low-objectness tiny 分桶、Phase 0 oracle | 存在可观 priority-ranking 上限 |
| RQ2：频谱证据是否互补 | H2 | spectral/saliency branch | 参数匹配 Evidence Ablation | fixed budget 下改善目标子群 recall/AP |
| RQ3：双证据融合是否优于单证据 | H3 | fusion priority head | Fusion Ablation、coverage/priority correlation | 优于 objectness-only 与 spectral-only |
| RQ4：能否单模型多预算 | H5 | budget embedding | 单预算模型 vs 统一模型 | 多预算 Pareto 接近独立 oracle |
| RQ5：coverage 下界是否保护 tiny | H4 | object-level coverage loss | fixed patch count 的 Constraint Ablation | miss rate 降低且不靠多保留区域 |
| RQ6：FLOPs 能否变成 latency | H6 | latency lookup、hard routing | 端到端设备计时 | 净 latency 为正 |
| RQ7：新增成本是否值得 | H7 | 轻量 spectral/fusion/router | microbenchmark、break-even curve | overhead 满足预设上限 |
| RQ8：等预算下是否仍有增益 | H3、H6 | fixed-capacity routing | fixed patch/FLOPs/latency | 至少两种对齐下有一致收益 |
| RQ9：是否跨 backend 成立 | H8 | Patch/Query/Mask adapters | §7.12、Phase 4/6 | 至少 ESOD + QueryDet 同方向有效 |

### 11.2 阶段依赖与停止条件

```text
Phase 0：确认存在 selector-priority 上限
   ↓ Go
Phase 1：固定预算 semantic MVP
   ↓ Go
Phase 2：双证据融合
   ↓ Go
Phase 3：单模型多预算
   ├──────────────→ Phase 4：QueryDet 通用性验证
   │                         ↓ Go
   └──────────────→ Phase 5：结构化动作与外部验证
                              ↓ Optional
                         Phase 6：CEASC 分层稀疏扩展
```

依赖规则：

1. Phase 0 的 GT coverage oracle 若没有明显上限，停止完整 BCRS；
2. Phase 1 在 fixed patch count 下不优于 objectness，不能进入频谱和复杂动作扩展；
3. Phase 2 只有在双证据获得等预算增益、且轻量实现达到 break-even 后才能进入多预算；
4. QueryDet adapter 只依赖 Phase 1–3 的双证据优先级原则，不依赖 patch–FPN–context 复杂动作；
5. Phase 5 的结构化动作不能被用于替代核心等预算证据；
6. CEASC 是可选完整扩展，不是 conference 主结论或 QueryDet 验证的前置条件。

### Phase 0：基线复现与问题确认

- 复现 ESOD/objectness selector；
- 画出 objectness 分位数与 tiny miss 的关系；
- 测量 oracle GT coverage selector 上限；
- 建立端到端 latency 计时与 patch lookup。
- 分解 ESOD 的 selector、切片、检测和 merge latency；
- 测量 FFT、固定滤波、depthwise 频谱代理、fusion head 和 top-k 的独立成本；
- 建立不同分辨率、patch 数和 batch size 下的 break-even curve。

**Go/No-Go：** 如果 low-objectness 区域几乎没有真实 tiny objects，或 GT coverage oracle 不优于 objectness，则停止该方向。

### Phase 1：固定预算语义路由 MVP

- 固定 patch 和 FPN 路径；
- semantic selector + object-level coverage；
- fixed top-k 硬预算；
- AI-TOD + VisDrone；
- 与 objectness top-k 和随机 top-k 比较；
- 强制 fixed patch count，并补充 fixed measured latency 结果；
- 只使用共享浅层语义特征和单次 priority head。

### Phase 2：双证据融合

- 加入 learned spectral/saliency branch；
- 用 GT coverage、尺寸加权与检测难度直接监督 priority；
- 评估 selector recall、object coverage 和 priority correlation；
- 做低光、高纹理、low-objectness 分桶；
- 比较 FFT oracle、固定滤波与 learned depthwise spectral proxy；
- 比较 concatenation、gated fusion 与参数匹配卷积融合；
- 报告每种实现的精度—新增 latency Pareto。

### Phase 3：单模型多预算

- budget embedding；
- 离散多预算训练；
- 与独立单预算模型比较；
- 形成 AP–latency Pareto curve。

### Phase 4：QueryDet 跨框架验证

- 固定 CSQ 起始层与 context；
- 用 Query Adapter 替换 objectness ranking；
- 进行 fixed query count 和 fixed measured latency 比较；
- 验证后再开放动态 context；
- 不要求复用 ESOD 的已训练权重。

### Phase 5：结构化动作与外部验证

- patch + context；
- patch + FPN depth；
- latency-aware multiple-choice routing；
- UAVDT/TinyPerson 外部验证；
- 与 Phase 1 固定动作结果并列报告，防止复杂动作掩盖双证据排序贡献。

### Phase 6：CEASC 分层稀疏扩展

- 保留 CESC/CE-GN；
- 用 Mask Adapter 扩展 AMM；
- 先固定各层 activation budget，再学习跨 FPN 层分配；
- 建立 layer-specific latency lookup；
- 将其定位为 journal/完整通用性扩展，而不是前置依赖。

## 12. 预期贡献

1. **问题定义贡献：** 将高分辨率微小目标选择性计算建模为带对象覆盖约束的预算分配问题。
2. **方法贡献：** 提出语义—频谱双证据的轻量路由优先级选择器。
3. **优化贡献：** 在显式计算预算和 tiny-object recall 下界下，动态选择 patch、FPN 层与上下文。
4. **评估贡献：** 建立 selector quality、object coverage 与真实端到端 latency 联合评价协议。
5. **经验贡献：** 解释哪些微小目标会被 objectness-only selector 删除，以及频谱证据在哪些场景真正互补。
6. **框架贡献：** 以 Patch、Query 和 Mask adapters 验证同一双证据—预算原则是否能跨选择粒度成立，同时明确算法通用性与权重共享的边界。

## 13. 预期论文定位

最合适的叙事不是：

> 我们只是为 ESOD 增加了一个频谱分支。

而是：

> Existing selective detectors allocate compute according to foreground confidence. We instead fuse semantic and spectral evidence, and optimize object coverage under an explicit compute budget.

方法论文的主要证据链应为：

1. 证明 objectness-only selector 会遗漏一类真实微小目标；
2. 证明频谱证据能补充这些低语义响应区域；
3. 证明 coverage constraint 在固定预算下保护 tiny recall；
4. 证明双证据在目标子群上具有可解释互补性；
5. 证明理论预算转化为真实 latency Pareto 改善。
6. 先在 ESOD 形成完整闭环，再以 QueryDet 证明双证据原则跨选择粒度；CEASC 作为更强扩展，不反向成为主方法成立的必要条件。

## 14. 一句话总结

**BCRS 不把计算分给“最像目标”的区域，而是在显式预算下优先保留“跳过后最可能造成微小目标漏检”的区域，并以对象级召回约束保证选择过程的安全性。**

## 参考资料

- AutoFocus: Efficient Multi-Scale Inference, ICCV 2019.
- SelectiveNet: A Deep Neural Network with an Integrated Reject Option, ICML 2019.
- DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification, NeurIPS 2021.
- Selective Classification via One-Sided Prediction, AISTATS 2021.
- QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection, CVPR 2022.
- Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity, ICLR 2022.
- Focal Sparse Convolutional Networks for 3D Object Detection, CVPR 2022.
- A-ViT: Adaptive Tokens for Efficient Vision Transformer, CVPR 2022.
- AdaFocus V2: End-to-End Training of Spatial Dynamic Networks for Video Recognition, CVPR 2022.
- HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers, 2022.
- CEASC: Adaptive Sparse Convolutional Networks with Global Context Enhancement for Faster Object Detection on Drone Images, CVPR 2023.
- MSViT: Dynamic Mixed-Scale Tokenization for Vision Transformers, ICCV Workshops 2023.
- ESOD: Efficient Small Object Detection on High-Resolution Images, IEEE TIP 2024.
- Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models, 2024.
- SET: Spectral Enhancement for Tiny Object Detection, CVPR 2025.
- Frequency-Aware Token Reduction for Efficient Vision Transformer, NeurIPS 2025.
- A Novel Characterization of the Population Area Under the Risk Coverage Curve, ICML 2025.
- Dyna-ViT: Parameter-Free Pre-Encoder Token Pruning for Efficient Vision Transformers, CVPR Findings 2026.
