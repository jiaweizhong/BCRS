# 论文投稿目标规划 (Target Conferences Roadmap)

> **研究方向**：高分辨率图像微小目标检测 (Small Object Detection)、空间条件/选择性计算 (Selective Computation)、无人机与遥感边缘感知 (UAV & Remote Sensing Perception)  
> **论文成果**：HESOD (High-Resolution Efficient Small Object Detection)

---

## 1. 核心目标会议一览表

| 会议简称 | 会议全称 / 主办 | CCF分类 | CORE评级 | 举办地点 | 预计截稿时间 | 核心契合点与方向特点 | 推荐投稿侧重点 |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| **ICASSP 2027** | IEEE International Conference on Acoustics, Speech and Signal Processing (IEEE SPS) | **CCF-B** | **CORE-A** | **🇨🇦 加拿大 · 多伦多**<br>(Toronto) | **9月上旬<br>(2026年9月)** | **信号处理领域旗舰顶会**。IEEE SPS 规模最大、最具影响力的旗舰盛会。拥有专门的图像/多维信号处理（IVMSP）及自主感知计算轨道，对空频域联合特征、自适应滤波与小目标信号恢复高度认可。 | 突出**空域-频域双证据信号建模**（Spatial-Spectral Prior）与通道池化频域滤波算子的理论完备性，强调自适应选择性计算在保持信号覆盖度上的数学自洽性。 |
| **ICDIP 2027** | International Conference on Digital Image Processing (SPIE 出版) | **EI 核心** | **图像处理权威** | **🇨🇳 中国 · 北京**<br>(Beijing) | **11月10日左右<br>(2026年11月)** | **数字图像处理经典老牌会议**（第19届）。SPIE 官方出版并由 EI 核心检索。广泛接收高分辨率图像处理、空/频域特征提取、微小目标识别与高效计算架构。 | 突出**高分辨率图像空频域双特征提取**、软覆盖率损失优化与轻量化倒残差检测头（ISPP）的图像处理算法完备性与高效性。 |
| **ICME 2027** | IEEE International Conference on Multimedia and Expo | **CCF-B** | **CORE-A** | **🇨🇳 中国 · 厦门**<br>(Xiamen) | **3月中旬左右<br>(2027年)** | **多媒体/视觉方向极力推荐**。包含大量无人机目标检测、轻量化网络、视频小目标跟踪等工作，对小目标检测算法创新极其友好。 | 突出**双证据融合机制**（语义特征与浅层高频频谱特征结合），强调算法在航拍多媒体与复杂视觉流中的高效感知创新。 |
| **IROS 2027** | IEEE/RSJ International Conference on Intelligent Robots and Systems | **CCF-C** | **CORE-A** | **🇮🇹 意大利 · 佛罗伦萨**<br>(Florence) | **3月1日<br>(2027年)** | **无人机机器人应用方向极力推荐**。偏向无人机（UAV）平台搭载视觉算法的落地应用。非常看重算法在无人机、自主导航上的工程/落地/轻量化与实时性。 | 突出**端侧边缘部署与能耗节省**（Edge Deployment），强调空间选择性计算减少 50%+ 算力对无人机板载算力及续航的实际赋能。 |
| **IGARSS 2027** | IEEE International Geoscience and Remote Sensing Symposium (IEEE GRSS) | **遥感顶会** | **遥感旗舰** | **🇮🇸 冰岛 · 雷克雅未克**<br>(Reykjavík) | **1月上旬左右<br>(2027年)** | **遥感检测方向首选**。IEEE GRSS 旗舰盛会，在遥感领域认可度极高（常享有一区/顶级影响力）。无人机图像天然属于高空低空遥感，小目标检测（车辆、海面搜救行人、航拍船只、农作物）是其经典核心主题。 | 突出**超高分辨率全画幅遥感感知**（如 $2048\times 2048$ 不切图），强调复杂海陆下垫面背景下的微小目标高召回与覆盖安全性。 |
| **ICIP 2027** | IEEE International Conference on Image Processing (IEEE SPS) | **CCF-C** | **CORE-A** | **🇸🇬 新加坡**<br>(Singapore) | **1月下旬左右<br>(2027年)** | **图像处理领域经典盛会**。图像处理与信号处理标准顶级会议。 | 突出**频域/梯度滤波理论建模**（Channel-Pooled Spectral Branch）以及**软覆盖率损失函数 ($\mathcal{L}_{\mathrm{cover}}$)** 的数学推导与理论完整性。 |

---

## 2. 各会议定位与论文叙事调整策略 (Submission Strategy)

```mermaid
graph LR
    Core["HESOD Core<br/>(Dual-Evidence + Coverage Loss + ISPPHead)"]
    
    Core -->|强调信号处理理论与空频域先验| ICASSP["ICASSP 2027 (CCF-B / CORE-A)<br/>📍 加拿大·多伦多 (Toronto)<br/>Deadline: 2026年9月上旬"]
    Core -->|强调数字图像处理算法与特征优化| ICDIP["ICDIP 2027 (SPIE / EI核心)<br/>📍 中国·北京 (Beijing)<br/>Deadline: 2026年11月10日"]
    Core -->|强调频谱融合与算法创新| ICME["ICME 2027 (CCF-B / CORE-A)<br/>📍 中国·厦门 (Xiamen)<br/>Deadline: 2027年3月中旬"]
    Core -->|强调板载轻量化与实时部署| IROS["IROS 2027 (CCF-C / CORE-A)<br/>📍 意大利·佛罗伦萨 (Florence)<br/>Deadline: 2027年3月1日"]
    Core -->|强调高空遥感与复杂下垫面| IGARSS["IGARSS 2027 (遥感顶会)<br/>📍 冰岛·雷克雅未克 (Reykjavík)<br/>Deadline: 2027年1月上旬"]
    Core -->|强调图像处理理论与损失建模| ICIP["ICIP 2027 (CCF-C / CORE-A)<br/>📍 新加坡 (Singapore)<br/>Deadline: 2027年1月下旬"]
```

### 2.1 ICASSP 2027 (信号处理学会旗舰顶会 · 📍 加拿大 多伦多)
- **举办城市与场馆**：多伦多，加拿大（Metro Toronto Convention Centre, MTCC），2027年5月16–21日。
- **审稿偏好**：注重信号处理理论（空频域变换、自适应滤波、信号重构与稀疏感知）、数学严谨性与计算复杂度分析。
- **论文调整建议**：
  - 突出**“空域语义与频域结构双通道证据感知”**的信号学物理意义，严格形式化通道池化与高频梯度算子组（Laplacian/Sobel Bank）；
  - 深入剖析目标尺度崩塌过程中的信噪比（SNR）衰减，证明频域先验在极限尺度（$<16^2$ px）下的信号恢复能力。

### 2.2 ICDIP 2027 (数字图像处理经典国际会议 · 📍 中国 北京)
- **举办城市**：北京，中国，2027年4月16–18日（SPIE 官方出版，EI Compendex 核心检索）。
- **审稿偏好**：注重数字图像处理算法完整性、图像增强/滤波设计、空间与频域特征表达及计算效率。
- **论文调整建议**：
  - 强化**高分辨率图像空频域双重证据提取**的算法完备性，重点展示通道池化与高频边缘算子的特征响应图；
  - 突出 ISPP 倒残差解耦头带来的算力压缩（FLOPs 减少 20.7%，参数减少 27.6%）与实时高通量推理表现。

### 2.3 ICME 2027 (多媒体大顶会 · 📍 中国 厦门)
- **举办城市**：厦门，中国，2027年7月13–17日。
- **审稿偏好**：注重视觉算法创新的有效性与完备性，关注多模态/多特征融合及多媒体流应用。
- **论文调整建议**：
  - 引言与方法部分强化**“语义特征”与“频域高频结构特征”的互补性机制**；
  - 重点展示 SeaPerson 与 UAVDT 上的尺寸分桶增益（尤其是 Tiny/Very Tiny 目标的显著召回跃升）。

### 2.4 IROS 2027 (机器人旗舰会 · 📍 意大利 佛罗伦萨)
- **举办城市与场馆**：佛罗伦萨，意大利（Fortezza da Basso），2027年9月26日–10月1日。
- **审稿偏好**：极度看重无人机（UAV）、自主移动机器人（AMR）的机载闭环落地价值与实时延迟（Latency）。
- **论文调整建议**：
  - 强化“边缘计算平台（如 Jetson Orin / Xavier）实际推理延迟、FPS 与显存占用”实验；
  - 强调非对称误删代价在无人机自主搜救（Search-and-Rescue）中的致命性，突出 Soft-Coverage 的“召回安全保障（Recall Safety）”。

### 2.5 IGARSS 2027 (遥感旗舰会 · 📍 冰岛 雷克雅未克)
- **举办城市与场馆**：雷克雅未克，冰岛（Harpa Conference and Concert Centre），2027年7月11–16日。
- **审稿偏好**：关注地球科学、低空航拍遥感、海洋监测与特定地物（如微小行人、车辆、浮标）解译。
- **论文调整建议**：
  - 突出无需切图切片（Without Pre-tiling）直接在 $2048\times 2048$ 甚至更高分辨率遥感全景图上的高通量检测能力；
  - 增加海面杂波（Sea clutter）、光照反光与开阔荒野场景的定性可视化对比。

### 2.6 ICIP 2027 (图像处理经典盛会 · 📍 新加坡)
- **举办城市**：新加坡，2027年秋季。
- **审稿偏好**：偏好清晰优雅的图像处理算子设计、损失函数数学形式化与信号/特征分析。
- **论文调整建议**：
  - 保持目前 IEEE ICIP 4+1 页的严密数学表述（Laplacian/Sobel 算子形式化、Wasserstein 距离与 Soft-Coverage 概率推导）。

---

## 3. 投稿准备时间线与行动路线 (Timeline)

- **2026年 8月 - 9月上旬**：
  - **9月上旬**：首发冲刺 **ICASSP 2027**（📍 加拿大·多伦多，IEEE SPS 旗舰顶会，4+1 页格式，侧重空频域信号先验与稀疏选择计算）；
- **2026年 10月 - 11月上旬**：
  - **11月10日左右**：投稿 **ICDIP 2027**（📍 中国·北京，SPIE / EI 核心检索，侧重数字图像处理算法创新与轻量化架构）；
- **2026年 11月 - 12月**：
  - 完善补充多 GPU / 边缘端（Jetson）FPS 与算力能耗测试；
  - 准备 IGARSS 2027（📍 冰岛·雷克雅未克）与 ICIP 2027（📍 新加坡）的后续批次投稿稿件。
- **2027年 1月**：
  - **1月上旬**：完成 **IGARSS 2027** 投稿提交；
  - **1月下旬**：完成 **ICIP 2027** 投稿提交。
- **2027年 2月 - 3月**：
  - 根据实验补充与针对性包装（机器人应用 vs. 多媒体视觉算法）；
  - **3月1日**：完成 **IROS 2027**（📍 意大利·佛罗伦萨）投稿提交；
  - **3月中旬**：完成 **ICME 2027**（📍 中国·厦门）投稿提交。
