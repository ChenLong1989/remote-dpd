# ILC DPD 先行工作与 legacy 实现审计

## 1. 范围与结论

本文审计 `origin/main` 的提交 `1115c5b03bb459d38148a7f4b5d899ccba7c2bbb`，并将其中的
`remote_dpd/algorithms.py`、`remote_dpd/dsp.py` 与公开的 linear ILC、瞬时增益 ILC 和
BLA-inverse ILC 对照。文献内容以论文原文、作者机构仓储或出版者元数据为准；元数据核验日期为
2026-08-22。

结论如下。

1. legacy 更新式在一组收窄配置下具有 scalar linear ILC 的代数形式，但它作用于逐轮重新做
   RMS/全局相位归一化后的反馈，而不是未经变换的物理 PA 输出。只有当该校准本身为恒等变换时，
   才与公开的原始信号域 linear ILC 完全相同。
2. legacy 的可选相位补偿只近似消除逐点复增益的相位，不除以增益幅度，因而不是公开的完整
   instantaneous-gain inverse；它也没有估计或应用 Schoukens 等人的频率相关 BLA inverse。
3. “先拟合 PA 正向模型，再把损失反向传播穿过该模型”已有明确先例。本文不得声称首次使用
   PA 模型、PA backward、Jacobian、memory backpropagation 或 direct learning。
4. 本项目可检验的贡献边界应收窄到：面向逐波形 ILC 的单位 Jacobian 失效机制，组合每轮在线
   LS 正向模型、真实测量残差的 real-linear VJP、matrix-free 阻尼 Gauss--Newton/LM，以及
   预测下降和输入安全保护；是否具有足够新颖性仍须由更完整的系统检索和实验结果支持。

本文不报告本项目的实验结果，也不把仿真结论外推到真实 PA 硬件。

## 2. 公开 ILC 基线

令第 `k` 轮 PA 输入、输出和期望输出分别为 `u_k`、`y_k` 与 `d`，并统一定义跟踪误差

```text
e_k = d - y_k.
```

### 2.1 Scalar linear ILC

Chani-Cahuana 等人把波形本身作为学习变量，给出的 linear ILC 为

```text
u_(k+1) = u_k + gamma * e_k.
```

其推导在相应 Jacobian 条件下给出 `0 < gamma < 2 / J_max` 的收敛范围；这不是对任意强非线性、
饱和或带记忆 PA 的无条件保证。该工作还先用 ILC 得到最优逐波形输入，再以标准 LS 等方法从
参考波形映射到该输入，从而形成可部署的 DPD 模型
([Chani-Cahuana et al., 2016](https://doi.org/10.1109/TMTT.2016.2588483))。

### 2.2 Instantaneous-gain ILC

同一工作给出的逐点增益学习矩阵为

```text
G_k = diag(y_k / u_k)
u_(k+1) = u_k + inverse(G_k) * e_k.
```

它以测得的瞬时复增益近似局部、对角、无记忆的 plant inverse。零或低幅度样点需要工程门限，
小增益还会放大噪声，因此实际对照宜明确使用门限与 Tikhonov 阻尼，例如

```text
delta[n] = mu * conj(g_hat[n]) * e_k[n]
           / (abs(g_hat[n])**2 + lambda),
g_hat[n] = y_k[n] / u_k[n].
```

这个带阻尼版本是可复现的工程基线，不应表述成对原论文未正则化公式的逐项复刻。Wei 等人后来
也明确提出基于 instantaneous complex gain 的逐点源信号补偿，并在取得目标输入后再拟合 GMP
预失真器；因此“ILC 首次使用 PA 瞬时复增益”不属于本项目
([Wei et al.](https://doi.org/10.1109/LMWT.2025.3620316))。

### 2.3 BLA-inverse plant-inversion ILC

Schoukens 等人在频域使用标准 plant-inversion ILC：

```text
E_k(jw) = D(jw) - Y_k(jw)
U_(k+1)(jw) = Q(jw) * (U_k(jw) + L(jw) * E_k(jw)),
Q(jw) = 1,
L(jw) = 1 / G_BLA(jw).
```

`G_BLA` 是给定激励类下 PA 的 best linear approximation。它是频率相关的线性逆，而不是单位
learning matrix。原文同时明确指出：当 PA 非线性很强、线性近似与真实行为相差过大时，ILC
可能无法得到高质量解。这已经构成“线性逆在强非线性下存在失效边界”的公开先例
([Schoukens et al., 2017](https://doi.org/10.1109/TMTT.2017.2694822))。

## 3. 仓库 legacy 路径的精确等式

在被审计提交中，令

```text
g = 10 ** (dpdGainDb / 20)
d = g * r
epsilon_k = y_tilde_k - d = -e_k,
```

其中 `r` 是参考波形，`y_tilde_k` 是 resampling、时延对齐、逐轮 RMS 与全局相位校正，并在十次
packed capture 情形下平均之后的反馈。令 `P_k` 表示可选的幅度加权相位预条件器，`F_err` 与
`F_tx` 表示可选的 centered circular FIR，则代码的更新可写成

```text
u_(k+1) = F_tx(
    g * alpha * d
    + (1 - alpha) * u_k
    - g * mu * F_err(P_k * epsilon_k)
).
```

因为 `d = g*r`，第一项实际为 `alpha * g**2 * r`，并非通常的 `alpha*d`。关闭相位补偿与两个
FIR 后，上式化为

```text
u_(k+1) = (1 - alpha) * u_k + alpha * g**2 * r
          + g * mu * (d - y_tilde_k).
```

在 `alpha=0`、`dpdGainDb=0`、`phaseCompensate=false`、两个 FIR 均关闭、采样率一致且不触发
packed-capture 截断时，更新具有

```text
u_(k+1) = u_k + mu * (r - y_tilde_k)
```

的 scalar linear ILC 形式。这里必须保留 `y_tilde_k` 下标：`align_and_average()` 每轮都用 `r`
重新估计复校准系数，使对齐反馈具有 `r` 的 RMS 和全局相位。因此，它严格等于“对动态归一化后
plant 做 linear ILC”；只有 `y_tilde_k=y_k` 时才严格等于原始 PA 域公开公式。

legacy 的相位项在安全样点近似为

```text
P_k[n] = w_k[n] * conj(sign(y_tilde_k[n] / u_k[n]))
         + (1 - w_k[n]).
```

它只旋转误差且带平滑权重，不包含 `1/abs(y/u)`，所以不能标注为完整 instantaneous-gain ILC。
同理，可选 FIR 没有在 legacy 配置中被约束为 `1/G_BLA(jw)`，不能仅凭存在 FIR 就把该路径称为
BLA-inverse ILC。

### 3.1 对照表

| 项目 | 公开方法 | legacy 实现 | 审计判断 |
| --- | --- | --- | --- |
| 误差符号 | `e=d-y` | 先算 `epsilon=y-d`，更新时取负 | 代数等价 |
| Linear learning matrix | `gamma*I` | 收窄配置下为 `g*mu*I` | `g=1` 时匹配 |
| Instantaneous gain | 除以逐点复增益，或其正则化逆 | 仅可选相位旋转与软门限 | 不等价 |
| BLA inverse | 频率相关 `1/G_BLA` | 未估计 BLA；FIR 来源不受约束 | 不等价 |
| 校准 | 植物域固定后分析 | 每轮重新估计 RMS/全局相位 | 优化对象不同 |
| `alpha` 混合 | 上述论文基线无此项 | 含 `(1-alpha)u + alpha*g**2*r` | legacy 扩展 |
| 捕获协议 | 单个重复任务波形 | 支持十路 packed capture、平均与复制 | 传输扩展 |
| 初始化 | Chani 使用 `d/g_avg` | 无历史输出时使用 `r` | 仅特定增益约定相同 |

因此，论文和图例必须把既有路径写为 `legacy_ilc`，另以清晰名称实现 `linear_ilc`、
`instantaneous_gain_ilc` 和需要时的 `bla_inverse_ilc`。不能把 legacy 默认配置直接写成某篇论文
的原样复现。

## 4. PA 模型与反向传播先例

### 4.1 Memory polynomial / GMP

Morgan 等人从 Volterra、Wiener、Hammerstein 等结构出发，系统化了 memory polynomial，并提出
包含领先/滞后包络交叉项的 GMP。它是 PA/DPD 行为建模的经典来源
([Morgan et al., 2006](https://doi.org/10.1109/TSP.2006.879264))。本项目选用较简单的
complex-coefficient memory polynomial 作为在线正向模型，属于结构取舍，不是模型结构创新；
若后续使用 GMP，也必须引用该来源。

### 4.2 通过 PA 正向模型反向传播

Tarver 等人先训练 neural-network PA 模型，再冻结该模型，把 neural-network DPD 与其级联，
由级联输出损失反向传播穿过 PA 模型来更新 DPD 参数
([Tarver et al., 2019](https://doi.org/10.1109/IEEECONF44664.2019.9048910))。这已经覆盖
“PA forward model + backward + direct DPD training”的核心概念。

Loebl 等人进一步明确研究了通过带记忆 PA 模型进行 backpropagation 的 neural-network direct
learning，并在宽带 CMOS PA 测量上评价该方法
([Loebl et al., 2023](https://doi.org/10.1109/IMS37964.2023.10187912))。因此，本项目也不能把
“backward 覆盖 PA memory”本身作为首创点。

上述两项工作主要更新参数化 DPD 的权重；本项目计划直接更新有限长度波形，并在每个 ILC 外层轮次
用当轮 `(u_k,y_k)` 数据重拟合 LS 正向模型。这是问题变量、模型刷新节奏和数值求解器上的差异，
但“差异”不自动等于“全球首次”。

## 5. 收窄后的研究定位

在这组先行工作之后，本项目适合提出以下可证伪、组合式研究问题。

1. 当 scalar linear ILC 把 PA backward 近似为单位映射时，高功率 AMAM 小斜率/饱和与低功率
   强 AMPM 分别怎样改变真实梯度方向、收敛速度与安全性？
2. 每轮仅依赖当轮 PA 输入和实测输出的 ridge-LS memory-polynomial 正向模型，能否提供足够准确的
   real-linear JVP/VJP，从而改善逐波形 ILC，而不把梯度穿过 LS 求解或校准估计？
3. 相比 raw VJP，使用同一模型的 matrix-free 阻尼 Gauss--Newton/LM、真实测量残差、
   anchored model delta、trust region、回溯与硬输入投影，能否在可达强失真场景中稳定改善预注册
   指标，并在不可达饱和区正确报告 `saturation_limited`？
4. 这些结论在噪声、捕获平均、模型阶数/记忆失配、动态 out-of-family PA 与校准方式变化下的边界
   是什么？

适合使用的谨慎表述是：

> 本文研究在线 LS PA 正向模型的 real-linear VJP/JVP 如何用于逐波形 ILC，并构造带预测与输入
> 安全保护的 matrix-free LM 更新，以分析单位 Jacobian 近似在两类强失真机制中的失效边界。

以下表述不应使用：

- “首次把 ILC 用于 PA/DPD”；
- “首次在 ILC 中使用瞬时 PA 增益、plant inverse 或 PA 信息”；
- “首次通过 PA 模型反向传播训练 DPD”；
- “首次处理带记忆 PA 的 backward”；
- “首次提出 memory polynomial/GMP”；
- 在没有系统检索证据时使用笼统的 “the first model-based ILC” 或 “the first Jacobian ILC”；
- 在只有仿真时声称真实 GaN/Doherty PA、PAE、产业部署或 3GPP 合规收益。

## 6. 文献元数据与核验记录

| BibTeX key | 出版信息 | 权威/原始核验入口 |
| --- | --- | --- |
| `chani2016ilc` | IEEE T-MTT, 64(9), 2778--2789, 2016 | [DOI](https://doi.org/10.1109/TMTT.2016.2588483)、[Chalmers 机构记录](https://research.chalmers.se/en/publication/245055) |
| `schoukens2017preinverse` | IEEE T-MTT, 65(11), 4266--4273, 2017 | [DOI](https://doi.org/10.1109/TMTT.2017.2694822)、[作者原稿](https://arxiv.org/abs/1606.08663) |
| `morgan2006gmp` | IEEE T-SP, 54(10), 3852--3860, 2006 | [DOI](https://doi.org/10.1109/TSP.2006.879264) |
| `tarver2019backprop` | 53rd Asilomar, 358--362, 2019 | [DOI](https://doi.org/10.1109/IEEECONF44664.2019.9048910)、[作者稿（NSF PAR）](https://par.nsf.gov/biblio/10202730) |
| `loebl2023memorybackprop` | IEEE IMS, 791--794, 2023 | [DOI](https://doi.org/10.1109/IMS37964.2023.10187912)、[Technion 机构记录](https://cris.technion.ac.il/en/publications/direct-learning-neural-network-digital-predistortion-using-backpr/) |
| `wei2026icg` | IEEE MWTL, 36(1), 7--10, 2026 | [DOI](https://doi.org/10.1109/LMWT.2025.3620316)、[IEEE Xplore](https://ieeexplore.ieee.org/document/11216079/) |

Wei 等人的 DOI 字符串包含 `2025`，且工作于 2025 年进入 Early Access；IEEE/Crossref 的正式卷期
元数据是 2026 年第 36 卷第 1 期、7--10 页，因此 `references.bib` 按正式卷期记为 2026。正文首次
提及时可写作 “Wei et al. (Early Access 2025; issue publication 2026)” 以避免年份歧义。

这六篇工作足以否定若干宽泛的“首次”表述，但不构成全领域穷尽检索。投稿前仍应围绕
`model-based ILC`、`Newton ILC`、`Gauss--Newton ILC`、`adjoint ILC`、`Jacobian-free ILC` 与
`PA waveform optimization` 继续做数据库级系统检索。
