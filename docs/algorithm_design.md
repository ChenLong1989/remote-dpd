# PA 模型反向传播 ILC 算法设计

## 1. 文档目的与范围

本文档定义 `remote_dpd` 当前实现中的五类 ILC 更新、在线 PA 正向模型、复数实线性
Jacobian、校准、安全投影和回退语义。它是代码行为的数学说明，不包含实验结果，也不把仿真结论
外推到真实功放、产业部署、PAE 或 3GPP 合规性。

本文采用以下基本记号：

- `r`：输入参考波形；
- `G = 10^(gain_db/20)`：配置的线性幅度增益；
- `d = G r`：非 legacy 策略的期望 PA 输出；
- `u_k`：第 `k` 轮送入 PA 的 DPD 波形；
- `y_k`：反馈经过重采样、时延对齐和指定校准后的 PA 输出；
- `e_k = y_k - d`：输出端残差；
- `f_hat_k`：使用本轮 `(u_k, y_k)` 在线拟合并在本轮更新中冻结的 PA 正向模型。

除 legacy 的 PyTorch 兼容路径外，新策略及 PA 模型默认使用 NumPy `complex128`。实验层也可将
状态、目标、测量及 `PAForwardModelConfig.numeric_dtype` 一并固定为 `complex64`，此时模型系数、
预测、JVP/VJP、CG 解和学习输出保持 `complex64`；标量诊断仍转为 Python `float`。复向量只是
`2N` 维实向量的紧凑编码，内积统一定义为

```text
<a, b>_R = real(vdot(a, b)).
```

因此，本文中的 `J^T` 指上述实内积下的转置，而不是把非解析 PA 模型误当作复解析映射后得到的
普通 Hermitian 导数。

## 2. 算法分层与选择

实现注册了以下 engine 名称：

| engine | 内部模式 | 作用 |
| --- | --- | --- |
| `ilc` | 读取 `backward_mode` | 文件服务的兼容入口 |
| `legacy_ilc` | `legacy` | 保留原工程更新、动态归一化、FIR 和 packed capture 语义 |
| `linear_ilc` | `linear` | 单位 Jacobian 的公开 scalar linear ILC 基线 |
| `instantaneous_gain_ilc` | `instantaneous_gain` | 带门限和 Tikhonov 阻尼的逐点复增益基线 |
| `model_vjp_ilc` | `model_vjp` | 在线模型的 raw VJP 更新，用于隔离反向传播作用 |
| `model_lm_ilc` | `model_lm` | 在线模型的受保护、matrix-free LM 更新 |

显式 engine 名称覆盖 `backward_mode`；`ilc` 才按配置中的模式分派。MAT 配置未指定模式时默认为
`legacy`，因此原文件服务不会静默改用新算法。

### 2.1 公开 scalar linear ILC

`linear_ilc` 使用

```text
u_(k+1) = u_k - mu e_k
          = u_k + mu (d - y_k),     mu > 0.
```

这等价于用 `J = I` 近似 PA 的局部 Jacobian。候选波形随后经过输入安全投影。代码中的线性预测
`y_k + (u_(k+1)-u_k)` 只用于诊断，不参与该基线的接受或拒绝。

这一基线对应 RF PA ILC 文献中的标量学习更新；项目采用明确的 `linear_ilc` 名称，将其与带额外
预处理的 legacy 工程路径分开。相关公开描述见 Chani-Cahuana et al.（2016）和
Schoukens et al.（2017）。

### 2.2 瞬时复增益 ILC

`instantaneous_gain_ilc` 在满足 `|u_k[n]| > 1e-8` 且比值有限的样点上估计

```text
g_k[n] = y_k[n] / u_k[n],
delta[n] = -mu * conj(g_k[n]) * e_k[n]
           / (|g_k[n]|^2 + lambda),
u_(k+1) = Project(u_k + delta).
```

其中 `lambda` 复用 `lm_damping`，默认 `1e-2`。无效样点的更新为零。该式是逐点复数
Tikhonov 逆；它比单位 Jacobian 包含更多 PA 信息，但 `y/u` 是割线增益而非严格微分增益，且不描述
记忆耦合。它是项目定义的、可复现的正则化 instantaneous-gain 对照，不声称逐行复现某篇论文的
全部预处理、增益估计或保护逻辑。

若残差非零但修正量相对残差小于局部响应门限，返回 `saturation_limited` 并保持当前输入。

### 2.3 Legacy 工程更新

legacy 路径先把反馈逐轮对齐并按参考 `r` 重新估计全局 RMS 与相位校正；注意这里的校准参考是
`r`，而误差目标是 `d = G r`。记校准后的反馈为 `y_bar_k`，初始残差为

```text
e_bar_k = y_bar_k - d.
```

可选相位预条件器按 `u_k`、`y_bar_k` 的幅度生成平滑权重，并用
`conj(sign(y_bar_k/u_k))` 旋转高置信度样点的误差；随后可施加 centered circular
`error_fir`。最终更新为

```text
u_(k+1) = TX_FIR(
    G * alpha * d
    + (1-alpha) * u_k
    - G * mu * ERROR_FIR(e_bar_k)
).
```

由于 `d = G r`，参考分支实际为 `alpha G^2 r`。只有在

```text
G = 1, alpha = 0, phase_compensate = false,
error_fir = None, tx_fir = None
```

时，该路径才严格退化为 `u_(k+1)=u_k+mu(r-y_bar_k)`。逐轮 RMS/相位归一化还会改变被优化的
物理域，因此 legacy 的一般配置不能作为公开 scalar linear ILC 的同义词，也不能直接套用固定 PA
Jacobian 的收敛分析。

legacy 兼容路径还保留十路 packed capture：当反馈长度为 `327680` 且参考至少有 `32768`
个样点时，只在前 `32768` 个参考/当前输入样点上学习，把反馈按 MATLAB Fortran 顺序拆成十次
捕获，分别对齐后平均，并把得到的 `32768` 点输出连续复制十次。

## 3. 在线 PA 正向模型

### 3.1 Memory polynomial

本轮模型采用复系数、循环记忆的 memory polynomial：

```text
z_m[n] = roll(u_k, m)[n] = u_k[(n-m) mod N]

f_hat_k(u_k)[n]
  = sum_(p in P) sum_(m=0..M-1)
      c_(p,m) z_m[n] (|z_m[n]| / s)^(p-1).
```

engine 使用 `P = {1,3,...,P_max}`，默认 `P_max=9`、`M=3`。`P_max` 必须是 `[1,21]`
内的奇数，`M` 必须在 `[1,16]`。循环边界与周期波形 ILC 相匹配；它不是零填充卷积模型。

包络尺度 `s` 是训练样本 `|u_k|` 的 `99.9%` 分位数，并至少为 `1e-12`。这一尺度同时用于
模型预测和解析 Jacobian。

### 3.2 确定性 train/validation 划分

样本按连续 256 点分块，块编号从 1 开始。每五个块的第 5 块全部作为 validation，其余块全部
作为 train：

```text
validation(n) iff (floor(n / 256) + 1) mod 5 == 0.
```

validation 样本不进入 LS。该划分无随机状态；不足一个 validation 块时拟合失败，而不是改用训练
误差冒充泛化误差。

### 3.3 带 ridge 的增广复数 LS

令 `X` 为完整波形上的复设计矩阵。先在训练集上计算每列 RMS `q_j`，并形成
`X_bar[:,j] = X[:,j]/q_j`。若任一 `q_j <= 1e-14`，模型以 `zero_design_column`
失败。随后对未加 ridge 的训练矩阵执行 SVD，检查数值秩与条件数；默认最大允许条件数为 `1e12`。

系数通过增广最小二乘求解，而不显式形成 normal equation：

```text
theta = argmin ||X_bar_train theta - y_train||_2^2
                 + ridge ||theta||_2^2

        = lstsq([X_bar_train; sqrt(ridge) I],
                [y_train; 0]),

c_j = theta_j / q_j.
```

默认 `ridge=1e-6`；配置允许 `ridge=0`，用于无正则消融。ridge 作用于列归一化后的系数
`theta`，而非未经缩放的 `c`。

模型分别记录 train/validation NMSE：

```text
NMSE(y, y_hat) = 10 log10(||y_hat-y||_2^2 / ||y||_2^2).
```

文件服务默认要求 validation NMSE `<= -20 dB`。配置键名
`PAModelMinValidationNmseDb` 沿用接口命名，但代码语义是“可接受的最大 NMSE 数值”：结果高于该阈值
即 `validation_nmse_exceeded`。低层 `PAForwardModelConfig` 独立使用时默认阈值为 `0 dB`；engine
会明确传入服务配置的阈值。

长度不一致、样本不足、validation 为空、输入/输出非有限、尺度或设计矩阵无效、零设计列、SVD/LS
失败、秩不足、条件数超限、系数/预测非有限以及 validation 不合格，都会返回
`model=None` 和可观测的 `fallback_reason`。LS 系数拟合完成后在本轮被视为常量；反向传播不穿过
LS 求解、时延估计或校准估计。

## 4. Complex real-linear Jacobian

PA 基函数依赖 `|z|`，因此不是复解析函数。对单个基函数

```text
phi_p(z) = c z (|z|/s)^(p-1)
```

在复扰动 `h` 上的实线性微分为

```text
D phi_p(z)[h] = a_p(z) h + b_p(z) conj(h),

a_p(z) = c * (p+1)/2 * (|z|/s)^(p-1),

b_p(z) = c * (p-1)/2 * (z/s)^2 * (|z|/s)^(p-3).
```

当 `p=1` 时，`b_p` 显式为零，不计算带负指数的表达式。把同一 delay 的各阶贡献相加得到
`a_m[n]`、`b_m[n]`，则解析 JVP 为

```text
J(u) h = sum_m [
    a_m * roll(h, m)
    + b_m * conj(roll(h, m))
].
```

在 `<.,.>_R` 下满足

```text
<v, J h>_R = <J^T v, h>_R,
```

相应 VJP 为

```text
J(u)^T v = sum_m roll(
    conj(a_m) * v + b_m * conj(v),
    -m
).
```

这里 `b_m * conj(v)` 前没有额外共轭；delay 的 adjoint 使用反向循环位移 `-m`。实现先在
`u_k` 处生成不可变的 `MemoryPolynomialLinearization`，同一次 CG 求解中的所有 JVP/VJP 都复用该
线性化，避免 operator 在 Krylov 迭代内部变化。

## 5. 模型反向传播更新

### 5.1 Raw VJP

`model_vjp_ilc` 使用平方误差梯度

```text
L(u_k) = 1/2 ||y_k-d||_2^2,
gradient_k = J_hat_k(u_k)^T e_k,
delta_k = -mu gradient_k,
u_(k+1) = Project(u_k + delta_k).
```

该模式只用于验证“把残差从 PA 输出位置传回输入位置”这一机制。它没有 trust region 或预测下降
接受条件；anchored prediction 仅作为诊断。因此 raw VJP 在模型失配、Jacobian 尺度很大或局部
非线性强时仍可能不稳定。

### 5.2 Damped Gauss--Newton / LM

`model_lm_ilc` 在固定线性化处求解

```text
(J_hat^T J_hat + lambda I) delta_LM = -J_hat^T e_k,
lambda > 0.
```

公共 LM 求解器和 engine 都要求 `lambda >= 1e-8`；文件配置中的 `lm_damping` 也执行同一下限。
instantaneous-gain 复用该配置作为逐点逆的正则项，但低层 instantaneous-gain API 只要求其为正数。

实现不显式构造 `2N x 2N` Jacobian，而用

```text
q -> JVP(q) -> VJP(JVP(q)) + lambda q
```

作为 normal operator。CG 从零向量开始，默认最多 8 步，相对残差容限 `1e-3`。残差平方、曲率
分母、CG `alpha` 和 `beta` 全部使用 `<.,.>_R`，因而均为实标量。非正曲率、operator/残差非有限
或模型调用失败会立即停止并保持当前输入；达到最大步数但仍保持有限时，当前截断 CG 解仍可进入
后续保护流程。

engine 把 `mu` 用作 LM 的 `step_size`。完整 CG 步先裁到 RMS trust region，得到唯一的
`base_update`；每个候选再取 `0.5^b base_update`，其中 `b=0,...,8`。这种顺序保证超大 CG 步不会在
每轮回溯中被重复裁成同一个边界步。随后依次执行：

1. 输入 peak/RMS/PAPR 硬约束投影；
2. 投影后重新检查 `RMS(delta) <= trust_region_ratio * RMS(u_k)`，越界则拒绝并继续缩步；
3. 使用 anchored model prediction 检查严格预测下降。

anchored prediction 定义为

```text
y_pred(delta)
  = y_k + f_hat_k(u_k + delta) - f_hat_k(u_k).
```

它保留真实测量 `y_k` 作为零步锚点，从而抵消模型在 `u_k` 处的静态偏差。默认接受条件是

```text
1/2 ||y_pred(delta)-d||_2^2 < 1/2 ||e_k||_2^2.
```

所有九个候选均未通过时保持 `u_k`，返回 `prediction_rejected`。当前实现的 `lambda` 在一次更新内
固定，只做候选缩步，并不会在回溯中自动增大 damping；调用方可在下一轮修改配置，但服务本身不含
自适应阻尼状态。

## 6. 校准域

所有非 legacy 策略先把反馈按
`input_sample_rate_hz / feedback_sample_rate_hz` 重采样，再逐捕获估计 1/32 样点分辨率的循环时延。
若反馈恰为参考长度的十倍，则按十次捕获分别处理后平均。

复校准系数的估计式为

```text
c = phase * RMS(reference) / RMS(delay_aligned_feedback),
phase = vdot(delay_aligned_feedback, reference)
        / |vdot(delay_aligned_feedback, reference)|,
y = c * delay_aligned_feedback.
```

三种显式校准语义如下：

| 模式 | 行为 |
| --- | --- |
| `legacy_dynamic` | 每轮、每次捕获重新估计时延和复校准 |
| `frozen_first` | 首个有效反馈按捕获估计；把各捕获系数的复数均值存入会话，后续只重估时延并复用该系数 |
| `explicit` | 后续只重估时延，始终乘配置中已验证的非零有限复系数 |

非 legacy 策略中的 `auto` 解析为 `frozen_first`。实现冻结的是“首个有效反馈”，并不自动判断它是否
为低功率捕获；若研究协议要求低功率校准，调用方必须保证首轮测量条件。`legacy_ilc` 不走上述模式
分派，而始终保持逐轮动态校准。

固定校准使 `d`、`y_k`、模型和误差指标处于同一会话域。动态校准会消除一部分真实幅度/相位变化，
因此只能作为 legacy 兼容行为或校准消融，不能与固定物理域结果混为一谈。

## 7. 安全约束、停止与回退

### 7.1 输入投影

可配置的硬限制为 `max_input_rms`、`max_input_peak` 和 `max_input_papr_db`；三者默认均不启用。
投影顺序是：

1. 若 PAPR 超限，保持相位并用幅度截断降低 PAPR；
2. 用统一幅度缩放同时满足 RMS 和 peak 上限；
3. 以数值容差重新验证所有限制。

PAPR 定义为

```text
PAPR_dB(u) = 20 log10(max|u| / RMS(u)).
```

线性、instantaneous-gain 和 raw VJP 在生成候选后执行一次安全投影。LM 除投影外还检查投影后的
有效更新仍位于 RMS trust region。非有限候选或无法验证约束时不输出该候选。

### 7.2 局部无响应与不可达边界

当 `||e_k|| > 0` 且模型梯度或 instantaneous-gain 修正的范数不超过

```text
1e-10 * max(||e_k||, machine_tiny)
```

时，实现返回 `saturation_limited` 并保持输入。该标志是局部数值保护，不是目标全局不可达的数学
证明；模型失配也可能造成很小的估计梯度。

若 PA 已进入硬饱和，目标输出超出 PA 可达集合，或者硬输入约束排除了所需输入，则不存在能令残差
为零的可行 `u`。此时：

- scalar linear ILC 仍可能沿单位 Jacobian 方向累积输入并触发约束；
- raw VJP 在 `J -> 0` 时出现梯度消失；
- LM 的正 damping 会限制病态逆，但不能恢复不存在的物理解；
- 安全的预期行为是保持、投影或报告 `saturation_limited`，而不是宣称收敛到不可达目标。

同样，低包络处的相位和 `y/u` 比值天然病态；instantaneous-gain 的输入门限会保持这些样点。在线
模型能利用邻近样本和记忆结构提供平滑导数，但仍受训练覆盖、模型阶数和信噪比限制。所有模型方法
都是局部更新，不提供非凸 PA 逆问题的全局收敛保证。

### 7.3 模型回退

只有“在线拟合未成功”会触发 `PAModelFallback`：

- `linear`（默认）：用同一 `mu` 和同一安全限制执行一次 scalar linear ILC；
- `hold`：输出 `u_k`，更新为零，原因记为 `model_failure`。

服务指标会保留原模型 `fallback_reason`，并把更新原因标记为
`model_fallback_<step_reason>`。模型已经拟合成功后，如果 VJP/JVP、CG、预测或投影阶段失败，当前
实现选择保持输入并报告相应原因，不再二次降级为 linear ILC。配置冲突也不会静默回退：模型模式与
legacy 预条件器冲突时在配置或 engine 构造阶段直接拒绝。

## 8. 配置契约

MAT `configDPD` 的主要算法键如下。括号中为默认值。

| MAT 键 | 运行字段 | 约束与语义 |
| --- | --- | --- |
| `ILCBackwardMode` | `backward_mode` | `legacy`；也可为 `linear`、`instantaneous_gain`、`model_vjp`、`model_lm` |
| `LearningRate` / `ILCMu` / `mu` | `mu` | 正有限数，默认 `0.5`；Zilink 且未显式提供时为 `0.3` |
| `dpdGainDb` | `gain_db` | 有限 dB，目标幅度 `G=10^(gain_db/20)` |
| `alpha` | `alpha` | `[0,1]`，默认 `0`；只进入 legacy 更新 |
| `phaseCompensate` | `phase_compensate` | 只进入 legacy 误差预条件；Zilink 默认开启，其他 supplier 默认关闭 |
| `phaseCompThr` | `phase_threshold` | 正有限数，默认 `0.15`，相对于目标 RMS 的门限 |
| `errFirHd` / `txFirHd` | `error_fir` / `tx_fir` | 多 tap 有限复 FIR，只进入 legacy 路径；单 tap 按未配置处理 |
| `InternalSamplingRate` / `FeedbackSamplingRate` | 输入/反馈采样率 | 正有限 MHz；反馈按两者之比重采样 |
| `ILCCalibrationMode` | `calibration_mode` | `auto`、`legacy_dynamic`、`frozen_first`、`explicit` |
| `ILCCalibrationCoefficient` | `calibration_coefficient` | `explicit` 必填，必须有限且非零 |
| `PAModelOrder` | `pa_model_order` | 奇数 `[1,21]`，默认 `9` |
| `PAModelMemoryDepth` | `pa_model_memory_depth` | 整数 `[1,16]`，默认 `3` |
| `PAModelRidge` | `pa_model_ridge` | 非负有限数，默认 `1e-6` |
| `PAModelMinValidationNmseDb` | `pa_model_min_validation_nmse_db` | 不大于 `0 dB`，默认 `-20 dB`；validation NMSE 必须不高于它 |
| `ILCLMDamping` | `lm_damping` | 有限且至少 `1e-8`，默认 `1e-2`；也用于 instantaneous-gain 的分母 |
| `ILCCGMaxIterations` | `cg_max_iterations` | 整数 `[1,128]`，默认 `8` |
| `ILCCGTolerance` | `cg_tolerance` | `(0,1)`，默认 `1e-3` |
| `ILCTrustRegionRatio` | `trust_region_ratio` | `(0,1]`，默认 `0.25` |
| `ILCMaxInputRms` | `max_input_rms` | 可选正有限数 |
| `ILCMaxInputPeak` | `max_input_peak` | 可选正有限数 |
| `ILCMaxInputPaprDb` | `max_input_papr_db` | 可选正有限 dB |
| `PAModelFallback` | `pa_model_fallback` | `linear` 或 `hold`，默认 `linear` |

模式别名在解析后归一化，例如 `identity -> linear`、`raw_vjp`/`vjp -> model_vjp`、
`lm_vjp -> model_lm`，以及带 `_ilc` 后缀的对应名称。

`model_vjp` 和 `model_lm` 必须同时满足：

```text
alpha = 0
phase_compensate = false
tx_fir = None
error_fir = None
```

否则配置被拒绝。原因是当前 normal operator 没有把这些 legacy 预条件器及其 adjoint 纳入 Jacobian；
静默叠加会破坏文档中的目标函数和 LM 对称正定语义。`gain_db`、固定校准和安全投影仍可用于模型模式。
`alpha`、相位预条件器和两个 FIR 在 `linear`、`instantaneous_gain` 策略中也不参与更新；研究配置应
保持它们关闭，避免把无效配置误认为算法组成。当前只实现 memory polynomial；即使 MAT 载荷包含
`PAModelType`，也不会据此选择其他模型族。

`ILCConfig.dtype` 默认 `complex64`，只控制 legacy PyTorch 路径。NumPy 数值核心默认
`complex128`；研究消融必须显式把全部波形边界转成 `complex64`，并设置
`PAForwardModelConfig.numeric_dtype="complex64"`。只转换输入而仍用 complex128 模型不算
complex64 消融；实验产物必须记录模型与学习输出的实际 dtype。

## 9. 可观测指标与停止原因

非 legacy engine 输出以下公共诊断：

- 对齐域：`aligned_nmse_db`、校准系数实部/虚部与幅度 dB、逐捕获时延和捕获数；
- 波形：`feedback_rms`、`output_rms`、`update_rms`；
- 更新：`gradient_rms`、`stop_reason`、`update_accepted`、回溯数、trust region/输入投影是否激活；
- LM：`lm_damping`、CG 步数与相对残差；
- 模型：train/validation NMSE、秩、条件数、样本数、系数数、包络尺度、系数范数和回退原因。

`aligned_nmse_db` 按固定或所选校准域中的 `d` 与 `y_k` 计算。它不是未经校准的 PA 物理输出
NMSE；比较不同校准模式时必须明确这一点。生产文件 engine 没有解析 PA oracle，因而不提供可解释的
梯度方向余弦。研究 runner 在合成 PA 中分别计算 `identity_gradient_cosine` 和
`learned_gradient_cosine`；前者评估单位 Jacobian 方向，后者评估在线模型 VJP 方向，二者不能当作
真实生产捕获中的已有测量。

常见停止原因包括 `accepted`、`converged`、`saturation_limited`、`safety_limited`、
`prediction_rejected`、`model_failure`、`projection_failed`、非有限输入/更新，以及带 `cg_` 前缀的
CG 数值失败。停止原因是算法状态，不应作为过滤轨迹或删除失败样本的依据。

## 10. 与公开方法的关系及表述边界

本项目的比较口径为：

- `linear_ilc` 是明确的单位 Jacobian scalar ILC 基线；
- `instantaneous_gain_ilc` 是本项目带门限、阻尼和安全投影的逐点复增益实现；
- `legacy_ilc` 是为文件协议和历史配置保留的工程变体，不代表纯公开基线；
- `model_vjp_ilc` 验证实线性 PA Jacobian 的输出到输入反向传播；
- `model_lm_ilc` 在同一模型上加入阻尼 normal solve、trust region、anchored prediction 与硬约束。

“通过 PA 模型反向传播训练 DPD”并非新概念。本文实现可主张和检验的是：在逐波形 ILC 中，以每轮
在线 ridge LS 正向模型近似 PA，用真实测量残差驱动解析 VJP，并以 matrix-free、实空间 LM 和明确
保护机制研究单位 Jacobian 近似的失效边界。不得声称首次把 PA 信息、Jacobian 或 backward 用于
DPD。

公开方法背景：

1. Chani-Cahuana et al., *Iterative Learning Control for RF Power Amplifier Linearization*,
   IEEE T-MTT, 2016, https://doi.org/10.1109/TMTT.2016.2588483
2. Schoukens et al., *Obtaining the Preinverse of a Power Amplifier Using Iterative Learning Control*,
   IEEE T-MTT, 2017, https://doi.org/10.1109/TMTT.2017.2694822
3. Morgan et al., *A Generalized Memory Polynomial Model for Digital Predistortion of RF Power Amplifiers*,
   IEEE TSP, 2006, https://doi.org/10.1109/TSP.2006.879264
4. Tarver et al., *Neural Network DPD via Backpropagation through a Neural Network Model of the PA*,
   2019, https://doi.org/10.1109/IEEECONF44664.2019.9048910
5. Loebl et al., *Direct Learning Neural Network Digital Predistortion Using Backpropagation Through a
   Memory Power Amplifier Model*, 2023, https://doi.org/10.1109/IMS37964.2023.10187912
6. Wei et al., *Iterative Learning for RF Power Amplifier Linearization Based on Instantaneous Complex Gain*,
   IEEE Microwave and Wireless Technology Letters, 2025,
   https://doi.org/10.1109/LMWT.2025.3620316

## 11. 实现对应关系与验证契约

| 设计对象 | 实现位置 | 必须保持的验证性质 |
| --- | --- | --- |
| MP 拟合、预测、JVP/VJP | `remote_dpd/pa_model.py` | 确定性 split；validation 不入 LS；有限差分和 real-adjoint identity |
| 线性、瞬时增益、raw VJP、LM、投影 | `remote_dpd/learning.py` | identity/复增益退化；CG 使用实标量；预测拒绝与硬约束不泄漏 |
| engine 分派、legacy 兼容、在线拟合 | `remote_dpd/algorithms.py` | 默认 legacy 不变；显式方法名不混淆；packed capture 数值兼容 |
| MAT 配置与严格校验 | `remote_dpd/config.py` | 枚举、范围、显式校准和模型模式冲突在执行前失败 |
| 时延与复校准 | `remote_dpd/dsp.py`、`remote_dpd/state.py` | `frozen_first` 跨轮复用；`legacy_dynamic` 逐轮估计 |

核心数学验证应覆盖 identity、任意线性复增益、纯共轭映射、循环 delay、有限差分 JVP、
`<v,Jh>_R=<J^T v,h>_R`、显式小尺寸 `2N x 2N` real Jacobian 对照，以及 LM normal operator 的
实空间对称性。上述性质比仅观察单条收敛曲线更能发现共轭、位移方向或因子约定错误。
