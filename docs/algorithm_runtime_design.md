# DPD Runtime 与数字安全设计

本文描述 `remote_dpd/runtime.py` 和 `remote_dpd/safety.py` 当前实现。runtime 契约已经接入 `ClosedLoopController`，并由新文件命令服务驱动；旧引擎和旧文件监听实现已经移除。

## 1. Runtime 契约

当前逻辑 API 版本为 `1.0`。`DPDRuntime` 生命周期如下：

```text
created -> initialize(config) -> step(...) * N -> reset()
                                  |                |
                                  +---- close() <--+
```

- 未初始化时不能执行 `step()`；重复初始化被拒绝。
- `reset()` 清除私有状态和配置，之后必须重新初始化。
- `close()` 幂等且是终态，关闭后不能初始化、执行或重置。
- 初始化配置和每步配置必须在归一化后相等，防止运行中静默改变算法参数。

`RuntimeStepInput` 使用只读防御性副本保存：

- 固定参考 `x`；
- 当前已经实际发送的 `y_current`；
- 与之对应的预处理反馈 `z_current`；
- 非负迭代号；
- 本次不可变配置。

三个波形必须一维、非空、有限、数值型且等长；其中固定 `x` 已由 controller 完成配置化 RMS conditioning。波形副本使用不可写底层缓冲区，不能通过重新打开 NumPy write flag 修改。配置和指标仅接受可递归冻结的标量、mapping、sequence 和非 object NumPy array；嵌套 mapping、sequence 和 array 均与调用方分离且不可变，非有限数值和任意可变对象被拒绝。`RuntimeStepResult` 返回同样不可变的候选 `y_candidate` 和结构化指标。runtime 不得读取设备、原始抓取、网页或 MAT 文件，也不得自行做时延/相位/增益预处理、AGC、归一化、削峰或安全裁剪。

## 2. 内置 runtime

### 2.1 基础 ILC

`BasicILCRuntime` 注册名为 `basic_ilc`，唯一参数为正有限实数 `mu`，默认值 `0.5`。更新公式是：

```text
y_candidate = y_current - mu * (z_current - x)
```

输出指标包括外部迭代号、runtime 内部步数、`mu`、误差 RMS 和候选 RMS。任何非有限候选都作为契约错误拒绝。

该形式把 PA Jacobian 近似为单位阵：误差从 PA 输出位置原样搬到 PA 输入位置。PA 深度压缩使局部斜率趋零甚至变负时，该方向严重偏离真梯度，收敛停滞乃至发散；减小 `mu` 只能推迟恶化。

### 2.2 前向模型梯度 ILC

`ForwardModelILCRuntime` 注册名为 `forward_model_ilc`，每步先在残差形式下拟合 PA 前向模型，再把误差经模型 Jacobian 伴随（反向传播）折算回 PA 输入位置：

```text
z_hat       = y + Phi(y) w                              (residual FLF model, see below)
w           = argmin ||z - y - Phi w||^2 + lambda||w||^2 + lambda_d||D beta||^2
g           = e + A_Phi^H e + conj(B_Phi^H e)           (e = z - x)
y_candidate = y_current - mu * g
```

残差化是设计核心：恒等项 `y` 直连使模型 Jacobian 恒含单位阵，ILC 固定点从 `J_fit^H e = 0`（模型类失配时偏离 `e=0`，真机 MP 基时代 ≈-32 dB 停滞的根因）回到 `(I + J_Phi)^H e = 0`，被恒等项拉回 `e≈0`；`w→0`（或正则极大）时更新严格退化为 basic ILC，模型失配不再制造偏置固定点。恒等直连在反向传播中表现为 `g` 的 `e` 直连项。

#### 2.2.1 FLF 基（`remote_dpd/forward_model.py`）

基函数是 dpd-compass 仓库 FLF（FIR-LUT-FIR）算法结构的 numpy 移植（见该仓库 `docs/algorithm_design.md` §13 与 `models/flf.py`；polyphase FIR、三角 LUT、系数布局均同源）。tap 集合为三对对角延迟 `(d_x, d_p) ∈ {(-1,-1), (0,0), (1,1)}`（内部 2x 采样率单位，用户决策，替代 dpd-compass 的 46 对参考 S-matrix 阶梯）。FLF 对系数线性，因此改写为 `z_hat = y + Phi(y) w` 的线性形式后用块状 Gram 正规方程直接求解（dpd-compass 原实现为梯度下降训练，本项目的 LS 变体即为此构建）。结构：

```text
u[2n]   = y[n]                        # polyphase 0: identity
u[2n+1] = H1{y}[n]                    # polyphase 1: fixed 9-tap center-aligned FIR
X_d[n]  = u[2n+r+d]                   # integer tap; half tap -> mean of floor/ceil
A_d[n]  = |u[2n+r+d]|                 # magnitude FIRST, then average for half taps
tau_k(a)= triangular hat on grid j/Q  # interior knots k=1..Q-2, endpoints fixed zero,
                                      # amplitude above (Q-1)/Q -> all hats zero
q_r[n]  = sum_d alpha[r,d] X_d[n] + sum_t sum_k beta[r,t,k] X_{dx[t]}[n] tau_k(A_{dp[t]}[n])
z_hat[n]= y[n] + G0{q_0}[2n] + G1{q_1}[2n+1]     # G0 identity, G1 fixed FIR
```

- tap 集合：三对对角 `(d_x, d_p) = (-1,-1), (0,0), (1,1)`，嵌套前缀选择 `tap_count ∈ {1, 3}`（1 = 仅对齐 tap）；独立线性族延迟 `D = unique(d_x) = {-1, 0, 1}`。系数 `w = (alpha(2,U), beta(2,T,Q-2))`，`U = |D|`、`T = tap_count`，总复列数 `K = 2*(U + T*(Q-2))`（默认 3/32 时 K=186）。
- 与 dpd-compass 的差异：tap 集合为三对角延迟（非其 46 对 S-matrix）；序列边缘用周期 `roll`（本项目 ILC 波形为整周期，周期边界精确；dpd-compass 为 zero-pad + 截断）；系数用 LS 直解而非迭代训练。
- 幅度网格不做数据归一化（忠实 dpd-compass 语义）。本机记录波形峰值 0.52-0.61，位于 Q=32 网格上界 31/32 内余量充足；候选峰值超出网格时该样点 LUT 贡献置零、恒等项兜底（优雅退化）。

#### 2.2.2 LS 求解

`min_w ||z - y - Phi w||^2 + lambda||w||^2 + lambda_d||D beta||^2`：朴素 Tikhonov `lambda` 防奇异（FLF 基结构性秩亏），`lambda_d` 为 **LUT 差分平滑正则**（`D` 是 beta knot 轴一阶差分，加在 Gram 上即 `Gram + lambda_d D^H D`）。差分正则直接压制 hat 斜率 `dLUT/da = Q(beta_hi - beta_lo)` 被 knot 数放大造成的伴随梯度尖峰：真机数据上朴素正则下种子步梯度尖峰比高达 142（候选峰值超安全限，与 GMP 时代的 `candidate_peak_exceeded` 同类病根），`lambda_d=1e-3` 把尖峰压到 ~9-13 而残差仅损 ~0.1 dB；加大朴素 lambda 压尖峰代价高且非单调。Gram 与右端按每块约 800 万基样点分块累积，`numpy.linalg.solve` 求解、奇异时回退 `lstsq`；基样点或目标能量非有限时 fail-closed 拒绝。

#### 2.2.3 伴随（反向传播）

`g = e + A_Phi^H e + conj(B_Phi^H e)`：恒等项贡献 `e` 直连；FLF 通路链为 `e -> G_r 伴随 -> v_r ->`（每 tap 两条路径：X 信号路径全纯散射，权重 `conj(LUT_{r,t}(A))`；幅度路径权重 `conj(X)·conj(dLUT/da)·(1/2)(u/|u|)` 及其共轭部；半整数 tap 在 floor/ceil 两端点各 0.5）`-> polyphase FIR 伴随 -> y`。非全纯映射必须包含共轭项 `conj(B^H e)`（与 MP 时代结论一致）；方向公式已用中心有限差分与全数值 Jacobian 转置对拍验证（相对误差 ~1e-7，即 FD 精度极限），并作为单元测试保留。

#### 2.2.4 固定点与稳定边界

更新 `y_candidate = y - mu*g`；固定点 `(I + J_Phi)^H e = 0`。稳定边界 `mu < 2 / lambda_max(J^H J)` 与 PA 工作点有关；恒等项使 `lambda_max >= 1`，`mu` 上界不再由病态模型 Jacobian 主导，但高增益工作点仍需相应下调 `mu`。

#### 2.2.5 配置字段

| 字段 | 默认 | 约束 |
| --- | --- | --- |
| `mu` | `1.0` | 有限正实数 |
| `tap_count` | `3` | `{1, 3}` 之一 |
| `lut_size` | `32` | 整数 `>= 3` |
| `ridge` | `1e-8` | `[1e-12, 1e-2]` 内有限实数（朴素，防奇异底） |
| `lut_ridge` | `1e-3` | `[1e-12, 1e-2]` 内有限实数（LUT 差分平滑） |

`tap_count x lut_size` 组合的复列数上限 6144（更大的组合被拒绝，防止 Gram 内存失控）。17-tap S-matrix 阶梯版本的取舍依据真机数据离线实验（记录于 `current_plan.md` 2026-09-04 节）：it0 残差 -39.2 dB（MP 基 -27.4、GMP-105 -32.8）、种子步梯度尖峰 ~9、mu=1.0 候选峰 0.766、全长拟合 ~35 s；46/32 残差 -41.4 dB 但拟合 ~105 s 超真机每轮周期。现行 3-tap 集合的用户决策与真机回放数据见同节追加记录。

配置字段相对旧版（`orders/memory_depths/ridge`）为**破坏性变更**：旧显式配置会被未知字段检查拒绝；Web/file 快速启动只传 `mu`，不受影响。

#### 2.2.6 指标

在基础字段（迭代号、步数、`mu`、误差 RMS、候选 RMS）之外输出梯度 RMS、模型拟合残差 RMS（对 `z` 的残差，语义与旧版一致）和 `model_coefficients` 摘要：`alpha` 全量 2×U 复系数（`phase/delay/real/imag`）+ `beta` 的 `count/rms/max`（beta 可达数千系数，逐项输出对逐轮 JSON 过重，且旧 `model_terms` 字段无下游消费方）。runtime 不做缩放、裁剪或预处理，数字安全边界不变。

## 3. 注册表

`register_runtime()` 只接受 API 版本兼容的 `DPDRuntime` 子类，名称归一化为小写，并限制为字母起始的字母、数字、下划线或连字符。`create_runtime()` 每次返回全新、未初始化实例，避免任务之间共享 runtime 状态。`list_runtimes()` 返回确定性排序名称。

当前注册表只解决进程内 Python runtime。进程外程序、共享库、容器或模型文件加载器要等首个真实产物形态确定后再设计，不属于 API 1.0 已实现能力。

## 4. 数字安全

数字安全是 runtime 之后、TX 设备调用之前的独立边界。检查函数只测量并报告，不改变输入数组。

参考 `x` 必须是一维、非空、有限数值向量且：

```text
max(abs(x)) <= 1.0
```

每个候选还必须与 `x` 等长，并同时满足：

```text
max(abs(y_candidate)) <= 1.0
RMS(y_candidate) <= RMS(x) * 10 ** (2 / 20)
```

边界值允许通过。`check_reference()` 和 `check_candidate()` 返回 `DigitalSafetyReport`；`validate_reference()` 和 `validate_candidate()` 在失败时抛出携带同一报告的 `DigitalSafetyError`。报告包含样点数、峰值、RMS、限制值和稳定错误代码；不可测量的数值使用 `None`，`to_dict()` 可被严格 JSON 序列化，供后续控制器、文件接口和网页直接消费。

安全模块不做 AGC、缩放或削峰。零 RMS 参考只允许零 RMS 候选；物理 dBm 功率安全由后续功率控制器处理。
