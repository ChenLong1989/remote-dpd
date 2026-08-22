# PA 模型反向传播 ILC 冻结实验协议

## 1. 状态与适用范围

本文档记录仓库当前实现的冻结仿真协议，协议名为 `pa-model-backprop-ilc`，修订日为
`2026-08-22`。科学常量以 `experiments/config.py` 中的 `ExperimentProtocol`、固定候选表和
`enumerate_study()` 为可执行依据；调度器实现见 `experiments/runner.py`。本文不包含实验结果，
也不得用来暗示已经完成硬件验证。

研究问题限定为：单位 Jacobian 的逐波形 ILC 在强 AM/AM 或低功率 AM/PM 非线性下何时失效，
以及“每轮在线 PA 正向模型 + 真实测量残差 VJP + 受保护 matrix-free LM”能否改善该失效。
本协议仅覆盖确定性合成 PA 与合成捕获噪声；不覆盖真实 GaN/Doherty PA、PAE、热效应、
产业部署或 3GPP 合规性。

算法数学定义见 [algorithm_design.md](algorithm_design.md)，运行与恢复说明见
[reproducibility.md](reproducibility.md)。

## 2. 波形、随机性与固定域

### 2.1 NR-like OFDM 波形

| 项目 | 冻结值 |
| --- | --- |
| 调制 | 独立、单位平均能量的 256-QAM 星座点 |
| IFFT | `NFFT=2048`，unitary normalization |
| 占用子载波 | FFT 顺序中的 `-600..-1` 与 `1..600`，共 1200 个；DC 与 guard 为零 |
| symbol 数 | 16 |
| CP | 关闭；用于周期 ILC，而非标准 NR 帧 |
| 样点数 | `N=32768` |
| 时域功率 | 每条生成波形归一化到单位均方功率；频域已知 grid 同比例缩放 |

生成器不实现信道编码、标准资源映射、同步器或标准接收机，因此文档和图表只能使用
“NR-like OFDM”，不得写成“3GPP-compliant NR”。

### 2.2 Seed 派生与配对

根熵固定为 `SeedSequence(20260822)`。实现不依赖 `SeedSequence.spawn()` 的调用顺序，而把
`study/role/index` 的稳定 SHA-256 名称映射写入 child `spawn_key`。`seeds.csv` 保存每个 PA、
波形和噪声流的 state words 与 spawn key；动态 PA 另保存 `dynamic_pa_wiener` 与
`dynamic_pa_hammerstein` 流。

同一 cell 和 seed 下所有方法共享 PA seed、波形 seed 与基础噪声流。统计重复单位是 PA/波形
seed 对，不把样点或 OFDM symbol 当作独立重复。

### 2.3 固定 unity synthetic native-domain 校准

研究 runner 在 manifest 中明确写入：

```json
{
  "fixed_calibration": {
    "coefficient_real": 1.0,
    "coefficient_imag": 0.0,
    "domain": "synthetic_pa_native_domain"
  }
}
```

这是仿真专用简化：合成 PA 的输入、输出和目标本来就在同一个数值域中，不存在额外测量链增益、
公共相位或采样时延，因此使用显式 unity 校准，而不是虚构一次低功率硬件标定。PA 自身的 AM/AM、
AM/PM 和动态响应不会被该校准消除。文件服务仍支持 `legacy_dynamic`、`frozen_first` 和
`explicit` 校准；这些生产路径语义不能与研究 runner 的 unity native-domain 选择混写。

## 3. PA 场景

### 3.1 主要 AM/AM 扫描

平滑 Rapp PA 固定为 `A_sat=1.0`、`p=4`，目标峰值/饱和值扫描
`0.55, 0.75, 0.90, 0.97`。四个目标均严格小于饱和值，runner 在每个实际 seed 上构造目标、
用解析逆求 `required_input_peak`，并把可达性、所需输入峰值和安全限制写入 manifest 的
`cell_instances`。预注册主要单元为 `amam/0.97`。

### 3.2 主要低功率 AM/PM 扫描

AM/AM 为 unity，AM/PM 相位函数为

```text
phase(r) = phase_max * exp(-(r / r0)^2).
```

目标 RMS 固定为 `0.35`，`r0=0.21`，`phase_max` 扫描 `0,45,90,135` 度。预注册主要单元为
`ampm/135`。每条轨迹使用由输入安全峰值上限确定的 20 个固定包络分箱；低功率相位 RMSE 固定
汇总 `bin_center <= r0` 的样本，按每箱有效相位样本数加权，不允许逐 seed 或逐轮改变 `r0`。

### 3.3 Out-of-family 动态 PA

动态研究包含两个 cell：

- `amam_dynamic/0.97`：Wiener 输入 FIR 后接平滑 Rapp 非线性；
- `ampm_dynamic/135`：平滑 Rapp/指数 AM/PM 非线性后接 Hammerstein 输出 FIR。

基准 taps 分别为 `[0.94+0j, 0.18-0.08j, -0.05+0.03j]` 与
`[0.92+0j, 0.12+0.06j, -0.04+0.02j]`。每个 PA seed 对非首 tap 的实部和虚部分别施加
`Uniform(-5%,5%)` 的确定性相对扰动，再按复数 DC gain 归一化为 1。该研究检验模型族失配，
不是主要假设检验。

### 3.4 独立压力场景

压力场景不进入主要 NMSE 优势推断：

| cell | 冻结参数 | 目的 |
| --- | --- | --- |
| `hard_saturation/2` | 理想硬限幅 `A_sat=1.0`；`target_peak_ratio=2.00`；`reachable=false` | 明确不可达目标下的安全停滞、饱和受限和约束行为 |
| `gain_rolloff/0.4` | `target_peak=0.40`、`initial_input_peak=2.50`、`turnover=0.70`；`reachable=true` | 从局部负 AM/AM slope 区域启动，检验单位 Jacobian 的局部错误方向 |

gain-rolloff 目标位于低输入分支的可达范围内；runner 计算并记录该分支的
`required_input_peak`。`identity_negative_local_fraction` 与
`identity_negative_inner_magnitude_fraction` 用于描述局部负方向，不把全局方向余弦的符号当作
唯一判据。

## 4. 方法与固定更新预算

### 4.1 核心方法

确认扫描和 stress 固定包含 7 种方法：

1. `no_dpd`：保持初始输入；
2. `linear_ilc`：单位 Jacobian scalar ILC；
3. `legacy_ilc`：保留历史动态增益/相位校准和相位预条件语义；
4. `instantaneous_gain_ilc`：带门限、阻尼的逐点复增益更新；
5. `oracle_lm`：使用合成 PA 的解析 JVP/VJP，给出模型误差之外的参考上界；
6. `model_vjp_ilc`：每轮在线模型的 raw VJP；
7. `model_lm_ilc`：每轮在线模型的 safeguarded matrix-free LM。

外层固定执行 `K=30` 次更新，并保存 `k=0..30` 共 31 次 PA 评估。发现收敛后仍继续执行到
`k=30`；`final` 始终指 `k=30`，不得事后选择最佳轮。

### 4.2 在线模型与 LM 常量

| 项目 | 冻结值 |
| --- | --- |
| MP orders | `1,3,5,7,9` |
| memory depth | 3 |
| ridge | `1e-6` |
| 包络 scale | 训练输入幅度 `99.9%` 分位数，最小 `1e-12` |
| train/validation split | 连续 256-sample blocks；每 5 个 block 的第 5 个仅作 validation |
| condition 上限 | `1e12` |
| validation 接受门限 | `<= 0 dB` |
| 拟合失败回退学习率 | `0.05` 的显式 linear fallback |
| CG | 最多 8 步，relative tolerance `1e-3` |
| damping 下限 | `1e-8` |
| backtracking | 最多 8 次，因子 `0.5` |

模型系数在本轮拟合后冻结；JVP/VJP 不穿过 LS、校准或时延估计。LM 的预测使用锚定形式
`measured + model(input+delta) - model(input)`。消融 `unanchored_prediction` 只改变预测器，
仍保留其他保护。

### 4.3 输入安全限制

每个实际 cell/seed 的限制由初始输入和可达性元数据物化：

```text
max_rms  = 2.0 * initial_rms
max_peak = max(1.15 * required_input_peak, 2.0 * initial_peak)
max_papr = initial_papr_db + 4.0 dB
```

数值容差外的投影后违规被标记为 `constraint_violation` 和发散，并保持当前输入。关闭 trust region
的消融仍保留最终硬输入投影。

## 5. Pilot 与 resolved config

Pilot 只使用两个主要单元、6 个与确认集命名空间隔离的 seed。候选固定为：

| 方法 | 候选 |
| --- | --- |
| `linear_ilc` | `mu=0.025,0.05,0.1,0.2,0.4,0.8` |
| `instantaneous_gain_ilc` | `(mu,damping)=(0.1,1e-2),(0.2,1e-2),(0.4,1e-2),(0.8,1e-2),(0.2,1e-3),(0.4,1e-3)` |
| `model_vjp_ilc` | `mu=0.01,0.03,0.1,0.3,0.6,1.0` |
| `model_lm_ilc` | `(step,damping,trust)=(0.25,1e-2,0.1),(0.5,1e-2,0.25),(1.0,1e-2,0.25),(0.25,1e-3,0.1),(0.5,1e-3,0.25),(1.0,1e-3,0.25)` |

选择顺序为：排除任一安全失败的候选；最小化两个主要单元合并后的 median AUEC；进入最佳值
2% 范围的候选视为 tie；先选预注册计算成本更低者，仍相同时按表中索引。当前
`PILOT_COMPUTE_COSTS` 对同一方法的六个候选均显式写为 `1.0`，因为它们具有相同模型规模、CG
上限与回溯上限；因此当前 cost tie 最终落到表序，但该顺序仍经过显式成本表。

`--select-pilot` 只接受完整且校验通过的 288 条 pilot shards。输出的 `resolved_config.json`
包含完整 7 方法参数、候选索引和 tie set，并绑定：

- 当前 `protocol_hash`；
- pilot 的 `code/configuration/protocol/matrix/environment` 五个 SHA-256；
- pilot run directory 的相对路径；
- `manifest.json` 与 `expected_ids.json` 的文件 SHA-256；
- 已验证 records 的稳定 hash；
- resolved payload 自身的 `resolved_hash`。

加载 resolved config 时会重新读取 pilot artifacts、重算选择并逐字段比较，不能手工替换候选。
`no_dpd`、`legacy_ilc` 和 `oracle_lm` 使用预注册固定参数，不参加 pilot 调优。

## 6. 固定实验矩阵

| study | cell / 方法 / seed | 轨迹数 | 推断角色 |
| --- | --- | ---: | --- |
| `smoke` | 2 个主要 cell × 7 方法 × 1 seed | 14 | 端到端检查，不作正式推断 |
| `pilot` | 2 cell × 4 方法 × 6 候选 × 6 seeds | 288 | 锁定超参数 |
| `confirmatory` | 8 severity cells × 7 方法 × 40 seeds | 2240 | 主要比较与完整扫描 |
| `robustness` | 2 cell × 4 SNR × 2 capture counts × 3 方法 × 20 seeds | 960 | 捕获噪声/平均稳健性 |
| `mismatch` | 2 cell × 4 maximum orders × 3 depths × 12 seeds | 288 | 模型阶数/记忆失配 |
| `ablation` | 2 cell × 8 消融 × 12 seeds | 192 | 单因素机制诊断 |
| `dynamic` | 2 动态 cell × 4 方法 × 20 seeds | 160 | Out-of-family 动态 PA |
| `stress` | 2 压力 cell × 7 方法 × 12 seeds | 168 | 安全边界，仅描述性 |

robustness 的 SNR 为 `Inf,50,40,30 dB`，capture count 为 `1,10`，方法为
`linear_ilc`、`instantaneous_gain_ilc`、`model_lm_ilc`。mismatch 的 maximum order 为
`3,5,7,9`，memory depth 为 `1,3,5`。八个消融为：

```text
raw_vjp
no_ridge
frozen_first_model
three_iteration_replay
unanchored_prediction
no_trust_region
complex64
legacy_dynamic_calibration
```

dynamic 只比较 `linear_ilc`、`instantaneous_gain_ilc`、`oracle_lm`、`model_lm_ilc`。
除 `smoke` 和 `pilot` 外的全部研究必须使用同一份验证后的 `resolved_config.json`；即使主要假设
失败，次要矩阵也不事后删减。

## 7. 指标、失败和端点

### 7.1 主要端点

- 固定 native-domain tracking NMSE；
- `AUEC = mean(k=0..30, 10^(NMSE_k/10))`，在线性功率比域计算；
- 最终 NMSE，固定取 `k=30`；
- 成功率、发散率、约束违规率和算法失败率；
- 达到并保持 `NMSE <= -35 dB` 连续 3 轮的首次迭代。

非收敛轨迹在 `k=30` 右删失。精确零误差在线性域保留为零；需要有限 dB 统计时使用
`-300 dB` 下限，导出元数据必须声明该规则。

### 7.2 次要端点

逐轮保存双边 sampled-band ACLR、known-grid raw/one-tap EVM、固定分箱 AM/AM 与 AM/PM 误差、
低功率相位 RMSE、输入 RMS/peak/PAPR、模型 train/validation NMSE、秩、条件数、CG 步数与残差、
回溯和安全状态。研究 runner 分开记录：

- `identity_gradient_cosine`：单位 Jacobian 方向相对合成 PA oracle 的方向余弦；
- `learned_gradient_cosine`：在线模型 VJP 方向相对同一 oracle 的方向余弦；
- 局部负方向比例和按内积幅度加权的局部负比例。

这些字段只在具有解析合成 PA 的研究 runner 中有 oracle 意义；生产文件服务没有 oracle，不能从
真实捕获推导或宣称上述方向余弦。

### 7.3 失败编码

runner 必须物化完整 `k=0..30` 序列。算法失败后停止更新、保持当前输入，并继续记录后续评估；
失败轨迹不删除、不缩短、不按算法失败重试。PA 或捕获产生非有限值时，当前轮及后续相同失败评估
使用固定、有限的 `300 dB` NMSE penalty，相关 ACLR/EVM/分箱指标写为缺失，同时标记
`algorithm_failure` 与 `diverged`。基础设施异常最多重试 2 次，与算法失败严格分开。

发散定义为 NaN/Inf 的数值失败、投影后约束违规，或连续 3 轮 NMSE 相对初始值恶化超过 `3 dB`。
成功必须同时满足已经出现 3 轮收敛事件、没有发散、没有约束违规且状态为 `completed`。

## 8. 统计分析与预注册成功标准

主要比较只在 `amam/0.97` 和 `ampm/135` 中比较 `model_lm_ilc` 与 `linear_ilc`。使用 40 个
完整配对 seed，以 `pa_seed_index` 为 bootstrap cluster，执行 10,000 次确定性层级 paired
percentile bootstrap，置信度 95%。两个主要 cell 的 AUEC p 值组成一个 family，按 Holm
step-down、`alpha=0.05` 校正。

每个主要 cell 的预注册效果标准为：

- paired median AUEC 相对降低至少 25%；
- paired median 最终 NMSE 改善至少 `3 dB`；
- 成功率提高至少 20 个百分点；
- 发散率差不超过 `+10` 个百分点；
- 约束违规率不增加。

这些阈值是可证伪标准。未通过时必须照实报告，不能更换指标、删 seed、只展示最佳迭代或把
stress 结果并入主要推断。ACLR、EVM、分箱、梯度余弦和次要研究均为解释性结果。

## 9. 资源门与不可变产物

运行时资源设置不改变科学矩阵：默认 6 个独立 worker，允许范围 `1..8`；每个 worker 固定
NumPy/BLAS/PyTorch 单线程。单 worker 峰值 RSS 门为 `2.5 GiB`，产物总预算 `5 GiB`，输出盘
可用空间门为 `50 GiB`，预计协议超过 72 小时时停止继续调度。worker 可因 RSS 门递减，不能通过
减少 cell、seed、迭代或方法绕过容量门。

每个 study 在运行前固定并交叉绑定五个 SHA-256：`code_hash`、`configuration_hash`、
`protocol_hash`、`matrix_hash` 与 `environment_hash`。manifest 另记录 Python/平台/依赖、线程环境变量、Git commit 与
dirty 状态、完整 argv/cwd、所有实际 cell 的可达性、安全限制和分箱边界。调度期间反复重算 live
hash；科学源代码、协议、resolved 参数或矩阵发生变化即停止，禁止混合不同版本 shards。

普通轨迹只保存逐轮标量；代表 seed 0 在 `k=0,1,2,5,10,20,30` 保存输入、测量和目标波形。
其他波形由 seed、spec 与协议重建。每条轨迹使用独立、自校验、原子替换的 JSON shard；checkpoint
以 JSON 元数据绑定一个带 SHA-256 的压缩数组文件。恢复及分析必须先验证哈希、checksum、完整
expected ID 集和无重复 ID，不能手工拼接修复前后的结果。
