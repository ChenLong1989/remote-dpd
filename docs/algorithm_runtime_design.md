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

`ForwardModelILCRuntime` 注册名为 `forward_model_ilc`，每步在更新前先拟合 PA 前向模型并把误差经模型 Jacobian 伴随（反向传播）折算回 PA 输入位置：

```text
basis_k[n]  = y[n-m_k] * |y[n-m_k]|**(p_k-1)            (periodic roll boundary)
z_model     = argmin_c ||sum_k c_k basis_k - z||^2       (complex LS, see below)
A-weight_k  = (p_k+1)/2 * |y|**(p_k-1)                   (d/dy part of basis_k)
B-weight_k  = (p_k-1)/2 * conj(y)^2 * |y|**(p_k-3)       (d/dy* part of basis_k)
g           = sum_k conj(c_k)*A-weight_k*roll(e,-m_k) + c_k*conj(B-weight_k*roll(e,-m_k))
y_candidate = y_current - mu * g
```

要点：

- 拟合只用当前轮 `(y_current, z_current)`，不跨轮累积；`z` 是已对齐反馈，因此线性系数自然吸收衰减、反馈增益/相位和残余对齐等复合线性路径，无需显式建模。
- LS 采用列 RMS 归一化 + 相对岭正则（`ridge * N` 加在归一化 Gram 对角）的正规方程，`numpy.linalg.solve` 求解；Gram 与右端分块累积（每块约 800 万基样点）以限制长波形的峰值内存。全零列直接赋零系数，病态矩阵回退 `lstsq`。
- `g` 是 `||F(y) - x||^2` 相对复参数化 `y` 的精确实梯度方向。非全纯映射必须包含共轭项 `conj(B^H e)`：只取 `A^H e` 或误用 `(A+B)^H e` 都会使方向倾斜，深度压缩下足以导致停滞或发散。伴随中 A/B 权重按输入样点 `y[j]` 计算，与输出误差 `e[j+m]` 配对；方向公式已用中心有限差分验证。
- 模型精度只影响方向，不用于预测输出，因此粗拟合即可。默认基 `{1,3,5} x {0,1,2}` 已覆盖强压缩场景；继续加阶数或深度对默认场景改善小于 `0.1 dB`。
- 收敛稳定边界为 `mu < 2 / lambda_max(J^H J)`，与具体 PA 工作点有关。超出边界时发散是渐进的，不会立即爆炸，但 `mu` 仍应按 PA 严重程度选取。

配置字段（全部严格校验）：

| 字段 | 默认 | 约束 |
| --- | --- | --- |
| `mu` | `1.0` | 有限正实数 |
| `orders` | `[1, 3, 5]` | 1..9 的正奇数、严格升序、至多 5 项 |
| `memory_depths` | `[0, 1, 2]` | 0..16 的整数、严格升序、至多 8 项 |
| `ridge` | `1e-8` | `[1e-12, 1e-2]` 内有限实数 |

`orders x memory_depths` 至多 16 个基项。指标在基础字段（迭代号、步数、`mu`、误差 RMS、候选 RMS）之外增加梯度 RMS、模型拟合残差 RMS 和逐项拟合系数（`p`、`m`、实部、虚部）。runtime 不做缩放、裁剪或预处理，数字安全边界不变。

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
