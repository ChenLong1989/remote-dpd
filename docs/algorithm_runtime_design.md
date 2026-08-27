# DPD Runtime 与数字安全设计

本文描述 `remote_dpd/runtime.py` 和 `remote_dpd/safety.py` 当前实现。新 runtime 契约尚未接入现有文件监听服务；旧服务仍调用 `remote_dpd/algorithms.py` 的旧引擎接口。

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

三个波形必须一维、非空、有限、数值型且等长。波形副本使用不可写底层缓冲区，不能通过重新打开 NumPy write flag 修改。配置和指标仅接受可递归冻结的标量、mapping、sequence 和非 object NumPy array；嵌套 mapping、sequence 和 array 均与调用方分离且不可变，非有限数值和任意可变对象被拒绝。`RuntimeStepResult` 返回同样不可变的候选 `y_candidate` 和结构化指标。runtime 不得读取设备、原始抓取、网页或 MAT 文件，也不得做时延/相位/增益预处理、AGC、归一化、削峰或安全裁剪。

## 2. 基础 ILC

内置 `BasicILCRuntime` 注册名为 `basic_ilc`，唯一参数为正有限实数 `mu`，默认值 `0.5`。更新公式是：

```text
y_candidate = y_current - mu * (z_current - x)
```

输出指标包括外部迭代号、runtime 内部步数、`mu`、误差 RMS 和候选 RMS。任何非有限候选都作为契约错误拒绝。

首期不实现 `alpha`、TX FIR、误差 FIR、动态相位补偿或其他 ILC 变体。

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
