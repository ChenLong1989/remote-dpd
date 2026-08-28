# MAT 文件命令接口设计

本文描述 `remote_dpd/file_interface.py` 的当前版本化外部契约。该接口只负责命令投送、状态和结果交付；所有闭环行为由同一个 `ClosedLoopController` 实现。

本接口不兼容旧 `Config_file.mat`、`DPD_in.mat`、`FB_Signal.mat`、ACK、心跳、`safeBack` 或特殊十段输出规则。

## 1. 目录和生产者规则

```text
<exchange-root>/
├── inbox/command_<command_id>.mat
└── outbox/
    ├── status_<command_id>.mat
    └── result_<command_id>.mat
```

`command_id` 长度为 1 至 64，只允许 ASCII 字母、数字、下划线和连字符，并且首字符必须是字母或数字。命令文件必须是 inbox 的直接普通文件，文件名中的 ID 必须与 MAT 变量一致；符号链接和目录逃逸被拒绝。

生产者必须在 inbox 外或使用非正式临时文件名完成写入，再原子重命名为正式 `command_<command_id>.mat`。watcher 处理创建和移动事件，不把“持续修改同一个正式文件”作为有效生产协议。

## 2. 输入 MAT

公共变量：

| 变量 | 约束 |
| --- | --- |
| `schema_version` | 整数标量，当前只能为 1 |
| `command_id` | 字符串标量，与文件名一致 |
| `action` | 严格小写动作名 |
| `x` | `load` 或 `run` 可带；非空有限数值行/列向量，归一化为一维 `complex128` |
| `config_json` | `configure` 或 `run` 可带；严格 JSON 字符串 |

未知 MAT 变量被拒绝。支持动作：

| 动作 | 行为 |
| --- | --- |
| `load` | 保存或替换参考 `x`，使旧校准和迭代失效 |
| `configure` | 根据 `device_type` 创建新 controller，连接设备并应用配置 |
| `power_tune` | 必要时先发送 `x`，再完成初始功率调节 |
| `calibrate` | 必要时重启已调功率的 `x`，监控功率并提交第 0 轮 |
| `step` | 执行一次完整 ILC 发送、监控、抓取和提交 |
| `run` | 可携带配置和/或 `x`，补齐前置步骤并自动运行到最大次数 |
| `stop` | 立即转发停止 Event，不等待当前串行 worker |
| `reset` | 安全复位 controller 并结束当前临时 recorder |
| `export` | 对当前 `COMPLETED` snapshot 生成正式 MAT |

`run` 不带某项输入时复用当前已加载配置或 `x`；两者都不存在时明确失败。

## 3. `config_json`

顶层只允许：

```json
{
  "device_type": "simulated",
  "device_config": {},
  "runtime_name": "basic_ilc",
  "runtime_config": {"mu": 0.5},
  "max_iterations": 5
}
```

`device_config` 字段名必须属于 `DeviceConfig`，设备专属值位于其中的 `device_options`。设备 schema 会在 controller 应用配置时补齐默认 PA 系数及其他仿真项。

JSON 拒绝重复 key、NaN/Infinity、未知字段和非法类型。`ClosedLoopConfig.to_dict()` 产生的特殊值使用：

- `{"$type":"complex","real":...,"imag":...}`；
- `{"$type":"ndarray","dtype":...,"shape":[...],"data":...}`。

解码器限制 dtype、shape 和最大元素数，拒绝 object、structured、datetime、bytes 和扩展精度数组。

## 4. 状态 MAT

每个可安全识别 ID 的命令最终都会得到 `status_<command_id>.mat`：

| 变量 | 含义 |
| --- | --- |
| `schema_version` | 整数 1 |
| `command_id` | 命令 ID |
| `accepted` | `uint8` 0/1 |
| `state` | `accepted`、`busy` 或 controller 状态 |
| `iteration` | 当前完整轮次，无轮次时为 -1 |
| `message` | 人类可读说明 |
| `error_code` | 稳定错误代码，成功为空字符串 |
| `timestamp` | UTC ISO 8601 时间 |
| `run_id` | 当前命令关联的临时 run ID；尚未形成 run 时为空字符串 |

状态采用唯一临时 MAT 和 `os.replace()` 原子更新。自动 `run` 期间，仅当 controller 状态或完整轮次变化时更新进度；完成或失败后写最终状态。独立 `stop` 命令若初始返回 `stopping`，后台 monitor 会按被停止命令的持久状态把该 stop 状态更新为 `completed`、`stopped` 或 `failed`。

## 5. 幂等、并发和重启

`status_<command_id>.mat` 是持久幂等记录。状态文件已经存在时，重复命令不再次调用设备或算法。自动 `run` 总是创建 `run_id == command_id` 的专属临时 run；即使从已有的分步校准继续，也会先把当前完整 snapshot 复制到新 run，再发生任何新的 RF 操作。其他分步命令通过状态中的 `run_id` 记录关联。

服务只恢复已经持久化的交付状态，不恢复 controller、runtime 或设备会话，也不重放硬件动作：

- 若 outbox 已有通过完整契约校验的结果，补写 `completed` 状态；
- 若 run 内已有有效 `final_result.mat` 缓存，补交 `completed` manifest、原子发布 outbox 结果，再补状态；
- 若 run 已是 `stopped` 或 `failed`，从 manifest 恢复对应命令终态；
- 若只有非终态状态/run 且没有有效最终缓存，将二者收敛到 `failed/service_restarted`；
- 若 outbox 和缓存都存在但均损坏，返回 `failed/recovery_artifact_invalid`。

只要存在状态、同 ID run 或结果证据，就不会把命令当作新命令执行。已完成的普通分步命令状态保持不变。

一个非 `stop` 命令由单 worker 执行。任务运行中到达的其他非 `stop` 命令立即写 `accepted=0, state=busy, error_code=busy`，不会排队后意外执行。`stop` 绕过该限制，直接调用 processor 的取消锁存和当前/pending controller 的线程安全停止入口；停止状态 monitor 绑定被停止的命令 ID，后续新 controller 不会改变旧 stop 命令的结果。

服务启动时先开始 watchdog，再扫描 inbox 中已经存在且文件名有效的完整命令。这样可接管启动前已经原子落盘、但尚无任何持久证据的命令。服务停止时先关闭新命令投送、停止并等待 observer，再请求当前 controller 停止、关闭 executor 并等待 worker 收尾；即使当时处于分步命令之间，只要 RF 仍在发射也会安全停止。同步和后台非 stop 入口都不会在关闭窗口新认领命令。

## 6. 运行记录与结果

配置和 `x` 都可用后，processor 以形成该运行的命令 ID 创建临时 run。替换配置、参考、复位或开始专属自动 run 时，只通过存储 API 终结旧 recorder，不会为了更新 manifest 而停止仍需复用的 controller。命令执行前后以及自动运行状态变化时同步 snapshot；完成、失败或停止后 manifest 成为终态。

成功的 `run` 自动生成 `result_<run-command-id>.mat`。分步流程在完成后通过 `export` 生成 `result_<export-command-id>.mat`。导出期间使用 run 的 export guard，契约详见 `storage_design.md`。
