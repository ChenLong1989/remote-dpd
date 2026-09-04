# 运行存储与最终结果设计

本文描述 `remote_dpd/storage.py` 和 `remote_dpd/result_export.py` 的当前实现。临时运行记录面向排错、历史浏览和后续网页展示；正式 MAT 只保存一次已完成任务的最终结果。

## 1. 临时目录

`RunStore` 在受控根目录下只管理 `runs` 子目录：

```text
<runtime-root>/runs/<run_id>/
├── manifest.json
├── config.json
├── x.npy
├── events.json
├── power_trace.json
├── snapshot.json
├── final_result.mat          # 仅完成任务
└── iterations/
    └── 000000/
        ├── record.json
        ├── y.npy
        ├── z.npy
        └── aligned_average.npy
```

`create_run()` 保存实际生效的 `ClosedLoopConfig` 和 controller 已完成 RMS conditioning 的生效参考 `x`，返回创建后立即处于 active 清理保护的 `RunRecorder`。源 MAT 幅度不复制到 run；归一化来源由配置和 snapshot 报告追溯。默认 `run_id` 为 UUID；外部指定值只能包含受限 ASCII 字符，不能构造路径层级。

## 2. Manifest 和完整迭代

manifest schema 当前为 `1.0`，固定包含：

- `run_id`、`status` 和可空 `device_type`；
- UTC ISO 8601 的 `created`、`updated` 和可空 `completed`；
- 配置、参考、事件、功率轨迹、最新 snapshot 和可恢复最终结果的相对路径；
- 已提交迭代的有序 artifact 索引。

`record_snapshot()` 将新增内容按 append-only 语义同步：

- 状态转换和结构化错误进入 `events.json`；
- 初始调功率的完整轨迹进入 `power_trace.json`；
- 每个 `IterationRecord` 保存 `y`、`z`、增益应用前 `aligned_average` 和完整 JSON 指标；
- snapshot 元数据包含当前状态、固定增益、锁定衰减、最新功率、reference normalization 报告和最后错误。

同一 snapshot 重放不改文件时间。相同轮次若出现不同波形或元数据，抛出 `RunConflictError`，不会覆盖已提交 artifact。运行达到 `completed`、`failed` 或 `stopped` 且 manifest 成功落盘后，recorder 自动释放其初始 active guard。

完成任务使用两阶段提交：先把包含全部迭代索引、最终 snapshot 和设备类型的 manifest 写成 `finalizing`，再原子生成 `final_result.mat` 缓存，最后提交 `completed` manifest。进程若在中间退出，重启逻辑可以根据缓存补交完成状态；没有有效缓存的 `finalizing` run 只会失败收尾，不会重放设备操作。`mark_terminal()` 可在不改变 controller 或硬件状态的情况下把被替换、被中断的 recorder 收敛到终态。

临时存储不保存原始 `raw_capture`，因为当前控制器只在内存中持有预处理结果。

## 3. 原子性与路径安全

JSON、NumPy 和 MAT artifact 都先写同目录唯一临时文件，再使用 `os.replace()` 原子替换；JSON/NumPy 写入还执行 flush/fsync。不可变迭代文件若已存在会比较内容，相同则幂等，不同则冲突。

存储根、`runs`、run 目录和 iteration 目录必须是真实目录且解析后仍位于受控父目录。清理和读取忽略未知目录、损坏 manifest 和符号链接；若运行期间 `runs` 根被替换或逃逸，操作直接失败，不触碰替代目标。

## 4. 保留与周期清理

默认保留时间为 7 天，周期为 24 小时。指向同一个规范化 runs root 的多个 `RunStore` 实例共享锁、active guard 和 export guard，读取整组 run 元数据或执行崩溃恢复时也持有 active guard，避免另一个实例的清理与读/恢复/导出交错。`cleanup_expired()` 只删除：

- manifest 结构和 `run_id` 均有效；
- `updated` 已达到保留边界；
- 当前没有 active guard；
- 当前没有 export guard；
- 仍是受控 `runs` 的直接真实子目录。

`start_cleanup()` 创建最多一个 daemon 清理线程，重复启动无副作用；单次清理异常会被记录并在下一周期重试。`stop_cleanup()`/`close()` 可重复调用。

## 5. 正式 MAT

`export_final_mat()` 只接受 `COMPLETED` snapshot，并验证：

- 存在第 0 轮和最终实际评价轮；
- 最终轮等于 `max_iterations`；
- `x`、最终 `y`、最终 `z` 等长且有限；
- 第 0 轮 `y` 与由 `(x, 生效配置)` 确定性重建的 ILC 种子波形（默认 `x`+种子噪声，见 `controller_design.md` §1.1；关闭噪声时即 `x`）逐位一致；
- 数字安全报告与最终 `y` 一致；
- 最终预处理仍复用第 0 轮固定增益；
- NMSE、功率、衰减和抓取计数均有效。

新生成正式文件 schema 为 3，固定变量如下；reader 和崩溃恢复仍接受既有 schema 1/2 缓存。

| 变量 | 类型与含义 |
| --- | --- |
| `schema_version` | 新文件为整数 3；旧文件 1/2 只读兼容 |
| `x` | 复数列向量，RMS conditioning 后的生效参考 |
| `y` | 复数列向量，最终已发射并评价波形 |
| `z` | 复数列向量，对应最终预处理反馈 |
| `metrics` | MATLAB struct，包含最终轮、NMSE、数字 RMS/峰值、物理功率、衰减、固定增益、抓取计数、source/effective RMS dBFS 和 reference scale dB |
| `config` | 严格 JSON 字符串，保存实际生效配置；MATLAB 使用 `jsondecode` 读取 |
| `status` | `completed` |
| `completed_at` | UTC ISO 8601 字符串 |

`config` 顶层包含设备注册名 `device_type`，其余字段与实际生效的 `ClosedLoopConfig` 一致，包括归一化开关与目标，以及 schema 3 新增的四项 ILC 种子噪声字段（`seed_noise_enabled/psd_db/bandwidth_hz/seed`），可直接作为新文件入口的配置重新解析。schema 2 config 缺少种子噪声字段，schema 1 config/metrics 没有归一化字段，均按各自 legacy 字段集只读；已有完成缓存仍可补交。`completed_at` 来自 controller 实际进入终态的时刻，不是用户稍后点击导出的时刻。

正式结果不包含迭代历史。正常完成时同一内容先进入 run 内的可恢复缓存；文件入口在 `RunStore.export_guard()` 保护下把缓存原子发布到 outbox。无存储的程序化调用仍可直接原子导出；失败不会留下部分正式结果。
