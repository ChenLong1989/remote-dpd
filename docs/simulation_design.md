# 仿真 RF Bench 设计

本文描述 `remote_dpd/simulation.py` 的当前实现。`SimulatedRFBench` 同时实现 `RFBench`、`Transmitter`、`Receiver` 和 `PowerSensor`，用于在没有射频硬件时验证与真实设备相同的闭环调用边界。

设备注册表内置 `simulated` 名称；`create_rf_bench("simulated")` 每次返回一个新的独立 `SimulatedRFBench`。

## 1. 生命周期

仿真设备使用严格顺序：

```text
connect -> configure -> upload_waveform -> start_transmission
                                      ^             |
                                      +-- stop -----+
```

- `configure()` 和 `upload_waveform()` 仅允许在停止发射时调用。
- `set_attenuation_db()` 可以在发射中调用，用于第 0 轮功率调节。
- `capture()` 和 `measure_power_dbm()` 仅允许在已有波形且正在发射时调用。
- `safe_shutdown()` 幂等停止 RF 输出但保留连接和配置；`disconnect()` 同时清理波形、配置和仿真状态。
- 所有可能阻塞的方法都要求正有限 `timeout_seconds`，与未来真实设备契约一致。

每次重新配置都会恢复初始衰减、清除已上传波形并按配置随机种子重建噪声生成器。

## 2. 动态配置 schema

`SIMULATED_DEVICE_SCHEMA` 的设备类型为 `simulated`，版本为 2，公开以下可由后续网页动态编辑的字段：

| 字段 | 当前默认值 | 含义 |
| --- | --- | --- |
| `pa_coefficients` | 两个一阶项和两个三阶项 | 全部复系数，每行含正奇数 `p`、非负 `m`、`real` 和 `imag` |
| `system_gain_db` | `-6.0` | 反馈采集路径固定幅度增益 |
| `system_phase_rad` | `0.35` | 反馈采集路径固定整体相位 |
| `delay_samples` | `2.25` | 周期小数时延 |
| `noise_dbfs` | `-80.0` | 复高斯噪声的复包络 RMS |
| `random_seed` | `42` | 可复现噪声随机种子 |
| `power_reference_dbm` | `1.0` | 无噪 PA 输出 RMS 为 1 时的功率标定值 |
| `max_capture_samples` | `1000000` | 单次 `capture()` 最大样点数 |

schema 拒绝未知字段、空 PA 系数表、偶数/非正阶数、负记忆深度、非有限数和越界值。`DeviceConfig` 中未填写的仿真专属字段由 schema 补入默认值，生效配置与调用方对象完全分离。

## 3. 发射和 PA 模型

上传波形按原样保存，不执行 AGC、归一化或削峰。当前 TX 衰减先作用于数字波形：

```text
u[n] = y[n] * 10 ** (-attenuation_db / 20)
```

PA 使用周期边界复系数记忆多项式：

```text
pa[n] = sum(a[p,m] * u[n-m] * abs(u[n-m]) ** (p-1))
```

索引通过 `np.roll()` 循环，因此段首与段尾连续，符合周期波形闭环假设。任意配置导致的非有限输出会作为仿真错误拒绝，而不是裁剪。

默认 PA 系数固定为：

| `p` | `m` | 系数 |
| ---: | ---: | ---: |
| 1 | 0 | `1.0+0.0j` |
| 1 | 1 | `0.04+0.015j` |
| 3 | 0 | `-0.36+0.075j` |
| 3 | 1 | `-0.06+0.03j` |

在当前 `10×20 MHz @ 491.52 MS/s` 默认 waveform、`-10 dBm` 目标和 Web quick-start 的 `mu=0.35`、15 次 ILC 下，功率调节锁定约 `1.6 dB`。以左右最外侧 20 MHz 载波为 main、紧邻外侧 20 MHz 为 adjacent，第 0 轮分别为 `-30.73/-31.80 dBc`，第 15 轮为 `-57.98/-59.83 dBc`；NMSE 从约 `-27.19 dB` 改善到 `-54.89 dB`，最终数字峰值约 `0.985`。该结论只保证当前默认 waveform 和预设，不保证用户任意系数组合收敛。

## 4. 反馈与抓取

每个反馈周期依次应用 PA、固定反馈路径增益、固定相位和周期小数时延。一次 `capture()` 将该周期连续重复到请求段数，再添加由固定种子驱动的独立复高斯噪声。

返回的 `CaptureBatch` 总是标记 `coherent_within_batch=true`，表示同一次抓取的所有段共享时延和相位。不同 `capture()` 调用会继续消耗同一随机序列；重新应用相同配置后从同一随机序列起点重新开始。

请求段长必须严格等于当前上传波形长度，总样点数不得超过 `max_capture_samples`。应用控制器负责在超过限制时拆成多个批次。

## 5. 功率测量

功率传感器测量无噪 PA 输出，不包含反馈采集路径的 `system_gain_db`、相位、时延或噪声：

```text
power_dbm = power_reference_dbm + 20 * log10(RMS(pa))
```

因此降低 TX 衰减会提高实测功率，功率调节器可以按真实设备方向从较大衰减向目标逼近。零 RMS 输出返回负无穷，随后由功率安全层作为非有限无效测量拒绝。
