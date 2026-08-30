# 反馈预处理设计

本文描述 `remote_dpd/preprocessing.py` 当前实现。预处理模块与设备调用、DPD runtime、任务状态和文件协议完全解耦。

## 1. 输入契约

`FeedbackPreprocessor` 在构造时固定原始周期参考波形 `x` 和采样率。`x` 必须是一维、非空、数值型、有限且具有非零 RMS 的复数向量；模块保存只读防御性副本。

每个 `CaptureBatch` 表示一次设备调用返回的连续周期数据：

| 字段 | 约束 |
| --- | --- |
| `iq` | 一维、非空、有限数值向量；内部保存为只读 `complex128` 副本 |
| `segment_length` | 正整数，处理时必须等于 `len(x)` |
| `segment_count` | 正整数，`len(iq)` 必须严格等于二者乘积 |
| `sample_rate_hz` | 正有限数，处理时必须与参考采样率一致 |
| `coherent_within_batch` | true 表示批次内各段复用同一时延和相位变换 |

模块不执行采样率转换。设备或未来设备边界必须按配置返回与 `x` 相同的采样率。

## 2. 处理顺序

一次 `process()` 接收一个或多个批次：

1. 检查批次类型、段长和采样率。
2. 使用周期相关将反馈段相对固定参考 `x` 对齐到 `1/32` 样点分辨率。实现把高分辨率相关拆成 32 个 fractional-offset 的原长度 IFFT；各 offset 共同覆盖完整周期的 `1/32` 样点网格，并按高分辨率索引顺序选择全局峰值。计算不会构造 32 倍长度的时域或频域数组。
3. 在时延对齐后根据复相关计算单位模相位旋转，只校正整体相位，不改变幅度。
4. `coherent_within_batch=true` 时只从第一段估计变换并复用于本批次其余段；false 时每段分别估计。不同批次始终分别估计。
5. 将全部已对齐段做复数相干平均，不做异常段判断或剔除。
6. 未传 `gain_correction` 时视为第 0 轮，计算正实数：

   ```text
   gain_correction = RMS(x) / RMS(aligned_average)
   ```

7. 后续轮次显式传回同一 `gain_correction`；模块只复用，不重新估计。
8. 输出 `z = aligned_average * gain_correction`。

这一区分保证每轮采集链路不稳定的时延和相位被消除，但第 0 轮之后的硬件幅度增益变化仍进入 `z`，由 DPD 算法补偿。

## 3. 输出与诊断

`PreprocessingResult` 保存只读 `z`、增益校正前的 `aligned_average`、固定增益及其 dB 值、RMS、段数和 NMSE。

每个 `BatchDiagnostic` 包含批次平均指标和全部 `SegmentDiagnostic`。逐段诊断记录：

- 该段是否实际估计了对齐参数；
- 应用的时延、单位模相位系数及其角度；
- 输入/对齐后 RMS 和对齐后 NMSE。

对 coherent 批次，后续段的诊断会记录复用的参数并将 `alignment_estimated` 标记为 false。

## 4. 错误边界

以下情况直接拒绝本轮，不生成部分结果：

- 没有批次、类型错误、长度或采样率不匹配；
- 任一输入或对齐结果含 NaN/Inf；
- 第 0 轮平均反馈 RMS 为零，无法计算增益；
- 外部传入的固定增益不是正有限实数；
- 增益应用后出现非有限值。

模块不负责异常段识别、重抓、设备重试、输出安全或任务终态，这些属于应用控制层。
