# 真机 RF bench 适配器设计

本文描述 `remote_dpd/real_bench.py` 实现的真实硬件适配器，使 `remote-dpd` 可以驱动本机
NI RFIC 测试站的完整闭环链路完成 ILC DPD 验证：

- 发射与接收：NI PXIe-5842 VST（`PXI2Slot2`），经 NI-RFSG / NI-RFSA 驱动 API；
- 功率测量：Agilent N1912A 功率计（`TCPIP0::192.168.255.40::inst0::INSTR`），经 PyVISA/SCPI；
- PA 主电源：Agilent N5767A（`GPIB1::5`），仅 44 V 输出关断控制；
- 辅助电源：Agilent E3648A ×2（`GPIB1::7/8`，GaN PA 偏置），只读，见电源红线。

适配器通过既有注册表以 `vst5842` 名称注册；上层（controller、预处理、ILC、Web）零改动，
`simulated` 与既有入口行为不变。模块顶层不 import 驱动包（延迟导入），未安装
`real-hardware` 可选依赖的环境仍可加载与注册本适配器，只有实际 `connect` 才需要驱动。

## 1. 电源控制红线（硬约束）

以下约束由用户明确给定，写入实现并由测试锁定：

1. 被测件为 GaN PA，偏置上电顺序错误会导致硬件烧毁。
2. E3648A ×2（8 V / 12 V 偏置）至始至终保持上电，只允许只读控制：适配器代码不实现
   任何针对 E3648A 的写方法；联锁校验只调用 `OUTP?` / `VOLT?` / `MEAS:VOLT?` 查询，
   且只在发射前临时打开资源、查完即关。
3. N5767A 是唯一允许写的电源，语义仅限 44 V 输出关断（`OUTP OFF`）。
4. 适配器不提供"开启 44 V"的任何代码路径：PA 上电由人工完成（先偏置后 44 V 的顺序由
   人工保证）。软件侧只做两件事：
   - 安全收尾（`safe_shutdown` / `disconnect` 路径）按配置发送 `OUTP OFF`；
   - 发射前只读联锁校验（两路偏置 `OUTP? == 1` 且实测电压与设定差 ≤ 0.5 V，否则拒绝发射）。
5. 下电顺序软件侧同样收敛为"只关 44 V"；偏置下电由人工在确认 44 V 已断开后执行。
6. 单元测试断言：E3648A 资源上只出现查询类命令；全部 VISA 会话中不存在 `OUTP ON`/
   `OUTP 1`；模块源码不包含输出使能字面量。

## 2. 模块结构

`remote_dpd/real_bench.py` 单文件模块：

```text
Vst5842RFBench(RFBench)          # 聚合对象，注册名 vst5842
├── _Vst5842Instrument           # 一体化核心，同时实现 Transmitter 与 Receiver
│   ├── nirfsg Session           # 波形下载、循环发射、电平（衰减映射）
│   └── nirfsa Session           # IQ 抓取（单次连续采集后按段构造批次）
├── _N1912APowerSensor           # PowerSensor，PyVISA 资源
└── _N5767ASupplyGuard           # 44 V 守卫（只读持有 + 安全收尾关断 + 发射前联锁）
```

- `transmitter` 与 `receiver` 返回同一个 `_Vst5842Instrument` 实例（契约允许一体机这样做）。
- `RFBench.connect()` 按顺序建立：RFSG/RFSA 驱动会话 → 功率计 VISA → 守卫 VISA；
  任一失败即回滚已建立的连接。守卫对 N5767A 的连接是只读持有，不做任何写。
- `configure()` 用 `VST5842_DEVICE_SCHEMA.validate_options()` 归一化选项后重建
  `DeviceConfig`，先停止发射再逐组件下发；发射前联锁检查通过
  `set_pretransmit_check()` 注入 `_Vst5842Instrument.start_transmission()`，
  只有 `enable_supply_interlock=true` 时安装。
- 适配器参数运行期不可变；`update_settings()` 在 schema 重新校验后整体替换，
  守卫在连接打开期间拒绝更换 `supply_resource`。
- 模块导入即注册（幂等），`device.py` 不反向依赖本模块，与 `simulated` 同构。

## 3. 能力实现映射

### 3.1 Transmitter（nirfsg）

| 契约方法 | 映射 |
| --- | --- |
| `upload_waveform(w)` | `_copy_waveform()` 精确拷贝（一维、非空、有限、complex128、只读）后 `write_arb_waveform("rdpdWave", data)` + `write_script()` 写循环脚本 `repeat forever / generate rdpdWave` |
| `start_transmission()` | 先执行注入的联锁检查（如启用）→ `output_enabled = True` → `initiate()` |
| `stop_transmission()` | `output_enabled = False`（RF 先断）→ `abort()`；未发射时为无害空操作 |
| `get_attenuation_db()` | `reference_power_dbm - rfsg.power_level` |
| `set_attenuation_db(v)` | 校验衰减范围后 `rfsg.power_level = reference_power_dbm - v`，发射中可调 |

衰减 0 dB 对应 RFSG 满电平工作点 `reference_power_dbm`（默认 `-17.0`，与 GUI 一致）。
衰减范围默认 `0/40 dB`，`initial_attenuation_db` 默认 `20 dB`（首发射约 -37 dBm，
对应 PA 输出约 +18 dBm，从远低于 +38 dBm 工作点的下方爬升）。

### 3.2 Receiver（nirfsa）

- `configure()` 设置 `acquisition_type=IQ`、`center_frequency`、`iq_rate`（与公共
  `sample_rate_hz` 一致）、`reference_level`（默认 `55.0`，与 GUI 一致，隐含 RX 前端
  外部衰减折算；反馈绝对标度由预处理第 0 轮固定增益校正吸收）、
  `start_trigger_type=NONE`（立即触发；周期对齐由预处理完成）。
- `capture(request)`：校验 `segment_length == 已上传波形长度` 且总样点不超过
  `max_capture_samples`；要求发射进行中；设 `number_of_samples` 后一次
  `read_iq_single_record_into()` 连续采集（complex64 缓冲，转 complex128 校验有限），
  按 `CaptureBatch` 构造，`coherent_within_batch=True`（同一次连续采集即同相干批次，
  满足预处理"批次内复用首段时延/相位"的假设）。
- `max_capture_samples` 是本地缓存常量（schema 字段，默认 64M 复样点 ≈ 512 MiB
  float32 IQ），不在 property 读取中做硬件 I/O。

### 3.3 PowerSensor（N1912A，PyVISA）

`measure_power_dbm()`：先写 `SENS1:FREQ = <center_frequency_hz>`（消除探头校准频率
与实际发射频率的偏差；写测量设置不属于电源红线范围）与 `SENS1:AVER:COUN = <平均次数>`
（默认 64，与站点当前一致），再 `READ1?` 读取。平均化 READ 会阻塞至完成，读取超时按
`max(call_timeout, average × 0.25 s + 5 s)` 放大；非有限读数 fail-closed 抛错。
测量点语义（用户确认）：探头直测 PA 输出，读数即准确 PA 输出功率，因此
`target_power_dbm` / `safety_power_limit_dbm` 均以 N1912A 读数为准。

### 3.4 44 V 守卫

- `connect()` 打开 N5767A 资源并只读持有（`enable_supply_shutdown=false` 时同样持有，
  只是不写）。
- `check_aux_supplies()`（发射前联锁）：逐台临时打开 E3648A 资源，仅查询
  `OUTP?` / `VOLT?` / `MEAS:VOLT?`，校验输出开启且电压偏差 ≤ 0.5 V，查完即关；
  失败立即抛错并阻止发射。
- `shutdown_drain()`：`enable_supply_shutdown=true` 时发送 `OUTP OFF`；这是守卫唯一的
  写命令，模块中不存在任何使能输出的代码路径（见 §1 红线）。

## 4. 适配器参数 schema

`VST5842_DEVICE_SCHEMA`（device_type `vst5842`，schema_version 1）：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `vst_resource` | string | `PXI2Slot2` | NI-RFSG/RFSA 设备名 |
| `reference_power_dbm` | number | `-17.0` | 衰减 0 dB 对应的 RFSG 输出电平 |
| `reference_level_dbm` | number | `55.0` | RFSA 参考电平（与 GUI 一致） |
| `power_meter_resource` | string | `TCPIP0::192.168.255.40::inst0::INSTR` | N1912A VISA 地址 |
| `power_meter_average` | integer | `64` | 功率计平均次数 |
| `supply_resource` | string | `GPIB1::5::INSTR` | N5767A VISA 地址 |
| `aux_supply_resources` | string array | `["GPIB1::7::INSTR","GPIB1::8::INSTR"]` | 只读联锁校验的偏置电源 |
| `enable_supply_shutdown` | boolean | `true` | 安全收尾时关闭 44 V |
| `enable_supply_interlock` | boolean | `true` | 发射前只读校验偏置在位 |
| `max_capture_samples` | integer | `64_000_000` | 单次抓取样点上限 |

模块同时导出 `VST5842_RECOMMENDED_CONFIG`：站点 5G NR 100 MHz 工作点基准
（1.84 GHz / 491.52 MS/s、`average_segment_count=16`、`target_power_dbm=38.0`、
`safety_power_limit_dbm=39.0`、衰减 `0/40 dB` 初始 `20 dB`、`settle_seconds=0.5`），
供冒烟脚本与文档引用；运行配置仍由命令显式提供。

## 5. 参考波形（临时过渡方案）

**定位**：本节为过渡措施。项目后续将另行开发数据源生成工具，届时参考波形由该工具直接
供给；适配器与上层不感知波形来源（契约只认 MAT 变量 `x`，见 `waveforms.py` 加载边界），
数据源工具对接同一 MAT 契约即可，无耦合。

过渡期做法：站点 GUI 使用 `NR_DL_TM1.1.tdms`（本机路径
`C:\Users\Public\Documents\National Instruments\RFIC SCPI\Waveforms\NR_DL_TM1.1.tdms`）。
实测该文件为 NI RFmx `InterleavedIQCluster` 布局：单 float32 通道交织 I/Q，
共 1,228,800 复样点 @ **122.88 MS/s**（完整 10 ms NR 帧），信号带宽 98.304 MHz，
PAPR 11.75 dB，`NI_RF_RuntimeScaling = -1.5`。

一次性转换脚本 `scripts/convert_tdms_to_mat.py`（临时工具，依赖 `nptdms` 临时安装，
不入正式依赖组；数据源工具上线后移除）执行三步显式、幅度保持的处理：

1. 解交织 I/Q 为复数序列；
2. 有理重采样 122.88 MS/s → 491.52 MS/s（精确 4× 上采样，`resample_poly`，
   通带覆盖信号带宽，增益为 1，不应用 RuntimeScaling）；
3. 从帧起始（slot 边界）截取默认 245,760 样点（491.52 MS/s 下 0.5 ms，即一个完整
   NR slot = 14 个 OFDM 符号），输出复数列向量 MAT `x`。

整帧 10 ms（4.915M 样点）会让控制器"参考长度 × 轮数 ≤ 2000 万样点"与
"每轮抓取 ≤ 1000 万样点"的上限只剩 2 轮余量，不实用；0.5 ms slot 段配合
`average_segment_count=16` 单轮约 3.9M 样点，留有充分余量。slot 边界截取保证段内
OFDM 符号完整；循环播放的拼接边界可能引入轻微谱再生，属 DPD 训练常规做法，
冒烟阶段以 ACLR 实测评估影响。

## 6. 依赖与部署

`pyproject.toml` 可选依赖组（不进入默认安装；npTDMS 不在此列，见 §5 临时方案）：

```toml
[project.optional-dependencies]
real-hardware = [
    "PyVISA>=1.15",
    "nirfsg>=1.2",
    "nirfsa>=1.0",
    "nitclk>=1.5",
]
```

NI-VISA 与 NI-RFSG/RFSA 系统驱动由本机 NI 软件栈提供（已确认安装），不从 PyPI 分发。
本机部署使用独立 venv（Python 3.13）+ `pip install -e .[real-hardware]`；
`cli.py` 无需改动——模块注册不依赖驱动安装。

## 7. 验证方式

1. **无硬件单测**（`tests/test_real_bench.py`，22 例，进 CI）：以注入 `sys.modules` 的
   fake `nirfsg`/`nirfsa`/`hightime`/`pyvisa` 模块实现——注册与 schema、
   configure 的驱动状态映射（SCRIPT 模式、频率、采样率、电平 `reference_power_dbm -
   attenuation`、参考电平、立即触发）、上传精确拷贝与循环脚本、发射前联锁顺序
   （偏置查询 → RF on → initiate）、停止顺序（RF off 先于 abort）、衰减双向映射与
   越界拒绝、抓取切段/相干标记/长度校验/超限拒绝/未发射拒绝、功率计校准频率与
   平均写入及读数解析、**电源红线**（E3648A 仅查询命令、全程无输出使能命令、
   `enable_supply_shutdown=false` 不写 OUTP OFF、源码无使能字面量）、联锁失败阻断
   发射、生命周期与资源释放。
2. **真机冒烟**（需用户释放 VST 会话后人工执行，逐步、全程不超过当前工作点电平）：
   连接与配置 → 20 dB 衰减发射 → 功率计核对（预期 ≈ +18 dBm）→ 按控制环爬升至
   +38 dBm 目标 → 抓取 IQ 并检查对齐/谱 → 停止发射与安全收尾。
3. **闭环验证**：冒烟通过后执行完整 ILC DPD 闭环（低 mu、少轮次先行）。

## 8. 边界与后续事项

1. VST 驱动会话为独占式：联调期间需停止 `NIRficScpiServer` / `InstrumentStudio` 对
   VST 的占用，窗口由用户安排；真机冒烟与闭环验证尚未执行（等待联调窗口）。
2. 5842 板载内存与抓取上限的保守值（64M 复样点）需联调时用驱动实际能力校准。
3. NR100 TM1.1 slot 段循环边界的谱再生影响需 ACLR 实测评估（§5）。
4. RFSG 侧 FIR/滤波与 TClk 精细同步未做专项配置，默认设置下若 ACLR 底噪异常，
   在联调阶段补充。
