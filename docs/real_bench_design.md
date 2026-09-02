# 真机 RF bench 适配器设计

本文描述 `remote_dpd/real_bench.py` 实现的真实硬件适配器，使 `remote-dpd` 可以驱动本机
NI RFIC 测试站的完整闭环链路完成 ILC DPD 验证：

- 发射与接收：NI PXIe-5842 VST，经 **NI RFIC SCPI 服务器**（loopback VXI-11 资源
  `TCPIP0::127.0.0.1::inst0::INSTR`，与 InstrumentStudio / 站点 MATLAB 客户端同一路径）；
- 功率测量：Agilent N1912A 功率计（`TCPIP0::192.168.255.40::inst0::INSTR`），经 PyVISA/SCPI；
- PA 主电源：Agilent N5767A（`GPIB1::5`），仅 44 V 输出关断控制；
- 辅助电源：Agilent E3648A ×2（`GPIB1::7/8`，GaN PA 偏置），只读，见电源红线。

适配器通过既有注册表以 `vst5842` 名称注册；上层（controller、预处理、ILC、Web）零改动，
`simulated` 与既有入口行为不变。模块顶层不 import `pyvisa`/`nptdms`（延迟导入），未安装
`real-hardware` 可选依赖的环境仍可加载与注册本适配器，只有实际 `connect` / 上传波形才需要。

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
   `OUTP 1`（RF 输出使能走 RFSG 专用命令 `SOURce:RFSG:OUTPut:ENABled`，与电源无关）；
   模块源码不包含输出使能字面量。

## 2. 为什么是 SCPI 服务器而非裸驱动 API

2026-09-02 联调在本机确认，裸 `nirfsg`/`nirfsa` 驱动会话在这套硬件上不可行：

1. 5842 + PXIe-5655 系统的接收 LO / 下变频资源由 NI RFIC 软件栈统一管理。直接 nirfsa
   会话会遇到 `downconverter_frequency_offset_mode=USER_DEFINED` 下 LO 卡死 6.5 GHz 的
   持久毒状态（`reset_device` 清不掉）、`SG_SA_SHARED`/`LO_IN` 的 LO 资源预约冲突、
   5655 Cross Switch 被占等问题。
2. RFIC SCPI 服务器（`NIRficScpiServer.exe`）是 NI 提供的官方共享路径，InstrumentStudio
   与站点 MATLAB 客户端（`labCtrlClient.m`）都通过它访问 VST。

因此收发统一走 SCPI 服务器，`nirfsg`/`nirfsa`/`nitclk` 依赖从可选组移除。

### 2.1 服务依赖链（联调环境前提）

SCPI 服务器需要按序拉起（异常时重启两者最干净；顺序错则服务器起不来）：

1. `C:\Program Files\National Instruments\Shared\NI gRPC Device Server\ni_grpc_device_server.exe
   server_config.json`（监听 31763）；
2. `C:\Program Files\National Instruments\RFIC Test Software\SCPI\NIRficScpiServer.exe`
   （必须以独立进程启动，重定向 stdin 会令其崩溃退出）。

适配器 `connect()` 时探测 loopback 资源并查询 `*IDN?`；不可达时报错并在错误信息中
给出上述启动顺序。IS 打开期间勿发 SCPI 测量命令（IS 与服务器可共存但测量互斥）；
用户桌面的功率计 / GPIB 常驻脚本可能与仪器读数竞争，联调异常读数多源于此。

### 2.2 数据格式根因（记录以防回退）

`FETCh:SPECan:RESult:IQ:TRACe:DATA?` 返回 `<t0>,<dt>,#<n><长度><二进制块>`，块内容为
**big-endian float32 交织 I/Q**（8 字节/样点；NI 文档 "Complex Blockdata Single
Precision"）。曾按小端 int16 解码产生"统计上伪装成平坦宽带噪声"的乱码，造成长期
"反馈无信号"误判。适配器解码固定 `np.frombuffer(payload, dtype=">f4")` 解交织。

TX 不发射时 FETCh 返回空块（SCPI 采集与发射绑定，非错误）；适配器将空块与短块都
fail-closed 抛错。读取块数据时必须临时清除 `read_termination`（块内 0x0A 字节会被
`\n` 终止符截断），读完恢复。

### 2.3 波形播放行为（2026-09-02 联调确认）

- **缓存与板载内存是两级**：`MMEMory:LOAD:WAVeform` 只把 TDMS 读入服务器缓存；
  `SOURce:RFSG:LOAD:WAVeform:MEMory <名>`（= niRFSGPlayback Download User Waveform）
  才把缓存波形下载到 RFSG 板载内存——缺这一步 `INITiate:RFSG` 报"波形不存在"。
- **波形名字符约束**：服务器把名字编入播放脚本，只接受字母数字（下划线等符号报
  "invalid character"）；缓存目录 `MEMory:WAVeform:CATalog?` 对名字做大写化处理。
  适配器校验 `waveform_name` 仅含 ASCII 字母数字并统一大写。
- **ARB 重选需先停止生成任务**：`SELect`/`GMODe` 在任务 running 时被拒；upload 在
  生成任务已发起时先 `ABORt:RFSG`，configure/disconnect 无条件 ABORt 兜底（对 idle
  任务为无害 no-op），下次 start 重新 `INITiate`。
- **RMS 归一化（重要）**：服务器把任意上传波形归一化到 `SOURce:RFSG:POWer:LEVel`
  播放——波形整体幅度缩放不改变实际输出功率（实测 0.5× 波形功率不变、频谱形状完整
  保留）。因此 ILC 的**形状修正**完整生效，而**增益项**由功率电平与预处理固定增益
  校正承担；`reference_power_dbm - attenuation` 的衰减映射保持精确（实测每 1 dB
  衰减对应 PA 输出 1 dB）。

## 3. 模块结构

`remote_dpd/real_bench.py` 单文件模块：

```text
Vst5842RFBench(RFBench)          # 聚合对象，注册名 vst5842
├── _Vst5842Instrument           # 一体化核心，同时实现 Transmitter 与 Receiver
│   └── PyVISA loopback SCPI 会话    # RFSG 波形发射 + RFmx SpecAn IQ 抓取
├── _N1912APowerSensor           # PowerSensor，PyVISA 资源
└── _N5767ASupplyGuard           # 44 V 守卫（只读持有 + 安全收尾关断 + 发射前联锁）
```

- `transmitter` 与 `receiver` 返回同一个 `_Vst5842Instrument` 实例（契约允许一体机这样做）。
- `RFBench.connect()` 按顺序建立：SCPI 会话（含服务探测）→ 功率计 VISA → 守卫 VISA；
  任一失败即回滚已建立的连接。守卫对 N5767A 的连接是只读持有，不做任何写。
- 所有下发的 SCPI 写命令后跟 `SYSTem:ERRor?` 轮询（服务器对非法命令不抛异常，只入
  错误队列）；非零错误码立即抛 `RuntimeError`（含命令与错误文本），fail-closed。
- `configure()` 用 `VST5842_DEVICE_SCHEMA.validate_options()` 归一化选项后重建
  `DeviceConfig`，先停止发射再逐组件下发；发射前联锁检查通过
  `set_pretransmit_check()` 注入 `_Vst5842Instrument.start_transmission()`，
  只有 `enable_supply_interlock=true` 时安装。
- 适配器参数运行期不可变；`update_settings()` 在 schema 重新校验后整体替换，
  守卫在连接打开期间拒绝更换 `supply_resource`。
- 模块导入即注册（幂等），`device.py` 不反向依赖本模块，与 `simulated` 同构。

## 4. 能力实现映射

### 4.1 Transmitter（RFSG over SCPI）

| 契约方法 | 映射 |
| --- | --- |
| `upload_waveform(w)` | 校验拷贝（一维、非空、有限、complex128、只读）→ 若生成任务已发起先 `ABORt:RFSG` → 写 NI RFmx 风格 TDMS 到 `waveforms_directory/rdpd_wave.tdms`（§4.2）→ `MMEMory:LOAD:WAVeform "<绝对路径>","<名>",0`（磁盘→缓存）→ `SOURce:RFSG:LOAD:WAVeform:MEMory "<名>"`（缓存→板载内存，必需）→ `SOURce:RFSG:WAVeform:REPeat:MODE "<名>",CONTINUOUS` → `SOURce:RFSG:ARB:WAVeform:SELect "<名>"` → `MEMory:WAVeform:DX?` 回读采样周期并校验等于 `1/sample_rate_hz` |
| `start_transmission()` | 先执行注入的联锁检查（如启用）→ 未发起（或 ABORt/configure 重载后）`INITiate:RFSG` → `SOURce:RFSG:OUTPut:ENABled 1` |
| `stop_transmission()` | `SOURce:RFSG:OUTPut:ENABled 0`；不 ABORt（快速暂停语义；upload 需要重选时自行 ABORt）。未发射时为无害空操作 |
| `get_attenuation_db()` | `reference_power_dbm - SOURce:RFSG:POWer:LEVel?` |
| `set_attenuation_db(v)` | 校验衰减范围后 `SOURce:RFSG:POWer:LEVel = reference_power_dbm - v`，发射中可调 |

衰减 0 dB 对应 RFSG 满电平工作点 `reference_power_dbm`（默认 `-17.0`，与 GUI 一致）。
衰减范围默认 `0/40 dB`，`initial_attenuation_db` 默认 `20 dB`（首发射约 -37 dBm，
对应 PA 输出约 +18 dBm，从远低于 +38 dBm 工作点的下方爬升）。

### 4.2 TDMS 波形写入（nptdms）

上传的任意 numpy 复波形按站点原生 `NR_DL_TM1.1.tdms` 的结构写出（NI RFmx 波形
TDMS，已实测该布局服务器可加载）：

- group `waveforms`：属性 `Application="NI-RFmx Waveform Creator"`、
  `NI_RF_WaveformFileVersion="2.0.0"`；
- channel `Channel 0`：单通道 **float32 交织 I/Q**（`[Re, Im, Re, Im, ...]`），
  属性 `NI_RF_IQRate = sample_rate_hz`、`NI_RF_WaveformType="InterleavedIQCluster"`、
  `NI_RF_SignalBandwidth = 0.8 × 采样率`、`NI_RF_PAPR`（按数据实算）、
  `NI_RF_RuntimeScaling = 0.0`（上层参考 `x` 即最终发射样本，不做任何额外缩放）、
  `dt = 1/采样率`（**复样点周期**，与原生文件一致；`MEMory:WAVeform:DX?` 直接回读
  该值，适配器据此校验加载结果）、`t0 = 0`。

ILC 每轮迭代重新上传候选波形（同长度），上传路径即"写 TDMS → 服务器加载"，
与波形来源零耦合（上层只认 MAT `x` 契约）。`MEMory:LOAD:WAVeform` 支持绝对路径，
文件名固定 `rdpd_wave.tdms` 每次覆盖。

### 4.3 Receiver（RFmx SpecAn IQ over SCPI）

- `configure()` 加载 InstrumentStudio 导出的接收配置
  `MMEMory:INSTr:LOAD:STATe "Instrument_2_PXI2Slot2.rfmxconfig",1`（内含 ACP+IQ 使能、
  参考电平与外部衰减基线、PXI 参考时钟、无触发自由运行采集），随后显式下发
  `CONFigure:SPECan:FREQuency` / `RLEVel` / `EATTenuation`（与 rfmxconfig 基线一致，
  保证 schema 参数真实生效）、`SOURce:RFSG:FREQuency` / `GMODe ARBWAVEFORM` /
  `POWer:LEVel`。周期对齐由预处理时延估计完成，无需硬件触发。
- `capture(request)`：校验 `segment_length == 已上传波形长度` 且总样点不超过
  `max_capture_samples`；要求发射进行中；下发
  `CONFigure:SPECan:MEASurement:SELect 1,IQ`、`CONFigure:SPECan:IQ:ACQuisition:TIME =
  (样点数 + 32 guard) / sample_rate`（服务器按整样点取整，guard 保证不短）、
  `CONFigure:SPECan:IQ:SRATe = sample_rate_hz`、`INITiate:SPECan`，等待采集时长加固定
  结算余量（2 s）后 `FETCh:SPECan:RESult:IQ:TRACe:DATA?`；按 §2.2 解码大端块为
  complex128，截取请求样点数构造 `CaptureBatch`，`coherent_within_batch=True`
  （单次连续采集即同相干批次，满足预处理"批次内复用首段时延/相位"的假设）。
- `max_capture_samples` 是本地缓存常量（schema 字段，默认 64M 复样点 ≈ 512 MiB
  float32 IQ），不在 property 读取中做硬件 I/O。

### 4.4 PowerSensor（N1912A，PyVISA）

`measure_power_dbm()`：先写 `SENS1:FREQ = <center_frequency_hz>`（消除探头校准频率
与实际发射频率的偏差；写测量设置不属于电源红线范围）与 `SENS1:AVER:COUN = <平均次数>`
（默认 64，与站点当前一致），置 `INIT1:CONT ON` 后等一个平均周期，用 `FETC1?`
取最近完成的平均读数（N1912A 在连续初始化模式下 `READ1?` 会永久挂起，不可用）；
非有限读数 fail-closed 抛错。测量点语义（用户确认）：探头直测 PA 输出，读数即准确
PA 输出功率，因此 `target_power_dbm` / `safety_power_limit_dbm` 均以 N1912A 读数为准。

### 4.5 44 V 守卫

- `connect()` 打开 N5767A 资源并只读持有（`enable_supply_shutdown=false` 时同样持有，
  只是不写）。
- `check_aux_supplies()`（发射前联锁）：逐台临时打开 E3648A 资源，仅查询
  `OUTP?` / `VOLT?` / `MEAS:VOLT?`，校验输出开启且电压偏差 ≤ 0.5 V，查完即关；
  失败立即抛错并阻止发射。
- `shutdown_drain()`：`enable_supply_shutdown=true` 时发送 `OUTP OFF`；这是守卫唯一的
  写命令，模块中不存在任何使能输出的代码路径（见 §1 红线）。

## 5. 适配器参数 schema

`VST5842_DEVICE_SCHEMA`（device_type `vst5842`，schema_version 2；v1 的
`vst_resource` 驱动名已随 SCPI 重构移除）：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `scpi_resource` | string | `TCPIP0::127.0.0.1::inst0::INSTR` | RFIC SCPI 服务器 loopback 资源 |
| `instrument_config_file` | string | `Instrument_2_PXI2Slot2.rfmxconfig` | configure 时加载的 IS 导出接收配置 |
| `reference_power_dbm` | number | `-17.0` | 衰减 0 dB 对应的 RFSG 输出电平 |
| `reference_level_dbm` | number | `50.0` | SpecAn 参考电平（PA 输出尺度，与 GUI 一致） |
| `external_attenuation_db` | number | `53.5` | PA 输出至 VST 输入的外部衰减（GUI 一致；读数标度经此校正） |
| `waveform_name` | string | `RDPD1` | 服务器波形名（仅字母数字，统一大写；见 §2.3） |
| `waveforms_directory` | string | `C:\Users\Public\Documents\National Instruments\RFIC SCPI\Waveforms` | 上传 TDMS 写入目录 |
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

## 6. 参考波形（临时过渡方案）

**定位**：本节为过渡措施。项目后续将另行开发数据源生成工具，届时参考波形由该工具直接
供给；适配器与上层不感知波形来源（契约只认 MAT 变量 `x`，见 `waveforms.py` 加载边界），
数据源工具对接同一 MAT 契约即可，无耦合。

过渡期做法：站点 GUI 使用 `NR_DL_TM1.1.tdms`（本机路径
`C:\Users\Public\Documents\National Instruments\RFIC SCPI\Waveforms\NR_DL_TM1.1.tdms`）。
实测该文件为 NI RFmx `InterleavedIQCluster` 布局：单 float32 通道交织 I/Q，
共 1,228,800 复样点 @ **122.88 MS/s**（完整 10 ms NR 帧），信号带宽 98.304 MHz，
PAPR 11.75 dB，`NI_RF_RuntimeScaling = -1.5`。

一次性转换脚本 `scripts/convert_tdms_to_mat.py`（临时工具；数据源工具上线后移除）
将 TDMS 转为项目 MAT 契约 `x`：解交织 → 有理重采样 122.88 → 491.52 MS/s（精确 4×，
增益 1，不应用 RuntimeScaling）→ 截取 245,760 样点（491.52 MS/s 下一个完整 0.5 ms
NR slot = 14 个 OFDM 符号）输出复数列向量。

注意：**SCPI 路径发射时不需要该 4× 上采样**——适配器直接按 491.52 MS/s 写 TDMS
（`NI_RF_IQRate` 携带采样率，服务器自行播放），转换脚本只服务于上层参考 `x` 的
生成。发射波形与上层参考都在同一 491.52 MS/s / 245,760 样点域，`segment_length`
契约语义与仿真一致。

整帧 10 ms（4.915M 样点）会让控制器"参考长度 × 轮数 ≤ 2000 万样点"与
"每轮抓取 ≤ 1000 万样点"的上限只剩 2 轮余量，不实用；0.5 ms slot 段配合
`average_segment_count=16` 单轮约 3.9M 样点，留有充分余量。slot 边界截取保证段内
OFDM 符号完整；循环播放的拼接边界可能引入轻微谱再生，属 DPD 训练常规做法，
冒烟阶段以 ACLR 实测评估影响。

## 7. 依赖与部署

`pyproject.toml` 可选依赖组（不进入默认安装）：

```toml
[project.optional-dependencies]
real-hardware = [
    "PyVISA>=1.15",
    "nptdms>=1.11",
]
```

NI-VISA、NI RFIC Test Software（含 SCPI 服务器与 gRPC 设备服务器）由本机 NI 软件栈
提供（已确认安装），不从 PyPI 分发。本机部署使用独立 venv（Python 3.13）+
`pip install -e .[real-hardware]`；`cli.py` 无需改动——模块注册不依赖驱动安装。

## 8. 验证方式与 2026-09-02 真机冒烟结果

1. **无硬件单测**（`tests/test_real_bench.py`，40 例，进 CI）：以注入 `sys.modules` 的
   fake `pyvisa`（可编程 SCPI/仪器资源：命令记录、错误队列、大端块响应、read_termination
   语义）与 fake `nptdms`（记录写出的组/通道/属性/数据）实现——注册与 schema v2、
   connect 服务探测与启动顺序提示、configure 完整命令序列（含兜底 ABORt）与逐命令
   错误轮询/错误传播、上传 TDMS 结构（交织 float32、`NI_RF_IQRate`、`dt`、
   `RuntimeScaling=0`）与缓存/板载绑定/选择/DX 校验序列及 ABORt 时序、波形名字符集
   校验与大写化、真实 nptdms 写出文件 round-trip（装了 nptdms 才跑）、发射前联锁顺序
   （偏置查询 → INITiate → 输出使能）、停止只关门控、configure 重载后重新 initiate、
   衰减双向映射与越界拒绝、抓取 **大端解码**/回显请求/guard 截断/短块与空块
   拒绝/长度校验/超限拒绝/未发射拒绝、功率计频率与平均写入及 FETC 读数、
   **电源红线**（E3648A 仅查询命令、全程无输出使能命令、`enable_supply_shutdown=false`
   不写 OUTP OFF、源码无使能字面量）、生命周期与资源释放。
2. **真机冒烟（已通过，2026-09-02）**：
   - 衰减爬升线性：衰减 25→5 dB 每步 PA 输出实测 +5 dB（14.32/19.66/24.72/29.50/
     34.03 dBm），全程低于 +39 dBm 安全限；
   - 抓取质量：单次连续采集段相干 0.9997（0.5 ms 波形周期）；带内/带外频谱对比
     +43.3 dB（NR100 形状清晰）；
   - 波形内容切换：NR ↔ 10 MHz 单音往返切换，反馈谱峰精确跟随（+10.001 MHz）；
   - 闭环 ILC DPD：功率自动调谐 19 步爬至 +34.83 dBm（目标 35，gap +0.17 dB），
     标定建立增益校正，3 轮迭代（mu=0.1）error_rms 单调下降
     （0.01265 → 0.01152 → 0.01053），干净收尾（TX off、44 V 未触碰）。
3. SCPI 服务器状态机易被错误序列污染：现象异常时先重启两个服务进程（先 gRPC 后
   SCPI 服务器）再复测，勿反复盲试命令。

## 9. 边界与后续事项

1. 波形热切换已验证（NR↔单音谱峰切换 + 闭环 3 轮迭代逐轮重上传）；注意 RMS 归一化
   行为（§2.3）——发射波形幅度缩放不改变输出功率，判定播放内容须用频谱形状类判据。
2. `NI_RF_SignalBandwidth` / `NI_RF_PAPR` 为信息性属性，按 0.8 × 采样率与实算 PAPR
   写入；如服务器播放行为与这些属性相关，联调时校正。
3. NR100 TM1.1 slot 段循环边界的谱再生影响需 ACLR 实测评估（§6）。
4. 板载内存与抓取上限的保守值（64M 复样点）已按 SCPI 块传输可行性设定，联调若遇
   传输瓶颈再收紧。
5. 用户桌面常驻脚本（功率计/GPIB）与仪器读数偶发竞争（冒烟中 FETC1? 出现过双读数
   拼接），正式闭环以 N1912A 读数做安全判据时注意复核合理性。
