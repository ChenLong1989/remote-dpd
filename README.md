# remote-dpd

`remote-dpd` 是一个不依赖 MATLAB 的设备直控 ILC DPD 闭环服务。当前版本提供：

- 可扩展的发射、接收、功率测量和组合式 `RFBench` 契约；
- 带有记忆多项式 PA、TX 衰减、反馈扰动和功率测量的确定性仿真设备；v2 默认 PA 在当前 10×20 MHz waveform 上产生约 `-30 dBc` 初始外邻道失真；
- 独立反馈预处理、基础 ILC runtime、数字波形安全和物理功率安全；
- 分步或自动的单任务闭环控制器；
- 自动清理的完整临时运行记录和最终 MAT 正式结果；
- 版本化 inbox/outbox MAT 文件命令入口；
- 与文件入口共享单任务仲裁、具备专业 RF 分析工作区的可信网络 Web 控制台。

当前唯一内置设备是 `simulated`。真实仪器适配器可通过相同设备注册表逐个增加；Web 页面按设备 schema 动态生成专属配置，因此后续真实设备不需要复制控制台流程。

## 安装与启动

项目要求 Python 3.10 或更高版本：

```bash
python -m pip install -e .
remote-dpd --exchange-root /opt/remote-dpd/exchange
```

默认临时运行数据位于 `<exchange-root>/runtime`，保留 7 天并每 24 小时清理一次。可以显式覆盖：

```bash
remote-dpd \
  --exchange-root /opt/remote-dpd/exchange \
  --runtime-root /var/lib/remote-dpd \
  --retention-days 7 \
  --cleanup-interval-seconds 86400
```

使用 `--once` 可同步扫描当前 inbox 后退出，适合测试和批处理：

```bash
remote-dpd --exchange-root /tmp/remote-dpd-exchange --once
```

## Web 控制台

Web 模式默认只监听 `127.0.0.1`，同时保留同一进程的 MAT inbox watcher：

```bash
remote-dpd \
  --exchange-root /tmp/remote-dpd-exchange \
  --mode web \
  --waveform-root /data/dpd-waveforms \
  --web-port 8000
```

浏览器打开 `http://127.0.0.1:8000/`。页面支持：

- 受限 waveform root 内的 MAT `x` 浏览、安全 preview 和加载；
- 固定单屏显示频谱、核心结果、RF 状态和单一辅助测量 pane，页面本身无需滚动；
- 弹窗式公共设备配置、动态专属 schema、仿真 PA 系数和 measurement-band 编辑；
- connect/disconnect、load/configure、start/stop TX、power tune、calibrate、ILC step、reset/export 分步操作和一键自动闭环；
- 默认 simulated bench 与首个 waveform 的一键完整闭环、固定 RF abort、controller/channel 状态、功率调节和对齐诊断；
- 全新页面从服务端获得确定的 Web quick-start profile，不读取浏览器存储或历史 run；当前 profile 使用 `491.52 MS/s`、十段平均、1000 万单次抓取上限、`mu=0.35` 和 15 次 ILC；
- 完整周期 `Z₀/Zₙ/Eₙ` 频谱、Trace/Marker、默认 10TX+2 adjacent ACLR 模板、分 reference 的 dBc/channel-power bar、PAPR、AM/AM、AM/PM 和 NMSE 改善；
- 临时 run、任意历史轮次重新分析、结构化事件和最终 MAT 下载。

可信局域网可显式启动：

```bash
remote-dpd \
  --exchange-root /tmp/remote-dpd-exchange \
  --mode web \
  --waveform-root /data/dpd-waveforms \
  --web-host 0.0.0.0 \
  --web-allowed-host 192.168.3.100 \
  --web-port 8765
```

此时从 `http://192.168.3.100:8765/` 访问。非 loopback bind 必须显式提供至少一个私网 `--web-allowed-host`；参数可重复，通配符和公网地址被拒绝。loopback Host 始终保留。

Web 与外部 MAT 命令共用同一个普通命令 worker 和 stop latch，任一入口繁忙时另一入口不会绕过单任务边界。控制台没有登录鉴权或 TLS；可信 LAN 内能访问端口的主机可以控制 RF 任务和下载结果，因此不得用于不可信网络、反向代理或公网部署。服务保留精确 Host、同源 Origin、JSON、自定义控制头和 CSP 校验，不启用 CORS，也不自动修改系统防火墙。完整契约见 [`docs/web_console_design.md`](docs/web_console_design.md)。`--once` 只适用于默认 `file` 模式。

## 文件命令入口

服务使用以下目录：

```text
<exchange-root>/
├── inbox/
│   └── command_<command_id>.mat
├── outbox/
│   ├── status_<command_id>.mat
│   └── result_<command_id>.mat
└── runtime/
    └── runs/<run_id>/
```

命令文件公共变量为：

- `schema_version=1`；
- `command_id`，必须与文件名一致；
- `action`；
- 按动作可选的参考波形 `x` 和严格 JSON 字符串 `config_json`。

支持动作：`connect`、`disconnect`、`load`、`configure`、`start_transmission`、`stop_transmission`、`power_tune`、`calibrate`、`step`、`run`、`stop`、`reset` 和 `export`。首次 `configure` 会创建并连接设备；显式 connect/disconnect 用于后续会话生命周期控制。`run` 可同时携带 `x` 与完整配置完成自动闭环。

生产者必须先写临时文件，再原子重命名为 `command_<command_id>.mat`。`command_id` 是持久幂等键；已有状态、同 ID 临时 run 或正式结果的命令不会重复执行硬件动作。已完成但尚未交付完毕的结果可从校验后的 run 缓存补写到 outbox。完整契约见 [`docs/file_interface_design.md`](docs/file_interface_design.md)。

本接口不兼容旧 `Config_file.mat`、`DPD_in.mat`、`FB_Signal.mat`、ACK、心跳或 `safeBack` 协议。

## 最终结果

成功的 `run` 或显式 `export` 生成不含迭代历史的正式 MAT：

- `x`：原始参考波形；
- `y`：最终实际发射并评价的 DPD 波形；
- `z`：对应的最终预处理反馈；
- `metrics`：最终 NMSE、数字功率、物理功率、衰减、固定增益和抓取计数；
- `config`：可由 MATLAB `jsondecode` 解析的实际生效配置；
- `status`、`schema_version` 和 `completed_at`。

完整迭代只保存在临时运行目录，正式结果不会包含历史波形。

## 核心模块

- `device.py`：设备能力、公共配置、动态 schema 和设备注册表。
- `simulation.py`：仿真 RF bench 和周期有记忆 PA。
- `preprocessing.py`：时延/相位对齐、相干平均和首轮固定幅度增益。
- `runtime.py`：版本化 DPD runtime 和基础 ILC。
- `safety.py` / `power_control.py`：数字与物理功率安全。
- `controller.py`：分步/自动闭环、状态机、停止和安全收尾。
- `storage.py` / `result_export.py`：临时记录、清理和最终 MAT。
- `file_interface.py`：新 MAT 命令服务。
- `waveforms.py` / `web_analysis.py` / `web_bridge.py` / `web.py`：安全 waveform repository、完整周期 RF 分析、共享 Web 命令桥和 FastAPI 控制台。

文档入口见 [`docs/README.md`](docs/README.md)。
