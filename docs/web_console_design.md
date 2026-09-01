# 可信网络 Web 控制台设计

本文描述 `remote_dpd/web.py`、`remote_dpd/web_bridge.py`、`remote_dpd/web_analysis.py`、`remote_dpd/waveforms.py` 和 `remote_dpd/web_static/` 的当前实现。控制台面向本机或可信单用户局域网操作，不提供登录、多用户、远程公网或多设备并行能力。

## 1. 启动和部署边界

CLI 使用 `--mode web` 启动 FastAPI/uvicorn，同时保留同进程 MAT inbox watcher：

```bash
remote-dpd \
  --exchange-root /opt/remote-dpd/exchange \
  --mode web \
  --waveform-root /opt/remote-dpd/waveforms \
  --web-port 8000
```

默认 `--web-host=127.0.0.1`。可信 LAN 使用 `--web-host 0.0.0.0` 并至少显式提供一个非 loopback `--web-allowed-host`，例如：

```bash
remote-dpd \
  --exchange-root /opt/remote-dpd/exchange \
  --mode web \
  --waveform-root /opt/remote-dpd/waveforms \
  --web-host 0.0.0.0 \
  --web-allowed-host 192.168.3.100 \
  --web-port 8765
```

`--web-host` 只接受 `127.0.0.1`、私网 IPv4 或 `0.0.0.0`；非 loopback bind 必须声明私网 allowed host，绑定某个具体私网 IP 时该地址也必须在白名单内。`--web-allowed-host` 可重复，拒绝 `*`、`0.0.0.0`、公网、组播、保留地址和非 IP 文本。TrustedHost 始终保留 `127.0.0.1/localhost/[::1]`，再追加显式白名单。两种模式均使用单 worker、关闭 proxy headers；`waveform-root` 未指定时使用 `<exchange-root>/waveforms`。

FastAPI 提供原生 HTML/CSS/JavaScript 单页，不需要 Node 构建链。Web 进程和文件入口共享一个 `FileCommandService`、一个 `FileCommandProcessor`、一个普通命令 worker 和同一 stop latch；不能分别创建 Web worker 后并发调用 controller。

## 2. 共享命令仲裁

`WebCommandBridge` 把严格 Web 请求转换为带 `web-` 前缀的版本化命令，并使用通用原子 MAT writer 发布到同一 inbox，再显式交给 `FileCommandService.process_file()`：

```text
Browser REST ──> WebCommandBridge ─┐
                                   ├─> FileCommandService ─> FileCommandProcessor
External MAT producer ─────────────┘                         ├─> ClosedLoopController
                                                             └─> RunStore
```

这条内部持久命令路径只承担控制和幂等，不传递设备反馈；发射、功率测量和抓取仍由 controller 直接调用 RF bench。其结果是：

- Web 与文件普通命令严格互斥，冲突立即返回 `409 busy`，不会排队后执行；
- 外部 MAT stop 绕过普通 worker；Web safety stop 在等待普通命令 MAT 序列化前先设置独立 immediate-stop barrier 并通知 current/pending controller，再持久化正式 stop 命令；
- 命令状态、run ID、最终结果和重启后的已完成交付恢复沿用同一契约；
- Web 路由不直接访问 controller 变更方法，也不复制功率、预处理或算法流程。

支持 Web 动作为 `connect/disconnect/load/configure/start_transmission/stop_transmission/power_tune/calibrate/step/run/reset/export`；safety stop 使用独立端点。首次 `configure` 完成设备创建、连接和应用配置，后续可显式断开与重连。自动 `run` 可同时携带 waveform path 和完整配置，也可由 API 调用方省略其中一项以复用当前会话。

## 3. Waveform repository

`WaveformRepository` 在启动时打开并持有真实 waveform root 的目录描述符。所有浏览和加载使用 `dir_fd`、`O_DIRECTORY`、`O_NOFOLLOW` 和最终文件 `O_NONBLOCK`：

- API 只接受规范化 POSIX 相对路径；拒绝绝对路径、`.`、`..`、重复分隔符、反斜线、盘符和 NUL；
- 每一级父目录和最终文件都不跟随符号链接；root 路径启动后被替换也不会切换到新目录；
- 列表只返回真实目录和普通 `.mat` 文件，忽略隐藏项、FIFO、socket、device 和其他扩展名；
- 默认文件上限 256 MiB，`x` 上限 1000 万样点，preview 上限 4096 点；
- MAT 先用目录元数据检查 `x` 的 shape/样点上限，再只解压变量 `x`；拒绝 scalar、矩阵、logical、cell、struct、object、sparse、空数组、NaN/Inf 和零 RMS；source 峰值可超过 `0 dBFS`，但生效参考仍必须通过 controller 安全检查；
- 可选 MAT v7.3/HDF5 路径只接受内部 hard link 指向的普通 dataset，拒绝 external/virtual storage、可扩展 shape、扩展宽度 dtype 和大于逻辑数据集的 chunk；
- `preview` 和 `load` 都重新从已锚定文件描述符读取并完整校验，不把之前 preview 当作信任依据。

成功的 source `x` 转换为独立、不可写的一维 `complex128`，随后文件命令 parser 和 controller 数字安全边界仍会再次校验。preview 返回 source 安全统计；页面结合当前 draft 显示预计 scale、effective RMS/peak。controller 在归一化开启时校验缩放后的 `x`，关闭时校验 source，任何实际生效峰值超限仍 fail closed。

## 4. REST 和 SSE

版本前缀为 `/api/v1`：

| 方法与路径 | 行为 |
| --- | --- |
| `GET /health` | 进程健康，不返回本地绝对路径 |
| `GET /session` | 当前 active command、run 和有界 controller/metrics 元数据 |
| `GET /session/preview` | 当前 `x/y/z/error` 的有界抽样 |
| `POST /session/analysis` | 对当前不可变 snapshot 执行只读完整周期 RF 分析 |
| `GET /devices` | 注册设备、动态专属 schema 和公共默认配置 |
| `GET /waveforms` | 受限目录浏览 |
| `GET /waveforms/preview` | 安全报告和 `x` 抽样 |
| `POST /commands` | 提交一条普通命令，成功受理返回 202 |
| `GET /commands/{id}` | 查询持久命令状态 |
| `POST /stop` | 立即安全停止，不进入普通 worker |
| `GET /events` | latest-state SSE，状态变化时发送，15 秒 heartbeat |
| `GET /runs` | 有界临时 run 摘要 |
| `GET /runs/{id}` | 配置、snapshot、功率轨迹和结构化事件 |
| `GET /runs/{id}/iterations/{n}/preview` | 有界迭代 `x/y/z/error` 抽样和诊断 |
| `POST /runs/{id}/analysis` | 在 cleanup guard 内对指定历史轮次执行只读 RF 分析 |
| `GET /runs/{id}/result.mat` | 在完整 export guard 内流式下载正式结果 |
| `GET /results/{command_id}.mat` | 下载 Web/file export 命令生成的正式结果 |

session/SSE 不包含完整 `x/y/z`。最多返回最近 256 条轻量迭代摘要和最近 256 个功率调节点；只有最新一轮附带前 8 个 batch、每 batch 前 8 个 segment 以及递归受限的 runtime metrics。run detail 的 config、snapshot、events 和 iteration 索引也分别使用递归节点预算及截断标记，单轮 preview metadata 使用独立预算。完整波形不经 Web 返回；preview 只返回有界时域抽样，analysis 在服务端读取完整波形并只返回有界频谱和汇总结果。最多允许 8 个 SSE 客户端，断开只释放订阅，不会隐式停止任务。

run 结果下载在整个 ASGI response 生命周期内持有 `RunStore.export_guard()`；即使发送 response header 时失败，也会释放 guard。run 与 outbox 下载均从启动时锚定的目录描述符以 `O_NOFOLLOW` 打开普通文件，校验和流式发送复用同一文件描述符，避免路径检查后被替换。

## 5. 请求和可信网络安全

服务不启用 CORS，并使用 `TrustedHostMiddleware` 只接受 loopback 和显式私网 Host。所有修改请求必须同时满足：

- `Content-Type: application/json`；
- `X-Remote-DPD-Request: 1` 自定义头；
- 没有压缩 Content-Encoding；
- Origin 缺省或与请求 Host 完全同源，拒绝 `null` 和外部 Origin；
- body 实际流式累计不超过 1 MiB，不只信任 `Content-Length`；
- JSON 使用 UTF-8，拒绝重复 key、NaN/Infinity、超过 32 层或 10 万节点的结构；
- 未知命令字段、动作不匹配字段和配置类型被拒绝。

Web 另外限制平均段数和功率调节次数不超过 10000、稳定时间不超过 60 秒、设备调用超时不超过 300 秒、ILC 最大迭代不超过 1000、PA 系数不超过 256 项、阶数不超过 99、记忆延迟不超过 4096。controller 进一步限制每轮反馈总抓取不超过 1000 万样点、保留轮次乘参考长度不超过 2000 万样点；设备 schema 和数字安全仍是最终业务校验者。

所有响应设置 no-sniff、no-referrer、DENY frame 和只允许同源脚本/样式/连接的 CSP；页面、静态资源、API 和下载使用 `Cache-Control: no-store`，CSS/JavaScript URL 另带显式版本参数，避免更新后混用旧资源。LAN 模式仍为无鉴权明文 HTTP，能访问端口的可信主机可以控制 RF 和下载结果；应用不自动修改防火墙。本安全边界只适用于本机或可信单用户局域网，不等价于公网部署加固。

## 6. 页面交互

单页控制台使用固定 `100dvh` 仪表布局，document 不滚动。persistent header 持续显示 TX/RX/PWR、Controller、RF Output、中心频率、采样率、功率、衰减、迭代以及 `CONFIG/RUNS/RF OFF`。主区同时显示 `Z₀/Zₙ/Eₙ` 频谱、核心 DPD result 和唯一主 CTA；底部单一辅助 pane 在 Convergence、ACLR、AM/AM、AM/PM、Power Tune、Alignment 间切换。

Configuration、Expert Manual Control 和 Runs/Inspector 使用原生 dialog；配置按 Signal、Power & Safety、Analysis Bands、Simulation DUT 页签分组。默认 simulated 配置和首个安全 waveform 加载成功后，首页一次点击 `START DEFAULT SIMULATION` 即向现有 run 命令提交完整 waveform/config，无需先 Load、Configure 或 Connect。

新页面的 simulated 默认值来自 `/api/v1/devices` 返回的确定性 Web-only quick-start profile。初始化、刷新、切换设备和 `RESET DEFAULTS` 都从 `default_configuration` 重建公共字段和动态 `device_options`；不读取浏览器存储、run 历史或 controller 状态。当前 profile 固定匹配默认 `491.52 MS/s` waveform，使用 reference RMS normalization `true/-15 dBFS`、物理 Target power `-15 dBm`、十段平均、1000 万单次抓取上限、`mu=0.35` 和 15 次 ILC。simulated schema v3 默认 PA 的完整实测可完成第 15 轮，最终峰值约 `0.710`。数字安全上限和 runtime 通用默认保持不变。

Signal 配置页显示 normalization 开关和 target RMS；关闭时禁用 target 输入。waveform preview 显示 source RMS/peak 以及按当前 draft 预计的 scale、effective RMS/peak，预计峰值超过 0 dBFS 时使用告警色。浏览器只预估和展示，不修改 IQ；最终缩放与峰值安全由 controller 完成。公共物理 Target power 仍在 Power & Safety 页编辑。

页面使用 SSE 更新状态；浏览器不支持 EventSource 或流暂时断开时，每秒 session polling 仍会纠正状态。按钮根据 active command 和 controller 状态禁用，服务端仍独立执行完整互斥和状态校验。原生 Canvas 绘图层负责工程坐标、单位、trace、marker、Auto Set 和相对/绝对频率，不引入 Node 构建链或外部 CDN。

## 7. 生命周期和限制

启动顺序为 RunStore cleanup、共享 FileCommandService watcher、FastAPI；关闭顺序为停止 HTTP 接入、停止 watcher、取消 active/pending controller、等待唯一 worker、停止命令间仍在发射的 RF、关闭 processor 和 RunStore。uvicorn graceful shutdown 最多等待 10 秒，避免长期 SSE 客户端使本机进程无法退出。

当前结构化 `events.json` 作为 run 日志展示，不新增任意路径文本日志。功率调节轨迹由 controller 在调节完成或失败后一次性提交，因此页面不会保证逐测量点实时动画；最终完整轨迹仍会显示。未完成硬件任务在重启后不恢复，只按已有持久证据安全收敛。

## 8. 专业射频工作台与分析

控制台以信号路径、RF 发射安全、频谱和 DPD 前后对比为中心；现有部署、安全、命令仲裁和设备控制边界保持不变。

### 8.1 单屏仪表区

header、主频谱、核心结果/控制和辅助测量构成固定单屏。`RF OFF / ABORT` 始终位于 header；dialog 内提供等价 abort。主 CTA 默认执行完整 simulated run，connect/load/configure/start/stop TX/power tune/calibrate/step/reset 保留在 Expert dialog。Runs dialog 提供历史选择、重新分析、结构化诊断和正式 MAT 下载。

### 8.2 R&S 参考视觉语言

主监控区参考 R&S FSW 的频谱、channel bar、trace/marker 和 MultiView 信息结构；配置区参考 R&S SMW200A 的信号路径 block diagram。实现采用原创标识和布局，不复制 R&S 商标或产品面板：

- 深灰仪表底色、黑色绘图区、清晰但低干扰的灰色 major/minor grid；
- 主 trace 黄色，目标对比 trace 青色，其他 trace 使用有限且全局一致的颜色；绿色只表示 ready/pass，琥珀表示 warning/active RF，红色只表示 fault/abort 等高优先级状态；
- 紧凑扁平控件、细分隔线、小圆角、工程数字等宽字体，不使用装饰性 glow、大面积渐变或营销式空白；
- 主图必须具有刻度、单位、trace 名称、显示范围、marker readout、空数据/计算中/错误状态；辅助图通过固定 tab 复用同一 pane；
- 英文仪器术语为主，单位使用 SI 工程前缀；`1920×1080` 和 `1366×768` 下 document 均无横向/纵向 overflow。

### 8.3 Trace、频谱与单位

主频谱默认比较第 0 轮反馈 `Z₀`、当前或用户选择轮反馈 `Zₙ`，并显示 `Eₙ = Zₙ - X` error spectrum，用于同时观察 DPD 前后的 PA 输出和目标轮带内 waveform distortion。服务端从完整、已预处理的目标轮 `z` 与固定 reference `x` 直接相减后执行同一 DFT；error trace 随 Target iteration 更新，单位为 `dBFS/bin`，不代表 demod EVM。`x` reference 和对应轮 `y` DPD drive 作为可选 trace，最大可同时请求 5 条。trace 标签必须写明逻辑含义和轮次，不能只显示单字母颜色图例。

频谱基于完整周期的矩形窗 DFT，先 `fftshift` 再按采样率生成频率轴。归一化定义为 `X[k] / N`，单位幅度复正弦落在单个 DFT bin 时为 `0 dBFS/bin`；显示 `FFT Size=N` 和 `Bin BW=sample_rate/N`。相对模式以 baseband center 为 `0 Hz`，绝对模式在频率轴加有效中心频率。系统不把 Bin BW 标成 RBW，也不显示没有物理实现的 VBW、Sweep Time、Max Hold 或校准 `dBm/Hz`。

服务端在任何显示压缩前完成 band power 积分；绘图 trace 使用保峰值的有界聚合，避免窄带 spur 被平均掉。归一化数字频谱与功率传感器 `power_dbm` 分别显示，后者只代表校准总输出功率，不用于推导每 bin `dBm`。

### 8.4 Analysis Profile 与 measurement bands

`AnalysisProfile` 是只读、版本化、逐请求提交的 Web 分析参数，不属于 RF 控制配置，不写入 waveform MAT、controller、文件命令、run manifest 或正式结果 MAT。本轮不提供 profile 的跨会话 Save/Recall；同一个 profile 可以应用到当前 session 或任意历史 run/iteration。

profile 包含显示点数、相对/绝对频率模式和有界 measurement-band 表。每行包含：

| 字段 | 约束与含义 |
| --- | --- |
| `label` | 非空、长度受限、在一次请求内唯一 |
| `role` | `main`、`adjacent` 或 `other` |
| `center_offset_hz` | 相对载波中心的有限频率 offset |
| `integration_bandwidth_hz` | 正有限积分带宽 |
| `enabled` | 是否参与绘图和积分 |

页面默认创建一个多载波 ACLR template：中心 `-90` 到 `+90 MHz` 的 10 个 20 MHz `role=main` TX channel，以及中心 `-110/+110 MHz` 的两个 20 MHz adjacent，分别通过 `reference_label` 引用最左 `TX1` 和最右 `TX10`。允许多个启用的 main；adjacent/other 在多个 main 下必须显式引用一个启用的 main，不按距离猜测。只有一个 main 时省略 reference 仍按旧行为计算，保持旧 profile 兼容。

每个 band 返回积分 `power_dbfs`；有 reference 的 adjacent/other 返回 `relative_power_dbc = P_band - P_reference`，adjacent 另返回正值 `aclr_db = P_reference - P_band`。频谱显示 channel boundary、shading 和标签；Auxiliary ACLR tab 显示 `Z₀/Zₙ` grouped channel-power bar；结果表对 adjacent 显示负值 `dBc` 和改善量。TX channel power 使用数字 `dBFS`，不显示未校准 dBm。该模板仍为逐请求 Web Analysis Profile，不进入运行配置或结果契约；用户可在 Configuration 中编辑 TX/adjacent/reference。

band 区间不得越过 `[-sample_rate/2, sample_rate/2)`；边界与 DFT bin 不完全重合时按 bin 覆盖比例积分，避免仅用 bin center 导致宽度跳变。最大 band 数、label 长度、显示点数和嵌套节点均使用独立硬上限。

### 8.5 DPD 分析结果

- **NMSE**：继续使用预处理后 `z` 相对固定 reference `x` 的现有定义，展示第 0 轮、目标轮和改善量。
- **PAPR**：按 `20*log10(peak/RMS)` 计算选定 `x/y/z` 数字波形；不从稀疏 preview 估计。
- **AM/AM**：以同轮实际发射 `|y|` 为输入、已对齐且固定增益校正的 `|z|` 为输出，提供 output amplitude 与 normalized gain 视图。
- **AM/PM**：以 `angle(z)-angle(y)` 的包裹相位差为纵轴，单位 degree；低于相对峰值门限的输入样点排除，防止近零相位支配结果。
- **分箱**：第 0 轮和目标轮使用共同输入幅度 bin，返回中心统计量与离散范围，不向浏览器发送全量散点。完整 PAPR 仍使用全部样点；AM/AM、AM/PM 对超长波形使用等间隔确定性代表样点，最多 262144 点，并在响应中报告原始/分析样点数和 `sampled` 标记。
- **Power Tune**：衰减为横轴、传感器功率为纵轴，绘制 target、允许误差区间和 safety limit；完整轨迹仍来自 controller 已记录事实。
- **Alignment**：展示 delay、phase、fixed gain、batch/segment 计数和现有受限诊断；不改变预处理算法或增加异常段剔除。

本阶段不实现调制识别、符号同步、raw/demod EVM、频谱 mask pass/fail、连续扫频或瀑布图。NMSE 不得重命名为 EVM。

### 8.6 只读分析 API

`POST /api/v1/session/analysis` 读取当前 session，`POST /api/v1/runs/{run_id}/analysis` 读取指定 run 的基线/目标 iteration。由于 profile 包含可编辑 band 表，入口使用 `POST` 承载结构化查询，但不改变服务端状态；它们仍满足现有同源 Origin、精确 Host、`application/json`、自定义控制头、body/JSON 深度与节点预算，并返回 `Cache-Control: no-store`。

分析响应只包含：profile 摘要、基线/目标轮标识、频谱坐标与有界 trace、band 积分、ACLR/PAPR/NMSE 对比、AM/AM/AM/PM 分箱以及必要的 truncation/availability 标记。不得返回完整 `x/y/z`、绝对路径或未受限 runtime metadata。

当前 session 从 controller snapshot 读取；历史分析在 `RunStore.active_run()`/cleanup guard 内读取不可变 reference 和 iteration。分析模块不得取得设备对象、调用 controller 变更方法或写入 run。历史 run 在读取或计算期间被清理时必须产生稳定错误，不得使用部分数据。

### 8.7 资源、并发与安全

完整周期 FFT 最多面对 1000 万 complex 样点，因此响应点数限制不足以约束计算资源。实现必须：

- 在控制 worker 和事件循环之外执行数值计算，使用独立单并发分析 gate；已有分析占用 gate 时新请求立即返回 `429 analysis_busy`，不建立无界等待队列；
- 逐 trace 执行 FFT、band 积分和显示压缩，随后立即释放全量频谱中间数组；
- 以 run/session generation、iteration、trace 和 profile 指纹缓存有界最终结果，不缓存全量 FFT；当前最多缓存 16 项、总 JSON 预算 16 MiB；
- 对输入样点、请求 band、trace、显示点和并发等待设置硬限制，客户端断开不能造成无限排队；
- 保证 command、SSE/session polling 和 safety stop 不等待分析 gate；分析失败只影响分析响应，不改变 controller/run 状态。

profile 最多包含 32 个 band、5 条 trace 和 4096 个频谱显示点。最大允许 waveform 仍为 1000 万 complex 样点；实现使用标量 bin 边界积分，避免为每个 band 建立整轴权重数组。完整频谱绝不从稀疏时域 preview 估计。单并发分析 gate、逐 trace 释放和 16 项/16 MiB 有界最终结果缓存保持不变；1000 万点、5 trace、12 band 和 4096 显示点的本机基准为 5.70 秒、峰值 RSS `1400032 KiB`。

### 8.8 兼容性与验收

现有 REST 命令、SSE、preview、run detail、下载、文件入口、controller、正式 MAT 和可信 LAN 安全边界保持兼容。旧前端元素可以重构或移除，但所有动作必须继续通过 `WebCommandBridge`/独立 stop 端点，Analysis 代码不得复制闭环流程。

验收覆盖数值定义、band 边界、资源上限、API 安全、状态到推荐动作、RF 状态/abort、SSE 重连、trace/marker/iteration、measurement-band 编辑、无 profile/失败 run 和浏览器 `1920×1080`/`1366×768` 端到端。

## 9. 固定单屏与默认仿真

本节描述固定单屏 UI；它不改变第 1~8 节的设备控制、射频分析、命令、安全或资源契约。

### 9.1 固定视口

正式支持的最小视口为 `1366×768`。`html/body` 使用固定 `100dvh` 且禁止 document 级横向和纵向 overflow；主应用在扣除 header 后使用 CSS Grid 消耗全部剩余高度。用户完成默认运行、观察频谱/结果、切换辅助测量和发出 abort 均不需要页面滚动。

固定单屏由以下区域组成：

| 区域 | 固定内容 |
| --- | --- |
| Header/status | 产品、TX/RX/PWR、Controller、RF Output、Fc、Fs、Pout、Att、Iteration、Config、Runs、RF abort |
| Main spectrum | `Z₀/Zₙ/Eₙ`、可选 `x/y`、Marker、Peak Search、频率模式和 display controls |
| Result/control | NMSE、Pout、PAPR、ACLR、状态/错误、唯一主 CTA、Expert 入口 |
| Auxiliary | 单一绘图区和 `Convergence/ACLR/AM-AM/AM-PM/Power/Alignment` tab |

旧 `OPERATE/CONFIGURATION/DPD ANALYSIS/RUNS` 顶层 workspace、同时显示的多窗口和 maximize 交互删除。Auxiliary tab 只切换同一个有界 pane 的内容；已有 analysis 数据和选定 trace/iteration 不因切换而重新提交控制命令。

### 9.2 弹窗配置

`CONFIG`、`EXPERT`、`RUNS` 使用原生 `<dialog>`。配置 dialog 使用 `Signal`、`Power & Safety`、`Analysis Bands`、`Simulation DUT` 页签，每次只显示一组字段；footer 固定提供 `RESET DEFAULTS`、`CANCEL`、`APPLY DRAFT` 和 `START SIMULATION`。Apply 只保留前端 draft，只有 Configure/Run 动作才提交服务端。

默认字段、默认 PA 系数和默认 Main/L1/R1 行必须在 `1366×768` 对应 tab 中直接可见。任意扩展数量的 runs/events/coefficients/bands 可以在局部、有清晰边界的列表 pane 内滚动，但 dialog 与 document 本身不依赖页面滚动。dialog 打开时固定 header 中的 RF Output/abort 仍可见；Escape/Cancel 不修改 RF 或 controller。

### 9.3 默认 simulated 一键闭环

初始化选择首个安全 waveform、simulated bench 和服务端 Web `default_configuration`。资源加载成功后首页主 CTA 为 `START DEFAULT SIMULATION`，一次点击向现有 `/commands` 提交包含 waveform 与完整 configuration 的 `run`；用户不需要先打开配置、Load、Configure 或 Connect。全新浏览器、刷新页面、无痕窗口和其他可信 LAN 终端使用同一 profile，不依赖此前页面或运行历史。

主 CTA 状态固定为：

- initializing：`LOADING DEFAULT BENCH`，disabled；
- ready/idle：`START DEFAULT SIMULATION`；
- busy/running：显示 action/iteration，disabled，独立 RF abort 可用；
- completed：`RUN AGAIN`；
- failed/stopped：显示错误摘要并提供 `RESET` 或重新运行的明确动作；
- no waveform/default error：disabled 并显示唯一阻断原因。

一键路径继续复用 `WebCommandBridge → FileCommandService → ClosedLoopController`；UI 不新增快捷设备调用、隐式服务端状态或安全旁路。

### 9.4 验收

- `1366×768` 与 `1920×1080` 下 `documentElement.scrollHeight <= clientHeight` 且无水平溢出；header、RF abort、主 CTA、频谱、核心结果和辅助 tab 同时可见。
- 打开每个 dialog 后 document 仍不滚动；默认 tab 内容完整可用，只有扩展长列表出现局部滚动。
- 首次加载后不进入任何 dialog，单击主 CTA 可以使用默认 simulated 配置和首个 waveform 完成完整自动闭环并显示最终结果。
- Expert 全部既有动作、measurement-band 编辑、历史 run 重分析/下载、SSE 重连、Marker 和相对/绝对频率继续可用。
