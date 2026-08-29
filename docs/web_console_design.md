# 可信网络 Web 控制台设计

本文描述 `remote_dpd/web.py`、`remote_dpd/web_bridge.py`、`remote_dpd/waveforms.py` 和 `remote_dpd/web_static/` 的当前实现。控制台面向本机或可信单用户局域网操作，不提供登录、多用户、远程公网或多设备并行能力。

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
- MAT 先用目录元数据检查 `x` 的 shape/样点上限，再只解压变量 `x`；拒绝 scalar、矩阵、logical、cell、struct、object、sparse、空数组、NaN/Inf、零 RMS 和超过 `0 dBFS` 的参考；
- 可选 MAT v7.3/HDF5 路径只接受内部 hard link 指向的普通 dataset，拒绝 external/virtual storage、可扩展 shape、扩展宽度 dtype 和大于逻辑数据集的 chunk；
- `preview` 和 `load` 都重新从已锚定文件描述符读取并完整校验，不把之前 preview 当作信任依据。

成功的 `x` 转换为独立、不可写的一维 `complex128`，随后文件命令 parser 和 controller 数字安全边界仍会再次校验。

## 4. REST 和 SSE

版本前缀为 `/api/v1`：

| 方法与路径 | 行为 |
| --- | --- |
| `GET /health` | 进程健康，不返回本地绝对路径 |
| `GET /session` | 当前 active command、run 和有界 controller/metrics 元数据 |
| `GET /session/preview` | 当前 `x/y/z/error` 的有界抽样 |
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
| `GET /runs/{id}/result.mat` | 在完整 export guard 内流式下载正式结果 |
| `GET /results/{command_id}.mat` | 下载 Web/file export 命令生成的正式结果 |

session/SSE 不包含完整 `x/y/z`。最多返回最近 256 条轻量迭代摘要和最近 256 个功率调节点；只有最新一轮附带前 8 个 batch、每 batch 前 8 个 segment 以及递归受限的 runtime metrics。run detail 的 config、snapshot、events 和 iteration 索引也分别使用递归节点预算及截断标记，单轮 preview metadata 使用独立预算。完整波形只能通过有界 preview 获取。最多允许 8 个 SSE 客户端，断开只释放订阅，不会隐式停止任务。

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

所有响应设置 no-sniff、no-referrer、DENY frame 和只允许同源脚本/样式/连接的 CSP；API 和下载使用 `Cache-Control: no-store`。LAN 模式仍为无鉴权明文 HTTP，能访问端口的可信主机可以控制 RF 和下载结果；应用不自动修改防火墙。本安全边界只适用于本机或可信单用户局域网，不等价于公网部署加固。

## 6. 页面交互

单页控制台包括：

- waveform 目录递归浏览、样点数/峰值/RMS preview 和选择；
- 设备注册表选择、公共 RF/功率/衰减/抓取/通道/触发配置；
- 根据设备 schema 动态生成 string/integer/number/boolean/enum/array/object 字段；
- 仿真 PA 系数逐行添加、删除和编辑 `p/m/real/imag`；
- 一键自动闭环，以及 connect/disconnect、load/configure、start/stop TX、power tune、calibrate、ILC step、reset/export 和 safety stop；
- controller 状态、轮次、功率、锁定衰减、固定增益、NMSE、数字 RMS/峰值、时延、相位和功率调节轨迹；
- Canvas 绘制的有界 `x/y/z` 包络和 NMSE 历史；
- 临时 run 列表、配置/snapshot/结构化事件 inspector 和正式 MAT 下载。

页面使用 SSE 更新状态；浏览器不支持 EventSource 或流暂时断开时，每秒 session polling 仍会纠正状态。按钮根据 active command 和 controller 状态禁用，避免页面提交明显非法顺序；服务端仍独立执行完整互斥和状态校验。

## 7. 生命周期和限制

启动顺序为 RunStore cleanup、共享 FileCommandService watcher、FastAPI；关闭顺序为停止 HTTP 接入、停止 watcher、取消 active/pending controller、等待唯一 worker、停止命令间仍在发射的 RF、关闭 processor 和 RunStore。uvicorn graceful shutdown 最多等待 10 秒，避免长期 SSE 客户端使本机进程无法退出。

当前结构化 `events.json` 作为 run 日志展示，不新增任意路径文本日志。功率调节轨迹由 controller 在调节完成或失败后一次性提交，因此页面不会保证逐测量点实时动画；最终完整轨迹仍会显示。未完成硬件任务在重启后不恢复，只按已有持久证据安全收敛。
