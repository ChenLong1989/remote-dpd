# 复现实验与产物校验

## 1. 复现目标

本文给出从干净 checkout 到冻结实验、自动分析和可恢复产物的完整命令。科学协议见
[experiment_protocol.md](experiment_protocol.md)。命令默认在仓库根目录执行；Windows 示例使用
PowerShell，路径可替换为独立的大容量 D 盘目录。

复现的身份由结果目录中的五个 SHA-256 共同定义：

```text
code_hash
configuration_hash
protocol_hash
matrix_hash
environment_hash
```

只有五者完全一致的 manifest、expected IDs、checkpoint 和 shards 才能在同一次运行中恢复或汇总。
不要复制、手改或合并来自不同 hash 的 shards。

| hash | 计算范围 |
| --- | --- |
| `code_hash` | `remote_dpd/**/*.py` 与 `experiments/**/*.py` 的相对路径和原始字节 |
| `configuration_hash` | canonical JSON 形式的完整 `ExperimentProtocol`、resolved methods 与 resolved hash |
| `protocol_hash` | 协议名、修订日和全部科学参数 |
| `matrix_hash` | 按 trajectory ID 排序的完整 `TrajectorySpec` 集合 |
| `environment_hash` | Python、平台、解释器、关键依赖版本、BLAS/OpenMP backend 身份和线程限制的 canonical environment manifest |

文档、测试和依赖声明不进入 `code_hash`；manifest 另保存协议文档 SHA-256、Git commit/dirty 列表与
实际依赖版本，审计时必须一并保留，不能只抄录五个 hash。

## 2. 环境

### 2.1 支持范围

- Python `>=3.10`；
- 核心依赖：NumPy `>=1.24`、SciPy `>=1.10`、PyTorch `>=2.0`、watchdog `>=3.0`；
- 研究依赖：Matplotlib `>=3.8`、psutil `>=5.9`、threadpoolctl `>=3.5`；PDF/论文工具依赖见 `research` extra；
- CPU-only 主机优先使用 `requirements-cpu.txt`，避免下载 CUDA runtime。

当前开发与验证主机在 `2026-08-22` 的参考环境为：

| 组件 | 版本 |
| --- | --- |
| Python | `3.11.13`，64-bit，Anaconda packaged |
| platform string | `Windows-10-10.0.26100-SP0` |
| NumPy | `2.4.6` |
| SciPy | `1.17.1` |
| PyTorch | `2.13.0+cpu` |
| watchdog | `6.0.0` |
| Matplotlib | `3.11.1` |
| psutil | `7.2.2` |
| threadpoolctl | `3.6.0` |
| pdfplumber / pypdf / ReportLab | `0.11.10 / 6.16.1 / 5.0.1` |

上表是参考环境，不替代每次 run 的 `manifest.json.environment`。manifest 会记录精确 Python、平台、
解释器路径、依赖版本以及 BLAS/OpenMP 线程环境变量，是某一结果集的最终环境证据。

### 2.2 创建环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

`requirements-research.txt` 以 editable 方式安装 `.[research]`。Linux/macOS 使用
`.venv/bin/python` 替换 Windows 解释器路径。正式运行前建议从干净、已提交的 checkout 开始；runner
仍会记录 dirty worktree，但 dirty 状态会降低外部复核的便利性。

验证安装和测试：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 3. 运行前检查

先确认可执行矩阵规模；该命令不创建研究目录：

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiments --list-studies
```

期望计数为：

```json
{
  "ablation": 192,
  "confirmatory": 2240,
  "dynamic": 160,
  "mismatch": 288,
  "pilot": 288,
  "robustness": 960,
  "smoke": 14,
  "stress": 168
}
```

设置本次输出根目录并检查静态容量门：

```powershell
$runRoot = "artifacts\frozen_run"
$workers = 6
.\.venv\Scripts\python.exe -m experiments.run_experiments `
  --output $runRoot --workers $workers --capacity-only
```

静态门要求产物小于 `5 GiB`、输出盘可用空间至少 `50 GiB`、controller RSS 不超过
`2.5 GiB`。随后运行一次真实 learned-LM smoke 轨迹，测量 wall time 和峰值 RSS，并投影完整协议：

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiments `
  --output $runRoot --workers $workers --capacity-probe
```

capacity probe 按 code/protocol/resolved/spec/environment/worker hash 缓存；每条轨迹还通过
threadpoolctl 在运行时强制 BLAS/OpenMP 单线程，并把观测到的最大线程数写入 shard 与 probe 报告。
重复命令返回同一份已校验
报告，而不是把 resume 的近零耗时误作运行时。若报告 `allowed=false`，应先调整 `workers` 或资源，
不要通过减少科学 cell 绕过门。`--no-time-gate` 只关闭 72 小时投影门，不改变矩阵；正式运行仅在
明确接受预计时长后使用。

`--dry-run` 是纯矩阵检查，不写 manifest 或 shards：

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiments `
  --study smoke --output $runRoot --workers $workers --dry-run
```

锁定研究的 `--dry-run` 也必须提供有效的 `--resolved-config`。`--debug-limit` 只允许 smoke；由于它
产生不同 `matrix_hash`，调试子集必须使用独立输出根，例如 `artifacts\debug_run`，不能与完整
smoke 复用目录。

## 4. 冻结运行顺序

### 4.1 Smoke

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiments `
  --study smoke --output $runRoot --workers $workers
```

smoke 是工程门，不提供正式统计证据。检查输出摘要的 `complete=true`、14 条已完成轨迹及算法失败
计数；失败可以是科学结果，不能只凭非零失败计数删除轨迹。

### 4.2 Pilot 与参数锁定

运行完整 288 条 pilot：

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiments `
  --study pilot --output $runRoot --workers $workers
```

从经过 checksum、expected IDs 和五 hash 校验的 pilot 结果生成签名 resolved config：

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiments `
  --select-pilot --output $runRoot --workers $workers
$resolved = "$runRoot\pilot\resolved_config.json"
```

不要编辑该文件。后续加载会验证 resolved 自身 hash、pilot manifest/expected IDs 的文件 SHA-256、
records hash、完整候选表，并重新计算选择结果。

### 4.3 Confirmatory 与全部次要研究

```powershell
$studies = @(
  "confirmatory",
  "robustness",
  "mismatch",
  "ablation",
  "dynamic",
  "stress"
)
foreach ($study in $studies) {
  .\.venv\Scripts\python.exe -m experiments.run_experiments `
    --study $study `
    --output $runRoot `
    --workers $workers `
    --resolved-config $resolved
  if ($LASTEXITCODE -ne 0) { throw "study failed: $study" }
}
```

不要在看到 confirmatory 方法差异后修改协议、候选、seed 或次要矩阵。若科学代码修复改变
`code_hash`，应选择新的 `$runRoot` 重跑完整受影响 study，而不是把新旧 shards 混合。

## 5. 恢复语义

中断后，使用完全相同的 checkout、命令、输出根、worker 设置和 resolved config 再次执行该
study。worker 数属于运行资源，不进入轨迹科学身份，但使用相同值便于重现实测调度条件。runner
会按以下顺序恢复：

1. 验证现有 `manifest.json`、`expected_ids.json` 和 `specs.json` 的五 hash；
2. 验证所有已完成 shard 的 wrapper checksum、文件名、trajectory ID 和 hash；
3. 跳过已经完成的 trajectory；
4. 对未完成 trajectory 读取 `work/<key>/checkpoint.json` 与其 `arrays.<sha>.npz`；
5. 恢复 `u_k`、下一迭代号、PRNG state、LM damping、冻结/回放模型状态、逐轮 metrics 和终止状态；
6. 完成后重新验证 expected ID 集并原子写 `run_summary.json`。

如果全部 shards 已完成但 `run_summary.json` 丢失，重跑同一命令会从已验证 shards 重建摘要。
checkpoint 或 shard checksum 损坏会明确报错，不会静默忽略。基础设施异常最多自动重试 2 次；
算法失败已经编码为持久科学结果，不自动重试。

## 6. 产物布局与含义

典型目录如下：

```text
artifacts/frozen_run/
├── capacity_probe/
├── pilot/
│   └── resolved_config.json
├── smoke/
├── confirmatory/
├── robustness/
├── mismatch/
├── ablation/
├── dynamic/
└── stress/
    ├── manifest.json
    ├── expected_ids.json
    ├── specs.json
    ├── seeds.csv
    ├── run_summary.json
    ├── shards/<trajectory_id>.json
    ├── work/<short_key>/checkpoint.json
    └── waveforms/<trajectory_id>/kXXX.npz
```

每个 study 的关键文件：

| 文件 | 用途 |
| --- | --- |
| `manifest.json` | 科学协议、resolved 方法、五 hash、实际 cell、安全边界、环境、Git、argv/cwd |
| `expected_ids.json` | 完整且排序后的 trajectory ID 白名单 |
| `specs.json` | 每条 trajectory 的完整参数和 config hash |
| `seeds.csv` | 确定性 seed state 与 spawn key |
| `shards/*.json` | 自校验 wrapper 中的单轨迹逐轮数据和终点 |
| `work/*` | 可恢复 checkpoint；完成后仍可用于审计 |
| `waveforms/*` | 仅代表 seed 0 的 `k=0,1,2,5,10,20,30` 波形快照 |
| `run_summary.json` | 完成/恢复/算法失败/基础设施重试/资源摘要 |

普通轨迹波形由 `specs.json`、`seeds.csv` 和生成器重建，不需要复制 31 轮完整 IQ 数据。

## 7. 一键验证、分析与绘图

分析一个 study 时，输入必须是该 study 的 run directory。`plot_results` 会先验证 manifest、
expected IDs、trajectory specs、每个 shard checksum、五 hash、矩阵完整性与配对，再同时导出表格和
PNG/PDF 图；不能把多个 study 的 shards 放入一个分析 bundle。

对所有正式 study 执行一键分析：

```powershell
$analysisRoot = "artifacts\analysis"
$studies = @(
  "confirmatory",
  "robustness",
  "mismatch",
  "ablation",
  "dynamic",
  "stress"
)
foreach ($study in $studies) {
  .\.venv\Scripts\python.exe -m experiments.plot_results `
    "$runRoot\$study" `
    --output "$analysisRoot\$study" `
    --formats png pdf
  if ($LASTEXITCODE -ne 0) { throw "analysis failed: $study" }
}
```

只生成 JSON/CSV、不绘图时使用：

```powershell
.\.venv\Scripts\python.exe -m experiments.analysis `
  "$runRoot\confirmatory" `
  --output "artifacts\analysis\confirmatory"
```

每个分析目录固定包含：

```text
analysis_summary.json
trajectory_endpoints.csv
per_iteration_metrics.csv
cell_method_summary.csv
primary_comparisons.csv
ampm_fixed_r0_phase.csv
```

图名按 study 条件生成，包括 `convergence_main` 或 `convergence_study`、`endpoint_rates`、
`primary_effects`、`variant_endpoints`、`stress_diagnostics` 和 `ampm_fixed_r0_phase`。每张图输出
PNG 与 vector PDF，并在 metadata 中写入已验证 dataset SHA-256。次要研究的
`primary_comparisons.csv` 仅保留空表头，这是“无预注册主要推断”，不是分析失败。

`--allow-unlisted-jsonl` 仅用于外部、缺少 expected ID manifest 的 JSONL 审查。正式论文产物不得使用
该选项，因为它无法证明完整冻结矩阵。

## 8. 最小审计清单

正式引用结果前逐项确认：

- 所有目标 study 的 `run_summary.json.complete` 为 true；
- `completed_count == expected_count`，且计数与冻结矩阵一致；
- manifest 的 `git.commit`、`git.dirty`、environment 和完整 generation command 已归档；
- 六个锁定研究引用同一 `resolved_hash` 和可验证 pilot provenance；
- 输出根的 `frozen_resolved_lock.json` 校验通过，且六个锁定 manifest 均引用其同一 checksum；
- 没有 unexpected、missing 或 duplicate trajectory ID；
- 算法失败、发散、约束违规和右删失计数保留在表中；
- 主要结论只来自 confirmatory 的两个预注册 cell；
- stress、dynamic、mismatch、robustness 和 ablation 只按预注册角色解释；
- 论文中的数值和图均能追溯到对应分析目录的 dataset hash。
