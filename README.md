# remote-dpd

`remote-dpd` 是一个不依赖 MATLAB 的远程 ILC DPD 文件服务，同时提供可复现的 PA 正向模型反向传播
ILC 仿真研究框架。服务保留既有 `.mat` 文件交换边界；数值层显式区分经典 linear ILC、历史
legacy 路径、instantaneous-gain ILC、在线模型 raw VJP 和受保护 matrix-free LM。

项目当前没有真实 PA 闭环数据。`experiments/` 的结论边界是合成 PA 上的机制仿真，不代表硬件、
PAE、热稳定性、产业部署或 3GPP 合规结果；自生成波形应称为 NR-like OFDM。

## 安装

要求 Python 3.10 或更高版本。CPU-only 环境建议：

```bash
python -m venv .venv
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
```

需要运行冻结实验、绘图或生成论文产物时，再安装研究依赖：

```bash
python -m pip install -r requirements-research.txt
```

## 文件服务

默认监听 `<watch-root>/<supplier_name>`；`--path` 可直接指定目录：

```bash
remote-dpd Zilink --watch-root /opt/SharePoint
remote-dpd staging --path D:\DPD\staging --engine linear_ilc
```

可选 engine：

| engine | 行为 |
| --- | --- |
| `ilc` | 兼容入口，按 MAT 配置中的 `ILCBackwardMode` 分派；缺省为 `legacy` |
| `legacy_ilc` | 历史 `alpha/gain/phase/FIR/packed capture` 路径 |
| `linear_ilc` | `u_(k+1)=u_k+mu*(desired-measured)` |
| `instantaneous_gain_ilc` | 带门限和阻尼的逐点复增益更新 |
| `model_vjp_ilc` | 每轮在线 memory-polynomial 模型的 raw VJP |
| `model_lm_ilc` | 在线模型、matrix-free CG、trust region、预测回溯和输入投影 |

显式 engine 名称覆盖 `ILCBackwardMode`。默认 `ilc` 保持 legacy 行为，不会静默启用模型模式。
模型模式要求 `alpha=0`、`phaseCompensate=false`，并且不配置 `txFirHd` 或 `errFirHd`；冲突配置会被
明确拒绝。

### 文件交换

服务接受：

```text
Config_file.mat
DPD_in.mat
FB_Signal.mat
safeBack
```

服务写出：

```text
Config_file_ack.mat
ACK_DPDin.mat
DPDout_Nokia.mat
symbolEVM.mat
sync_dat.txt
```

现代生产者可在输入/反馈中携带 session、iteration 和 `DPDInputID/DPDOutputID` 绑定字段；字段存在时
服务严格校验，旧文件不带绑定字段时继续兼容并记录 `feedback_binding_verified=false`。反馈内容指纹
用于幂等去重。`safeBack` 会删除监听根目录中的普通文件并重置内存状态，因此监听目录必须专用且
可信。

MAT v5/v6 由 SciPy 读写。安装 `h5py` 后只提供有限的 v7.3/HDF5 顶层 numeric dataset 回退，
不承诺完整解析 MATLAB v7.3 struct、group 或 object reference。

### 关键配置

旧 `configDPD` struct 继续接受；常用新增字段如下：

| MAT 字段 | 示例/含义 |
| --- | --- |
| `ILCBackwardMode` | `legacy`, `linear`, `instantaneous_gain`, `model_vjp`, `model_lm` |
| `ILCCalibrationMode` | `auto`, `legacy_dynamic`, `frozen_first`, `explicit` |
| `ILCCalibrationCoefficient` | `explicit` 模式所需的有限、非零复系数 |
| `PAModelOrder` / `PAModelMemoryDepth` | 默认 9 / 3 |
| `PAModelRidge` | 默认 `1e-6` |
| `PAModelMinValidationNmseDb` | 默认 `-20 dB` |
| `ILCLMDamping` | 默认 `1e-2`，下限 `1e-8` |
| `ILCCGMaxIterations` / `ILCCGTolerance` | 默认 8 / `1e-3` |
| `ILCTrustRegionRatio` | 默认 `0.25` |
| `ILCMaxInputRms/Peak/PaprDb` | 可选硬输入限制 |
| `PAModelFallback` | `linear` 或 `hold` |

完整更新公式、复数实线性梯度、校准、配置约束和回退定义见
[算法设计](docs/algorithm_design.md)，文件协议与状态机见[系统设计](docs/system_design.md)。

## 冻结仿真实验

先查看固定矩阵并运行 smoke：

```bash
python -m experiments.run_experiments --list-studies
python -m experiments.run_experiments --study smoke --output artifacts/experiments --workers 6
```

正式流程必须先完成 pilot、生成带 provenance 的 `resolved_config.json`，再运行 confirmatory 和所有
次要研究：

```bash
python -m experiments.run_experiments --study pilot --output artifacts/experiments --workers 6
python -m experiments.run_experiments --select-pilot --output artifacts/experiments --workers 6
python -m experiments.run_experiments \
  --study confirmatory \
  --output artifacts/experiments \
  --workers 6 \
  --resolved-config artifacts/experiments/pilot/resolved_config.json
```

一条命令验证结果、导出 CSV/JSON 并生成 PNG/PDF 图：

```bash
python -m experiments.plot_results \
  artifacts/experiments/confirmatory \
  --output artifacts/analysis/confirmatory
```

PowerShell 完整命令、容量门、恢复语义、产物结构与多 study 分析流程见
[复现说明](docs/reproducibility.md)；冻结 cell、seed、失败编码和统计规则见
[实验协议](docs/experiment_protocol.md)。

冻结结果对应的论文与已逐页核验 PDF 位于 [paper/main.tex](paper/main.tex) 和
[output/pdf/pa_model_backprop_ilc.pdf](output/pdf/pa_model_backprop_ilc.pdf)。PDF 的数据驱动构建与文本
完整性检查命令见 [paper/README.md](paper/README.md)。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `remote_dpd/protocol.py` | MAT 文件边界与原子输出 |
| `remote_dpd/config.py` | legacy MAT 配置归一化与严格校验 |
| `remote_dpd/dsp.py` | 重采样、对齐、校准、循环 FIR 与兼容指标 |
| `remote_dpd/pa_model.py` | 确定性 ridge-LS memory polynomial、JVP/VJP |
| `remote_dpd/learning.py` | linear/instantaneous/VJP/LM 更新与安全投影 |
| `remote_dpd/algorithms.py` | engine 注册、文件服务数值编排和 legacy 兼容路径 |
| `remote_dpd/service.py` | 文件监听、会话、绑定、ACK、输出和心跳 |
| `experiments/` | 波形、合成 PA、冻结矩阵、runner、统计、分析与绘图 |

## 验证

```bash
python -m unittest discover -s tests -v
```

设计和相关工作边界还可查阅 [既有研究审查](docs/prior_art_review.md)。
