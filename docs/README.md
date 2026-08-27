# Remote DPD 文档导航

本文是仓库内版本化文档的稳定入口。行为判断的优先级为：已发布代码与测试、对应模块设计文档、`system_design.md`、顶层 `README.md`。`current_plan.md` 是本地过程文件，不属于发布契约。

## 当前状态

当前默认运行入口仍是旧 MAT 文件监听服务。设备直控重构的阶段 1 和阶段 2 已建立设备能力、反馈预处理、DPD runtime、数字安全、仿真 RF bench、功率控制和单任务闭环控制器；这些新能力目前通过 Python API 使用，尚未接入默认 CLI。运行存储与新文件协议、网页控制台将在后续阶段完成。

## 使用与消费

- [`../README.md`](../README.md)：当前可运行服务、安装方式和旧文件协议概览。
- [`system_design.md`](system_design.md)：当前系统实现及已批准的整体重构草案。

## 模块设计

- [`device_design.md`](device_design.md)：公共设备配置、动态参数 schema 和 RF 能力接口。
- [`preprocessing_design.md`](preprocessing_design.md)：反馈批次、时延/相位对齐、相干平均和固定增益校正。
- [`algorithm_runtime_design.md`](algorithm_runtime_design.md)：版本化 DPD runtime、基础 ILC 和数字安全边界。
- [`simulation_design.md`](simulation_design.md)：确定性有记忆 PA、仿真射频链路、抓取和功率模型。
- [`controller_design.md`](controller_design.md)：初始功率调节、分步/自动闭环、状态机、停止与安全收尾。

## 协作与 provenance

- [`user_agent_workflow.md`](user_agent_workflow.md)：用户与 Agent 的计划、批准、实现、PR 和收尾流程。
- `current_plan.md`：当前改动的本地计划与执行记录；该文件已加入 `.gitignore`，不应被外部消费者依赖。
