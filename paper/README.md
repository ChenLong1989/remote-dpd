# 论文源文件

`main.tex` 是中文主文，并提供英文标题、摘要与关键词。参考文献位于
`references.bib`。

正文结果已经由冻结发布目录 `artifacts/frozen_release` 下的完整分析产物填充。六个正式研究均通过
manifest、expected IDs、shard checksum，以及 code、configuration、environment、protocol 和
matrix 五类哈希验证。更新论文数字时必须重新运行同一验证入口；不要根据 smoke、部分轨迹或事后
最佳轮手工改写结果。

冻结 manifest 记录的 base commit 是 `1115c5b` 且 `dirty=true`；该 commit 不能单独复原运行。
复核时必须同时使用 manifest 中完整的 `code_hash` 及 configuration、environment、protocol、matrix
哈希。

建议的 TeX 工具链是 XeLaTeX 与 BibTeX：

```text
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

仓库同时保留数据驱动的 PDF 构建入口。它从已验证的冻结分析目录读取数字和图表，并生成提交版
`output/pdf/pa_model_backprop_ilc.pdf`：

```powershell
.venv\Scripts\python.exe paper\build_paper.py `
  --artifact-root artifacts\frozen_release `
  --output output\pdf\pa_model_backprop_ilc.pdf
.venv\Scripts\python.exe paper\verify_pdf.py output\pdf\pa_model_backprop_ilc.pdf
```

冻结实验目录体积较大且不纳入 Git；重建 PDF 前必须先按复现说明生成或恢复完整的
`artifacts/frozen_release`。提交版还需逐页渲染检查字体、公式、表格、图片与裁切；文本检查不能替代
视觉检查。
