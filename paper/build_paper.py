"""Build the verified simulation paper as a self-contained PDF."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


STUDIES = ("confirmatory", "robustness", "mismatch", "ablation", "dynamic", "stress")
ALGORITHM_LABELS = {
    "no_dpd": "No DPD",
    "linear_ilc": "Linear ILC",
    "legacy_ilc": "Legacy ILC",
    "instantaneous_gain_ilc": "Instantaneous-gain ILC",
    "model_vjp_ilc": "Learned raw VJP",
    "model_lm_ilc": "Learned safeguarded LM",
    "oracle_lm": "Oracle LM",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def fmt(value: Any, digits: int = 1) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def percent_interval(result: Mapping[str, Any]) -> str:
    return (
        f"{100.0 * float(result['estimate']):.1f}% "
        f"[{100.0 * float(result['confidence_low']):.1f}%, "
        f"{100.0 * float(result['confidence_high']):.1f}%]"
    )


def interval(result: Mapping[str, Any], suffix: str, digits: int = 1) -> str:
    return (
        f"{float(result['estimate']):.{digits}f}{suffix} "
        f"[{float(result['confidence_low']):.{digits}f}, "
        f"{float(result['confidence_high']):.{digits}f}]"
    )


def register_fonts() -> tuple[str, str]:
    regular_candidates = (
        Path("C:/Windows/Fonts/Deng.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    bold_candidates = (
        Path("C:/Windows/Fonts/Dengb.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    regular = next((path for path in regular_candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), None)
    if regular is None or bold is None:
        raise FileNotFoundError("a CJK TrueType font is required")
    pdfmetrics.registerFont(TTFont("PaperCJK", str(regular)))
    pdfmetrics.registerFont(TTFont("PaperCJKBold", str(bold)))
    return "PaperCJK", "PaperCJKBold"


def make_styles(font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=21,
            leading=31,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "PaperSubtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#355B7D"),
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "HeadingOneCJK",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "HeadingTwoCJK",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=12,
            leading=17,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=3 * mm,
            spaceAfter=1.8 * mm,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyCJK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.3,
            leading=15,
            alignment=TA_JUSTIFY,
            firstLineIndent=18,
            wordWrap="CJK",
            spaceAfter=2.2 * mm,
        ),
        "body_no_indent": ParagraphStyle(
            "BodyNoIndentCJK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.3,
            leading=15,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            wordWrap="CJK",
            spaceAfter=2.2 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCJK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.5,
            leading=10.3,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "small_center": ParagraphStyle(
            "SmallCenterCJK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.5,
            leading=10.3,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionCJK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8,
            leading=11.5,
            textColor=colors.HexColor("#404040"),
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
        ),
        "callout": ParagraphStyle(
            "CalloutCJK",
            parent=base["BodyText"],
            fontName=bold_font,
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#7F6000"),
            alignment=TA_LEFT,
            wordWrap="CJK",
            borderColor=colors.HexColor("#D6B656"),
            borderWidth=0.7,
            borderPadding=8,
            backColor=colors.HexColor("#FFF8E1"),
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
        "formula": ParagraphStyle(
            "FormulaCJK",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.4,
            leading=13,
            textColor=colors.HexColor("#203040"),
            leftIndent=8 * mm,
            rightIndent=8 * mm,
            borderColor=colors.HexColor("#CAD6E2"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#F5F8FB"),
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
    }


def paragraph(text: str, styles: Mapping[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, styles[name])


def table_cell(value: Any, styles: Mapping[str, ParagraphStyle], *, center: bool = False) -> Paragraph:
    style = styles["small_center" if center else "small"]
    return Paragraph(escape(str(value)), style)


def styled_table(
    rows: Iterable[Iterable[Any]],
    styles: Mapping[str, ParagraphStyle],
    widths: list[float],
    *,
    repeat_rows: int = 1,
) -> LongTable:
    converted = []
    for row_index, row in enumerate(rows):
        converted.append(
            [
                table_cell(value, styles, center=row_index == 0 or column_index > 0)
                for column_index, value in enumerate(row)
            ]
        )
    result = LongTable(converted, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("FONTNAME", (0, 0), (-1, 0), "PaperCJKBold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, colors.HexColor("#F7FAFC"))),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def add_figure(
    story: list[Any],
    path: Path,
    caption: str,
    styles: Mapping[str, ParagraphStyle],
    *,
    max_height: float = 128 * mm,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image(str(path))
    image._restrictSize(172 * mm, max_height)
    story.append(KeepTogether([image, paragraph(caption, styles, "caption")]))


def page_decorator(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#B7C9D9"))
    canvas.setLineWidth(0.4)
    canvas.line(19 * mm, height - 15 * mm, width - 19 * mm, height - 15 * mm)
    canvas.setFillColor(colors.HexColor("#51687A"))
    canvas.setFont("PaperCJK", 7.5)
    canvas.drawString(19 * mm, height - 11.8 * mm, "PA 正向模型反传与受保护逐波形 ILC")
    canvas.drawRightString(width - 19 * mm, 11 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build_document(artifact_root: Path, output: Path) -> None:
    font, bold_font = register_fonts()
    styles = make_styles(font, bold_font)
    analysis_root = artifact_root / "analysis"
    confirmatory = read_json(analysis_root / "confirmatory" / "analysis_summary.json")
    primary = {entry["scenario"]: entry for entry in confirmatory["primary_comparisons"]}
    confirmatory_cells = read_csv(
        analysis_root / "confirmatory" / "cell_method_summary.csv"
    )
    study_cell_summaries = {
        study: read_csv(analysis_root / study / "cell_method_summary.csv")
        for study in STUDIES
    }
    cell_index = {
        (row["scenario"], row["severity"], row["algorithm"]): row
        for row in confirmatory_cells
    }
    manifests = {study: read_json(artifact_root / study / "manifest.json") for study in STUDIES}
    summaries = {study: read_json(artifact_root / study / "run_summary.json") for study in STUDIES}
    locked_count = sum(int(summary["completed_count"]) for summary in summaries.values())
    total_runtime = sum(float(summary["elapsed_seconds"]) for summary in summaries.values())
    manifest = manifests["confirmatory"]
    hashes = manifest["hashes"]
    resolved_hash = manifest["pilot_lock"]["resolved_hash"]
    stress_cells = read_csv(analysis_root / "stress" / "cell_method_summary.csv")
    dynamic_cells = read_csv(analysis_root / "dynamic" / "cell_method_summary.csv")
    global_peak_rss_bytes = max(
        int(float(row["peak_rss_bytes"]))
        for rows in study_cell_summaries.values()
        for row in rows
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title="基于在线功率放大器正向模型反向传播的受保护逐波形迭代学习预失真",
        author="remote-dpd reproducible simulation study",
        subject="PA model-backpropagation ILC simulation mechanism study",
    )
    story: list[Any] = []

    story.extend(
        [
            Spacer(1, 18 * mm),
            paragraph(
                "基于在线功率放大器正向模型反向传播的<br/>受保护逐波形迭代学习预失真",
                styles,
                "title",
            ),
            paragraph(
                "Safeguarded Waveform Iterative Learning Predistortion via Backpropagation "
                "Through an Online Power-Amplifier Forward Model",
                styles,
                "subtitle",
            ),
            Spacer(1, 10 * mm),
            paragraph(
                "可复现仿真机制研究 · 中文主文 / English title and abstract",
                styles,
                "subtitle",
            ),
            Spacer(1, 12 * mm),
        ]
    )
    artifact_root_label = f"{artifact_root.parent.name}/{artifact_root.name}"
    identity_rows = [
        ("权威结果根", artifact_root_label),
        ("锁定轨迹", f"{locked_count} 条；confirmatory 2240 条"),
        ("Resolved hash", resolved_hash),
        ("Code hash", hashes["code_hash"]),
        ("Environment hash", hashes["environment_hash"]),
        ("生成日期", str(date.today())),
    ]
    identity_table = Table(
        [[table_cell(k, styles), table_cell(v, styles)] for k, v in identity_rows],
        colWidths=[36 * mm, 126 * mm],
        hAlign="CENTER",
    )
    identity_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            identity_table,
            Spacer(1, 11 * mm),
            paragraph(
                "证据边界：本报告只使用自生成 NR-like OFDM 与合成功率放大器。没有硬件闭环、"
                "真实 GaN/Doherty 器件、PAE、热稳定性、实时部署或 3GPP 合规证据。",
                styles,
                "callout",
            ),
            Spacer(1, 24 * mm),
            paragraph(
                "匿名技术报告 · remote-dpd · 结果由冻结矩阵与校验分析器生成",
                styles,
                "subtitle",
            ),
            PageBreak(),
        ]
    )

    amam = primary["amam"]
    ampm = primary["ampm"]
    story.extend(
        [
            paragraph("摘要", styles, "h1"),
            paragraph(
                "经典逐波形迭代学习控制（ILC）把输出位置的跟踪误差直接加到输入，等价于以单位"
                "映射近似功率放大器（PA）的局部 Jacobian。本文实现并检验另一条路径：每轮由"
                "当前输入和测量输出拟合带 ridge 的复系数 memory-polynomial 正向模型，以复数实"
                "内积定义 real-linear JVP/VJP，再用 matrix-free 阻尼 Gauss–Newton/LM 求解候选"
                "步，并施加 RMS trust region、锚定预测、回溯与输入 RMS/峰值/PAPR 约束。",
                styles,
            ),
            paragraph(
                f"冻结 confirmatory 数据含 2240 条轨迹。相对 linear ILC，learned safeguarded LM "
                f"在 AM/AM 0.97 cell 的 paired median AUEC 降低 "
                f"{percent_interval(amam['auec_relative_reduction'])}，最终 NMSE 改善 "
                f"{interval(amam['final_nmse_improvement_db'], ' dB')}；但两者成功率均为 100%，"
                "成功率改善为 0 个百分点，未达到预注册 +20 点门槛，因此该 cell 的联合假设失败。"
                f"在 AM/PM 135° cell，AUEC 降低 {percent_interval(ampm['auec_relative_reduction'])}，"
                f"最终 NMSE 改善 {interval(ampm['final_nmse_improvement_db'], ' dB')}，成功率提高 "
                f"{interval(ampm['success_rate_difference_points'], ' 点')}，联合假设通过。两 cell "
                "发散和约束违规差均为 0。",
                styles,
            ),
            paragraph(
                "重要边界结果是：instantaneous-gain ILC 在 AM/AM cell 的 AUEC 与 learned LM 只差"
                "约 4%、成功率同为 100%，而在 AM/PM cell 的 AUEC、最终 NMSE 与成功率均明显更好。"
                "所以结果只支持“相对 scalar linear ILC 的特定机制改善”，不支持一般 DPD 或所有"
                "ILC 变体上的最优性。硬饱和压力测试也确认 Jacobian 信息不能恢复物理上不可达目标。",
                styles,
            ),
            paragraph("关键词：数字预失真；迭代学习控制；功率放大器；VJP；Levenberg–Marquardt；可复现仿真", styles, "body_no_indent"),
            paragraph("English Abstract", styles, "h1"),
            paragraph(
                "This reproducible simulation study tests waveform ILC updates obtained by backpropagating "
                "a measured residual through an online memory-polynomial PA forward model. A real-linear "
                "JVP/VJP implementation feeds a matrix-free damped Gauss–Newton/LM solve protected by a trust "
                "region, anchored prediction, backtracking, and hard input constraints. Across 2,240 frozen "
                "confirmatory trajectories, the learned safeguarded LM reduced paired median AUEC relative to "
                "scalar linear ILC by 54.0% in the AM/AM 0.97 cell and 69.0% in the AM/PM 135-degree cell. "
                "The preregistered joint criterion failed in the former because both methods already achieved "
                "100% success, while it passed in the latter. Instantaneous-gain ILC was the materially "
                "stronger AM/PM endpoint baseline. The evidence is limited to generated NR-like OFDM and synthetic "
                "PA mechanisms; it is not a hardware, efficiency, deployment, or standards-compliance result.",
                styles,
                "body_no_indent",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            paragraph("1 研究问题与先行工作边界", styles, "h1"),
            paragraph(
                "逐波形 ILC 利用任务重复性，在第 k 轮观测 PA 输出后直接修正下一轮输入。Chani-Cahuana "
                "等给出了 scalar linear 与 instantaneous-gain ILC；Schoukens 等研究了借助 BLA inverse "
                "learning filter 获得 PA preinverse。因此，把经典路径笼统称为“没有 PA 信息”是不准确的。"
                "Memory polynomial/GMP、经 PA 模型反向传播训练 DPD，以及 instantaneous complex gain "
                "也已有公开工作。本文不声称首次引入 PA 模型、Jacobian 或 backward。",
                styles,
            ),
            paragraph(
                "问题被收窄为：在逐波形 ILC 中，单位 Jacobian 近似何时给出较差方向；在线拟合的"
                "正向模型及其 VJP 能否提供更贴近真实梯度的方向；而阻尼 LM、信赖域和预测回溯是否"
                "能把方向信息转化为稳定更新。方法必须严格退化到 identity 模型下的 scalar 更新，"
                "并必须允许预注册假设失败。",
                styles,
            ),
            paragraph("2 方法", styles, "h1"),
            paragraph("2.1 Real-linear 导数与 adjoint", styles, "h2"),
            paragraph(
                "复系数 memory polynomial 对复输入通常不是 complex-linear。实现使用实内积 "
                "〈a,b〉R = Re(vdot(a,b))，并把局部映射写成 J(q)=Aq+Bq*。对应 VJP 是 "
                "J^T(v)=A^H v+B^T v*；只有在该实内积下，J^T J+lambda I 才是 CG 所需的对称正定结构。CG 的"
                "步长、残差内积和曲率判定均显式取实数。",
                styles,
            ),
            paragraph("J(q) = A q + B q*    ;    J^T(v) = A^H v + B^T v*", styles, "formula"),
            paragraph("2.2 在线 PA 正向模型", styles, "h2"),
            paragraph(
                "每轮使用当轮输入与测量输出，以 256-sample block 构造有限阶 memory-polynomial 特征。"
                "训练 scale 由 99.9% envelope quantile 冻结，列 RMS 归一化后做 augmented ridge least "
                "squares；每五个 block 留一块验证。秩、条件数、训练/验证 NMSE 和回退原因全部进入 shard。"
                "scale、校准和 LS 系数在本轮导数中冻结，不通过它们反传。",
                styles,
            ),
            paragraph("2.3 受保护 LM 更新", styles, "h2"),
            paragraph(
                "raw VJP 只使用 J^T e，在平滑饱和区会随局部斜率趋零。主方法求解阻尼 normal equation，"
                "以截断 CG 只调用 JVP/VJP，不显式形成 Jacobian；阻尼有 1e-8 绝对下限。完整 CG 步先"
                "裁入 RMS trust region，再对这一基步回溯；每个候选同时满足 RMS/峰值/PAPR 输入约束和"
                "投影后的信赖域。预测采用 measured + f(candidate) - f(current) 的锚定增量，减少模型"
                "静态偏置影响。",
                styles,
            ),
            paragraph("(J^T J + lambda I) delta = -J^T e", styles, "formula"),
            PageBreak(),
        ]
    )

    protocol_rows = [
        ("项目", "冻结值"),
        ("波形", "32768 samples；16 个无 CP 周期 NR-like OFDM symbol"),
        ("外层预算", "K=30 更新；固定评估 k=0…30，共 31 点"),
        ("主要 cells", "AM/AM 0.97；低功率 AM/PM 135°"),
        ("主要配对", "每 cell 40 个 PA/波形 seed cluster"),
        ("主要对照", "Learned safeguarded LM vs scalar linear ILC"),
        ("统计", "10000 次 paired cluster bootstrap；95% CI；AUEC Holm family"),
        ("联合门槛", "AUEC ≥25%；final NMSE ≥3 dB；success ≥+20 点；divergence ≤+10 点；constraint 不增加"),
        ("失败口径", "算法失败不删除；保持输入并物化到 k=30；非有限端点按预注册策略编码"),
        ("校准", "合成原生域显式 unity calibration；不是硬件低功率标定"),
    ]
    count_rows = [("研究", "轨迹数", "角色")]
    roles = {
        "confirmatory": "预注册主要推断",
        "robustness": "SNR / capture count",
        "mismatch": "模型阶数/记忆失配",
        "ablation": "机制消融",
        "dynamic": "Wiener/Hammerstein out-of-family",
        "stress": "不可达与负斜率压力",
    }
    for study in STUDIES:
        count_rows.append((study, summaries[study]["completed_count"], roles[study]))
    story.extend(
        [
            paragraph("3 冻结实验与统计协议", styles, "h1"),
            paragraph(
                "所有方法在产生正式比较前完成 pilot 选择；六组锁定研究必须引用同一 resolved hash。"
                "每条轨迹含稳定 ID，shard 以原子写入和内容 checksum 持久化。分析从 metrics 重算 AUEC、"
                "最终 NMSE、三轮成功窗、发散和约束状态，而不信任可编辑的派生端点。",
                styles,
            ),
            styled_table(protocol_rows, styles, [35 * mm, 127 * mm]),
            Spacer(1, 4 * mm),
            styled_table(count_rows, styles, [42 * mm, 25 * mm, 95 * mm]),
            paragraph(
                "发散安全判据采用冻结协议中更具体的配对差 +10 个百分点非劣界；这消解了早期计划"
                "摘要中“不增加”与详细统计条款之间的文字冲突。约束违规率仍要求不增加。",
                styles,
                "callout",
            ),
            PageBreak(),
        ]
    )

    primary_rows = [
        ("端点", "AM/AM 0.97", "AM/PM 135°"),
        (
            "AUEC 相对降低 [95% CI]",
            percent_interval(amam["auec_relative_reduction"]),
            percent_interval(ampm["auec_relative_reduction"]),
        ),
        (
            "最终 NMSE 改善 [95% CI]",
            interval(amam["final_nmse_improvement_db"], " dB"),
            interval(ampm["final_nmse_improvement_db"], " dB"),
        ),
        (
            "成功率差 [95% CI]",
            interval(amam["success_rate_difference_points"], " 点"),
            interval(ampm["success_rate_difference_points"], " 点"),
        ),
        ("发散 / 约束差", "0.0 / 0.0 点", "0.0 / 0.0 点"),
        (
            "AUEC Holm 校正 p",
            f"{amam['holm']['adjusted_p_value']:.6f}",
            f"{ampm['holm']['adjusted_p_value']:.6f}",
        ),
        ("联合判定", "失败：success 改善不足", "通过"),
    ]
    story.extend(
        [
            paragraph("4 主要结果", styles, "h1"),
            paragraph(
                "两个 AUEC 原始 p 值均为 0.000200，Holm 校正后均为 0.000400。统计显著并不自动等于"
                "预注册联合成功：AM/AM cell 虽有更低 AUEC 和最终 NMSE，但 linear ILC 已在 40/40 seed "
                "成功，learned LM 无法再提高成功率，故必须判失败。AM/PM cell 的 linear ILC 成功率为 0%，"
                "learned LM 为 67.5%，五项阈值全部达到。",
                styles,
            ),
            styled_table(primary_rows, styles, [48 * mm, 57 * mm, 57 * mm]),
            Spacer(1, 4 * mm),
        ]
    )
    add_figure(
        story,
        analysis_root / "confirmatory" / "primary_effects.png",
        "图 1  Learned safeguarded LM 相对 linear ILC 的 paired 95% bootstrap 区间。橙色虚线为冻结效果或安全门槛。",
        styles,
        max_height=117 * mm,
    )
    story.append(PageBreak())

    algorithms = (
        "no_dpd",
        "legacy_ilc",
        "linear_ilc",
        "instantaneous_gain_ilc",
        "model_vjp_ilc",
        "model_lm_ilc",
        "oracle_lm",
    )
    endpoint_rows = [("方法", "AM/AM final NMSE", "success", "AM/PM final NMSE", "success")]
    for algorithm in algorithms:
        amam_row = cell_index[("amam", "0.97", algorithm)]
        ampm_row = cell_index[("ampm", "135", algorithm)]
        endpoint_rows.append(
            (
                ALGORITHM_LABELS[algorithm],
                f"{fmt(amam_row['median_final_nmse_db'], 1)} dB",
                f"{fmt(amam_row['success_rate_percent'], 1)}%",
                f"{fmt(ampm_row['median_final_nmse_db'], 1)} dB",
                f"{fmt(ampm_row['success_rate_percent'], 1)}%",
            )
        )
    story.extend(
        [
            paragraph("5 强基线与机制结果", styles, "h1"),
            paragraph(
                "主比较之所以只回答 learned LM 相对 scalar linear ILC 的问题，是因为更强基线显示了"
                "明显不同的排序。Instantaneous-gain ILC 在 AM/AM 0.97 的 median final NMSE 为 -116.6 dB，"
                "不及 learned LM 的 -261.0 dB，但两者 AUEC 只差约 4%、成功率同为 100%；在 AM/PM "
                "135°，instantaneous-gain 的 -318.7 dB 和 100% 成功则远优于 learned LM 的 -33.3 dB "
                "和 67.5%。Oracle LM 两 cell 均 100% 成功，说明"
                "learned LM 的 AM/PM 缺口主要来自在线模型/局部预测，而不是 solver 形式本身。",
                styles,
            ),
            styled_table(endpoint_rows, styles, [45 * mm, 32 * mm, 22 * mm, 37 * mm, 22 * mm]),
            Spacer(1, 4 * mm),
        ]
    )
    add_figure(
        story,
        analysis_root / "confirmatory" / "convergence_main.png",
        "图 2  两个主要 cell 的逐轮 tracking NMSE。线为 paired-seed 中位数，带为 IQR；显示下限裁到 -60 dB，但统计使用未裁值。",
        styles,
        max_height=82 * mm,
    )
    story.extend(
        [
            paragraph("5.1 梯度与模型诊断", styles, "h2"),
            paragraph(
                "所有 synthetic PA 路径统一记录 identity error 与 oracle VJP 的 direction cosine；模型"
                "方法另记录 learned VJP 与 oracle 的 cosine。解析 JVP/VJP 还由独立 PyTorch autograd "
                "oracle 和 finite difference 回归覆盖。结果不把 cosine 当作实际下降的充分条件：raw VJP "
                "在小斜率/饱和区会梯度消失，而 LM 的阻尼、预条件尺度、信赖域和锚定回溯共同决定能否"
                "接受更新。",
                styles,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            paragraph("6 稳健性、失配、消融与动态 PA", styles, "h1"),
            paragraph(
                "Robustness 扫描显示噪声地板直接限制最终 NMSE。在 AM/AM 0.97、SNR=30 dB 且单次"
                "capture 时，learned LM median final NMSE 约 -27.0 dB，成功率 0%，并出现 35% 发散；"
                "同 SNR 使用 10 次 capture 可恢复到约 -37.0 dB 和 100% 成功。这一负结果说明安全结论"
                "不能从无噪声 cell 外推到低 SNR、低平均次数。AM/PM 135° 的 learned LM 成功率也随"
                "SNR/capture 在 0%–90% 间变化。",
                styles,
            ),
        ]
    )
    add_figure(
        story,
        analysis_root / "robustness" / "variant_endpoints.png",
        "图 3  SNR 与 capture count 稳健性端点。次要研究为描述性结果，不升级为确认性推断。",
        styles,
        max_height=107 * mm,
    )
    story.extend(
        [
            paragraph(
                "Memory-polynomial 失配在 AM/AM cell 仍全部成功，但最终 NMSE 由约 -69 dB 到 -285 dB "
                "大幅变化；AM/PM cell 对阶数/记忆深度更敏感，12 个配置 cell 中仅少数组合达到非零"
                "成功率，范围 0%–91.7%。这与 online model validation 和 oracle LM 的差距一致。",
                styles,
            ),
            paragraph(
                "单因素消融没有产生“每个保护都必然更优”的简单排序。AM/PM cell 中 unanchored "
                "prediction 与 legacy dynamic calibration 的成功率均为 0%；raw VJP 也为 0%，支持"
                "“仅有 backward 方向并不足够”的结论。No-ridge、no-trust、replay、frozen model 与"
                "complex64 的端点差异则表明结果还受到拟合条件和数值路径影响；这些只作机制诊断。",
                styles,
            ),
            paragraph(
                "诊断审计发现 three-iteration replay 的一条轨迹在 k=21/24/27 记录了 model_fallback "
                "与 validation_nmse_exceeded，但遗漏 model_fallback_strategy 字符串。实际数值路径仍按"
                "预注册消融定义复用 stale model；31 个评价点、端点和冻结判据不受影响。该项属于非主要"
                "推断中的诊断 schema 缺口。",
                styles,
                "callout",
            ),
        ]
    )
    add_figure(
        story,
        analysis_root / "ablation" / "variant_endpoints.png",
        "图 4  预注册消融的最终 NMSE、成功与安全端点。每个变体只按其冻结定义解释。",
        styles,
        max_height=104 * mm,
    )
    story.append(PageBreak())

    dynamic_index = {(row["scenario"], row["algorithm"]): row for row in dynamic_cells}
    dynamic_rows = [("动态 PA", "方法", "final NMSE", "success")]
    for scenario in ("amam_dynamic", "ampm_dynamic"):
        for algorithm in ("linear_ilc", "instantaneous_gain_ilc", "model_lm_ilc", "oracle_lm"):
            row = dynamic_index[(scenario, algorithm)]
            dynamic_rows.append(
                (
                    scenario,
                    ALGORITHM_LABELS[algorithm],
                    f"{fmt(row['median_final_nmse_db'], 1)} dB",
                    f"{fmt(row['success_rate_percent'], 1)}%",
                )
            )
    story.extend(
        [
            paragraph("6.1 Out-of-family 动态 PA", styles, "h2"),
            paragraph(
                "在动态 AM/AM PA 中 learned LM 达到 100% 成功并优于 linear ILC；instantaneous-gain "
                "在该动态结构中反而为 0%。在动态 AM/PM PA 中 learned LM 仅 35% 成功，oracle LM 为"
                "100%，说明在线 memory-polynomial 对 Wiener/Hammerstein 类结构的近似仍是主限制。",
                styles,
            ),
            styled_table(dynamic_rows, styles, [38 * mm, 65 * mm, 31 * mm, 28 * mm]),
            Spacer(1, 4 * mm),
        ]
    )
    add_figure(
        story,
        analysis_root / "dynamic" / "convergence_study.png",
        "图 5  两个 out-of-family 动态 PA cell 的逐轮收敛。",
        styles,
        max_height=83 * mm,
    )
    story.append(PageBreak())

    stress_rows = [("压力场景", "方法", "final NMSE", "success", "divergence", "guarded stop", "sat.-limited")]
    for scenario in ("gain_rolloff", "hard_saturation"):
        for algorithm in ("linear_ilc", "instantaneous_gain_ilc", "model_vjp_ilc", "model_lm_ilc", "oracle_lm"):
            row = next(item for item in stress_cells if item["scenario"] == scenario and item["algorithm"] == algorithm)
            stress_rows.append(
                (
                    scenario,
                    ALGORITHM_LABELS[algorithm],
                    f"{fmt(row['median_final_nmse_db'], 1)}",
                    f"{fmt(row['success_rate_percent'], 1)}%",
                    f"{fmt(row['divergence_rate_percent'], 1)}%",
                    f"{fmt(row['guarded_safe_stop_rate_percent'], 1)}%",
                    f"{fmt(row['saturation_limited_trajectory_rate_percent'], 1)}%",
                )
            )
    story.extend(
        [
            paragraph("7 独立压力场景", styles, "h1"),
            paragraph(
                "Hard-saturation severity 2.0 把目标明确置于不可达区域。Oracle LM 与 No DPD 的 median "
                "final NMSE 相同（约 -19.93 dB），不是算法失败，而是局部 Jacobian 在硬限幅区无法"
                "提供可达方向。Oracle LM 100% 报告 saturation-limited/guarded stop；learned LM 虽保持"
                "安全，却没有识别该不可达状态（两率均 0%），说明平滑在线模型不能替代 oracle 可达性"
                "判定。Instantaneous-gain 和 legacy 路径则 100% 发散，说明不可达区的评价重点应是"
                "runaway guard，而不是要求 NMSE 归零。",
                styles,
            ),
            paragraph(
                "Gain-rolloff severity 0.4 在包络超过 turnover 后具有局部负斜率。全局 identity-vs-oracle "
                "cosine 未必为负，但 envelope-local negative fraction 明确非零；对 learned LM 轨迹，"
                "identity cosine 中位数为 -0.109，77.2% 观测为负，而 learned-vs-oracle cosine 中位数"
                "仅 0.00076、47.3% 为负。因此论文只声称冻结路径上的局部/轨迹方向错误，不声称所有"
                "方法的全局梯度都反转。Learned LM 把 median final NMSE 改善到约 -28.7 dB，仍只有"
                "8.3% 达到主成功阈值；oracle LM 为 100%。",
                styles,
            ),
            styled_table(stress_rows, styles, [27 * mm, 39 * mm, 22 * mm, 18 * mm, 20 * mm, 21 * mm, 21 * mm]),
            Spacer(1, 4 * mm),
        ]
    )
    add_figure(
        story,
        analysis_root / "stress" / "stress_diagnostics.png",
        "图 6  压力场景的安全停止、saturation-limited 与梯度方向诊断。底图的叉号为 identity-vs-oracle，圆点为 learned-vs-oracle。",
        styles,
        max_height=116 * mm,
    )
    story.append(PageBreak())

    environment = manifest["environment"]
    backend_lines = []
    for backend in environment.get("numeric_backends", []):
        backend_lines.append(
            f"{backend.get('internal_api')}/{backend.get('version')}/{backend.get('architecture')}"
        )
    audit_rows = [
        ("审计项", "结果"),
        ("六个 locked studies", f"{locked_count}/{locked_count} 完整；algorithm failure=0；infrastructure retry=0"),
        ("共同 resolved hash", resolved_hash),
        ("Code / protocol / environment", f"{hashes['code_hash'][:16]}… / {hashes['protocol_hash'][:16]}… / {hashes['environment_hash'][:16]}…"),
        ("数值后端", "; ".join(backend_lines)),
        ("锁定运行累计 elapsed", f"{total_runtime / 60.0:.1f} min（6 workers；含各 study runner 计时）"),
        (
            "容量门",
            "通过；单 worker 数值后端最大线程数=1；六个 locked studies 全局峰值 "
            f"{global_peak_rss_bytes:,} B（{global_peak_rss_bytes / (1024 ** 2):.2f} MiB）",
        ),
        ("分析完整性", "expected IDs、shard checksum、五 hash、matrix reconstruction、pilot provenance 与 root lock 全部重验"),
        ("单元/集成测试", "125/125 通过；含 independent PyTorch autograd JVP/VJP oracle"),
    ]
    story.extend(
        [
            paragraph("8 可复现性、资源与产物治理", styles, "h1"),
            paragraph(
                "复现身份由 code、configuration、protocol、matrix、environment 五个 SHA-256 共同定义。"
                "Environment hash 还包含 Python/NumPy/SciPy/Torch 版本、线程环境与 BLAS/OpenMP 后端身份。"
                "运行窗口持续重算科学源和环境；变化即停止调度。Pilot resolved config 绑定完整 pilot shards、"
                "候选表和重新计算的选择，六个 locked study 再由根级原子 lock 强制共用 resolved hash。",
                styles,
            ),
            styled_table(audit_rows, styles, [45 * mm, 117 * mm]),
            paragraph(
                "两次正式前试跑暴露并修复了线程指纹初始化顺序和 Windows 临时目录枚举竞态。两次修复"
                "都改变科学代码 hash，因此相关旧根被明确作废，全部 pilot 与锁定矩阵在新根重跑。本文"
                "只引用 artifacts/frozen_release。冻结 manifest 记录的 base commit 为 1115c5b，且"
                "dirty=true；因此该 commit 本身不足以复原运行，必须同时核验 manifest 中的完整 code hash。",
                styles,
                "callout",
            ),
            paragraph("9 局限性", styles, "h1"),
            paragraph(
                "（1）全部 PA、噪声和动态 taps 均为代码生成；unity calibration 是合成原生域选择，"
                "不等价于硬件标定。（2）NR-like OFDM 没有 CP、编码、标准资源映射或标准接收机；"
                "sampled-band ACLR 与 known-grid EVM 只用于同协议比较。（3）没有 DC 功耗、PAE、"
                "DAC/ADC、固定点、热效应和实时吞吐测量。（4）在线模型不含 GMP 交叉项，不能覆盖"
                "所有宽带 PA。（5）逐波形 ILC 依赖任务重复性，不能直接外推为任意数据流 DPD。",
                styles,
            ),
            paragraph("10 结论", styles, "h1"),
            paragraph(
                "受保护的 PA-model-backpropagation LM 在两个主要 synthetic cell 中均显著降低了相对"
                "scalar linear ILC 的 AUEC，但预注册联合判定一成一败：AM/PM 135° 通过，AM/AM 0.97 "
                "因成功率 ceiling 而失败。Instantaneous-gain ILC 在 AM/PM 的强端点表现否定了“主方法普遍优于"
                "经典 ILC”的宽泛叙述。Mechanism 证据支持 real-linear VJP、阻尼求解和保护必须作为"
                "组合评价；raw VJP 在饱和区会梯度消失，硬饱和不可达性不能靠 backward 恢复。",
                styles,
            ),
            paragraph(
                "因此，本工作可作为可复现的仿真机制论文/预印本证据，而不能作为真实 PA 线性化、"
                "效率提升、产业部署或标准合规结论。下一步应在冻结相同指标与失败口径的前提下接入"
                "硬件闭环或可调用 testbench，并加入 BLA-inverse 与更完整参数化 DPD 对照。",
                styles,
            ),
            PageBreak(),
        ]
    )

    references = [
        "[1] J. Chani-Cahuana, P. N. Landin, C. Fager, and T. Eriksson, “Iterative Learning Control for RF Power Amplifier Linearization,” IEEE TMTT, 64(9):2778–2789, 2016. doi:10.1109/TMTT.2016.2588483.",
        "[2] M. Schoukens, J. Hammenecker, and A. Cooman, “Obtaining the Preinverse of a Power Amplifier Using Iterative Learning Control,” IEEE TMTT, 65(11):4266–4273, 2017. doi:10.1109/TMTT.2017.2694822.",
        "[3] D. R. Morgan et al., “A Generalized Memory Polynomial Model for Digital Predistortion of RF Power Amplifiers,” IEEE TSP, 54(10):3852–3860, 2006. doi:10.1109/TSP.2006.879264.",
        "[4] C. Tarver et al., “Neural Network DPD via Backpropagation through a Neural Network Model of the PA,” Asilomar, pp. 358–362, 2019. doi:10.1109/IEEECONF44664.2019.9048910.",
        "[5] E. Loebl, N. Ginzberg, and E. Cohen, “Direct Learning Neural Network Digital Predistortion Using Backpropagation Through a Memory Power Amplifier Model,” IMS, pp. 791–794, 2023. doi:10.1109/IMS37964.2023.10187912.",
        "[6] X. Wei et al., “Iterative Learning for RF Power Amplifier Linearization Based on Instantaneous Complex Gain,” IEEE MWTL, 36(1):7–10, 2026. doi:10.1109/LMWT.2025.3620316.",
    ]
    story.append(paragraph("参考文献", styles, "h1"))
    for reference in references:
        story.append(paragraph(escape(reference), styles, "body_no_indent"))
    story.extend(
        [
            Spacer(1, 5 * mm),
            paragraph("附录：关键可复现路径", styles, "h1"),
            paragraph(
                "实现：remote_dpd/pa_model.py、remote_dpd/learning.py；协议与 runner：experiments/config.py、"
                "experiments/runner.py；统计与制图：experiments/statistics.py、experiments/analysis.py、"
                "experiments/plot_results.py；正式数据：artifacts/frozen_release；论文源：paper/main.tex；"
                "本 PDF 构建脚本：paper/build_paper.py。",
                styles,
                "body_no_indent",
            ),
        ]
    )

    document.build(story, onFirstPage=page_decorator, onLaterPages=page_decorator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts") / "frozen_release",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "pdf" / "pa_model_backprop_ilc.pdf",
    )
    args = parser.parse_args()
    build_document(args.artifact_root.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
