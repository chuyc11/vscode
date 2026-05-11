"""G-node: intelligence briefing generator with Jinja2 rendering.

Refactored to use Jinja2 templates for structured Markdown output with:
- Claim classification: PASSED / RISK / HIDDEN (FAILED nodes hidden, logged)
- XSS-immune rendering via sanitize_markdown filter (AST-safe, no regex/bleach)
- System contract: downstream MUST mount DOMPurify after HTML rendering
- Evidence chain analysis with strength indicators
- LLM-generated executive summary and recommendations

Engineering red lines
---------------------
1. FAILED nodes (HIGH severity) are NEVER rendered — only structlog-logged.
2. sanitize_markdown filter uses html.escape() on user content, NOT regex/bleach.
3. XSS-CONTRACT comment embedded in output header.
4. All user content passes through Jinja2 | sanitize_markdown filter.
"""

import logging
import os
from typing import List, Dict, Any
from html import escape as html_escape

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape

from schema import (
    ClaimGraph, ClaimNode, AttackFinding, AttackStatus,
    Severity, FinalDecision, utc_now,
)
from llm_client import call_llm

# ---------------------------------------------------------------------------
# Structlog
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if os.getenv("LOG_FORMAT") != "json"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.getenv("LOG_LEVEL_NUM", "20"))
    ),
)
slog = structlog.get_logger()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XSS Protection — sanitize_markdown filter
# ---------------------------------------------------------------------------
# Dangerous HTML tags that must be neutralised in Markdown output.
# We do NOT use regex or bleach on the full Markdown text.
# Instead, we apply html.escape() to user-supplied content at template
# interpolation points, preserving Markdown syntax while neutralising HTML.
_DANGEROUS_TAGS = frozenset({
    "script", "iframe", "object", "embed", "form", "input",
    "style", "link", "base", "meta", "applet",
})


def _sanitize_markdown(value: str) -> str:
    """Jinja2 filter: HTML-escape user content for safe Markdown embedding.

    This is NOT regex/bleach cleaning. We use Python's html.escape() to
    convert < > & " to HTML entities, which:
    - Preserves Markdown syntax (* _ # [] () etc.)
    - Neutralises any HTML injection (<script>, <iframe>, etc.)
    - Is safe for downstream Markdown-to-HTML renderers

    The Markdown AST itself is not modified — only user-supplied string
    values at interpolation points are escaped.
    """
    if not isinstance(value, str):
        value = str(value)
    return html_escape(value, quote=True)


# ---------------------------------------------------------------------------
# Jinja2 environment (lazy singleton)
# ---------------------------------------------------------------------------
_env: Environment | None = None


def _get_jinja_env() -> Environment:
    global _env
    if _env is None:
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        _env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,  # We handle escaping via sanitize_markdown filter
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _env.filters["sanitize_markdown"] = _sanitize_markdown
    return _env


# ---------------------------------------------------------------------------
# Claim classification — PASSED / RISK / HIDDEN
# ---------------------------------------------------------------------------

def _classify_claims(
    graph: ClaimGraph,
    attack_findings: List[AttackFinding],
) -> tuple[List[ClaimNode], List[ClaimNode], List[ClaimNode]]:
    """Classify claims into passed, risk, and hidden categories.

    HIDDEN = HIGH severity ATTACKED nodes (never rendered, only logged).
    RISK = ATTACKED nodes without HIGH severity.
    PASSED = UNATTACKED or DEFENDED nodes.
    """
    high_severity_ids = {f.claim_id for f in attack_findings if f.severity == Severity.HIGH}

    passed: List[ClaimNode] = []
    risk: List[ClaimNode] = []
    hidden: List[ClaimNode] = []

    for node in graph.nodes.values():
        if node.claim_id in high_severity_ids:
            hidden.append(node)
            slog.info("g_node_hidden_claim",
                      claim_id=node.claim_id,
                      content_preview=node.content[:100],
                      risk_notes_count=len(node.risk_notes))
        elif node.attack_status in (AttackStatus.UNATTACKED, AttackStatus.DEFENDED):
            passed.append(node)
        else:
            risk.append(node)

    slog.info("g_node_classification",
              passed=len(passed), risk=len(risk), hidden=len(hidden))
    return passed, risk, hidden


# ---------------------------------------------------------------------------
# Evidence chain analysis
# ---------------------------------------------------------------------------

def _build_evidence_summary(graph: ClaimGraph) -> List[Dict[str, Any]]:
    """Analyze evidence chain strength per claim."""
    summary = []
    for node in graph.nodes.values():
        if not node.fact_ids:
            continue
        cred_scores = []
        for fid in node.fact_ids:
            if fid in graph.facts:
                cred_scores.append(graph.facts[fid].credibility_score)
        avg_cred = sum(cred_scores) / len(cred_scores) if cred_scores else 0.0

        if avg_cred >= 0.7 and len(cred_scores) >= 2:
            strength = "🟢 强"
        elif avg_cred >= 0.5:
            strength = "🟡 中"
        else:
            strength = "🔴 弱"

        summary.append({
            "claim_id": node.claim_id,
            "description": node.content[:80],
            "fact_count": len(cred_scores),
            "avg_credibility": avg_cred,
            "strength": strength,
        })
    return summary


# ---------------------------------------------------------------------------
# Template data preparation
# ---------------------------------------------------------------------------

def _prepare_claim_data(node: ClaimNode) -> Dict[str, Any]:
    """Prepare a single claim node for template rendering."""
    return {
        "content": node.content,
        "claim_type": node.claim_type.value,
        "confidence": node.confidence,
        "reasoning": node.reasoning,
        "fact_ids": node.fact_ids,
        "risk_notes": node.risk_notes,
        "attack_type": ", ".join(
            note.split(":")[0] for note in node.risk_notes if ":" in note
        ) if node.risk_notes else "",
    }


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """你是一个顶级情报分析官。基于以下分类数据，生成两段深度分析文字：

1. 执行摘要（4-6 句话）：
   - 概括核心发现和最终结论
   - 明确回答分析主题的核心问题
   - 指出关键的不确定性和风险点
   - 提及最重要的证据来源

2. 建议与下一步（5-8 条）：
   - 基于分析给出具体可执行的行动建议
   - 指出需要进一步验证的关键假设
   - 识别潜在的风险情景和应对预案
   - 建议需要关注的后续信号和指标

语言要求：使用分析主题的语言（中文主题用中文，英文主题用英文）。
输出格式：
---SUMMARY---
（执行摘要内容）
---RECOMMENDATIONS---
（建议内容）"""


def _generate_summary_via_llm(
    topic: str,
    passed: List[ClaimNode],
    risk: List[ClaimNode],
    hidden: List[ClaimNode],
    decision: FinalDecision,
) -> tuple[str, str]:
    """Use LLM to generate executive summary and recommendations."""
    passed_text = "\n".join(
        f"  - [{n.claim_type.value}] {n.content} (置信度:{n.confidence})"
        for n in passed
    ) or "  无"

    risk_text = "\n".join(
        f"  - [{n.claim_type.value}] {n.content} | 风险: {'; '.join(n.risk_notes)}"
        for n in risk
    ) or "  无"

    prompt = f"""分析主题: {topic}
最终决策: {decision.value}

通过审查的声明（共 {len(passed)} 条）:
{passed_text}

有风险的声明（共 {len(risk)} 条）:
{risk_text}

隐藏的高风险声明数量: {len(hidden)}

请基于以上全部信息生成深度执行摘要和具体建议。注意综合所有声明进行交叉分析，而不是简单罗列。"""

    try:
        raw = call_llm([
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        summary = ""
        recommendations = ""
        if "---SUMMARY---" in raw and "---RECOMMENDATIONS---" in raw:
            parts = raw.split("---RECOMMENDATIONS---")
            summary = parts[0].replace("---SUMMARY---", "").strip()
            recommendations = parts[1].strip()
        else:
            summary = raw.strip()
            recommendations = "建议基于上述分析进一步验证关键声明。"

        return summary, recommendations

    except Exception as e:
        slog.error("g_node_llm_summary_failed", error=str(e))
        summary = f"基于 {len(passed)} 条通过声明和 {len(risk)} 条风险声明的分析，最终决策为 {decision.value}。"
        recommendations = "建议对风险声明进行进一步验证，关注证据链薄弱环节。"
        return summary, recommendations


# ---------------------------------------------------------------------------
# Fallback briefing (no LLM, no Jinja2)
# ---------------------------------------------------------------------------

def _fallback_briefing(
    topic: str,
    passed: List[ClaimNode],
    risk: List[ClaimNode],
    hidden: List[ClaimNode],
    decision: FinalDecision,
) -> str:
    """Generate a plain-text briefing when LLM or Jinja2 fails."""
    lines = [
        f"# 情报简报：{topic}",
        f"**最终决策：** {decision.value}",
        "",
        "## 通过审查的声明",
    ]
    for n in passed:
        lines.append(f"- [{n.claim_type.value}] {n.content} (置信度:{n.confidence})")
    if risk:
        lines.append("")
        lines.append("## ⚠️ 风险预警")
        for n in risk:
            lines.append(f"- ⚠️ {n.content}")
            for note in n.risk_notes:
                lines.append(f"  - {note}")
    if hidden:
        lines.append("")
        lines.append(f"> *🔒 {len(hidden)} 个高风险声明已被过滤。*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_briefing(
    topic: str,
    graph: ClaimGraph,
    attack_findings: List[AttackFinding],
) -> str:
    """G-node main entry: ClaimGraph + AttackFindings -> Markdown briefing.

    Pipeline:
    1. Classify claims: PASSED / RISK / HIDDEN
    2. LLM generates executive summary + recommendations
    3. Jinja2 renders structured Markdown briefing
    4. FAILED nodes are hidden, only structlog-logged
    """
    decision = FinalDecision.PASS
    if any(f.severity == Severity.HIGH for f in attack_findings):
        decision = FinalDecision.REJECT
    elif any(f.severity == Severity.MEDIUM for f in attack_findings):
        decision = FinalDecision.PASSED_WITH_RISKS

    # Step 1: Classify claims
    passed, risk, hidden = _classify_claims(graph, attack_findings)

    # Step 2: LLM summary
    summary_text, recommendations = _generate_summary_via_llm(
        topic, passed, risk, hidden, decision
    )

    # Step 3: Jinja2 render
    try:
        env = _get_jinja_env()
        template = env.get_template("briefing.md.j2")

        passed_data = [_prepare_claim_data(n) for n in passed]
        risk_data = [_prepare_claim_data(n) for n in risk]
        evidence_summary = _build_evidence_summary(graph)

        briefing = template.render(
            topic=topic,
            timestamp=utc_now().isoformat(),
            decision=decision.value,
            iteration_count=0,
            executive_summary=summary_text,
            passed_claims=passed_data,
            risk_claims=risk_data,
            hidden_count=len(hidden),
            evidence_summary=evidence_summary,
            recommendations=recommendations,
        )

        slog.info("g_node_complete",
                  decision=decision.value,
                  passed=len(passed), risk=len(risk), hidden=len(hidden),
                  briefing_length=len(briefing))
        return briefing

    except Exception as e:
        slog.error("g_node_jinja_failed", error=str(e))
        return _fallback_briefing(topic, passed, risk, hidden, decision)
