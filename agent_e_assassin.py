import json
import logging
from schema import (
    ClaimGraph, AssassinationReport, AttackFinding,
    AttackType, Severity, FinalDecision, ClaimType
)
from llm_client import call_llm

logger = logging.getLogger(__name__)

ATTACK_SYSTEM_PROMPT = """你是一个严格的情报审查专家（"刺客"）。你的任务是对每个 claim 进行深度语义攻击分析。

攻击维度：
1. 事实支撑度：声明与引用的事实是否真正支撑（而非仅有关联）？
2. 逻辑跳跃：推理链中是否有不合理的跳跃？
3. 选择性偏差：是否只选了有利的事实，忽略了反面证据？
4. 预测依据：预测性声明的依据是否充分？
5. 事实矛盾：多个引用的事实之间是否有矛盾？

对于每个发现的问题，返回：
- claim_id: 被攻击的 claim ID
- attack_type: contradiction/inconsistency/fabrication/misattribution/logical_leap/selection_bias/insufficient_evidence
- severity: low/medium/high
- description: 详细描述问题所在
- evidence_quote: 引用具体事实内容作为攻击证据（如果适用）

只返回 JSON 数组，不要其他内容。如果没有发现问题，返回空数组 []。"""


def deterministic_attack(graph: ClaimGraph) -> AssassinationReport:
    # Quick check: claims with no facts at all
    no_fact_claims = [
        node for node in graph.nodes.values() if not node.fact_ids
    ]
    if no_fact_claims:
        findings = [
            AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.FABRICATION,
                severity=Severity.HIGH,
                description="声明没有任何事实支撑，完全是无据声明。",
                evidence_quote=None
            )
            for node in no_fact_claims
        ]
        return AssassinationReport(
            findings=findings,
            final_decision=FinalDecision.REJECT,
            summary=f"拒绝：{len(findings)} 个声明完全没有事实支撑。"
        )

    # Build context for LLM attack
    claims_text = ""
    for node in graph.nodes.values():
        fact_contents = []
        for fid in node.fact_ids:
            if fid in graph.facts:
                f = graph.facts[fid]
                fact_contents.append(f"[{fid}] (可信度:{f.credibility_score}) {f.content}")
        facts_str = "\n    ".join(fact_contents) if fact_contents else "无"
        claims_text += f"""
Claim [{node.claim_id}]:
  类型: {node.claim_type.value}
  内容: {node.content}
  置信度: {node.confidence}
  推理过程: {node.reasoning}
  支撑事实:
    {facts_str}
"""

    prompt = f"""请对以下 claim 图谱进行严格审查：

{claims_text}

请逐个分析每个 claim，返回发现的问题列表（JSON 数组）。"""

    try:
        raw = call_llm([
            {"role": "system", "content": ATTACK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        # Extract JSON
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        items = json.loads(raw)
    except Exception as e:
        logger.warning("LLM attack analysis failed: %s, falling back to basic checks", e)
        items = []

    # Also run basic checks
    basic_findings = _basic_checks(graph)
    llm_findings = _parse_attack_items(items)

    all_findings = basic_findings + llm_findings

    if not all_findings:
        return AssassinationReport(
            findings=[],
            final_decision=FinalDecision.PASS,
            summary="所有声明通过审查，未发现显著问题。"
        )

    has_high = any(f.severity == Severity.HIGH for f in all_findings)
    has_medium = any(f.severity == Severity.MEDIUM for f in all_findings)

    if has_high:
        decision = FinalDecision.REJECT
        high_count = sum(1 for f in all_findings if f.severity == Severity.HIGH)
        summary = f"拒绝：发现 {high_count} 个高风险问题。"
    elif has_medium:
        decision = FinalDecision.PASSED_WITH_RISKS
        medium_count = sum(1 for f in all_findings if f.severity == Severity.MEDIUM)
        summary = f"有条件通过：发现 {medium_count} 个中等风险问题。"
    else:
        decision = FinalDecision.PASS
        summary = f"通过，但有 {len(all_findings)} 个低风险提示。"

    return AssassinationReport(
        findings=all_findings,
        final_decision=decision,
        summary=summary
    )


def _basic_checks(graph: ClaimGraph) -> list[AttackFinding]:
    findings = []
    for node in graph.nodes.values():
        # Low confidence claims
        if node.confidence < 0.4:
            findings.append(AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.INSUFFICIENT_EVIDENCE,
                severity=Severity.MEDIUM,
                description=f"声明置信度过低（{node.confidence}），推理基础薄弱。",
                evidence_quote=None
            ))
        # Prediction without multiple supporting facts
        if node.claim_type == ClaimType.PREDICTION and len(node.fact_ids) < 2:
            findings.append(AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.INSUFFICIENT_EVIDENCE,
                severity=Severity.MEDIUM,
                description="预测性声明仅有少量支撑事实，预测依据不足。",
                evidence_quote=None
            ))
        # Claims relying on low-credibility facts
        low_cred = [
            graph.facts[fid].content
            for fid in node.fact_ids
            if fid in graph.facts and graph.facts[fid].credibility_score < 0.5
        ]
        if low_cred:
            findings.append(AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.MISATTRIBUTION,
                severity=Severity.MEDIUM,
                description=f"声明依赖 {len(low_cred)} 条低可信度事实（< 0.5）。",
                evidence_quote=low_cred[0][:100]
            ))
    return findings


def _parse_attack_items(items: list) -> list[AttackFinding]:
    severity_map = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}
    type_map = {
        "contradiction": AttackType.CONTRADICTION,
        "inconsistency": AttackType.INCONSISTENCY,
        "fabrication": AttackType.FABRICATION,
        "misattribution": AttackType.MISATTRIBUTION,
        "logical_leap": AttackType.LOGICAL_LEAP,
        "selection_bias": AttackType.SELECTION_BIAS,
        "insufficient_evidence": AttackType.INSUFFICIENT_EVIDENCE,
    }
    findings = []
    for item in items:
        findings.append(AttackFinding(
            claim_id=item.get("claim_id", ""),
            attack_type=type_map.get(item.get("attack_type", ""), AttackType.INCONSISTENCY),
            severity=severity_map.get(item.get("severity", "medium"), Severity.MEDIUM),
            description=item.get("description", ""),
            evidence_quote=item.get("evidence_quote")
        ))
    return findings
