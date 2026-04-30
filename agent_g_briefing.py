import logging
import os
from openai import OpenAI
from typing import Dict, Any, List
from schema import ClaimGraph, AttackStatus, AttackFinding

logger = logging.getLogger(__name__)

def _get_client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not set")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

_model = os.getenv("NIM_DEEP_MODEL", "meta/llama-3.1-70b-instruct")

def _call_llm(messages: list) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=_model,
        messages=messages,
    )
    return response.choices[0].message.content

BRIEFING_SYSTEM_PROMPT = """你是一个高级情报分析官。你的任务是基于审查后的 claim 图谱和攻击分析结果，生成一份专业的情报简报。

简报结构（严格遵循）：

1. **执行摘要**（2-3 句话）
   - 概括核心发现和最终结论
   - 明确回答分析主题的核心问题

2. **关键发现**（按重要性排列）
   - 每个发现需有事实论据支撑
   - 标注置信度和推理路径

3. **证据链分析**
   - 展示 claim 之间的逻辑关系
   - 指出证据链中的强环节和弱环节

4. **风险与不确定性**
   - 明确哪些结论有风险
   - 说明风险来源（证据不足/逻辑跳跃/信息矛盾等）
   - 被攻击的 claim 需详细说明攻击理由

5. **建议与下一步**
   - 基于分析给出行动建议
   - 指出需要进一步验证的方向

语言要求：
- 使用分析主题的语言（中文主题用中文，英文主题用英文）
- 专业、客观、有深度
- 避免空洞的概括，每个观点都要有具体论据"""


def build_briefing_data(graph: ClaimGraph, attack_findings: List[AttackFinding]) -> Dict[str, Any]:
    passed = []
    attacked = []

    for node in graph.nodes.values():
        claim_data = {
            "content": node.content,
            "claim_type": node.claim_type.value,
            "confidence": node.confidence,
            "reasoning": node.reasoning,
            "fact_ids": node.fact_ids,
            "risk_notes": node.risk_notes
        }
        if node.attack_status in (AttackStatus.UNATTACKED, AttackStatus.DEFENDED):
            passed.append(claim_data)
        else:
            attacked.append(claim_data)

    facts_text = []
    for fact in graph.facts.values():
        facts_text.append({
            "content": fact.content,
            "credibility": fact.credibility_score,
            "relevance": fact.relevance_score,
            "summary": fact.summary
        })

    findings_text = []
    for f in attack_findings:
        finding = {
            "claim_id": f.claim_id,
            "type": f.attack_type.value,
            "severity": f.severity.value,
            "description": f.description
        }
        if f.evidence_quote:
            finding["evidence_quote"] = f.evidence_quote
        findings_text.append(finding)

    return {
        "passed_claims": passed,
        "attacked_claims": attacked,
        "facts": facts_text,
        "attack_findings": findings_text
    }


def generate_briefing(topic: str, graph: ClaimGraph, attack_findings: List[AttackFinding]) -> str:
    data = build_briefing_data(graph, attack_findings)

    claims_text = ""
    for i, c in enumerate(data["passed_claims"], 1):
        claims_text += f"\n  [{i}] ({c['claim_type']}, 置信度:{c['confidence']}) {c['content']}\n    推理: {c['reasoning']}"

    attacked_text = ""
    for i, c in enumerate(data["attacked_claims"], 1):
        attacked_text += f"\n  [{i}] ({c['claim_type']}, 置信度:{c['confidence']}) {c['content']}\n    推理: {c['reasoning']}\n    风险: {'; '.join(c['risk_notes'])}"

    facts_text = ""
    for f in data["facts"]:
        facts_text += f"\n  - (可信度:{f['credibility']}, 相关性:{f['relevance']}) {f['content']}\n    摘要: {f['summary']}"

    findings_text = ""
    for f in data["attack_findings"]:
        eq = f'\n    证据引用: "{f["evidence_quote"]}"' if f.get("evidence_quote") else ""
        findings_text += f"\n  - [{f['severity']}] {f['type']}: {f['description']}{eq}"

    prompt = f"""分析主题: {topic}

通过审查的声明:
{claims_text if claims_text else "  无"}

被攻击的声明:
{attacked_text if attacked_text else "  无"}

支撑事实:
{facts_text}

攻击发现:
{findings_text if findings_text else "  无"}

请基于以上数据生成一份完整的情报简报。"""

    try:
        briefing = _call_llm([
            {"role": "system", "content": BRIEFING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        return briefing
    except Exception as e:
        logger.error("Briefing generation failed: %s", e)
        return _fallback_briefing(topic, data)


def _fallback_briefing(topic: str, data: Dict[str, Any]) -> str:
    lines = [f"情报简报: {topic}\n"]
    lines.append("通过的声明:")
    for c in data["passed_claims"]:
        lines.append(f"  - {c['content']} (置信度: {c['confidence']})")
    if data["attacked_claims"]:
        lines.append("\n有风险的声明:")
        for c in data["attacked_claims"]:
            lines.append(f"  - {c['content']}")
            for r in c["risk_notes"]:
                lines.append(f"    风险: {r}")
    if data["attack_findings"]:
        lines.append("\n攻击发现:")
        for f in data["attack_findings"]:
            lines.append(f"  - [{f['severity']}] {f['description']}")
    return "\n".join(lines)
