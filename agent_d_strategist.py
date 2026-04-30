import logging
import os
from openai import OpenAI
from schema_draft import ClaimGraphDraft, ClaimDraft
from schema_views import FactCardViewForD
from typing import List

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
        response_format={"type": "json_schema", "json_schema": {"name": "claim_graph_draft", "schema": ClaimGraphDraft.model_json_schema()}}
    )
    return response.choices[0].message.content

ANALYSIS_SYSTEM_PROMPT = """你是一个高级情报分析策略师。你的任务是基于事实数据构建结构化的论证图谱。

分析框架：
1. 核心问题识别：从 topic 中提炼出需要回答的核心问题
2. 证据链提取：从 facts 中识别关键证据，按相关性和可信度排序
3. 因果推理：构建 claim 之间的逻辑依赖关系（parent_temp_ids 必须反映真实的逻辑依赖）
4. 置信度评估：基于证据质量和推理强度为每个 claim 打分

要求：
- 每个 claim 必须有明确的 reasoning 字段，解释你是如何从事实推导出这个结论的
- confidence 字段反映你对这个 claim 的信心程度（0.0-1.0）
- parent_temp_ids 必须体现真实的逻辑依赖：子 claim 的成立依赖于父 claim
- fact_ids 必须引用下方列出的事实 ID（如 "f_0", "f_1" 等），不能为空列表
- 至少构建 3-5 个 claim，涵盖 topic 的不同维度
- 包含至少一个 PREDICTION 类型的 claim（基于现有证据的合理推断）
- 如果 facts 之间存在张力或矛盾，创建 OPINION 类型 claim 来呈现不同视角"""


def _format_facts(fact_views: List[FactCardViewForD]) -> str:
    lines = []
    for f in fact_views:
        lines.append(
            f'事实 ID: {f.fact_id}\n'
            f'  可信度: {f.credibility_score} | 相关性: {f.relevance_score}\n'
            f'  内容: {f.content}\n'
            f'  摘要: {f.summary}'
        )
    return "\n\n".join(lines)


def _extract_keywords(text: str) -> set:
    """Extract meaningful words (3+ chars) from text for matching."""
    import re
    words = re.findall(r'[a-zA-Z一-鿿]{3,}', text.lower())
    return set(words)


def _match_claims_to_facts(draft: ClaimGraphDraft, fact_views: List[FactCardViewForD]) -> ClaimGraphDraft:
    """Deterministically match claims to facts based on content keyword overlap."""
    fact_keywords = {}
    for f in fact_views:
        fact_keywords[f.fact_id] = _extract_keywords(f.content + " " + f.summary)

    fixed = []
    for d in draft.drafts:
        # Start with LLM-provided fact_ids that are valid
        valid_ids = {f.fact_id for f in fact_views}
        cleaned_fids = [fid for fid in d.fact_ids if fid in valid_ids]

        # If no valid fact_ids, do content-based matching
        if not cleaned_fids:
            claim_kw = _extract_keywords(d.content)
            scores = []
            for fid, fkw in fact_keywords.items():
                overlap = len(claim_kw & fkw)
                if overlap > 0:
                    scores.append((overlap, fid))
            scores.sort(reverse=True)
            for _, fid in scores[:3]:
                cleaned_fids.append(fid)

        # If still no matches (cross-language), assign top facts by relevance
        if not cleaned_fids:
            sorted_facts = sorted(fact_views, key=lambda f: f.relevance_score, reverse=True)
            for f in sorted_facts[:2]:
                cleaned_fids.append(f.fact_id)

        # Ensure reasoning is not empty
        reasoning = d.reasoning
        if not reasoning or reasoning.strip() == "":
            if cleaned_fids:
                matched_contents = [
                    f.content[:60] for f in fact_views if f.fact_id in cleaned_fids
                ]
                reasoning = f"基于 {len(cleaned_fids)} 条事实的分析: {'; '.join(matched_contents)}"
            else:
                reasoning = "无直接事实支撑的推断"

        fixed.append(ClaimDraft(
            temp_id=d.temp_id,
            content=d.content,
            claim_type=d.claim_type,
            parent_temp_ids=d.parent_temp_ids,
            fact_ids=cleaned_fids,
            reasoning=reasoning,
            confidence=d.confidence
        ))
    return ClaimGraphDraft(drafts=fixed)


def generate_claim_graph_draft(fact_views: List[FactCardViewForD], topic: str) -> ClaimGraphDraft:
    facts_text = _format_facts(fact_views)

    prompt = f"""分析主题: {topic}

=== 可用事实数据（必须引用这些事实 ID）===
{facts_text}

请基于上述事实构建 ClaimGraphDraft。

关键要求：
- fact_ids 必须使用上方列出的事实 ID（原样引用，不要修改）
- 每个 claim 至少引用 1 个事实 ID
- reasoning 必须详细说明如何从事实推导到结论"""

    json_str = _call_llm([
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    draft = ClaimGraphDraft.model_validate_json(json_str)
    return _match_claims_to_facts(draft, fact_views)


def repair_claim_graph_draft(
    fact_views: List[FactCardViewForD],
    topic: str,
    validation_error: str
) -> ClaimGraphDraft:
    facts_text = _format_facts(fact_views)

    prompt = f"""分析主题: {topic}

=== 可用事实数据 ===
{facts_text}

之前的 draft 未通过验证，错误信息:
{validation_error}

请修复并重新生成一个有效的 ClaimGraphDraft。fact_ids 必须使用上方的事实 ID。"""

    json_str = _call_llm([
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    draft = ClaimGraphDraft.model_validate_json(json_str)
    return _match_claims_to_facts(draft, fact_views)


def reinforce_draft_after_attack(
    fact_views: List[FactCardViewForD],
    topic: str,
    current_draft: ClaimGraphDraft,
    attack_summary: str
) -> ClaimGraphDraft:
    facts_text = _format_facts(fact_views)
    draft_text = "\n".join(
        f"- [{d.temp_id}] {d.claim_type.value}: {d.content}\n  置信度: {d.confidence}\n  推理: {d.reasoning}\n  引用事实: {d.fact_ids}"
        for d in current_draft.drafts
    )

    prompt = f"""分析主题: {topic}

=== 可用事实数据（必须引用这些事实 ID）===
{facts_text}

=== 当前 draft ===
{draft_text}

=== 攻击分析发现的问题 ===
{attack_summary}

请根据攻击反馈加固 draft：
1. 对被攻击的 claim，补充更充分的推理和证据引用（fact_ids 必须引用上方事实 ID）
2. 如果某个 claim 无法加固，降低其 confidence 或删除
3. 如果攻击发现了逻辑跳跃，增加中间推理步骤的 claim
4. 确保所有 claim 的 reasoning 更加严谨
5. 每个 claim 的 fact_ids 不能为空

返回修复后的 ClaimGraphDraft。"""

    json_str = _call_llm([
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    draft = ClaimGraphDraft.model_validate_json(json_str)
    return _match_claims_to_facts(draft, fact_views)
