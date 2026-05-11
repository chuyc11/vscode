"""D-node: strategist — claim graph draft generation with temporal priority,
semantic routing, and Map-Reduce for large fact sets.

Refactored to use Pydantic + NVIDIA NIM API with:
- Strong temporal priority override (newest facts first)
- Semantic Routing (facts categorised → specialised prompts)
- Map-Reduce when fact count exceeds threshold
- tenacity Repair Loop with full 'sandwich prompt' error logging via structlog
- Strict guided_json output for ClaimGraphDraft

Engineering red lines
---------------------
1. All LLM calls go through call_llm_json (NVIDIA NIM guided_json).
2. Repair Loop uses tenacity; consecutive failures trigger structlog full-prompt dump.
3. Map-Reduce splits large fact sets into batches, generates partial drafts, merges.
4. Temporal priority: timestamp-sorted facts ensure newest evidence dominates.
"""

import logging
import os
import re
import uuid
from typing import List

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
)

from schema import ClaimGraphDraft, ClaimDraft, FactCardViewForD
from llm_client import call_llm_json

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
# Configuration
# ---------------------------------------------------------------------------
MAP_REDUCE_THRESHOLD = int(os.getenv("D_NODE_MAP_REDUCE_THRESHOLD", "15"))
MAP_BATCH_SIZE = int(os.getenv("D_NODE_MAP_BATCH_SIZE", "8"))
REPAIR_MAX_ATTEMPTS = int(os.getenv("D_NODE_REPAIR_MAX_ATTEMPTS", "3"))

_JSON_SCHEMA = {"name": "claim_graph_draft", "schema": ClaimGraphDraft.model_json_schema()}

# ---------------------------------------------------------------------------
# System prompt — the "sandwich" base
# ---------------------------------------------------------------------------
ANALYSIS_SYSTEM_PROMPT = """你是一个顶级情报分析策略师，负责构建深度结构化论证图谱。

分析框架（必须严格执行）：
1. 核心问题识别：从 topic 中提炼出 3-5 个需要回答的核心子问题
2. 证据链提取：从 facts 中识别所有关键证据，按相关性和可信度排序
3. 多维推理：从不同角度（政治、经济、军事、社会、历史、国际反应）分析同一问题
4. 因果推理：构建 claim 之间的逻辑依赖关系（parent_temp_ids 必须反映真实的逻辑依赖）
5. 置信度评估：基于证据质量和推理强度为每个 claim 打分
6. 张力分析：识别事实之间的矛盾和不同立场的冲突

要求：
- 至少构建 8-12 个 claim，涵盖 topic 的所有重要维度
- 每个 claim 的 reasoning 字段必须详细解释推理过程（至少 2-3 句话），不能只是复述事实
- 必须包含至少 2 个 PREDICTION 类型的 claim（基于现有证据的合理推断）
- 必须包含至少 2 个 OPINION 类型的 claim（呈现不同立场和视角）
- 如果 facts 之间存在张力或矛盾，必须创建 OPINION claim 来呈现冲突
- parent_temp_ids 必须体现真实的逻辑依赖：子 claim 的成立依赖于父 claim
- fact_ids 必须引用下方列出的事实 ID，不能为空列表
- confidence 字段反映你对这个 claim 的信心程度（0.0-1.0），不同 claim 之间必须有区分度"""

TEMPORAL_PRIORITY_ADDENDUM = """
【时序优先级覆盖】
- 事实已按时间戳降序排列（最新在前），最新事实具有更高的分析权重
- 对于时间敏感的 claim（如市场动态、事件进展），必须优先引用较新的事实
- 如果新旧事实存在矛盾，以较新的事实为准"""

SEMANTIC_ROUTING_ADDENDUMS = {
    "financial": """
【语义路由：金融分析】
- 重点关注股价、营收、估值、市场情绪等金融指标
- 创建 PREDICTION 类型 claim 时需引用具体数值
- 注意区分已确认数据和预期数据""",
    "geopolitical": """
【语义路由：地缘政治分析】
- 重点关注国家间关系、政策变化、国际事件、历史背景
- 从多方视角分析（当事国、盟友、对手、国际组织）
- 创建 OPINION 类型 claim 来呈现不同立场和潜在冲突
- 创建 PREDICTION 类型 claim 来推断事态发展方向
- 注意识别信息来源的立场偏向和宣传意图
- 分析事件的短期和长期地缘政治影响""",
    "technology": """
【语义路由：科技行业分析】
- 重点关注技术趋势、产品发布、竞争格局
- 创建 FACT 类型 claim 时需引用具体的技术指标
- 注意区分已发布产品和传闻""",
    "general": """
【语义路由：通用分析】
- 综合分析各维度的事实
- 确保 claim 覆盖 topic 的不同方面""",
}

MERGE_SYSTEM_PROMPT = """你是一个高级情报分析策略师。你的任务是合并多个部分的 ClaimGraphDraft 成一个完整的论证图谱。

合并规则：
1. 去除重复或高度相似的 claim（保留更完整的版本）
2. 建立跨批次的逻辑依赖关系（parent_temp_ids）
3. 确保最终图谱的 claim 覆盖所有批次的关键发现
4. 保持 temp_id 的唯一性
5. 每个 claim 的 fact_ids 不能为空"""


# ---------------------------------------------------------------------------
# Semantic Routing — categorise facts by content
# ---------------------------------------------------------------------------

_FINANCIAL_KEYWORDS = {
    "stock", "price", "revenue", "earnings", "market", "trading", "investor",
    "valuation", "pe ratio", "eps", "dividend", "ipo", "merger", "acquisition",
    "股价", "营收", "盈利", "市值", "估值", "投资", "上市", "并购",
}
_GEOPOLITICAL_KEYWORDS = {
    "sanction", "tariff", "embargo", "diplomat", "treaty", "conflict", "war",
    "military", "nato", "united nations", "geopolitical", "sovereignty",
    "制裁", "关税", "外交", "条约", "冲突", "军事", "地缘",
}
_TECH_KEYWORDS = {
    "ai", "artificial intelligence", "chip", "semiconductor", "quantum",
    "blockchain", "cloud", "software", "hardware", "algorithm", "neural",
    "芯片", "半导体", "人工智能", "量子", "算法", "大模型",
}


def _classify_facts(fact_views: List[FactCardViewForD]) -> str:
    """Classify the dominant semantic category of a batch of facts."""
    text_pool = " ".join(f.content.lower() + " " + f.summary.lower() for f in fact_views)
    scores = {
        "financial": sum(1 for kw in _FINANCIAL_KEYWORDS if kw in text_pool),
        "geopolitical": sum(1 for kw in _GEOPOLITICAL_KEYWORDS if kw in text_pool),
        "technology": sum(1 for kw in _TECH_KEYWORDS if kw in text_pool),
    }
    best = max(scores, key=scores.get)
    if scores[best] < 2:
        return "general"
    return best


# ---------------------------------------------------------------------------
# Temporal priority override — sort facts newest-first
# ---------------------------------------------------------------------------

def _sort_by_temporal_priority(fact_views: List[FactCardViewForD]) -> List[FactCardViewForD]:
    """Sort facts by timestamp descending (newest first) for temporal priority."""
    return sorted(fact_views, key=lambda f: f.timestamp, reverse=True)


# ---------------------------------------------------------------------------
# Fact formatting with temporal ordering markers
# ---------------------------------------------------------------------------

def _format_facts(fact_views: List[FactCardViewForD]) -> str:
    lines = []
    for i, f in enumerate(fact_views):
        priority_tag = "【最新】" if i == 0 else ""
        lines.append(
            f'事实 ID: {f.fact_id} {priority_tag}\n'
            f'  时间戳: {f.timestamp}\n'
            f'  可信度: {f.credibility_score} | 相关性: {f.relevance_score}\n'
            f'  来源层级: {f.source_tier} | 证据类型: {f.evidence_type}\n'
            f'  内容: {f.content}\n'
            f'  摘要: {f.summary}'
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Keyword extraction and claim-to-fact matching (deterministic)
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> set:
    """Extract meaningful words (3+ chars) from text for matching."""
    words = re.findall(r'[a-zA-Z一-鿿]{3,}', text.lower())
    return set(words)


def _match_claims_to_facts(draft: ClaimGraphDraft, fact_views: List[FactCardViewForD]) -> ClaimGraphDraft:
    """Deterministically match claims to facts based on content keyword overlap."""
    fact_keywords = {}
    for f in fact_views:
        fact_keywords[f.fact_id] = _extract_keywords(f.content + " " + f.summary)

    fixed = []
    for d in draft.drafts:
        valid_ids = {f.fact_id for f in fact_views}
        cleaned_fids = [fid for fid in d.fact_ids if fid in valid_ids]

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

        if not cleaned_fids:
            sorted_facts = sorted(fact_views, key=lambda f: f.relevance_score, reverse=True)
            for f in sorted_facts[:2]:
                cleaned_fids.append(f.fact_id)

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


# ---------------------------------------------------------------------------
# tenacity retry for Repair Loop with sandwich prompt logging
# ---------------------------------------------------------------------------

def _log_sandwich_on_failure(messages: list, trace_id: str) -> None:
    """Log the full sandwich prompt (system + user) via structlog on consecutive failures."""
    system_content = messages[0]["content"] if messages else ""
    user_content = messages[1]["content"] if len(messages) > 1 else ""
    slog.error(
        "d_node_sandwich_prompt_failure",
        trace_id=trace_id,
        system_prompt_length=len(system_content),
        user_prompt_length=len(user_content),
        system_prompt_preview=system_content[:500],
        user_prompt_preview=user_content[:500],
    )


def _is_retryable_llm(exc: BaseException) -> bool:
    """Retry on any LLM call failure."""
    return True  # Retry on all exceptions during repair


@retry(
    retry=retry_if_exception(_is_retryable_llm),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(REPAIR_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm_with_retry(messages: list, trace_id: str) -> str:
    """Call LLM with tenacity retry; on final failure, dump sandwich prompt."""
    try:
        return call_llm_json(messages, _JSON_SCHEMA)
    except Exception:
        _log_sandwich_on_failure(messages, trace_id)
        raise


# ---------------------------------------------------------------------------
# Core LLM call — builds sandwich prompt and parses response
# ---------------------------------------------------------------------------

def _generate_draft_from_llm(
    topic: str,
    facts_text: str,
    system_prompt: str,
    user_prompt_extra: str,
    trace_id: str,
) -> ClaimGraphDraft:
    """Build sandwich prompt, call LLM with guided_json, parse response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""分析主题: {topic}

=== 可用事实数据（必须引用这些事实 ID）===
{facts_text}

{user_prompt_extra}"""},
    ]

    json_str = _call_llm_with_retry(messages, trace_id)
    return ClaimGraphDraft.model_validate_json(json_str)


# ---------------------------------------------------------------------------
# Map-Reduce for large fact sets
# ---------------------------------------------------------------------------

def _split_into_batches(fact_views: List[FactCardViewForD], batch_size: int) -> List[List[FactCardViewForD]]:
    """Split facts into batches for Map-Reduce."""
    return [fact_views[i:i + batch_size] for i in range(0, len(fact_views), batch_size)]


def _map_batch(
    batch: List[FactCardViewForD],
    batch_idx: int,
    topic: str,
    trace_id: str,
) -> ClaimGraphDraft:
    """Map phase: generate partial draft from a single batch."""
    category = _classify_facts(batch)
    system_prompt = ANALYSIS_SYSTEM_PROMPT + TEMPORAL_PRIORITY_ADDENDUM + SEMANTIC_ROUTING_ADDENDUMS[category]

    facts_text = _format_facts(batch)
    user_extra = f"""这是第 {batch_idx + 1} 批事实数据（共 {len(batch)} 条）。
请基于这批事实生成部分 ClaimGraphDraft。注意：
- temp_id 使用 "batch{batch_idx}_claim_N" 格式以避免跨批次冲突
- fact_ids 必须使用上方列出的事实 ID"""

    slog.info("d_node_map_batch", trace_id=trace_id,
              batch_idx=batch_idx, batch_size=len(batch), category=category)

    return _generate_draft_from_llm(topic, facts_text, system_prompt, user_extra, trace_id)


def _reduce_drafts(
    partial_drafts: List[ClaimGraphDraft],
    topic: str,
    all_fact_views: List[FactCardViewForD],
    trace_id: str,
) -> ClaimGraphDraft:
    """Reduce phase: merge multiple partial drafts into one."""
    draft_text = ""
    for i, draft in enumerate(partial_drafts):
        draft_text += f"\n=== 第 {i + 1} 批 draft ===\n"
        for d in draft.drafts:
            draft_text += (
                f"- [{d.temp_id}] {d.claim_type.value}: {d.content}\n"
                f"  置信度: {d.confidence} | 推理: {d.reasoning}\n"
                f"  引用事实: {d.fact_ids}\n"
            )

    all_facts_text = _format_facts(all_fact_views)

    messages = [
        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"""分析主题: {topic}

=== 所有事实数据 ===
{all_facts_text}

=== 各批次 draft ===
{draft_text}

请合并以上所有批次的 draft 为一个完整的 ClaimGraphDraft。
要求：
1. 去除重复 claim
2. 建立跨批次的 parent_temp_ids 关系
3. 确保 temp_id 唯一（可以重新编号为 "claim_0", "claim_1" 等）
4. 每个 claim 的 fact_ids 必须引用上方的事实 ID"""},
    ]

    slog.info("d_node_reduce", trace_id=trace_id,
              partial_drafts=len(partial_drafts), total_drafts=sum(len(d.drafts) for d in partial_drafts))

    json_str = _call_llm_with_retry(messages, trace_id)
    return ClaimGraphDraft.model_validate_json(json_str)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_claim_graph_draft(
    fact_views: List[FactCardViewForD],
    topic: str,
    trace_id: str | None = None,
) -> ClaimGraphDraft:
    """Generate claim graph draft from facts.

    Pipeline:
    1. Temporal priority override (sort newest-first)
    2. Semantic routing (classify facts → specialised prompt)
    3. Map-Reduce if fact count exceeds threshold
    4. guided_json output via NVIDIA NIM
    5. Deterministic claim-to-fact matching
    """
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:12]

    slog.info("d_node_start", trace_id=trace_id, topic=topic, fact_count=len(fact_views))

    # Step 1: Temporal priority override
    sorted_views = _sort_by_temporal_priority(fact_views)

    # Step 2: Map-Reduce or direct generation
    if len(sorted_views) > MAP_REDUCE_THRESHOLD:
        slog.info("d_node_map_reduce_triggered", trace_id=trace_id,
                  threshold=MAP_REDUCE_THRESHOLD, fact_count=len(sorted_views))
        batches = _split_into_batches(sorted_views, MAP_BATCH_SIZE)
        partial_drafts = []
        for i, batch in enumerate(batches):
            partial = _map_batch(batch, i, topic, trace_id)
            partial_drafts.append(partial)
        draft = _reduce_drafts(partial_drafts, topic, sorted_views, trace_id)
    else:
        # Direct generation with semantic routing
        category = _classify_facts(sorted_views)
        system_prompt = ANALYSIS_SYSTEM_PROMPT + TEMPORAL_PRIORITY_ADDENDUM + SEMANTIC_ROUTING_ADDENDUMS[category]

        facts_text = _format_facts(sorted_views)
        user_extra = """请基于上述事实构建 ClaimGraphDraft。

关键要求：
- fact_ids 必须使用上方列出的事实 ID（原样引用，不要修改）
- 每个 claim 至少引用 1 个事实 ID
- reasoning 必须详细说明如何从事实推导到结论"""

        slog.info("d_node_direct_generation", trace_id=trace_id, category=category)
        draft = _generate_draft_from_llm(topic, facts_text, system_prompt, user_extra, trace_id)

    # Step 3: Deterministic claim-to-fact matching
    draft = _match_claims_to_facts(draft, sorted_views)

    slog.info("d_node_complete", trace_id=trace_id, claims=len(draft.drafts))
    return draft


def repair_claim_graph_draft(
    fact_views: List[FactCardViewForD],
    topic: str,
    validation_error: str,
    trace_id: str | None = None,
) -> ClaimGraphDraft:
    """Repair a draft that failed validation, with tenacity retry and error logging."""
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:12]

    slog.info("d_node_repair", trace_id=trace_id, error_preview=validation_error[:200])

    sorted_views = _sort_by_temporal_priority(fact_views)
    facts_text = _format_facts(sorted_views)

    system_prompt = ANALYSIS_SYSTEM_PROMPT + TEMPORAL_PRIORITY_ADDENDUM
    user_extra = f"""之前的 draft 未通过验证，错误信息:
{validation_error}

请修复并重新生成一个有效的 ClaimGraphDraft。fact_ids 必须使用上方的事实 ID。"""

    draft = _generate_draft_from_llm(topic, facts_text, system_prompt, user_extra, trace_id)
    draft = _match_claims_to_facts(draft, sorted_views)

    slog.info("d_node_repair_complete", trace_id=trace_id, claims=len(draft.drafts))
    return draft


def reinforce_draft_after_attack(
    fact_views: List[FactCardViewForD],
    topic: str,
    current_draft: ClaimGraphDraft,
    attack_summary: str,
    trace_id: str | None = None,
) -> ClaimGraphDraft:
    """Reinforce draft after adversarial attack, with tenacity retry."""
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:12]

    slog.info("d_node_reinforce", trace_id=trace_id, attack_preview=attack_summary[:200])

    sorted_views = _sort_by_temporal_priority(fact_views)
    facts_text = _format_facts(sorted_views)

    draft_text = "\n".join(
        f"- [{d.temp_id}] {d.claim_type.value}: {d.content}\n  置信度: {d.confidence}\n  推理: {d.reasoning}\n  引用事实: {d.fact_ids}"
        for d in current_draft.drafts
    )

    system_prompt = ANALYSIS_SYSTEM_PROMPT + TEMPORAL_PRIORITY_ADDENDUM
    user_extra = f"""=== 当前 draft ===
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

    draft = _generate_draft_from_llm(topic, facts_text, system_prompt, user_extra, trace_id)
    draft = _match_claims_to_facts(draft, sorted_views)

    slog.info("d_node_reinforce_complete", trace_id=trace_id, claims=len(draft.drafts))
    return draft
