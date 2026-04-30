import json
import logging
import os
from ddgs import DDGS
from openai import OpenAI
from schema import FactCard, SourceTier, EvidenceType, make_id, utc_now

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

def _search_raw(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results)

def _score_and_summarize(topic: str, raw_results: list[dict]) -> list[FactCard]:
    if not raw_results:
        return []

    items_text = "\n".join(
        f"[{i}] {r['title']}: {r['body']}" for i, r in enumerate(raw_results)
    )

    prompt = f"""你是一个情报分析专家。请对以下搜索结果进行评估。

分析主题: {topic}

搜索结果:
{items_text}

请对每条结果返回 JSON 数组，每条包含:
- index: 原始索引
- relevance: 与主题的相关性 (0.0-1.0)
- credibility: 来源可信度 (0.0-1.0)，根据来源权威性判断
- summary: 一句话说明这条信息与主题的关系

只返回 JSON 数组，不要其他内容。"""

    try:
        raw = _call_llm([{"role": "user", "content": prompt}])
        # Extract JSON from potential markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        scores = json.loads(raw)
    except Exception as e:
        logger.warning("LLM scoring failed: %s, using default scores", e)
        scores = [
            {"index": i, "relevance": 0.6, "credibility": 0.6, "summary": r.get("body", "")[:80]}
            for i, r in enumerate(raw_results)
        ]

    facts = []
    for item in scores:
        idx = item.get("index", 0)
        if idx >= len(raw_results):
            continue
        relevance = float(item.get("relevance", 0.5))
        if relevance < 0.5:
            continue
        r = raw_results[idx]
        fact = FactCard(
            fact_id=make_id(),
            content=f"{r['title']}: {r['body']}",
            source_tier=SourceTier.SECONDARY,
            evidence_type=EvidenceType.DOCUMENT,
            timestamp=utc_now(),
            credibility_score=float(item.get("credibility", 0.6)),
            relevance_score=relevance,
            summary=item.get("summary", "")
        )
        facts.append(fact)
    return facts

def search_real_news(query: str, max_results: int = 5) -> list[FactCard]:
    # Round 1: main query
    try:
        raw_results = _search_raw(query, max_results)
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning("Network search failed: %s", e)
        return _make_fallback_facts()

    if not raw_results:
        logger.warning("No results for main query, trying broader search")
        try:
            raw_results = _search_raw(f"{query} analysis report", max_results)
        except Exception:
            return _make_fallback_facts()

    facts = _score_and_summarize(query, raw_results)

    # Round 2: supplement with entity-focused search if few results
    if len(facts) < 3:
        try:
            supplement = _search_raw(f"{query} latest news trends", max_results)
            extra_facts = _score_and_summarize(query, supplement)
            # Deduplicate by content similarity (simple check)
            existing_contents = {f.content[:50] for f in facts}
            for ef in extra_facts:
                if ef.content[:50] not in existing_contents:
                    facts.append(ef)
                    existing_contents.add(ef.content[:50])
        except Exception as e:
            logger.info("Supplemental search failed: %s", e)

    if not facts:
        logger.warning("All scoring filtered out, using fallback")
        return _make_fallback_facts()

    logger.info("Retrieved %d relevant facts from %d raw results", len(facts), len(raw_results))
    return facts

def _make_fallback_facts() -> list[FactCard]:
    contents = [
        "Industry analysis shows AI chip demand continues to rise, benefiting major GPU manufacturers.",
        "Market analysts report increased capital expenditure in AI infrastructure across multiple sectors.",
        "Recent partnerships between AI companies and hardware suppliers may drive significant revenue growth.",
    ]
    return [
        FactCard(
            fact_id=make_id(),
            content=c,
            source_tier=SourceTier.TERTIARY,
            evidence_type=EvidenceType.DOCUMENT,
            timestamp=utc_now(),
            credibility_score=0.4,
            relevance_score=0.5,
            summary="备用情报（网络检索失败时启用）"
        )
        for c in contents
    ]
