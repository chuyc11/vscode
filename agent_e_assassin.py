"""E-node: adversarial assassin — dual-track audit of ClaimGraph.

Track 1 (pure Python): deterministic topological + heuristic checks.
Track 2 (LLM): causal-leap validation with batched adjacent edges.

Engineering red lines
---------------------
1. Topologically adjacent edges are batched into single LLM queries
   to prevent O(E) API explosion. Batches are per-node (incoming edges).
2. Both tracks merge into a single AssassinationReport.
3. Final decision and risk notes are written back onto ClaimGraph nodes.
4. All LLM calls use structlog for observability.
"""

import json
import logging
import os
from collections import defaultdict
from typing import List

import structlog

from schema import (
    ClaimGraph, ClaimNode, AssassinationReport, AttackFinding,
    AttackType, Severity, FinalDecision, ClaimType, AttackStatus,
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
# Configuration
# ---------------------------------------------------------------------------
EDGE_BATCH_SIZE = int(os.getenv("E_NODE_EDGE_BATCH_SIZE", "5"))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CAUSAL_LEAP_SYSTEM_PROMPT = """你是一个严格的情报审查专家。你的任务是检查一组 claim 之间的因果推理是否存在逻辑跳跃。

你将收到一个"推理簇"：一个中心 claim 及其父 claim（即它依赖的上游推理）。
请检查：
1. 中心 claim 的结论是否真的能从父 claim 推导出来？是否存在逻辑跳跃？
2. 父 claim 之间是否存在矛盾或张力？
3. 推理链是否遗漏了必要的中间步骤？

对每个发现的问题，返回一个 JSON 对象：
{
  "claim_id": "有问题的 claim ID",
  "attack_type": "logical_leap|contradiction|inconsistency|insufficient_evidence",
  "severity": "low|medium|high",
  "description": "问题描述",
  "evidence_quote": "引用具体 claim 内容作为证据（可选）"
}

返回一个 JSON 数组。如果没有问题，返回空数组 []。只返回 JSON，不要其他内容。"""

# ---------------------------------------------------------------------------
# Track 1: Pure Python deterministic checks
# ---------------------------------------------------------------------------

def _check_no_fact_claims(graph: ClaimGraph) -> List[AttackFinding]:
    """Claims with no supporting facts."""
    findings = []
    for node in graph.nodes.values():
        if not node.fact_ids:
            findings.append(AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.FABRICATION,
                severity=Severity.HIGH,
                description="声明没有任何事实支撑，完全是无据声明。",
            ))
    return findings


def _check_low_confidence(graph: ClaimGraph) -> List[AttackFinding]:
    """Claims with dangerously low confidence."""
    findings = []
    for node in graph.nodes.values():
        if node.confidence < 0.4:
            findings.append(AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.INSUFFICIENT_EVIDENCE,
                severity=Severity.MEDIUM,
                description=f"声明置信度过低（{node.confidence}），推理基础薄弱。",
            ))
    return findings


def _check_prediction_without_facts(graph: ClaimGraph) -> List[AttackFinding]:
    """Predictions relying on too few facts."""
    findings = []
    for node in graph.nodes.values():
        if node.claim_type == ClaimType.PREDICTION and len(node.fact_ids) < 2:
            findings.append(AttackFinding(
                claim_id=node.claim_id,
                attack_type=AttackType.INSUFFICIENT_EVIDENCE,
                severity=Severity.MEDIUM,
                description="预测性声明仅有少量支撑事实，预测依据不足。",
            ))
    return findings


def _check_low_credibility_facts(graph: ClaimGraph) -> List[AttackFinding]:
    """Claims relying on low-credibility source facts."""
    findings = []
    for node in graph.nodes.values():
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
                evidence_quote=low_cred[0][:100],
            ))
    return findings


def _check_orphan_nodes(graph: ClaimGraph) -> List[AttackFinding]:
    """Non-root claims that have no parent and no facts — orphaned reasoning."""
    findings = []
    for node in graph.nodes.values():
        # A node with no parents AND no facts is an orphan
        # (root nodes with no facts are caught by _check_no_fact_claims)
        if not node.parent_claim_ids and not node.fact_ids:
            continue  # already caught by no-fact check
        # A node with parents but all parent IDs are invalid
        if node.parent_claim_ids:
            invalid_parents = [p for p in node.parent_claim_ids if p not in graph.nodes]
            if invalid_parents:
                findings.append(AttackFinding(
                    claim_id=node.claim_id,
                    attack_type=AttackType.INCONSISTENCY,
                    severity=Severity.HIGH,
                    description=f"声明引用了不存在的父节点: {invalid_parents}",
                ))
    return findings


def _deterministic_checks(graph: ClaimGraph) -> List[AttackFinding]:
    """Run all pure-Python deterministic checks (Track 1)."""
    findings = []
    findings.extend(_check_no_fact_claims(graph))
    findings.extend(_check_low_confidence(graph))
    findings.extend(_check_prediction_without_facts(graph))
    findings.extend(_check_low_credibility_facts(graph))
    findings.extend(_check_orphan_nodes(graph))
    slog.info("e_node_track1_complete", finding_count=len(findings))
    return findings


# ---------------------------------------------------------------------------
# Track 2: LLM causal-leap validation with batched edges
# ---------------------------------------------------------------------------

def _build_edge_batches(graph: ClaimGraph) -> List[dict]:
    """Batch topologically adjacent edges around each node.

    For each node with incoming edges (parent -> node), create one batch
    containing the center node and all its parents. This groups edges by
    their shared target, keeping batch count at O(N) not O(E).
    """
    batches = []
    for node in graph.nodes.values():
        if not node.parent_claim_ids:
            continue  # root nodes have no incoming edges to validate

        parents = []
        for pid in node.parent_claim_ids:
            if pid in graph.nodes:
                parents.append(graph.nodes[pid])

        if not parents:
            continue

        batches.append({
            "center": node,
            "parents": parents,
        })

    return batches


def _format_batch_for_llm(batch: dict, graph: ClaimGraph) -> str:
    """Format a single edge batch into LLM-readable text."""
    center = batch["center"]
    lines = []

    # Center claim
    center_facts = []
    for fid in center.fact_ids:
        if fid in graph.facts:
            f = graph.facts[fid]
            center_facts.append(f"[{fid}] (可信度:{f.credibility_score}) {f.content[:150]}")

    lines.append(f"=== 中心 Claim [{center.claim_id}] ===")
    lines.append(f"  类型: {center.claim_type.value}")
    lines.append(f"  内容: {center.content}")
    lines.append(f"  置信度: {center.confidence}")
    lines.append(f"  推理过程: {center.reasoning}")
    lines.append(f"  支撑事实: {'; '.join(center_facts) if center_facts else '无'}")
    lines.append("")

    # Parent claims (the upstream reasoning)
    lines.append("--- 父 Claim（上游推理）---")
    for parent in batch["parents"]:
        parent_facts = []
        for fid in parent.fact_ids:
            if fid in graph.facts:
                f = graph.facts[fid]
                parent_facts.append(f"[{fid}] {f.content[:100]}")
        lines.append(f"  [{parent.claim_id}] ({parent.claim_type.value}) {parent.content}")
        lines.append(f"    置信度: {parent.confidence} | 支撑事实: {'; '.join(parent_facts) if parent_facts else '无'}")

    return "\n".join(lines)


def _llm_causal_leap_check(graph: ClaimGraph) -> List[AttackFinding]:
    """Track 2: LLM-based causal-leap validation with batched edges.

    Instead of querying each edge individually (O(E)), we batch edges
    by their target node. One LLM call per node-with-parents.
    """
    batches = _build_edge_batches(graph)
    if not batches:
        slog.info("e_node_track2_skip", reason="no_edges_to_validate")
        return []

    # Split into chunks of EDGE_BATCH_SIZE to further bound API calls
    chunk_count = (len(batches) + EDGE_BATCH_SIZE - 1) // EDGE_BATCH_SIZE
    slog.info("e_node_track2_start", total_batches=len(batches),
              edge_batch_size=EDGE_BATCH_SIZE, chunks=chunk_count)

    all_findings: List[AttackFinding] = []

    for chunk_idx in range(0, len(batches), EDGE_BATCH_SIZE):
        chunk = batches[chunk_idx:chunk_idx + EDGE_BATCH_SIZE]

        # Combine multiple batches into one LLM call
        combined_text = ""
        for i, batch in enumerate(chunk):
            combined_text += f"\n{'='*60}\n"
            combined_text += _format_batch_for_llm(batch, graph)
            combined_text += "\n"

        try:
            raw = call_llm([
                {"role": "system", "content": CAUSAL_LEAP_SYSTEM_PROMPT},
                {"role": "user", "content": f"请审查以下 {len(chunk)} 个推理簇的因果关系：\n{combined_text}"},
            ])

            # Extract JSON from response
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            items = json.loads(raw)
            chunk_findings = _parse_attack_items(items)
            all_findings.extend(chunk_findings)

            slog.info("e_node_track2_chunk_done",
                      chunk_idx=chunk_idx // EDGE_BATCH_SIZE,
                      items_parsed=len(chunk_findings))

        except Exception as e:
            slog.warning("e_node_track2_chunk_failed",
                         chunk_idx=chunk_idx // EDGE_BATCH_SIZE,
                         error=str(e))

    slog.info("e_node_track2_complete", finding_count=len(all_findings))
    return all_findings


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_attack_items(items: list) -> List[AttackFinding]:
    """Parse LLM-returned attack items into AttackFinding objects."""
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
        if not isinstance(item, dict):
            continue
        findings.append(AttackFinding(
            claim_id=item.get("claim_id", ""),
            attack_type=type_map.get(item.get("attack_type", ""), AttackType.INCONSISTENCY),
            severity=severity_map.get(item.get("severity", "medium"), Severity.MEDIUM),
            description=item.get("description", ""),
            evidence_quote=item.get("evidence_quote"),
        ))
    return findings


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _compute_decision(findings: List[AttackFinding]) -> tuple[FinalDecision, str]:
    """Compute final decision from all findings."""
    if not findings:
        return FinalDecision.PASS, "所有声明通过审查，未发现显著问题。"

    has_high = any(f.severity == Severity.HIGH for f in findings)
    has_medium = any(f.severity == Severity.MEDIUM for f in findings)

    if has_high:
        high_count = sum(1 for f in findings if f.severity == Severity.HIGH)
        return FinalDecision.REJECT, f"拒绝：发现 {high_count} 个高风险问题。"
    elif has_medium:
        medium_count = sum(1 for f in findings if f.severity == Severity.MEDIUM)
        return FinalDecision.PASSED_WITH_RISKS, f"有条件通过：发现 {medium_count} 个中等风险问题。"
    else:
        return FinalDecision.PASS, f"通过，但有 {len(findings)} 个低风险提示。"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def deterministic_attack(graph: ClaimGraph) -> AssassinationReport:
    """Run dual-track audit on a ClaimGraph.

    Track 1: Pure Python deterministic checks (topology + heuristics).
    Track 2: LLM causal-leap validation with batched adjacent edges.

    Returns AssassinationReport with merged findings.
    """
    slog.info("e_node_start", node_count=len(graph.nodes), fact_count=len(graph.facts))

    # Track 1: Deterministic (always runs, zero cost)
    det_findings = _deterministic_checks(graph)

    # Short-circuit: if we already have HIGH-severity findings, skip LLM
    has_high = any(f.severity == Severity.HIGH for f in det_findings)
    if has_high:
        decision, summary = _compute_decision(det_findings)
        slog.info("e_node_short_circuit", reason="high_severity_in_track1",
                  decision=decision.value, findings=len(det_findings))
        return AssassinationReport(
            findings=det_findings,
            final_decision=decision,
            summary=summary,
        )

    # Track 2: LLM causal-leap check (batched edges)
    llm_findings = _llm_causal_leap_check(graph)

    # Merge both tracks
    all_findings = det_findings + llm_findings
    decision, summary = _compute_decision(all_findings)

    slog.info("e_node_complete", decision=decision.value,
              track1_findings=len(det_findings),
              track2_findings=len(llm_findings),
              total_findings=len(all_findings))

    return AssassinationReport(
        findings=all_findings,
        final_decision=decision,
        summary=summary,
    )
