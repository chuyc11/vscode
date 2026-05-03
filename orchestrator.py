"""Orchestrator — LangGraph immutable state machine with Redis/zstd persistence.

Refactored from imperative run_pipeline() to LangGraph StateGraph:
- Immutable state: each node returns a partial state update (dict), never mutates.
- Large payloads stored in Redis with zstd compression; only key pointers in state.
- Lifecycle finally hooks clean up all Redis keys on completion or failure.
- D-E iteration loop via conditional edges.

Engineering red lines
---------------------
1. Large payloads (facts, graph, draft, report) stored in Redis, NOT in state.
2. Zstandard (zstd) level 3 compression before Redis write.
3. Absolute TTL (3600s) on all Redis keys.
4. Lifecycle finally hook deletes all temporary keys.
5. Redis must set maxmemory-policy volatile-lru.
6. Backward-compatible: run_pipeline(topic, facts, doc_dir) signature unchanged.
"""

import logging
import operator
from typing import TypedDict, Annotated, Dict, List

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from schema import (
    FactCard, FactCardBatch, QuantFactCardBatch, ClaimGraph, ClaimGraphDraft,
    AssassinationReport, AttackFinding, FinalDecision,
    build_fact_views_for_d,
)
from agent_c_financial import enrich_financial_data_sync as enrich_financial_data
from agent_d_strategist import (
    generate_claim_graph_draft,
    repair_claim_graph_draft,
    reinforce_draft_after_attack,
)
from graph_validation import validate_claim_dag
from graph_hydration import hydrate_claim_graph
from agent_e_assassin import deterministic_attack
from apply_assassination import apply_assassination_report_to_graph
from agent_g_briefing import generate_briefing
from repair_utils import format_validation_error
from state_store import StateStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph immutable state
# ---------------------------------------------------------------------------

MAX_DRAFT_REPAIRS = 2
MAX_ATTACK_ITERATIONS = 2


class PipelineState(TypedDict, total=False):
    """LangGraph immutable state — each node returns partial updates."""
    topic: str
    facts_ref: str                          # Redis key pointer
    quant_facts_ref: str                    # Redis key pointer
    local_fact_count: int
    draft_ref: str                          # Redis key pointer
    draft_validated: bool
    graph_ref: str                          # Redis key pointer
    report_ref: str                         # Redis key pointer
    all_attack_findings: Annotated[List[AttackFinding], operator.add]
    briefing: str
    errors: Annotated[List[str], operator.add]
    status: str                             # INIT | RUNNING | COMPLETED | FAILED
    final_decision: str                     # pass | passed_with_risks | failed | reject
    iteration_count: int
    doc_dir: str


# ---------------------------------------------------------------------------
# Store closure — shared across nodes within a single pipeline run
# ---------------------------------------------------------------------------

_store: StateStore | None = None
_keys_to_cleanup: list[str] = []


def _get_store() -> StateStore:
    global _store
    if _store is None:
        _store = StateStore()
    return _store


def _track_key(key: str) -> None:
    _keys_to_cleanup.append(key)


# ---------------------------------------------------------------------------
# Node functions — pure functions returning partial state updates
# ---------------------------------------------------------------------------

def node_f(state: PipelineState) -> dict:
    """F phase: local document ingestion."""
    if not state.get("doc_dir"):
        return {"local_fact_count": 0}

    logger.info("F phase: ingesting local documents from '%s'", state["doc_dir"])
    store = _get_store()

    try:
        facts = store.get(state["facts_ref"], FactCardBatch).facts
    except KeyError:
        return {"status": "FAILED", "errors": ["F phase: facts_ref expired"]}

    try:
        from agent_f_reader import read_local_documents_sync
        local_facts = read_local_documents_sync(state["doc_dir"], state["topic"])
        for f in local_facts:
            facts[f.fact_id] = f
        logger.info("F phase: %d local FactCards ingested", len(local_facts))

        new_ref = store.put(FactCardBatch(facts=facts))
        _track_key(new_ref)
        return {"facts_ref": new_ref, "local_fact_count": len(local_facts)}

    except Exception as e:
        logger.warning("F phase failed (non-fatal): %s", e)
        return {"errors": [f"F phase: {e}"]}


def node_c(state: PipelineState) -> dict:
    """C phase: financial enrichment."""
    logger.info("C phase: enriching financial data for topic '%s'", state["topic"])
    store = _get_store()

    try:
        facts = store.get(state["facts_ref"], FactCardBatch).facts
    except KeyError:
        return {"status": "FAILED", "errors": ["C phase: facts_ref expired"]}

    try:
        quant_facts = enrich_financial_data(facts)
        logger.info("C phase: %d QuantFactCards produced", len(quant_facts))
        ref = store.put(QuantFactCardBatch(quant_facts=quant_facts))
        _track_key(ref)
        return {"quant_facts_ref": ref}
    except Exception as e:
        logger.warning("C phase failed (non-fatal): %s", e)
        return {"errors": [f"C phase: {e}"]}


def node_d_initial(state: PipelineState) -> dict:
    """D phase: generate initial draft with validation + repair loop."""
    logger.info("D phase: generating claim graph draft")
    store = _get_store()

    try:
        facts = store.get(state["facts_ref"], FactCardBatch).facts
    except KeyError:
        return {"status": "FAILED", "draft_validated": False,
                "errors": ["D phase: facts_ref expired"]}

    fact_views = build_fact_views_for_d(facts)

    draft = None
    last_error = None
    for attempt in range(MAX_DRAFT_REPAIRS + 1):
        try:
            if attempt == 0:
                draft = generate_claim_graph_draft(fact_views, state["topic"])
            else:
                error_str = format_validation_error(last_error)
                logger.info("D phase: repair attempt %d", attempt)
                draft = repair_claim_graph_draft(fact_views, state["topic"], error_str)
            validate_claim_dag(draft)
            logger.info("D phase: draft validated with %d claims", len(draft.drafts))
            break
        except (ValidationError, ValueError) as e:
            last_error = e
            if attempt == MAX_DRAFT_REPAIRS:
                logger.warning("D phase: max repairs reached")
                return {
                    "status": "FAILED",
                    "draft_validated": False,
                    "final_decision": FinalDecision.PASSED_WITH_RISKS.value,
                    "errors": [f"D phase: {e}"],
                }

    if draft is None:
        return {"status": "FAILED", "draft_validated": False,
                "errors": ["D phase: no draft produced"]}

    ref = store.put(draft)
    _track_key(ref)
    return {"draft_ref": ref, "draft_validated": True, "status": "RUNNING"}


def node_d_repair(state: PipelineState) -> dict:
    """D phase: repair draft after failed validation."""
    logger.info("D phase: repairing draft")
    store = _get_store()

    try:
        facts = store.get(state["facts_ref"], FactCardBatch).facts
    except KeyError:
        return {"status": "FAILED", "draft_validated": False,
                "errors": ["D repair: facts_ref expired"]}

    fact_views = build_fact_views_for_d(facts)
    errors = state.get("errors", [])
    last_error = errors[-1] if errors else "Unknown validation error"

    try:
        draft = repair_claim_graph_draft(fact_views, state["topic"], last_error)
        validate_claim_dag(draft)
        ref = store.put(draft)
        _track_key(ref)
        return {"draft_ref": ref, "draft_validated": True}
    except (ValidationError, ValueError) as e:
        logger.warning("D repair failed: %s", e)
        return {"status": "FAILED", "draft_validated": False,
                "final_decision": FinalDecision.PASSED_WITH_RISKS.value,
                "errors": [f"D repair: {e}"]}


def node_hydrate(state: PipelineState) -> dict:
    """Hydration: ClaimGraphDraft → ClaimGraph."""
    logger.info("Hydrating claim graph")
    store = _get_store()

    try:
        facts = store.get(state["facts_ref"], FactCardBatch).facts
        draft = store.get(state["draft_ref"], ClaimGraphDraft)
    except KeyError as e:
        return {"status": "FAILED", "errors": [f"Hydration: {e}"]}

    try:
        graph = hydrate_claim_graph(draft, facts)
        ref = store.put(graph)
        _track_key(ref)
        return {"graph_ref": ref}
    except ValueError as e:
        logger.error("Hydration failed: %s", e)
        return {"status": "FAILED", "final_decision": FinalDecision.FAILED.value,
                "errors": [f"Hydration: {e}"]}


def node_e_attack(state: PipelineState) -> dict:
    """E phase: dual-track adversarial audit."""
    logger.info("E phase: running semantic attack analysis")
    store = _get_store()

    try:
        graph = store.get(state["graph_ref"], ClaimGraph)
    except KeyError:
        return {"status": "FAILED", "errors": ["E phase: graph_ref expired"]}

    report = deterministic_attack(graph)
    ref = store.put(report)
    _track_key(ref)

    logger.info("E phase: %d findings, decision=%s",
                len(report.findings), report.final_decision)

    return {
        "report_ref": ref,
        "final_decision": report.final_decision.value,
        "all_attack_findings": report.findings,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def node_d_reinforce(state: PipelineState) -> dict:
    """D phase: reinforce draft after adversarial attack."""
    logger.info("D phase: reinforcing draft")
    store = _get_store()

    try:
        facts = store.get(state["facts_ref"], FactCardBatch).facts
        draft = store.get(state["draft_ref"], ClaimGraphDraft)
        report = store.get(state["report_ref"], AssassinationReport)
    except KeyError as e:
        return {"status": "FAILED", "errors": [f"D reinforce: {e}"]}

    fact_views = build_fact_views_for_d(facts)
    attack_summary = "\n".join(
        f"- [{f.attack_type.value}] {f.description}" for f in report.findings
    )

    try:
        reinforced = reinforce_draft_after_attack(
            fact_views, state["topic"], draft, attack_summary
        )
        validate_claim_dag(reinforced)
        ref = store.put(reinforced)
        _track_key(ref)
        logger.info("D phase: reinforced draft validated")
        return {"draft_ref": ref, "draft_validated": True}
    except (ValidationError, ValueError) as e:
        logger.warning("D reinforcement failed: %s", e)
        return {"errors": [f"D reinforce: {e}"]}


def node_apply(state: PipelineState) -> dict:
    """Apply assassination results to graph nodes."""
    store = _get_store()

    try:
        graph = store.get(state["graph_ref"], ClaimGraph)
        report = store.get(state["report_ref"], AssassinationReport)
    except KeyError:
        return {"status": "COMPLETED"}

    updated_graph = apply_assassination_report_to_graph(graph, report)
    ref = store.put(updated_graph)
    _track_key(ref)
    return {"graph_ref": ref}


def node_g(state: PipelineState) -> dict:
    """G phase: generate intelligence briefing."""
    logger.info("G phase: generating intelligence briefing")
    store = _get_store()

    try:
        graph = store.get(state["graph_ref"], ClaimGraph)
    except KeyError:
        return {"briefing": "简报生成失败：图数据不可用。",
                "status": "COMPLETED", "errors": ["G phase: graph_ref expired"]}

    all_findings = state.get("all_attack_findings", [])
    briefing = generate_briefing(state["topic"], graph, all_findings)

    final_decision = state.get("final_decision", FinalDecision.PASS.value)
    briefing_mode = "normal" if final_decision == FinalDecision.PASS.value else "with_risks"

    logger.info("Pipeline completed: decision=%s, iterations=%d",
                final_decision, state.get("iteration_count", 0))

    return {"briefing": briefing, "status": "COMPLETED", "briefing_mode": briefing_mode}


# ---------------------------------------------------------------------------
# Conditional edge functions — routing logic
# ---------------------------------------------------------------------------

def _route_after_draft(state: PipelineState) -> str:
    """Route after D phase: validated → hydrate, failed → d_repair or END."""
    if state.get("status") == "FAILED":
        return END
    if state.get("draft_validated"):
        return "hydrate"
    return "d_repair"


def _route_after_attack(state: PipelineState) -> str:
    """Route after E phase: PASS → apply, else → d_reinforce (if iterations remain)."""
    if state.get("status") == "FAILED":
        return "apply"
    if state.get("final_decision") == FinalDecision.PASS.value:
        return "apply"
    if state.get("iteration_count", 0) >= MAX_ATTACK_ITERATIONS:
        return "apply"
    return "d_reinforce"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    g = StateGraph(PipelineState)

    # Nodes
    g.add_node("f", node_f)
    g.add_node("c", node_c)
    g.add_node("d_initial", node_d_initial)
    g.add_node("d_repair", node_d_repair)
    g.add_node("hydrate", node_hydrate)
    g.add_node("e_attack", node_e_attack)
    g.add_node("d_reinforce", node_d_reinforce)
    g.add_node("apply", node_apply)
    g.add_node("g", node_g)

    # Linear edges
    g.set_entry_point("f")
    g.add_edge("f", "c")
    g.add_edge("c", "d_initial")
    g.add_edge("hydrate", "e_attack")
    g.add_edge("apply", "g")
    g.add_edge("g", END)

    # D phase: validation routing
    g.add_conditional_edges("d_initial", _route_after_draft)
    g.add_conditional_edges("d_repair", _route_after_draft)

    # E phase: attack routing with D-E iteration loop
    g.add_conditional_edges("e_attack", _route_after_attack)

    # Reinforce loops back through hydration → attack
    g.add_edge("d_reinforce", "hydrate")

    return g


# ---------------------------------------------------------------------------
# Public entry point — backward-compatible signature
# ---------------------------------------------------------------------------

def run_pipeline(topic: str, facts: Dict[str, FactCard], doc_dir: str = "") -> dict:
    """Run the full A-J intelligence pipeline.

    Backward-compatible: same signature as before.
    Returns a ResearchState-like dict with all results.

    Lifecycle: finally hook cleans up all Redis keys.
    """
    global _store, _keys_to_cleanup
    _store = None  # reset so _get_store() creates fresh
    _keys_to_cleanup = []
    store = _get_store()

    try:
        # Serialize initial facts to Redis
        facts_ref = store.put(FactCardBatch(facts=facts))
        _keys_to_cleanup.append(facts_ref)

        initial_state: PipelineState = {
            "topic": topic,
            "facts_ref": facts_ref,
            "doc_dir": doc_dir,
            "local_fact_count": 0,
            "status": "INIT",
            "iteration_count": 0,
            "draft_validated": False,
            "errors": [],
            "all_attack_findings": [],
        }

        graph = build_graph().compile()
        result = graph.invoke(initial_state)

        # Hydrate result into backward-compatible ResearchState-like dict
        return _hydrate_result(result, store)

    finally:
        store.cleanup(*_keys_to_cleanup)
        _keys_to_cleanup = []


class _ResearchStateCompat:
    """Backward-compatible wrapper matching old ResearchState interface."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _hydrate_result(result: PipelineState, store: StateStore) -> _ResearchStateCompat:
    """Convert LangGraph state back to ResearchState-compatible object."""

    # Deserialize objects from Redis
    graph = None
    if result.get("graph_ref"):
        try:
            graph = store.get(result["graph_ref"], ClaimGraph)
        except KeyError:
            pass

    report = None
    if result.get("report_ref"):
        try:
            report = store.get(result["report_ref"], AssassinationReport)
        except KeyError:
            pass

    draft = None
    if result.get("draft_ref"):
        try:
            draft = store.get(result["draft_ref"], ClaimGraphDraft)
        except KeyError:
            pass

    facts = {}
    if result.get("facts_ref"):
        try:
            facts = store.get(result["facts_ref"], FactCardBatch).facts
        except KeyError:
            pass

    final_decision = None
    fd_str = result.get("final_decision")
    if fd_str:
        try:
            final_decision = FinalDecision(fd_str)
        except ValueError:
            pass

    return _ResearchStateCompat(
        topic=result.get("topic", ""),
        facts=facts,
        quant_facts=[],
        local_fact_count=result.get("local_fact_count", 0),
        draft=draft,
        graph=graph,
        report=report,
        all_attack_findings=result.get("all_attack_findings", []),
        briefing=result.get("briefing"),
        errors=result.get("errors", []),
        status=result.get("status", "INIT"),
        final_decision=final_decision,
        briefing_mode=result.get("briefing_mode"),
        iteration_count=result.get("iteration_count", 0),
    )
