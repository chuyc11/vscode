import logging
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Optional, List
from schema import FactCard, ClaimGraph, AssassinationReport, FinalDecision, AttackFinding
from schema_draft import ClaimGraphDraft
from schema_views import build_fact_views_for_d
from agent_d_strategist import generate_claim_graph_draft, repair_claim_graph_draft, reinforce_draft_after_attack
from graph_validation import validate_claim_dag
from graph_hydration import hydrate_claim_graph
from agent_e_assassin import deterministic_attack
from apply_assassination import apply_assassination_report_to_graph
from agent_g_briefing import generate_briefing
from repair_utils import format_validation_error
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class ResearchState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    facts: Dict[str, FactCard]
    draft: Optional[ClaimGraphDraft] = None
    graph: Optional[ClaimGraph] = None
    report: Optional[AssassinationReport] = None
    all_attack_findings: List[AttackFinding] = Field(default_factory=list)
    briefing: Optional[str] = None
    errors: list = Field(default_factory=list)
    status: str = "INIT"
    final_decision: Optional[FinalDecision] = None
    briefing_mode: Optional[str] = None
    iteration_count: int = 0

def run_pipeline(topic: str, facts: Dict[str, FactCard]) -> ResearchState:
    state = ResearchState(topic=topic, facts=facts)
    fact_views = build_fact_views_for_d(facts)

    # === D phase: initial draft ===
    logger.info("D phase: generating claim graph draft for topic '%s'", topic)
    max_revisions = 2
    for attempt in range(max_revisions + 1):
        try:
            if attempt == 0:
                state.draft = generate_claim_graph_draft(fact_views, topic)
            else:
                error_str = format_validation_error(state.errors[-1])
                logger.info("D phase: repair attempt %d", attempt)
                state.draft = repair_claim_graph_draft(fact_views, topic, error_str)
            validate_claim_dag(state.draft)
            logger.info("D phase: draft validated successfully with %d claims", len(state.draft.drafts))
            break
        except (ValidationError, ValueError) as e:
            state.errors.append(e)
            if attempt == max_revisions:
                logger.warning("D phase: max revisions reached, returning with risks")
                state.status = "COMPLETED"
                state.final_decision = FinalDecision.PASSED_WITH_RISKS
                state.briefing_mode = "with_risks"
                state.briefing = "Draft 未能通过验证，简报基于有限分析生成。"
                return state

    # === D-E iteration loop ===
    max_attack_iterations = 2
    for iteration in range(max_attack_iterations + 1):
        state.iteration_count = iteration + 1
        logger.info("=== Iteration %d ===", iteration + 1)

        # Hydrate
        logger.info("Hydrating claim graph")
        try:
            state.graph = hydrate_claim_graph(state.draft, facts)
        except ValueError as e:
            logger.error("Hydration failed: %s", e)
            state.status = "FAILED"
            state.final_decision = FinalDecision.FAILED
            return state

        # E phase: attack
        logger.info("E phase: running semantic attack analysis")
        state.report = deterministic_attack(state.graph)
        state.all_attack_findings.extend(state.report.findings)
        logger.info("E phase: %d findings, decision=%s", len(state.report.findings), state.report.final_decision)

        if state.report.final_decision == FinalDecision.PASS:
            logger.info("All claims passed attack analysis")
            break

        # REJECT or PASSED_WITH_RISKS: try to reinforce if iterations remain
        if iteration < max_attack_iterations:
            attack_summary = "\n".join(
                f"- [{f.attack_type.value}] {f.description}" for f in state.report.findings
            )
            logger.info("D phase: reinforcing draft (iteration %d)", iteration + 1)
            try:
                state.draft = reinforce_draft_after_attack(fact_views, topic, state.draft, attack_summary)
                validate_claim_dag(state.draft)
                logger.info("D phase: reinforced draft validated")
                continue
            except (ValidationError, ValueError) as e:
                logger.warning("D phase: reinforcement failed: %s", e)
                state.errors.append(e)
                break
        else:
            logger.info("Max iterations reached, proceeding to briefing")
            break

    # Apply final attack results to graph
    state.graph = apply_assassination_report_to_graph(state.graph, state.report)
    state.final_decision = state.report.final_decision

    # === G phase: always generate briefing ===
    logger.info("G phase: generating intelligence briefing")
    state.briefing = generate_briefing(topic, state.graph, state.all_attack_findings)
    state.status = "COMPLETED"
    state.briefing_mode = "normal" if state.final_decision == FinalDecision.PASS else "with_risks"

    logger.info("Pipeline completed: status=%s, decision=%s, iterations=%d",
                state.status, state.final_decision, state.iteration_count)
    return state
