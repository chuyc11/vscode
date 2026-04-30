from schema import ClaimGraph, AssassinationReport, AttackStatus
from copy import deepcopy

def apply_assassination_report_to_graph(graph: ClaimGraph, report: AssassinationReport) -> ClaimGraph:
    new_graph = deepcopy(graph)
    claim_findings = {}
    for finding in report.findings:
        if finding.claim_id not in claim_findings:
            claim_findings[finding.claim_id] = []
        claim_findings[finding.claim_id].append(finding)

    for claim_id, node in new_graph.nodes.items():
        if claim_id in claim_findings:
            node.attack_status = AttackStatus.ATTACKED
            for finding in claim_findings[claim_id]:
                note = f"{finding.attack_type.value}: {finding.description}"
                if finding.evidence_quote:
                    note += f' [证据: "{finding.evidence_quote}"]'
                node.risk_notes.append(note)

    return new_graph
