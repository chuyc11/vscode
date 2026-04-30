from pydantic import ValidationError
from typing import List
from schema import AttackFinding

def format_validation_error(validation_error: ValidationError) -> str:
    if isinstance(validation_error, ValidationError):
        errors = []
        for error in validation_error.errors():
            loc = ".".join(str(l) for l in error["loc"])
            msg = error["msg"]
            errors.append(f"{loc}: {msg}")
        return "; ".join(errors)
    else:
        return str(validation_error)

def format_assassination_findings(findings: List[AttackFinding]) -> str:
    if not findings:
        return "No findings."
    formatted = []
    for finding in findings:
        formatted.append(f"Claim {finding.claim_id}: {finding.attack_type.value} ({finding.severity.value}) - {finding.description}")
    return "\n".join(formatted)