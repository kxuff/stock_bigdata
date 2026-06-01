from typing import Any

from app.schemas.decision import ValidationResult
from app.schemas.request import AdvisoryDecisionRequest


FORBIDDEN_CLAIMS = ("guaranteed profit", "risk-free", "certain return")


def validate_financial_output(
    payload: dict[str, Any],
    *,
    request: AdvisoryDecisionRequest | None = None,
) -> ValidationResult:
    violations: list[str] = []

    if payload.get("not_financial_advice") is not True:
        violations.append("not_financial_advice must be true")

    _validate_forbidden_claims(payload, violations)

    allocations = payload.get("portfolio_allocation")
    if allocations is not None:
        _validate_portfolio_allocation(allocations, request=request, violations=violations)

    return ValidationResult(passed=not violations, violations=violations)


def _validate_portfolio_allocation(
    allocations: Any,
    *,
    request: AdvisoryDecisionRequest | None,
    violations: list[str],
) -> None:
    if not isinstance(allocations, list) or not allocations:
        violations.append("portfolio_allocation must be a non-empty list")
        return

    weights: list[float] = []
    for index, allocation in enumerate(allocations):
        if not isinstance(allocation, dict):
            violations.append(f"portfolio_allocation[{index}] must be an object")
            continue

        weight = allocation.get("weight_pct")
        if not isinstance(weight, int | float):
            violations.append(f"portfolio_allocation[{index}].weight_pct must be numeric")
            continue

        weight_float = float(weight)
        weights.append(weight_float)
        if weight_float < 0:
            violations.append(f"portfolio_allocation[{index}].weight_pct cannot be negative")

        symbol = str(allocation.get("symbol", "")).upper()
        if request is not None and symbol != "CASH":
            max_weight = request.user_context.max_single_asset_weight
            if weight_float > max_weight:
                violations.append(
                    f"portfolio_allocation[{index}].weight_pct exceeds max_single_asset_weight {max_weight}"
                )

    total_weight = sum(weights)
    if abs(total_weight - 100.0) > 0.01:
        violations.append(f"portfolio_allocation weight_pct must total 100, got {total_weight}")

    if request is not None:
        min_cash_weight = request.user_context.custom_constraints.get("min_cash_weight")
        if min_cash_weight is not None:
            cash_weight = sum(
                float(allocation.get("weight_pct", 0.0))
                for allocation in allocations
                if isinstance(allocation, dict)
                and str(allocation.get("symbol", "")).upper() == "CASH"
            )
            if cash_weight < float(min_cash_weight):
                violations.append(
                    f"cash allocation must be at least min_cash_weight {min_cash_weight}"
                )


def _validate_forbidden_claims(payload: Any, violations: list[str]) -> None:
    text = json_like_text(payload).lower()
    for forbidden_claim in FORBIDDEN_CLAIMS:
        if forbidden_claim in text:
            violations.append(f"forbidden financial claim: {forbidden_claim}")


def json_like_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return " ".join(json_like_text(value) for value in payload.values())
    if isinstance(payload, list):
        return " ".join(json_like_text(value) for value in payload)
    return str(payload)
