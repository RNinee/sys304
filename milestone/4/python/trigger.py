"""Decide whether a retraining cycle should start."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerDecision:
    run: bool
    reason: str


def should_retrain(
    *,
    now: float,
    last_run: float | None,
    interval_seconds: float,
    mean_confidence: float | None,
    confidence_floor: float,
    force: bool = False,
) -> TriggerDecision:
    """Fire when the time budget is spent or confidence has dropped.

    Either condition is enough. ``force`` skips both checks.
    """
    if force:
        return TriggerDecision(True, "forced")
    reasons: list[str] = []
    if last_run is None or now - last_run >= interval_seconds:
        reasons.append("time")
    if mean_confidence is not None and mean_confidence < confidence_floor:
        reasons.append("confidence")
    if not reasons:
        return TriggerDecision(False, "idle")
    return TriggerDecision(True, "+".join(reasons))
