from dataclasses import dataclass


@dataclass(frozen=True)
class ReplanDecision:
    should_replan: bool
    reason: str


def decide_replan(
    has_active_goals: bool,
    arrived_any: bool,
    now_ts: float,
    last_replan_ts: float,
    replan_dt: float,
    active_tick_dt: float,
    active_tick_enabled: bool,
) -> ReplanDecision:
    elapsed = now_ts - last_replan_ts
    if not has_active_goals:
        if elapsed < replan_dt:
            return ReplanDecision(False, "rate_limit_no_active_goal")
        return ReplanDecision(True, "no_active_goal")
    if arrived_any:
        if elapsed < replan_dt:
            return ReplanDecision(False, "rate_limit_arrived")
        return ReplanDecision(True, "arrived")

    if not active_tick_enabled:
        return ReplanDecision(False, "active_tick_disabled")
    if elapsed < active_tick_dt:
        return ReplanDecision(False, "rate_limit_active_tick")

    # Keep policy/planner coupling "warm" during transit with a lower active
    # tick frequency, so next goal can be pre-fetched without saturating CPU.
    return ReplanDecision(True, "active_tick")
