"""
Drug condition profiles for rat behavior analysis.

Provides normative behavior distributions per experimental condition,
condition-specific anomaly detection, and threshold adjustments.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConditionProfile:
    """Expected behavior distribution for a drug condition."""
    name: str
    # Expected fraction ranges (min, max) for key behaviors
    expected_exploration: tuple[float, float]
    expected_locomotor_burst: tuple[float, float]
    expected_stereotypy: tuple[float, float]
    expected_freezing: tuple[float, float]
    expected_grooming: tuple[float, float]
    expected_resting: tuple[float, float]
    expected_hesitation: tuple[float, float]
    # Behaviors to flag as clinically relevant
    flag_behaviors: list[str]
    # Description for reports
    description: str


CONDITION_PROFILES: dict[str, ConditionProfile] = {
    "baseline": ConditionProfile(
        name="Baseline",
        expected_exploration=(0.15, 0.45),
        expected_locomotor_burst=(0.02, 0.15),
        expected_stereotypy=(0.0, 0.05),
        expected_freezing=(0.0, 0.08),
        expected_grooming=(0.05, 0.20),
        expected_resting=(0.10, 0.35),
        expected_hesitation=(0.02, 0.10),
        flag_behaviors=["stereotypy_candidate"],
        description="Control session. Behavior serves as reference for drug comparisons.",
    ),
    "saline": ConditionProfile(
        name="Saline Control",
        expected_exploration=(0.15, 0.45),
        expected_locomotor_burst=(0.02, 0.15),
        expected_stereotypy=(0.0, 0.05),
        expected_freezing=(0.0, 0.08),
        expected_grooming=(0.05, 0.20),
        expected_resting=(0.10, 0.35),
        expected_hesitation=(0.02, 0.10),
        flag_behaviors=["stereotypy_candidate"],
        description="Saline vehicle control. Expected to match baseline.",
    ),
    "meth": ConditionProfile(
        name="Methamphetamine",
        expected_exploration=(0.05, 0.25),
        expected_locomotor_burst=(0.10, 0.40),
        expected_stereotypy=(0.10, 0.50),
        expected_freezing=(0.0, 0.05),
        expected_grooming=(0.0, 0.08),
        expected_resting=(0.0, 0.10),
        expected_hesitation=(0.0, 0.05),
        flag_behaviors=["stereotypy_candidate", "locomotor_burst"],
        description="Methamphetamine increases locomotor activity and stereotypy. Reduced resting and grooming expected.",
    ),
    "methamphetamine": ConditionProfile(
        name="Methamphetamine",
        expected_exploration=(0.05, 0.25),
        expected_locomotor_burst=(0.10, 0.40),
        expected_stereotypy=(0.10, 0.50),
        expected_freezing=(0.0, 0.05),
        expected_grooming=(0.0, 0.08),
        expected_resting=(0.0, 0.10),
        expected_hesitation=(0.0, 0.05),
        flag_behaviors=["stereotypy_candidate", "locomotor_burst"],
        description="Methamphetamine increases locomotor activity and stereotypy. Reduced resting and grooming expected.",
    ),
    "amphetamine": ConditionProfile(
        name="Amphetamine",
        expected_exploration=(0.08, 0.30),
        expected_locomotor_burst=(0.10, 0.35),
        expected_stereotypy=(0.05, 0.40),
        expected_freezing=(0.0, 0.05),
        expected_grooming=(0.01, 0.10),
        expected_resting=(0.02, 0.15),
        expected_hesitation=(0.0, 0.08),
        flag_behaviors=["stereotypy_candidate", "locomotor_burst"],
        description="Amphetamine produces dose-dependent locomotor activation and stereotypy.",
    ),
    "cocaine": ConditionProfile(
        name="Cocaine",
        expected_exploration=(0.10, 0.30),
        expected_locomotor_burst=(0.15, 0.40),
        expected_stereotypy=(0.05, 0.35),
        expected_freezing=(0.0, 0.05),
        expected_grooming=(0.0, 0.08),
        expected_resting=(0.02, 0.12),
        expected_hesitation=(0.0, 0.08),
        flag_behaviors=["stereotypy_candidate", "locomotor_burst"],
        description="Cocaine produces acute locomotor activation. Watch for rapid state transitions.",
    ),
    "alcohol": ConditionProfile(
        name="Alcohol",
        expected_exploration=(0.05, 0.20),
        expected_locomotor_burst=(0.0, 0.10),
        expected_stereotypy=(0.0, 0.05),
        expected_freezing=(0.03, 0.15),
        expected_grooming=(0.02, 0.12),
        expected_resting=(0.15, 0.50),
        expected_hesitation=(0.05, 0.20),
        flag_behaviors=["freezing_candidate", "hesitation", "resting"],
        description="Alcohol reduces exploration and may increase hesitation, freezing, and resting.",
    ),
    "ethanol": ConditionProfile(
        name="Ethanol",
        expected_exploration=(0.05, 0.20),
        expected_locomotor_burst=(0.0, 0.10),
        expected_stereotypy=(0.0, 0.05),
        expected_freezing=(0.03, 0.15),
        expected_grooming=(0.02, 0.12),
        expected_resting=(0.15, 0.50),
        expected_hesitation=(0.05, 0.20),
        flag_behaviors=["freezing_candidate", "hesitation", "resting"],
        description="Ethanol reduces exploration and may increase hesitation, freezing, and resting.",
    ),
    "morphine": ConditionProfile(
        name="Morphine",
        expected_exploration=(0.05, 0.25),
        expected_locomotor_burst=(0.05, 0.25),
        expected_stereotypy=(0.02, 0.15),
        expected_freezing=(0.02, 0.15),
        expected_grooming=(0.0, 0.08),
        expected_resting=(0.10, 0.40),
        expected_hesitation=(0.02, 0.12),
        flag_behaviors=["stereotypy_candidate", "freezing_candidate"],
        description="Morphine can produce both locomotor activation and catalepsy depending on dose.",
    ),
}


def get_condition_profile(condition: str | None) -> ConditionProfile | None:
    if not condition:
        return None
    return CONDITION_PROFILES.get(condition.strip().lower())


def detect_anomalies(
    behavior_share: dict[str, float],
    condition: str | None,
) -> list[dict]:
    """
    Detect behaviors that deviate from expected ranges for the condition.

    Returns a list of anomaly dicts with behavior, observed, expected_range, and severity.
    """
    profile = get_condition_profile(condition)
    if profile is None:
        return []

    anomalies = []
    behavior_ranges = {
        "exploration": profile.expected_exploration,
        "locomotor_burst": profile.expected_locomotor_burst,
        "stereotypy_candidate": profile.expected_stereotypy,
        "freezing_candidate": profile.expected_freezing,
        "grooming_candidate": profile.expected_grooming,
        "resting": profile.expected_resting,
        "hesitation": profile.expected_hesitation,
    }

    for behavior, (low, high) in behavior_ranges.items():
        observed = behavior_share.get(behavior, 0.0)
        if observed < low:
            deviation = low - observed
            anomalies.append({
                "behavior": behavior,
                "observed": round(observed, 3),
                "expected_range": [round(low, 3), round(high, 3)],
                "direction": "below_expected",
                "deviation": round(deviation, 3),
                "severity": "high" if deviation > 0.15 else "moderate" if deviation > 0.08 else "low",
            })
        elif observed > high:
            deviation = observed - high
            anomalies.append({
                "behavior": behavior,
                "observed": round(observed, 3),
                "expected_range": [round(low, 3), round(high, 3)],
                "direction": "above_expected",
                "deviation": round(deviation, 3),
                "severity": "high" if deviation > 0.15 else "moderate" if deviation > 0.08 else "low",
            })

    return sorted(anomalies, key=lambda a: a["deviation"], reverse=True)


def condition_adjusted_review_priority(
    base_priority: str,
    dominant_behavior: str,
    behavior_share: dict[str, float],
    condition: str | None,
) -> str:
    """Adjust review priority based on condition-specific expectations."""
    profile = get_condition_profile(condition)
    if profile is None:
        return base_priority

    # If dominant behavior is a flagged behavior for this condition, elevate priority
    if dominant_behavior in profile.flag_behaviors:
        if base_priority == "Low":
            return "Medium"

    # If anomalies are severe, elevate
    anomalies = detect_anomalies(behavior_share, condition)
    severe = [a for a in anomalies if a["severity"] == "high"]
    if severe:
        return "High"

    moderate = [a for a in anomalies if a["severity"] == "moderate"]
    if moderate and base_priority == "Low":
        return "Medium"

    return base_priority


def generate_condition_report(
    behavior_share: dict[str, float],
    condition: str | None,
    dominant_behavior: str,
) -> dict | None:
    """Generate a condition-aware analysis report section."""
    profile = get_condition_profile(condition)
    if profile is None:
        return None

    anomalies = detect_anomalies(behavior_share, condition)

    return {
        "condition": profile.name,
        "condition_description": profile.description,
        "flag_behaviors": profile.flag_behaviors,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "severe_anomaly_count": len([a for a in anomalies if a["severity"] == "high"]),
    }
