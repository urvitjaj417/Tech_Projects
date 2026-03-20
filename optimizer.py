# ml/optimizer.py
# Production workflow optimizer.
# Detects bottleneck machines, calculates OEE, and recommends actions.
#
# Usage: python optimizer.py

import numpy as np
import pandas as pd
from data_loader import load_data, summary_stats


def detect_bottlenecks(df: pd.DataFrame, fault_threshold: float = 15.0) -> pd.DataFrame:
    """
    A machine is flagged as a bottleneck if:
      - fault rate >= fault_threshold (%), OR
      - average output rate > 1.5 std devs below the fleet mean.
    """
    stats = summary_stats(df)
    mean_out = stats["avg_output"].mean()
    std_out  = stats["avg_output"].std() + 1e-6

    stats["output_deviation"] = (stats["avg_output"] - mean_out) / std_out
    stats["is_bottleneck"]    = (
        (stats["fault_rate_pct"] >= fault_threshold) |
        (stats["output_deviation"] < -1.5)
    )
    return stats


def suggest_workflow(df: pd.DataFrame) -> list:
    """Returns prioritised action list for the production floor."""
    stats      = detect_bottlenecks(df)
    bottleneck = stats[stats["is_bottleneck"]].copy()
    normal     = stats[~stats["is_bottleneck"]].copy()
    suggestions = []

    for _, row in bottleneck.sort_values("fault_rate_pct", ascending=False).iterrows():
        reasons = []
        if row["fault_rate_pct"] >= 15.0:
            reasons.append(f"fault rate {row['fault_rate_pct']}%")
        if row["output_deviation"] < -1.5:
            avg = stats["avg_output"].mean()
            reasons.append(f"output {row['avg_output']:.0f} uph vs fleet avg {avg:.0f}")

        suggestions.append({
            "machine":  row["machine_name"],
            "priority": "CRITICAL" if row["fault_rate_pct"] > 20 else "HIGH",
            "reasons":  reasons,
            "actions": [
                "Schedule immediate preventive maintenance",
                "Reduce feed rate by 15% until inspected",
                "Redirect load to standby unit if available",
            ],
        })

    if len(normal) >= 2:
        best = normal.sort_values("avg_output", ascending=False).iloc[0]
        suggestions.append({
            "machine":  best["machine_name"],
            "priority": "INFO",
            "reasons":  [
                f"Highest output ({best['avg_output']:.0f} uph, "
                f"fault rate {best['fault_rate_pct']}%)"
            ],
            "actions": ["Route priority orders through this machine"],
        })

    return suggestions


def oee_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified OEE (Availability x Performance x Quality) per machine.
    Based on sensor data heuristics -- replace with real downtime logs in production.
    """
    stats = summary_stats(df)

    # Availability: each fault record assumed to cause ~10 min downtime per 8-hr shift
    shift_min = 480.0
    stats["availability"] = (
        1 - (stats["fault_count"] * 10) / (stats["total_records"] * shift_min / 12)
    ).clip(0.5, 1.0)

    # Performance: output vs 95th percentile fleet max
    theoretical_max = df["output_rate_uph"].quantile(0.95)
    stats["performance"] = (stats["avg_output"] / theoretical_max).clip(0, 1.0)

    # Quality: proxy = 1 - fault rate
    stats["quality"] = 1 - (stats["fault_rate_pct"] / 100)

    stats["oee_pct"] = (
        stats["availability"] * stats["performance"] * stats["quality"] * 100
    ).round(2)

    return stats[["machine_name", "availability", "performance", "quality", "oee_pct"]]


if __name__ == "__main__":
    df = load_data()

    print("=" * 50)
    print("BOTTLENECK DETECTION")
    print("=" * 50)
    bn = detect_bottlenecks(df)
    print(bn[["machine_name", "fault_rate_pct", "avg_output", "is_bottleneck"]].to_string(index=False))

    print("\n" + "=" * 50)
    print("WORKFLOW RECOMMENDATIONS")
    print("=" * 50)
    for s in suggest_workflow(df):
        print(f"\n[{s['priority']}] {s['machine']}")
        print(f"  Reasons : {', '.join(s['reasons'])}")
        for a in s["actions"]:
            print(f"  -> {a}")

    print("\n" + "=" * 50)
    print("OEE REPORT")
    print("=" * 50)
    print(oee_report(df).to_string(index=False))
