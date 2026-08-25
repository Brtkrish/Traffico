"""Analyze Adaptive v4.1 diagnostic JSON."""

import json
import sys
from collections import Counter

filename = sys.argv[1] if len(sys.argv) > 1 else "adaptive_v41_diag_seed42.json"

with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

phases = data["phases"]

reasons = Counter(p["reason"] for p in phases)

durations = [p["green_seconds"] for p in phases]
queue_reductions = [p["queue_reduction"] for p in phases]

avg_duration = sum(durations) / len(durations) if durations else 0.0
avg_queue_reduction = (
    sum(queue_reductions) / len(queue_reductions)
    if queue_reductions else 0.0
)

effective_clear_phases = [
    p for p in phases
    if p["queue_reduction"] > 0
]

print(f"Diagnostic file: {filename}")
print(f"Seed: {data['seed']}")
print(f"Simulation steps: {data['steps']}")
print(f"Green phases: {data['phase_count']}")
print(f"Phase switches: {data['switch_count']}")
print(f"Total yellow/clearance: {data['total_yellow_time']:.1f} s")
print(f"Average green duration: {avg_duration:.2f} s")
print(f"Average queue reduction / phase: {avg_queue_reduction:.2f}")
print(
    f"Phases with positive queue reduction: "
    f"{len(effective_clear_phases)}/{len(phases)}"
)

print("\nSwitch reasons:")
for reason, count in reasons.most_common():
    print(f"  {reason}: {count}")

print("\nPhase durations:")
for i, p in enumerate(phases, start=1):
    print(
        f"  {i:>2}. {p['phase']:>2} "
        f"{p['green_seconds']:>5.1f}s "
        f"queue {p['start_halting']:>3} -> {p['end_halting']:>3} "
        f"Δ={p['queue_reduction']:>+4} "
        f"{p['reason']}"
    )

