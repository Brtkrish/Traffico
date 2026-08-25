"""Diagnostic version of Adaptive v4.1.

Purpose:
    Diagnose why v4.1 has strong throughput/waiting results but higher
    average trip duration/time loss than fixed-time.

This controller keeps the v4.1 decision logic unchanged, but records:
    - every green phase start/end;
    - phase duration;
    - switch reason;
    - queue at phase start/end;
    - vehicles/halting vehicles at phase start/end;
    - estimated queue reduction;
    - total number of phase switches;
    - cumulative yellow/clearance time.

Use seed 42 first:
    python adaptive_controller_v41_diagnostic.py --gui \
        --tripinfo-output adaptive_v41_diag_seed42.xml

Then:
    python analyze_v41_diagnostics.py adaptive_v41_diag_seed42.json
"""

import argparse
import json
import sys
import time
import traci

TLS_ID = "center"

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3

NS_LANES = ["north_in_0", "south_in_0"]
EW_LANES = ["east_in_0", "west_in_0"]

STEP_LENGTH = 1.0
MAX_STEPS = 600

MIN_GREEN = 8.0
MAX_GREEN = 35.0
BASE_GREEN = 10.0
YELLOW_TIME = 3.0

MAX_RED_WAIT = 60.0

COUNT_WEIGHT = 0.35
HALTING_WEIGHT = 1.75
WAIT_WEIGHT = 0.55

SECONDS_PER_VEHICLE = 0.45
SECONDS_PER_HALTING_VEHICLE = 0.65
SECONDS_PER_WAITING_SECOND = 0.02

MIN_REASSESS_INTERVAL = 5.0
SWITCH_SCORE_MARGIN = 9.0
QUEUE_CLEAR_FRACTION = 0.15
LOW_QUEUE_THRESHOLD = 2
MIN_EFFECTIVE_GREEN = 8.0


def lane_metrics(lanes):
    vehicles = 0
    halting = 0
    waiting = 0.0
    mean_speeds = []

    for lane_id in lanes:
        vehicles += traci.lane.getLastStepVehicleNumber(lane_id)
        halting += traci.lane.getLastStepHaltingNumber(lane_id)
        waiting += traci.lane.getWaitingTime(lane_id)

        speed = traci.lane.getLastStepMeanSpeed(lane_id)
        if speed >= 0:
            mean_speeds.append(speed)

    mean_speed = sum(mean_speeds) / len(mean_speeds) if mean_speeds else 0.0

    return {
        "vehicles": vehicles,
        "halting": halting,
        "waiting": waiting,
        "mean_speed": mean_speed,
    }


def demand_score(metrics, red_wait):
    return (
        metrics["vehicles"] * COUNT_WEIGHT
        + metrics["halting"] * HALTING_WEIGHT
        + red_wait * WAIT_WEIGHT
    )


def compute_green_time(metrics, red_wait):
    green = (
        BASE_GREEN
        + metrics["vehicles"] * SECONDS_PER_VEHICLE
        + metrics["halting"] * SECONDS_PER_HALTING_VEHICLE
        + red_wait * SECONDS_PER_WAITING_SECOND
    )
    return max(MIN_GREEN, min(MAX_GREEN, green))


def update_red_wait(red_wait, green_side):
    if green_side == "NS":
        red_wait["NS"] = 0.0
        red_wait["EW"] += STEP_LENGTH
    else:
        red_wait["EW"] = 0.0
        red_wait["NS"] += STEP_LENGTH


def other_side(side):
    return "EW" if side == "NS" else "NS"


def current_metrics_for(side, ns, ew):
    return ns if side == "NS" else ew


def run_clearance(phase_index, duration, red_wait, green_side):
    traci.trafficlight.setPhase(TLS_ID, phase_index)
    steps = int(round(duration / STEP_LENGTH))

    for _ in range(steps):
        traci.simulationStep()
        update_red_wait(red_wait, green_side)

    return steps * STEP_LENGTH


def should_switch(
    current,
    current_metrics,
    opposite_metrics,
    current_score,
    opposite_score,
    current_phase_elapsed,
    current_start_queue,
    current_previous_queue,
    red_wait,
):
    opposite = other_side(current)
    opposite_wait = red_wait[opposite]
    current_queue = current_metrics["halting"]

    if opposite_wait >= MAX_RED_WAIT and current_phase_elapsed >= MIN_GREEN:
        return True, "STARVATION"

    if current_phase_elapsed < MIN_EFFECTIVE_GREEN:
        return False, "MIN_GREEN"

    if current_phase_elapsed >= MAX_GREEN:
        return True, "MAX_GREEN"

    if current_start_queue > 0:
        total_reduction = (
            current_start_queue - current_queue
        ) / current_start_queue
        recent_improvement = max(
            0,
            current_previous_queue - current_queue,
        )

        if (
            total_reduction >= QUEUE_CLEAR_FRACTION
            and recent_improvement >= 0
            and current_queue > LOW_QUEUE_THRESHOLD
            and opposite_score <= current_score + SWITCH_SCORE_MARGIN
        ):
            return False, "CURRENT_CLEARING"

    if (
        current_queue <= LOW_QUEUE_THRESHOLD
        and opposite_metrics["halting"] >= 2
        and opposite_score > current_score + 2.0
    ):
        return True, "LOW_CURRENT_QUEUE"

    if opposite_score > current_score + SWITCH_SCORE_MARGIN:
        return True, "OPPOSING_DEMAND"

    return False, "HOLD"


def run(
    gui=False,
    tripinfo_output="adaptive_v41_diag_seed42.xml",
    diagnostics_output="adaptive_v41_diag_seed42.json",
    max_steps=MAX_STEPS,
    seed=42,
):
    sumo_binary = "sumo-gui" if gui else "sumo"

    traci.start([
        sumo_binary,
        "-c",
        "intersection.sumocfg",
        "--tripinfo-output",
        tripinfo_output,
        "--step-length",
        str(STEP_LENGTH),
        "--seed",
        str(seed),
    ])

    red_wait = {"NS": 0.0, "EW": 0.0}
    current = "NS"

    phase_elapsed = 0.0
    reassess_elapsed = 0.0
    step = 0

    ns = lane_metrics(NS_LANES)
    ew = lane_metrics(EW_LANES)

    current_start_metrics = ns.copy()
    current_previous_queue = ns["halting"]
    current_target = compute_green_time(ns, 0.0)

    phase_start_sim_step = 0
    switch_count = 0
    total_yellow_time = 0.0

    phases = []

    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)

    try:
        while (
            traci.simulation.getMinExpectedNumber() > 0
            and step < max_steps
        ):
            traci.simulationStep()
            step += 1

            phase_elapsed += STEP_LENGTH
            reassess_elapsed += STEP_LENGTH

            ns = lane_metrics(NS_LANES)
            ew = lane_metrics(EW_LANES)

            update_red_wait(red_wait, current)

            if phase_elapsed < MIN_EFFECTIVE_GREEN:
                current_previous_queue = current_metrics_for(
                    current, ns, ew
                )["halting"]
                continue

            if reassess_elapsed < MIN_REASSESS_INTERVAL:
                current_previous_queue = current_metrics_for(
                    current, ns, ew
                )["halting"]
                continue

            current_metrics = current_metrics_for(current, ns, ew)
            opposite_metrics = current_metrics_for(
                other_side(current), ns, ew
            )

            ns_score = demand_score(ns, red_wait["NS"])
            ew_score = demand_score(ew, red_wait["EW"])

            current_score = (
                ns_score if current == "NS" else ew_score
            )
            opposite_score = (
                ew_score if current == "NS" else ns_score
            )

            switch, reason = should_switch(
                current=current,
                current_metrics=current_metrics,
                opposite_metrics=opposite_metrics,
                current_score=current_score,
                opposite_score=opposite_score,
                current_phase_elapsed=phase_elapsed,
                current_start_queue=current_start_metrics["halting"],
                current_previous_queue=current_previous_queue,
                red_wait=red_wait,
            )

            if not switch:
                refreshed = compute_green_time(
                    current_metrics,
                    0.0,
                )
                current_target = min(
                    MAX_GREEN,
                    max(
                        current_target,
                        phase_elapsed + refreshed,
                    ),
                )
                reassess_elapsed = 0.0
                current_previous_queue = current_metrics["halting"]
                continue

            # Close out the green phase.
            end_metrics = current_metrics
            start_metrics = current_start_metrics

            phases.append({
                "phase": current,
                "start_step": phase_start_sim_step,
                "end_step": step,
                "green_seconds": phase_elapsed,
                "reason": reason,
                "start_vehicles": start_metrics["vehicles"],
                "end_vehicles": end_metrics["vehicles"],
                "start_halting": start_metrics["halting"],
                "end_halting": end_metrics["halting"],
                "queue_reduction": (
                    start_metrics["halting"]
                    - end_metrics["halting"]
                ),
                "start_waiting": start_metrics["waiting"],
                "end_waiting": end_metrics["waiting"],
                "start_mean_speed": start_metrics["mean_speed"],
                "end_mean_speed": end_metrics["mean_speed"],
                "ns_score": ns_score,
                "ew_score": ew_score,
            })

            next_side = other_side(current)
            next_metrics = (
                ew if next_side == "EW" else ns
            )
            next_green = compute_green_time(
                next_metrics,
                red_wait[next_side],
            )

            if current == "NS":
                yellow_elapsed = run_clearance(
                    PHASE_NS_YELLOW,
                    YELLOW_TIME,
                    red_wait,
                    current,
                )
                traci.trafficlight.setPhase(
                    TLS_ID,
                    PHASE_EW_GREEN,
                )
            else:
                yellow_elapsed = run_clearance(
                    PHASE_EW_YELLOW,
                    YELLOW_TIME,
                    red_wait,
                    current,
                )
                traci.trafficlight.setPhase(
                    TLS_ID,
                    PHASE_NS_GREEN,
                )

            total_yellow_time += yellow_elapsed
            switch_count += 1

            current = next_side
            red_wait[current] = 0.0
            phase_elapsed = 0.0
            reassess_elapsed = 0.0
            phase_start_sim_step = step
            current_start_metrics = next_metrics.copy()
            current_previous_queue = next_metrics["halting"]
            current_target = next_green

        # Add the final open green phase to diagnostics.
        if step > phase_start_sim_step:
            final_metrics = current_metrics_for(
                current, ns, ew
            )

            phases.append({
                "phase": current,
                "start_step": phase_start_sim_step,
                "end_step": step,
                "green_seconds": step - phase_start_sim_step,
                "reason": "SIMULATION_END",
                "start_vehicles": current_start_metrics["vehicles"],
                "end_vehicles": final_metrics["vehicles"],
                "start_halting": current_start_metrics["halting"],
                "end_halting": final_metrics["halting"],
                "queue_reduction": (
                    current_start_metrics["halting"]
                    - final_metrics["halting"]
                ),
                "start_waiting": current_start_metrics["waiting"],
                "end_waiting": final_metrics["waiting"],
                "start_mean_speed": current_start_metrics["mean_speed"],
                "end_mean_speed": final_metrics["mean_speed"],
                "ns_score": demand_score(
                    ns, red_wait["NS"]
                ),
                "ew_score": demand_score(
                    ew, red_wait["EW"]
                ),
            })

        diagnostics = {
            "seed": seed,
            "steps": step,
            "switch_count": switch_count,
            "total_yellow_time": total_yellow_time,
            "phase_count": len(phases),
            "phases": phases,
        }

        with open(
            diagnostics_output,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                diagnostics,
                f,
                indent=2,
            )

        print("\nDiagnostic summary")
        print(f"Seed: {seed}")
        print(f"Simulation steps: {step}")
        print(f"Green phases recorded: {len(phases)}")
        print(f"Phase switches: {switch_count}")
        print(f"Yellow/clearance time: {total_yellow_time:.1f}s")
        print(f"Diagnostics: {diagnostics_output}")
        print(f"Tripinfo: {tripinfo_output}")

    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--tripinfo-output",
        default="adaptive_v41_diag_seed42.xml",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="adaptive_v41_diag_seed42.json",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run(
        gui=args.gui,
        tripinfo_output=args.tripinfo_output,
        diagnostics_output=args.diagnostics_output,
        max_steps=args.max_steps,
        seed=args.seed,
    )
