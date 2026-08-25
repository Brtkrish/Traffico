"""Adaptive NS/EW traffic-signal controller for SUMO via TraCI.

v4 is based on phase effectiveness rather than repeatedly choosing the larger
aggregate score. The controller:

- keeps the safe 2-phase NS/EW signal plan;
- enforces a minimum green;
- measures whether the current phase is actually clearing its queue;
- extends green while the current phase is effective and the opposing demand
  is not strong enough to justify a switch;
- switches when the current phase is no longer clearing well, the opposing
  side has meaningful demand, or the maximum green is reached;
- uses a hard maximum red-wait limit as a final starvation guard.

Run:
    python adaptive_controller_v4.py
    python adaptive_controller_v4.py --gui
    python adaptive_controller_v4.py --gui --tripinfo-output adaptive_v4_tripinfo.xml
"""

import argparse
import sys
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

# Demand score weights. Queue/halting traffic matters most.
COUNT_WEIGHT = 0.35
HALTING_WEIGHT = 1.75
WAIT_WEIGHT = 0.55

# Extension model.
SECONDS_PER_VEHICLE = 0.45
SECONDS_PER_HALTING_VEHICLE = 0.65
SECONDS_PER_WAITING_SECOND = 0.02

# Phase-effectiveness / switching parameters.
OPPOSING_MIN_DEMAND = 4.0
SWITCH_SCORE_MARGIN = 6.0
QUEUE_CLEAR_FRACTION = 0.15
LOW_QUEUE_THRESHOLD = 2
MIN_EXTEND_INTERVAL = 5.0


def lane_metrics(lanes):
    """Aggregate live SUMO metrics for a group of incoming lanes."""
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
    """Estimate how strongly a phase currently needs green."""
    return (
        metrics["vehicles"] * COUNT_WEIGHT
        + metrics["halting"] * HALTING_WEIGHT
        + red_wait * WAIT_WEIGHT
    )


def compute_initial_green(metrics, red_wait):
    """Initial green allocation for a newly selected phase."""
    green = (
        BASE_GREEN
        + metrics["vehicles"] * SECONDS_PER_VEHICLE
        + metrics["halting"] * SECONDS_PER_HALTING_VEHICLE
        + red_wait * SECONDS_PER_WAITING_SECOND
    )
    return max(MIN_GREEN, min(MAX_GREEN, green))


def update_red_wait(red_wait, green_side):
    """Advance the continuously-red side's wait timer by one simulation step."""
    if green_side == "NS":
        red_wait["NS"] = 0.0
        red_wait["EW"] += STEP_LENGTH
    else:
        red_wait["EW"] = 0.0
        red_wait["NS"] += STEP_LENGTH


def run_clearance(phase_index, duration, red_wait, green_side):
    """Run yellow clearance while continuing red-wait accounting."""
    traci.trafficlight.setPhase(TLS_ID, phase_index)
    steps = int(round(duration / STEP_LENGTH))
    for _ in range(steps):
        traci.simulationStep()
        update_red_wait(red_wait, green_side)


def other_side(side):
    return "EW" if side == "NS" else "NS"


def should_switch(
    current,
    current_metrics,
    opposite_metrics,
    current_score,
    opposite_score,
    current_phase_elapsed,
    current_start_queue,
    red_wait,
):
    """Decide whether the current phase has become ineffective or unsafe to hold."""
    opposite = other_side(current)
    opposite_wait = red_wait[opposite]
    current_queue = current_metrics["halting"]

    # Final starvation protection.
    if opposite_wait >= MAX_RED_WAIT and current_phase_elapsed >= MIN_GREEN:
        return True, "STARVATION"

    if current_phase_elapsed < MIN_EXTEND_INTERVAL:
        return False, "MIN_INTERVAL"

    # Always respect the hard maximum green once the minimum is satisfied.
    if current_phase_elapsed >= MAX_GREEN:
        return True, "MAX_GREEN"

    # If the current queue has mostly cleared and the other side has real demand,
    # don't waste green time on an almost-empty phase.
    if current_start_queue > 0:
        queue_reduction = (current_start_queue - current_queue) / current_start_queue
        if (
            queue_reduction >= QUEUE_CLEAR_FRACTION
            and current_queue <= LOW_QUEUE_THRESHOLD
            and opposite_metrics["halting"] > 0
            and opposite_score >= OPPOSING_MIN_DEMAND
        ):
            return True, "QUEUE_CLEARED"

    # If the opposing side is clearly more urgent, switch after the minimum green.
    if opposite_score >= OPPOSING_MIN_DEMAND and opposite_score > current_score + SWITCH_SCORE_MARGIN:
        return True, "OPPOSITE_DEMAND"

    # If there is no meaningful current queue but opposite traffic exists, switch.
    if current_queue <= LOW_QUEUE_THRESHOLD and opposite_metrics["halting"] >= 2:
        return True, "LOW_CURRENT_QUEUE"

    return False, "HOLD"


def run(gui=False, tripinfo_output="adaptive_v4_tripinfo.xml", max_steps=MAX_STEPS):
    sumo_binary = "sumo-gui" if gui else "sumo"
    command = [
    sumo_binary,
    "-c",
    "intersection.sumocfg",
    "--seed",
    "46",
    "--tripinfo-output",
    tripinfo_output
]


    traci.start(command)

    red_wait = {"NS": 0.0, "EW": 0.0}
    current = "NS"
    phase_elapsed = 0.0
    last_reassessment = 0.0
    step = 0

    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
    current_metrics = lane_metrics(NS_LANES)
    current_start_queue = max(0, current_metrics["halting"])
    current_target = compute_initial_green(current_metrics, red_wait["NS"])

    header = (
        f"{'step':>6} | {'phase':>5} | {'NS_v':>5} | {'NS_q':>5} | {'NS_wait':>7} | "
        f"{'NS_score':>9} | {'EW_v':>5} | {'EW_q':>5} | {'EW_wait':>7} | "
        f"{'EW_score':>9} | {'green':>6} | reason"
    )
    print(header)
    print("-" * len(header))

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            step += 1
            phase_elapsed += STEP_LENGTH
            last_reassessment += STEP_LENGTH

            ns = lane_metrics(NS_LANES)
            ew = lane_metrics(EW_LANES)
            update_red_wait(red_wait, current)

            ns_score = demand_score(ns, red_wait["NS"])
            ew_score = demand_score(ew, red_wait["EW"])

            # Reassess at least every 5 s after the minimum green.
            if phase_elapsed >= MIN_GREEN and last_reassessment >= MIN_EXTEND_INTERVAL:
                metrics = ns if current == "NS" else ew
                opposite_metrics = ew if current == "NS" else ns
                current_score = ns_score if current == "NS" else ew_score
                opposite_score = ew_score if current == "NS" else ns_score

                switch, reason = should_switch(
                    current=current,
                    current_metrics=metrics,
                    opposite_metrics=opposite_metrics,
                    current_score=current_score,
                    opposite_score=opposite_score,
                    current_phase_elapsed=phase_elapsed,
                    current_start_queue=current_start_queue,
                    red_wait=red_wait,
                )

                if not switch and phase_elapsed < current_target:
                    continue

                if switch:
                    next_side = other_side(current)
                    next_metrics = ew if next_side == "EW" else ns
                    next_green = compute_initial_green(next_metrics, red_wait[next_side])

                    print(
                        f"{step:>6} | {current:>5} | {ns['vehicles']:>5} | {ns['halting']:>5} | "
                        f"{red_wait['NS']:>7.0f} | {ns_score:>9.2f} | {ew['vehicles']:>5} | "
                        f"{ew['halting']:>5} | {red_wait['EW']:>7.0f} | {ew_score:>9.2f} | "
                        f"{next_green:>6.1f} | {reason}"
                    )

                    if current == "NS":
                        run_clearance(PHASE_NS_YELLOW, YELLOW_TIME, red_wait, current)
                        traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
                    else:
                        run_clearance(PHASE_EW_YELLOW, YELLOW_TIME, red_wait, current)
                        traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)

                    current = next_side
                    red_wait[current] = 0.0
                    phase_elapsed = 0.0
                    last_reassessment = 0.0
                    current_start_queue = max(0, next_metrics["halting"])
                    current_target = next_green
                else:
                    # Current phase remains valuable. Refresh its extension window,
                    # but never beyond the hard max green.
                    remaining_target = compute_initial_green(metrics, 0.0)
                    current_target = min(MAX_GREEN, max(current_target, phase_elapsed + remaining_target))
                    last_reassessment = 0.0

            # Hard safety: if the dynamic target is reached, reassess on the next tick.
            if phase_elapsed >= current_target and phase_elapsed >= MIN_GREEN:
                continue

        print("\nSimulation finished.")
        print(f"Steps: {step}")
        print(f"Tripinfo: {tripinfo_output}")

    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="run with sumo-gui")
    parser.add_argument(
        "--tripinfo-output",
        default="adaptive_v4_tripinfo.xml",
        help="output XML filename for SUMO tripinfo",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="maximum simulation steps",
    )
    args = parser.parse_args()
    run(gui=args.gui, tripinfo_output=args.tripinfo_output, max_steps=args.max_steps)
