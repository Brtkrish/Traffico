"""Adaptive v4.1 SUMO controller.

v4.1 keeps the validated v4 NS/EW phase structure, but adds phase-efficiency
and switch-cost logic. The goal is to preserve v4's throughput/fairness gains
while reducing unnecessary phase changes and travel-time loss.

Key changes versus v4:
- keeps the same 2-phase NS/EW signal plan;
- keeps queue-clearing and starvation protection;
- adds a switching-cost / hysteresis margin;
- requires the opposing side to be meaningfully better before switching;
- tracks whether the current phase is still clearing its queue;
- avoids switching when both sides are close in demand;
- keeps the same network, routes, 600 s horizon, and seedable SUMO setup.

First test:
    python adaptive_controller_v4_1.py --gui --tripinfo-output adaptive_v41_seed42.xml
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

# Demand score inherited from v4.
COUNT_WEIGHT = 0.35
HALTING_WEIGHT = 1.75
WAIT_WEIGHT = 0.55

# Green-time allocation inherited from v4.
SECONDS_PER_VEHICLE = 0.45
SECONDS_PER_HALTING_VEHICLE = 0.65
SECONDS_PER_WAITING_SECOND = 0.02

# v4.1: stronger hysteresis / switching-cost logic.
# A switch must have a clear benefit rather than merely a slightly larger score.
MIN_REASSESS_INTERVAL = 5.0
SWITCH_SCORE_MARGIN = 9.0

# If the current queue is improving by at least this fraction, prefer to hold
# unless the opposing side is substantially more urgent.
QUEUE_CLEAR_FRACTION = 0.15
LOW_QUEUE_THRESHOLD = 2

# Minimum useful green before an extension/switch decision.
MIN_EFFECTIVE_GREEN = 8.0


def lane_metrics(lanes):
    """Aggregate live SUMO metrics for a lane group."""
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


def run_clearance(phase_index, duration, red_wait, green_side):
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
    current_previous_queue,
    red_wait,
):
    opposite = other_side(current)
    opposite_wait = red_wait[opposite]
    current_queue = current_metrics["halting"]

    # 1. Hard starvation safeguard.
    if opposite_wait >= MAX_RED_WAIT and current_phase_elapsed >= MIN_GREEN:
        return True, "STARVATION"

    # 2. Do not make decisions before a useful minimum green.
    if current_phase_elapsed < MIN_EFFECTIVE_GREEN:
        return False, "MIN_GREEN"

    # 3. Respect hard maximum green.
    if current_phase_elapsed >= MAX_GREEN:
        return True, "MAX_GREEN"

    # 4. Determine whether the current phase is actually clearing the queue.
    if current_start_queue > 0:
        total_reduction = (current_start_queue - current_queue) / current_start_queue
        recent_improvement = max(0, current_previous_queue - current_queue)

        # If the current phase is clearly reducing its queue and the other side
        # is not strongly better, continue serving the current phase.
        if (
            total_reduction >= QUEUE_CLEAR_FRACTION
            and recent_improvement >= 0
            and current_queue > LOW_QUEUE_THRESHOLD
            and opposite_score <= current_score + SWITCH_SCORE_MARGIN
        ):
            return False, "CURRENT_CLEARING"

    # 5. If current queue is nearly empty, allow a switch when the other side
    # actually has demand. This avoids wasting green.
    if (
        current_queue <= LOW_QUEUE_THRESHOLD
        and opposite_metrics["halting"] >= 2
        and opposite_score > current_score + 2.0
    ):
        return True, "LOW_CURRENT_QUEUE"

    # 6. Main v4.1 hysteresis rule: the opponent must be meaningfully better.
    if opposite_score > current_score + SWITCH_SCORE_MARGIN:
        return True, "OPPOSING_DEMAND"

    return False, "HOLD"


def run(
    gui=False,
    tripinfo_output="adaptive_v41_seed42.xml",
    max_steps=MAX_STEPS,
    seed=42,
):
    sumo_binary = "sumo-gui" if gui else "sumo"

    # Same proven startup structure as v4/v6.
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

    current_start_queue = ns["halting"]
    previous_queue = current_start_queue
    current_target = compute_green_time(ns, 0.0)

    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)

    print(
        f"{'step':>5} | {'phase':>5} | "
        f"{'NS_v':>4} {'NS_q':>4} {'NS_score':>8} | "
        f"{'EW_v':>4} {'EW_q':>4} {'EW_score':>8} | "
        f"{'green':>6} | reason"
    )
    print("-" * 86)

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            step += 1

            phase_elapsed += STEP_LENGTH
            reassess_elapsed += STEP_LENGTH

            ns = lane_metrics(NS_LANES)
            ew = lane_metrics(EW_LANES)

            update_red_wait(red_wait, current)

            if phase_elapsed < MIN_EFFECTIVE_GREEN:
                previous_queue = current_metrics_for(current, ns, ew)["halting"]
                continue

            if reassess_elapsed < MIN_REASSESS_INTERVAL:
                previous_queue = current_metrics_for(current, ns, ew)["halting"]
                continue

            current_metrics = current_metrics_for(current, ns, ew)
            opposite_metrics = current_metrics_for(other_side(current), ns, ew)

            ns_score = demand_score(ns, red_wait["NS"])
            ew_score = demand_score(ew, red_wait["EW"])

            current_score = ns_score if current == "NS" else ew_score
            opposite_score = ew_score if current == "NS" else ns_score

            switch, reason = should_switch(
                current=current,
                current_metrics=current_metrics,
                opposite_metrics=opposite_metrics,
                current_score=current_score,
                opposite_score=opposite_score,
                current_phase_elapsed=phase_elapsed,
                current_start_queue=current_start_queue,
                current_previous_queue=previous_queue,
                red_wait=red_wait,
            )

            if not switch:
                # Extend only when the current phase still has useful work.
                refreshed = compute_green_time(current_metrics, 0.0)
                current_target = min(
                    MAX_GREEN,
                    max(current_target, phase_elapsed + refreshed),
                )
                reassess_elapsed = 0.0
                previous_queue = current_metrics["halting"]
                continue

            next_side = other_side(current)
            next_metrics = ew if next_side == "EW" else ns
            next_green = compute_green_time(
                next_metrics,
                red_wait[next_side],
            )

            print(
                f"{step:>5} | {current:>5} | "
                f"{ns['vehicles']:>4} {ns['halting']:>4} {ns_score:>8.2f} | "
                f"{ew['vehicles']:>4} {ew['halting']:>4} {ew_score:>8.2f} | "
                f"{next_green:>6.1f} | {reason}"
            )

            if current == "NS":
                run_clearance(
                    PHASE_NS_YELLOW,
                    YELLOW_TIME,
                    red_wait,
                    current,
                )
                traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
            else:
                run_clearance(
                    PHASE_EW_YELLOW,
                    YELLOW_TIME,
                    red_wait,
                    current,
                )
                traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)

            current = next_side
            red_wait[current] = 0.0
            phase_elapsed = 0.0
            reassess_elapsed = 0.0

            current_start_queue = next_metrics["halting"]
            previous_queue = current_start_queue
            current_target = next_green

        print("\nSimulation finished.")
        print(f"Seed: {seed}")
        print(f"Steps: {step}")
        print(f"Tripinfo: {tripinfo_output}")

    finally:
        traci.close()
        sys.stdout.flush()


def current_metrics_for(side, ns, ew):
    return ns if side == "NS" else ew


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--tripinfo-output",
        default="adaptive_v41_seed42.xml",
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
        max_steps=args.max_steps,
        seed=args.seed,
    )
