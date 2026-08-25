"""Adaptive v6 SUMO controller: movement-aware NS/EW control.

v6 keeps the validated v4 two-phase signal plan (NS / EW), but it no longer
treats every approach as one undifferentiated queue.

For every vehicle waiting on an incoming lane, SUMO route information is used
to classify the intended movement as:
    straight / left / right

The controller then:
- measures movement-specific vehicle and halting counts;
- gives more pressure to movement-specific queues (especially left turns);
- uses the movement-aware phase pressure to choose/extend NS or EW;
- retains v4's queue-clearing and starvation protection logic;
- prints movement breakdowns so the behavior can be inspected.

This version DOES NOT create separate protected left-turn phases yet.
It is a controlled movement-aware upgrade of v4 while keeping the same
2-phase NS/EW signal structure.

Run:
    python adaptive_controller_v6.py
    python adaptive_controller_v6.py --gui
    python adaptive_controller_v6.py --gui --tripinfo-output adaptive_v6_seed42.xml
"""

import argparse
import sys
from collections import defaultdict
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

# v4-style demand weights, now movement-aware.
COUNT_WEIGHT = 0.30
HALTING_WEIGHT = 1.60
LEFT_HALTING_BONUS = 0.70
LEFT_COUNT_BONUS = 0.20
WAIT_WEIGHT = 0.55

SECONDS_PER_VEHICLE = 0.40
SECONDS_PER_HALTING_VEHICLE = 0.60
SECONDS_PER_LEFT_HALTING = 0.30
SECONDS_PER_WAITING_SECOND = 0.02

OPPOSING_MIN_DEMAND = 4.0
SWITCH_SCORE_MARGIN = 6.0
QUEUE_CLEAR_FRACTION = 0.15
LOW_QUEUE_THRESHOLD = 2
MIN_EXTEND_INTERVAL = 5.0

# Map incoming edge -> outgoing edge -> movement.
# This follows the actual junction connections in intersection.net.xml.
MOVEMENT_MAP = {
    "north_in": {
        "center_out_s": "straight",
        "center_out_e": "left",
        "center_out_w": "right",
        "center_out_n": "uturn",
    },
    "south_in": {
        "center_out_n": "straight",
        "center_out_w": "left",
        "center_out_e": "right",
        "center_out_s": "uturn",
    },
    "east_in": {
        "center_out_w": "straight",
        "center_out_s": "left",
        "center_out_n": "right",
        "center_out_e": "uturn",
    },
    "west_in": {
        "center_out_e": "straight",
        "center_out_n": "left",
        "center_out_s": "right",
        "center_out_w": "uturn",
    },
}

MOVEMENTS = ("straight", "left", "right", "uturn")


def empty_movement_counts():
    return {m: 0 for m in MOVEMENTS}


def movement_metrics(lanes):
    """Return aggregate metrics plus straight/left/right/UTurn counts."""
    vehicles = 0
    halting = 0
    waiting = 0.0
    speed_sum = 0.0
    speed_n = 0

    by_movement = empty_movement_counts()
    halting_by_movement = empty_movement_counts()

    for lane_id in lanes:
        vehicles += traci.lane.getLastStepVehicleNumber(lane_id)
        halting += traci.lane.getLastStepHaltingNumber(lane_id)
        waiting += traci.lane.getWaitingTime(lane_id)

        lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)
        if lane_speed >= 0:
            speed_sum += lane_speed
            speed_n += 1

        incoming_edge = lane_id.rsplit("_", 1)[0]
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

        for vehicle_id in vehicle_ids:
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)

            # We only need vehicles still on an incoming approach.
            if not route or route_index < 0 or route_index + 1 >= len(route):
                continue

            out_edge = route[route_index + 1]
            movement = MOVEMENT_MAP.get(incoming_edge, {}).get(out_edge)
            if movement is None:
                continue

            by_movement[movement] += 1

            if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                halting_by_movement[movement] += 1

    mean_speed = speed_sum / speed_n if speed_n else 0.0

    return {
        "vehicles": vehicles,
        "halting": halting,
        "waiting": waiting,
        "mean_speed": mean_speed,
        "by_movement": by_movement,
        "halting_by_movement": halting_by_movement,
    }


def demand_score(metrics, red_wait):
    """Movement-aware pressure score for an NS or EW phase."""
    move = metrics["by_movement"]
    halted = metrics["halting_by_movement"]

    return (
        metrics["vehicles"] * COUNT_WEIGHT
        + metrics["halting"] * HALTING_WEIGHT
        + halted["left"] * LEFT_HALTING_BONUS
        + move["left"] * LEFT_COUNT_BONUS
        + red_wait * WAIT_WEIGHT
    )


def compute_initial_green(metrics, red_wait):
    """Allocate green from current movement-aware demand."""
    move = metrics["by_movement"]
    halted = metrics["halting_by_movement"]

    green = (
        BASE_GREEN
        + metrics["vehicles"] * SECONDS_PER_VEHICLE
        + metrics["halting"] * SECONDS_PER_HALTING_VEHICLE
        + halted["left"] * SECONDS_PER_LEFT_HALTING
        + red_wait * SECONDS_PER_WAITING_SECOND
    )

    # A phase with many left-turners gets a modest additional extension,
    # but the hard maximum remains in force.
    green += move["left"] * 0.10

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
    red_wait,
):
    opposite = other_side(current)
    opposite_wait = red_wait[opposite]
    current_queue = current_metrics["halting"]

    if opposite_wait >= MAX_RED_WAIT and current_phase_elapsed >= MIN_GREEN:
        return True, "STARVATION"

    if current_phase_elapsed < MIN_EXTEND_INTERVAL:
        return False, "MIN_INTERVAL"

    if current_phase_elapsed >= MAX_GREEN:
        return True, "MAX_GREEN"

    if current_start_queue > 0:
        queue_reduction = (current_start_queue - current_queue) / current_start_queue
        if (
            queue_reduction >= QUEUE_CLEAR_FRACTION
            and current_queue <= LOW_QUEUE_THRESHOLD
            and opposite_metrics["halting"] > 0
            and opposite_score >= OPPOSING_MIN_DEMAND
        ):
            return True, "QUEUE_CLEARED"

    if (
        opposite_score >= OPPOSING_MIN_DEMAND
        and opposite_score > current_score + SWITCH_SCORE_MARGIN
    ):
        return True, "OPPOSITE_MOVEMENT_DEMAND"

    if current_queue <= LOW_QUEUE_THRESHOLD and opposite_metrics["halting"] >= 2:
        return True, "LOW_CURRENT_QUEUE"

    return False, "HOLD"


def movement_text(metrics):
    move = metrics["by_movement"]
    halted = metrics["halting_by_movement"]
    return (
        f"S {move['straight']}/{halted['straight']} "
        f"L {move['left']}/{halted['left']} "
        f"R {move['right']}/{halted['right']}"
    )


def run(
    gui=False,
    tripinfo_output="adaptive_v6_seed42.xml",
    max_steps=MAX_STEPS,
    seed=42,
):
    sumo_binary = "sumo-gui" if gui else "sumo"
    command = [
        sumo_binary,
        "-c",
        "intersection.sumocfg",
        "--tripinfo-output",
        tripinfo_output,
        "--step-length",
        str(STEP_LENGTH),
        "--seed",
        str(seed),
    ]

    traci.start(command)

    red_wait = {"NS": 0.0, "EW": 0.0}
    current = "NS"
    phase_elapsed = 0.0
    last_reassessment = 0.0
    step = 0

    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
    current_metrics = movement_metrics(NS_LANES)
    current_start_queue = max(0, current_metrics["halting"])
    current_target = compute_initial_green(current_metrics, red_wait["NS"])

    header = (
        f"{'step':>6} | {'phase':>5} | "
        f"{'NS_v':>4} {'NS_q':>4} {'NS_score':>8} | "
        f"{'EW_v':>4} {'EW_q':>4} {'EW_score':>8} | "
        f"{'green':>6} | movement demand | reason"
    )
    print(header)
    print("-" * len(header))

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            step += 1
            phase_elapsed += STEP_LENGTH
            last_reassessment += STEP_LENGTH

            ns = movement_metrics(NS_LANES)
            ew = movement_metrics(EW_LANES)
            update_red_wait(red_wait, current)

            ns_score = demand_score(ns, red_wait["NS"])
            ew_score = demand_score(ew, red_wait["EW"])

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
                    next_green = compute_initial_green(
                        next_metrics,
                        red_wait[next_side],
                    )

                    print(
                        f"{step:>6} | {current:>5} | "
                        f"{ns['vehicles']:>4} {ns['halting']:>4} {ns_score:>8.2f} | "
                        f"{ew['vehicles']:>4} {ew['halting']:>4} {ew_score:>8.2f} | "
                        f"{next_green:>6.1f} | "
                        f"NS[{movement_text(ns)}] EW[{movement_text(ew)}] | "
                        f"{reason}"
                    )

                    if current == "NS":
                        run_clearance(
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
                        run_clearance(
                            PHASE_EW_YELLOW,
                            YELLOW_TIME,
                            red_wait,
                            current,
                        )
                        traci.trafficlight.setPhase(
                            TLS_ID,
                            PHASE_NS_GREEN,
                        )

                    current = next_side
                    red_wait[current] = 0.0
                    phase_elapsed = 0.0
                    last_reassessment = 0.0
                    current_start_queue = max(0, next_metrics["halting"])
                    current_target = next_green

                else:
                    remaining_target = compute_initial_green(metrics, 0.0)
                    current_target = min(
                        MAX_GREEN,
                        max(current_target, phase_elapsed + remaining_target),
                    )
                    last_reassessment = 0.0

            if phase_elapsed >= current_target and phase_elapsed >= MIN_GREEN:
                continue

        print("\nSimulation finished.")
        print(f"Seed: {seed}")
        print(f"Steps: {step}")
        print(f"Tripinfo: {tripinfo_output}")

    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--tripinfo-output",
        default="adaptive_v6_seed42.xml",
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
        help="SUMO random seed",
    )
    args = parser.parse_args()

    run(
        gui=args.gui,
        tripinfo_output=args.tripinfo_output,
        max_steps=args.max_steps,
        seed=args.seed,
    )
