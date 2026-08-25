"""Adaptive NS/EW traffic-signal controller for SUMO via TraCI.

Adaptive v3 focuses on the weakness found in v2: excessive waiting on the
neglected phase. It keeps the same physically safe 2-phase NS/EW signal plan,
but adds:

- hard starvation protection (a side waiting too long gets the next green),
- a lower maximum continuous green,
- queue/halting/vehicle/wait based scoring,
- reassessment at the end of every green allocation,
- bounded green duration without allowing one side to hold green forever.

Run:
    python adaptive_controller_v3.py
    python adaptive_controller_v3.py --gui
    python adaptive_controller_v3.py --tripinfo-output adaptive_v3_tripinfo.xml
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

# Signal timing parameters.
MIN_GREEN = 8.0
MAX_GREEN = 35.0          # v3: reduced from 45s to limit long starvation elsewhere
BASE_GREEN = 10.0
YELLOW_TIME = 3.0
MAX_RED_WAIT = 60.0       # v3: hard fairness cap

# Adaptive-controller weights.
# Queue/halting vehicles are deliberately stronger than raw vehicle presence.
COUNT_WEIGHT = 0.50
HALTING_WEIGHT = 2.00
WAIT_WEIGHT = 0.80

# Green-time extension weights.
SECONDS_PER_VEHICLE = 0.70
SECONDS_PER_HALTING_VEHICLE = 0.85
SECONDS_PER_WAITING_SECOND = 0.04

STEP_LENGTH = 1.0
MAX_STEPS = 600


def lane_metrics(lanes):
    """Return aggregate traffic metrics for a group of incoming lanes."""
    vehicles = 0
    halting = 0
    total_waiting = 0.0
    mean_speeds = []

    for lane_id in lanes:
        vehicles += traci.lane.getLastStepVehicleNumber(lane_id)
        halting += traci.lane.getLastStepHaltingNumber(lane_id)
        total_waiting += traci.lane.getWaitingTime(lane_id)
        speed = traci.lane.getLastStepMeanSpeed(lane_id)
        if speed >= 0:
            mean_speeds.append(speed)

    mean_speed = sum(mean_speeds) / len(mean_speeds) if mean_speeds else 0.0

    return {
        "vehicles": vehicles,
        "halting": halting,
        "waiting": total_waiting,
        "mean_speed": mean_speed,
    }


def demand_score(metrics, red_wait):
    """Score current need for green for one phase."""
    return (
        metrics["vehicles"] * COUNT_WEIGHT
        + metrics["halting"] * HALTING_WEIGHT
        + red_wait * WAIT_WEIGHT
    )


def compute_green_time(metrics, red_wait):
    """Allocate bounded green time from current demand."""
    green = (
        BASE_GREEN
        + metrics["vehicles"] * SECONDS_PER_VEHICLE
        + metrics["halting"] * SECONDS_PER_HALTING_VEHICLE
        + red_wait * SECONDS_PER_WAITING_SECOND
    )
    return max(MIN_GREEN, min(MAX_GREEN, green))


def update_red_wait(red_wait, green_side):
    """Advance the continuously-red side's wait timer by one simulation second."""
    if green_side == "NS":
        red_wait["NS"] = 0.0
        red_wait["EW"] += STEP_LENGTH
    else:
        red_wait["EW"] = 0.0
        red_wait["NS"] += STEP_LENGTH


def run_clearance(phase_index, duration, red_wait, green_side):
    """Run a yellow phase while maintaining red-wait accounting."""
    traci.trafficlight.setPhase(TLS_ID, phase_index)
    steps = int(round(duration / STEP_LENGTH))
    for _ in range(steps):
        traci.simulationStep()
        update_red_wait(red_wait, green_side)


def choose_next_side(current, ns_score, ew_score, ns_wait, ew_wait):
    """Choose the next side with hard starvation protection first."""
    # Hard fairness rule: if either side has waited too long, it gets priority.
    if ns_wait >= MAX_RED_WAIT and ew_wait >= MAX_RED_WAIT:
        # If both are starved, pick the one with the larger score.
        return "NS" if ns_score >= ew_score else "EW"

    if ns_wait >= MAX_RED_WAIT:
        return "NS"
    if ew_wait >= MAX_RED_WAIT:
        return "EW"

    # Otherwise use adaptive demand score, preserving the current side on ties.
    if ns_score > ew_score:
        return "NS"
    if ew_score > ns_score:
        return "EW"
    return current


def run(gui=False, tripinfo_output="adaptive_v3_tripinfo.xml", max_steps=MAX_STEPS):
    sumo_binary = "sumo-gui" if gui else "sumo"

    command = [
        sumo_binary,
        "-c", "intersection.sumocfg",
        "--tripinfo-output", tripinfo_output,
        "--step-length", str(STEP_LENGTH),
    ]

    traci.start(command)

    red_wait = {"NS": 0.0, "EW": 0.0}
    current = "NS"
    green_elapsed = 0.0
    current_green_target = BASE_GREEN
    step = 0

    print(
        f"{'step':>6} | {'NS_v':>5} | {'NS_q':>5} | {'NS_wait':>7} | {'NS_score':>9} | "
        f"{'EW_v':>5} | {'EW_q':>5} | {'EW_wait':>7} | {'EW_score':>9} | {'phase':>6} | {'green':>6}"
    )

    try:
        traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)

        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            step += 1
            green_elapsed += STEP_LENGTH

            ns = lane_metrics(NS_LANES)
            ew = lane_metrics(EW_LANES)
            update_red_wait(red_wait, current)

            ns_score = demand_score(ns, red_wait["NS"])
            ew_score = demand_score(ew, red_wait["EW"])

            if green_elapsed >= current_green_target:
                next_side = choose_next_side(
                    current,
                    ns_score,
                    ew_score,
                    red_wait["NS"],
                    red_wait["EW"],
                )

                next_metrics = ns if next_side == "NS" else ew
                next_wait = red_wait[next_side]
                next_green = compute_green_time(next_metrics, next_wait)

                forced = (
                    next_side != current
                    and ((next_side == "NS" and red_wait["NS"] >= MAX_RED_WAIT)
                         or (next_side == "EW" and red_wait["EW"] >= MAX_RED_WAIT))
                )

                marker = " FORCE" if forced else ""
                print(
                    f"{step:>6} | {ns['vehicles']:>5} | {ns['halting']:>5} | {red_wait['NS']:>7.0f} | {ns_score:>9.2f} | "
                    f"{ew['vehicles']:>5} | {ew['halting']:>5} | {red_wait['EW']:>7.0f} | {ew_score:>9.2f} | "
                    f"{next_side:>6} | {next_green:>6.1f}{marker}"
                )

                if next_side == current:
                    # Recalculate a new bounded green window using fresh conditions.
                    # It cannot exceed MAX_GREEN, so one side cannot hold green forever.
                    current_green_target = next_green
                    green_elapsed = 0.0
                else:
                    if current == "NS":
                        run_clearance(PHASE_NS_YELLOW, YELLOW_TIME, red_wait, current)
                    else:
                        run_clearance(PHASE_EW_YELLOW, YELLOW_TIME, red_wait, current)

                    current = next_side
                    if current == "NS":
                        traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
                    else:
                        traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)

                    red_wait[current] = 0.0
                    current_green_target = next_green
                    green_elapsed = 0.0

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
        default="adaptive_v3_tripinfo.xml",
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
