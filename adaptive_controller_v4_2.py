"""Adaptive v4.2 SUMO controller.

v4.2 is a targeted refinement of validated v4.1.

What changed from v4.1:
- Removed the aggressive LOW_CURRENT_QUEUE switching behavior.
- Requires opposing demand to persist across two reassessments before switching.
- Adds a post-switch stability lock so the controller cannot immediately
  reverse direction.
- Keeps the same scoring, queue-clearing logic, starvation protection,
  2-phase NS/EW signal plan, and green-time model.

Goal:
    Reduce unnecessary phase switching / yellow-clearance loss while
    preserving v4.1 throughput and waiting-time performance.

First test:
    python adaptive_controller_v4_2.py --gui --tripinfo-output adaptive_v42_seed42.xml --seed 42
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

COUNT_WEIGHT = 0.35
HALTING_WEIGHT = 1.75
WAIT_WEIGHT = 0.55

SECONDS_PER_VEHICLE = 0.45
SECONDS_PER_HALTING_VEHICLE = 0.65
SECONDS_PER_WAITING_SECOND = 0.02

# v4.2 stability controls
MIN_REASSESS_INTERVAL = 5.0
POST_SWITCH_LOCK = 10.0
OPPOSING_DEMAND_MARGIN = 9.0
OPPOSING_PERSISTENCE_REQUIRED = 2

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


def run_clearance(phase_index, duration, red_wait, green_side):
    traci.trafficlight.setPhase(TLS_ID, phase_index)

    for _ in range(int(round(duration))):
        traci.simulationStep()
        update_red_wait(red_wait, green_side)


def other_side(side):
    return "EW" if side == "NS" else "NS"


def current_metrics_for(side, ns, ew):
    return ns if side == "NS" else ew


def switch_decision(
    current,
    current_metrics,
    opposite_metrics,
    current_score,
    opposite_score,
    phase_elapsed,
    current_start_queue,
    current_previous_queue,
    red_wait,
    post_switch_lock_remaining,
    opposing_persistence,
):
    opposite = other_side(current)
    opposite_wait = red_wait[opposite]
    current_queue = current_metrics["halting"]

    # Hard starvation protection overrides the stability lock.
    if (
        opposite_wait >= MAX_RED_WAIT
        and phase_elapsed >= MIN_GREEN
        and post_switch_lock_remaining <= 0
    ):
        return True, "STARVATION"

    # Never switch too early.
    if phase_elapsed < MIN_EFFECTIVE_GREEN:
        return False, "MIN_GREEN"

    # Respect hard maximum green, unless we're still inside the short
    # post-switch stability lock.
    if phase_elapsed >= MAX_GREEN and post_switch_lock_remaining <= 0:
        return True, "MAX_GREEN"

    # During the post-switch lock, hold unless starvation is urgent.
    if post_switch_lock_remaining > 0:
        return False, "POST_SWITCH_LOCK"

    # If current phase is clearing well, prefer to continue.
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
            and opposite_score <= current_score + OPPOSING_DEMAND_MARGIN
        ):
            return False, "CURRENT_CLEARING"

    # Main v4.2 change:
    # opposing demand must be meaningfully larger and persist across multiple
    # reassessments before we switch.
    if (
        opposite_score > current_score + OPPOSING_DEMAND_MARGIN
        and opposite_metrics["halting"] > 0
    ):
        opposing_persistence += 1
        if opposing_persistence >= OPPOSING_PERSISTENCE_REQUIRED:
            return True, "PERSISTENT_OPPOSING_DEMAND"

    return False, "HOLD"


def run(
    gui=False,
    tripinfo_output="adaptive_v42_seed42.xml",
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
    post_switch_lock_remaining = 0.0
    step = 0

    # Count how many consecutive reassessments the opposing side has been
    # clearly better than the current side.
    opposing_persistence = 0

    ns = lane_metrics(NS_LANES)
    ew = lane_metrics(EW_LANES)

    current_start_metrics = ns.copy()
    current_previous_queue = ns["halting"]

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

            post_switch_lock_remaining = max(
                0.0,
                post_switch_lock_remaining - STEP_LENGTH,
            )

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
            opposite_side_name = other_side(current)
            opposite_metrics = current_metrics_for(
                opposite_side_name, ns, ew
            )

            ns_score = demand_score(ns, red_wait["NS"])
            ew_score = demand_score(ew, red_wait["EW"])

            current_score = ns_score if current == "NS" else ew_score
            opposite_score = ew_score if current == "NS" else ns_score

            # Persistence tracking is reset when the opposing side is no longer
            # clearly better.
            if opposite_score > current_score + OPPOSING_DEMAND_MARGIN:
                opposing_persistence += 1
            else:
                opposing_persistence = 0

            # Avoid double-counting when passing the persistence into the helper:
            # helper receives the count already accumulated this reassessment.
            switch = False
            reason = "HOLD"

            # Starvation.
            if (
                red_wait[opposite_side_name] >= MAX_RED_WAIT
                and phase_elapsed >= MIN_GREEN
                and post_switch_lock_remaining <= 0
            ):
                switch = True
                reason = "STARVATION"

            elif post_switch_lock_remaining > 0:
                reason = "POST_SWITCH_LOCK"

            elif phase_elapsed < MIN_EFFECTIVE_GREEN:
                reason = "MIN_GREEN"

            elif phase_elapsed >= MAX_GREEN:
                switch = True
                reason = "MAX_GREEN"

            else:
                current_queue = current_metrics["halting"]

                # Keep current phase if it is still clearing effectively.
                if current_start_metrics["halting"] > 0:
                    total_reduction = (
                        current_start_metrics["halting"] - current_queue
                    ) / current_start_metrics["halting"]

                    recent_improvement = max(
                        0,
                        current_previous_queue - current_queue,
                    )

                    if (
                        total_reduction >= QUEUE_CLEAR_FRACTION
                        and recent_improvement >= 0
                        and current_queue > LOW_QUEUE_THRESHOLD
                        and opposite_score
                        <= current_score + OPPOSING_DEMAND_MARGIN
                    ):
                        switch = False
                        reason = "CURRENT_CLEARING"

                # Only persistent strong opposing demand can cause a normal switch.
                if (
                    not switch
                    and opposite_score
                    > current_score + OPPOSING_DEMAND_MARGIN
                    and opposite_metrics["halting"] > 0
                    and opposing_persistence
                    >= OPPOSING_PERSISTENCE_REQUIRED
                ):
                    switch = True
                    reason = "PERSISTENT_OPPOSING_DEMAND"

            if not switch:
                refreshed = compute_green_time(current_metrics, 0.0)
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
            reassess_elapsed = 0.0
            current_start_metrics = next_metrics.copy()
            current_previous_queue = next_metrics["halting"]
            current_target = next_green

            # This is the new v4.2 anti-chatter mechanism.
            post_switch_lock_remaining = POST_SWITCH_LOCK

            # Reset persistence after an actual switch.
            opposing_persistence = 0

        print("\nSimulation finished.")
        print(f"Seed: {seed}")
        print(f"Steps: {step}")
        print(f"Tripinfo: {tripinfo_output}")

    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    parser.add_argument(
        "--tripinfo-output",
        default="adaptive_v42_seed42.xml",
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
