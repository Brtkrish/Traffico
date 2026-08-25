"""Adaptive v7: protected left-turn phases.

This version keeps the same SUMO demand and movement-aware sensing, but adds
four signal phases:
  0 NS through/right
  1 NS protected left
  2 EW through/right
  3 EW protected left

Yellow phases are 1, 3, 5, 7 respectively.

The controller selects the next safe phase from current movement demand.
Run first with seed 42:
    python adaptive_controller_v7.py --gui --tripinfo-output adaptive_v7_seed42.xml
"""

import argparse
import sys
import traci

TLS_ID = "center"

# 8-phase program in intersection_v7.net.xml
PHASE_NS_TR = 0
PHASE_NS_TR_Y = 1
PHASE_NS_LEFT = 2
PHASE_NS_LEFT_Y = 3
PHASE_EW_TR = 4
PHASE_EW_TR_Y = 5
PHASE_EW_LEFT = 6
PHASE_EW_LEFT_Y = 7

NS_LANES = ["north_in_0", "south_in_0"]
EW_LANES = ["east_in_0", "west_in_0"]

STEP_LENGTH = 1.0
MAX_STEPS = 600

MIN_GREEN_TR = 8.0
MAX_GREEN_TR = 35.0
MIN_GREEN_LEFT = 6.0
MAX_GREEN_LEFT = 20.0
BASE_GREEN_TR = 10.0
BASE_GREEN_LEFT = 6.0
YELLOW_TIME = 3.0

MAX_RED_WAIT = 60.0
MIN_EXTEND_INTERVAL = 4.0
SWITCH_MARGIN = 5.0

COUNT_W = 0.25
HALTING_W = 1.50
WAIT_W = 0.45
LEFT_WAIT_BONUS = 0.85


def other_approach(side):
    return "EW" if side.startswith("NS") else "NS"


def other_through(side):
    return "EW_TR" if side == "NS_TR" else "NS_TR"


def other_left(side):
    return "EW_LEFT" if side == "NS_LEFT" else "NS_LEFT"


def phase_group(side):
    return "NS" if side.startswith("NS") else "EW"


def movement_counts(lanes):
    counts = {"straight": 0, "left": 0, "right": 0}
    halting = {"straight": 0, "left": 0, "right": 0}
    waiting_by_move = {"straight": 0.0, "left": 0.0, "right": 0.0}

    vehicle_total = 0
    halt_total = 0
    wait_total = 0.0

    movement_map = {
        "north_in": {"center_out_s": "straight", "center_out_e": "left", "center_out_w": "right"},
        "south_in": {"center_out_n": "straight", "center_out_w": "left", "center_out_e": "right"},
        "east_in": {"center_out_w": "straight", "center_out_s": "left", "center_out_n": "right"},
        "west_in": {"center_out_e": "straight", "center_out_n": "left", "center_out_s": "right"},
    }

    for lane_id in lanes:
        vehicle_total += traci.lane.getLastStepVehicleNumber(lane_id)
        halt_total += traci.lane.getLastStepHaltingNumber(lane_id)
        wait_total += traci.lane.getWaitingTime(lane_id)

        incoming = lane_id.rsplit("_", 1)[0]
        ids = traci.lane.getLastStepVehicleIDs(lane_id)
        for vid in ids:
            route = traci.vehicle.getRoute(vid)
            idx = traci.vehicle.getRouteIndex(vid)
            if idx < 0 or idx + 1 >= len(route):
                continue
            out_edge = route[idx + 1]
            move = movement_map.get(incoming, {}).get(out_edge)
            if move is None:
                continue

            counts[move] += 1
            speed = traci.vehicle.getSpeed(vid)
            if speed < 0.1:
                halting[move] += 1

            # SUMO vehicle waiting time is cumulative, so this is useful
            # as a relative movement-pressure indicator.
            waiting_by_move[move] += traci.vehicle.getAccumulatedWaitingTime(vid)

    return {
        "vehicles": vehicle_total,
        "halting": halt_total,
        "waiting": wait_total,
        "counts": counts,
        "halting_by_move": halting,
        "waiting_by_move": waiting_by_move,
    }


def phase_score(metrics, red_wait, kind):
    if kind == "LEFT":
        move = metrics["counts"]["left"]
        halted = metrics["halting_by_move"]["left"]
        mwait = metrics["waiting_by_move"]["left"]
        return (
            move * COUNT_W
            + halted * HALTING_W
            + red_wait * WAIT_W
            + mwait * 0.01
            + halted * LEFT_WAIT_BONUS
        )

    straight = metrics["counts"]["straight"]
    right = metrics["counts"]["right"]
    halted_straight = metrics["halting_by_move"]["straight"]
    halted_right = metrics["halting_by_move"]["right"]

    return (
        (straight + right) * COUNT_W
        + (halted_straight + halted_right) * HALTING_W
        + red_wait * WAIT_W
        + (metrics["waiting_by_move"]["straight"] +
           metrics["waiting_by_move"]["right"]) * 0.005
    )


def green_time(metrics, red_wait, kind):
    if kind == "LEFT":
        raw = (
            BASE_GREEN_LEFT
            + metrics["counts"]["left"] * 0.65
            + metrics["halting_by_move"]["left"] * 0.80
            + red_wait * 0.02
        )
        return max(MIN_GREEN_LEFT, min(MAX_GREEN_LEFT, raw))

    tr_count = metrics["counts"]["straight"] + metrics["counts"]["right"]
    tr_halt = metrics["halting_by_move"]["straight"] + metrics["halting_by_move"]["right"]
    raw = (
        BASE_GREEN_TR
        + tr_count * 0.40
        + tr_halt * 0.60
        + red_wait * 0.02
    )
    return max(MIN_GREEN_TR, min(MAX_GREEN_TR, raw))


def is_demanded(metrics, kind):
    if kind == "LEFT":
        return metrics["halting_by_move"]["left"] >= 1
    return (
        metrics["halting_by_move"]["straight"] +
        metrics["halting_by_move"]["right"]
    ) >= 1


def set_green(side):
    phases = {
        "NS_TR": PHASE_NS_TR,
        "NS_LEFT": PHASE_NS_LEFT,
        "EW_TR": PHASE_EW_TR,
        "EW_LEFT": PHASE_EW_LEFT,
    }
    traci.trafficlight.setPhase(TLS_ID, phases[side])


def set_yellow(side):
    phases = {
        "NS_TR": PHASE_NS_TR_Y,
        "NS_LEFT": PHASE_NS_LEFT_Y,
        "EW_TR": PHASE_EW_TR_Y,
        "EW_LEFT": PHASE_EW_LEFT_Y,
    }
    traci.trafficlight.setPhase(TLS_ID, phases[side])


def run(gui=False, tripinfo_output="adaptive_v7_seed42.xml", max_steps=600, seed=42):
    sumo_binary = "sumo-gui" if gui else "sumo"
    command = [
        sumo_binary,
        "-c",
        "intersection_v7.sumocfg",
        "--tripinfo-output",
        tripinfo_output,
        "--step-length",
        str(STEP_LENGTH),
        "--seed",
        str(seed),
    ]

    traci.start(command)

    red_wait = {"NS": 0.0, "EW": 0.0}
    current = "NS_TR"
    phase_elapsed = 0.0
    last_check = 0.0
    current_target = MIN_GREEN_TR
    step = 0

    set_green(current)

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            step += 1
            phase_elapsed += STEP_LENGTH
            last_check += STEP_LENGTH

            ns = movement_counts(NS_LANES)
            ew = movement_counts(EW_LANES)

            active_group = phase_group(current)
            if active_group == "NS":
                red_wait["NS"] = 0.0
                red_wait["EW"] += STEP_LENGTH
            else:
                red_wait["EW"] = 0.0
                red_wait["NS"] += STEP_LENGTH

            if phase_elapsed < min_green_for(current) or last_check < MIN_EXTEND_INTERVAL:
                continue

            ns_tr = phase_score(ns, red_wait["NS"], "TR")
            ns_l = phase_score(ns, red_wait["NS"], "LEFT")
            ew_tr = phase_score(ew, red_wait["EW"], "TR")
            ew_l = phase_score(ew, red_wait["EW"], "LEFT")

            candidate = [
                ("NS_TR", ns_tr),
                ("NS_LEFT", ns_l),
                ("EW_TR", ew_tr),
                ("EW_LEFT", ew_l),
            ]

            # Don't schedule an empty left phase just because its score has a
            # little inherited waiting-time pressure.
            filtered = []
            for side, score in candidate:
                data = ns if phase_group(side) == "NS" else ew
                kind = "LEFT" if side.endswith("LEFT") else "TR"
                if is_demanded(data, kind) or red_wait[phase_group(side)] >= MAX_RED_WAIT:
                    filtered.append((side, score))

            if not filtered:
                filtered = candidate

            filtered.sort(key=lambda x: x[1], reverse=True)
            best_side, best_score = filtered[0]

            current_kind = "LEFT" if current.endswith("LEFT") else "TR"
            current_metrics = ns if phase_group(current) == "NS" else ew
            current_score = (
                ns_l if current == "NS_LEFT" else
                ns_tr if current == "NS_TR" else
                ew_l if current == "EW_LEFT" else
                ew_tr
            )

            opposite_wait = red_wait[other_approach(current)]

            # Hold the current phase if it is still clearly useful.
            hard_starvation = opposite_wait >= MAX_RED_WAIT
            maxed = phase_elapsed >= max_green_for(current)

            switch = hard_starvation or maxed
            reason = "STARVATION" if hard_starvation else ("MAX_GREEN" if maxed else "HOLD")

            if not switch and best_side != current and best_score > current_score + SWITCH_MARGIN:
                switch = True
                reason = "BEST_MOVEMENT"

            if not switch and phase_elapsed >= current_target:
                switch = best_side != current
                reason = "TARGET_REACHED" if switch else "EXTEND"

            if not switch:
                current_target = min(
                    max_green_for(current),
                    max(current_target, phase_elapsed + 3.0)
                )
                last_check = 0.0
                continue

            # Yellow then transition.
            set_yellow(current)
            for _ in range(int(YELLOW_TIME)):
                traci.simulationStep()
                step += 1
                if phase_group(current) == "NS":
                    red_wait["NS"] = 0.0
                    red_wait["EW"] += STEP_LENGTH
                else:
                    red_wait["EW"] = 0.0
                    red_wait["NS"] += STEP_LENGTH

            current = best_side
            set_green(current)
            phase_elapsed = 0.0
            last_check = 0.0

            next_metrics = ns if phase_group(current) == "NS" else ew
            current_target = green_time(
                next_metrics,
                red_wait[phase_group(current)],
                "LEFT" if current.endswith("LEFT") else "TR",
            )

            print(
                f"{step:>4} | {current:>7} | "
                f"NS S/L/R {ns['counts']['straight']}/{ns['counts']['left']}/{ns['counts']['right']} "
                f"| EW S/L/R {ew['counts']['straight']}/{ew['counts']['left']}/{ew['counts']['right']} "
                f"| target {current_target:>5.1f}s | {reason}"
            )

        print("\nSimulation finished.")
        print(f"Seed: {seed}")
        print(f"Steps: {step}")
        print(f"Tripinfo: {tripinfo_output}")

    finally:
        traci.close()
        sys.stdout.flush()


def min_green_for(side):
    return MIN_GREEN_LEFT if side.endswith("LEFT") else MIN_GREEN_TR


def max_green_for(side):
    return MAX_GREEN_LEFT if side.endswith("LEFT") else MAX_GREEN_TR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tripinfo-output", default="adaptive_v7_seed42.xml")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        gui=args.gui,
        tripinfo_output=args.tripinfo_output,
        max_steps=args.max_steps,
        seed=args.seed,
    )
