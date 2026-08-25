"""
Adaptive traffic signal controller running against a SUMO simulation via TraCI.

This mirrors the decision logic from the YOLOv8-based project:
  - instead of vehicle counts coming from CV detection on video frames,
    counts come live from SUMO's simulated lanes each step.
  - the same GREEN / YELLOW / ALL_RED style state machine decides how long
    each direction gets, using vehicle count + a starvation/wait-time score.
  - the decision is applied back to SUMO via traci.trafficlight.setPhase(),
    so you can WATCH (in sumo-gui) queues actually build and clear in
    response to your algorithm -- the closed loop your stock video couldn't give you.

Run:
    python3 adaptive_controller.py            # headless
    python3 adaptive_controller.py --gui      # with sumo-gui (visual)
"""

import argparse
import sys
import traci

# --- Phase indices from the auto-generated tlLogic (see intersection.net.xml) ---
# phase 0 = North/South GREEN   (42s default, we override this)
# phase 1 = North/South YELLOW  (fixed, 3s)
# phase 2 = East/West GREEN     (42s default, we override this)
# phase 3 = East/West YELLOW    (fixed, 3s)
TLS_ID = "center"
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3

NS_LANES = ["north_in_0", "south_in_0"]
EW_LANES = ["east_in_0", "west_in_0"]

MIN_GREEN = 8      # seconds - floor so a light isn't absurdly short
MAX_GREEN = 45      # seconds - ceiling so the other side never fully starves
YELLOW_TIME = 3
BASE_GREEN = 10      # baseline green before adding count-based extension
SECONDS_PER_VEHICLE = 2.5   # how much extra green each waiting vehicle earns
STARVATION_BONUS_PER_STEP = 0.3   # grows priority for a side that's been waiting


def get_lane_counts(lanes):
    """Live vehicle count per lane group, straight from the simulation (no CV needed here)."""
    return sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)


def compute_green_time(vehicle_count, starvation_score):
    """
    Same idea as the YOLOv8 controller's allocation rule:
    more waiting vehicles -> longer green, clamped to [MIN_GREEN, MAX_GREEN],
    with a starvation bonus so a long-neglected side gets pushed up faster.
    """
    green = BASE_GREEN + vehicle_count * SECONDS_PER_VEHICLE + starvation_score
    return int(max(MIN_GREEN, min(MAX_GREEN, green)))


def run(gui=False):
    sumo_binary = "sumo-gui" if gui else "sumo"
    traci.start([sumo_binary, "-c", "intersection.sumocfg"])

    starvation = {"NS": 0.0, "EW": 0.0}
    current = "NS"  # which side currently has green
    step = 0

    print(f"{'step':>6} | {'NS_count':>8} | {'EW_count':>8} | {'green_to':>8} | {'green_secs':>10}")

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < 600:
            ns_count = get_lane_counts(NS_LANES)
            ew_count = get_lane_counts(EW_LANES)

            # whichever side is red accrues starvation pressure
            if current == "NS":
                starvation["EW"] += STARVATION_BONUS_PER_STEP
                starvation["NS"] = 0.0
            else:
                starvation["NS"] += STARVATION_BONUS_PER_STEP
                starvation["EW"] = 0.0

            # decide which side should get green next: higher demand + starvation wins
            ns_priority = ns_count + starvation["NS"]
            ew_priority = ew_count + starvation["EW"]
            next_side = "NS" if ns_priority >= ew_priority else "EW"

            green_count = ns_count if next_side == "NS" else ew_count
            green_secs = compute_green_time(green_count, starvation[next_side])

            print(f"{step:>6} | {ns_count:>8} | {ew_count:>8} | {next_side:>8} | {green_secs:>10}")

            # --- apply decision to SUMO ---
            if next_side == "NS":
                if current == "EW":
                    traci.trafficlight.setPhase(TLS_ID, PHASE_EW_YELLOW)
                    for _ in range(YELLOW_TIME):
                        traci.simulationStep(); step += 1
                traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
            else:
                if current == "NS":
                    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_YELLOW)
                    for _ in range(YELLOW_TIME):
                        traci.simulationStep(); step += 1
                traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)

            current = next_side

            for _ in range(green_secs):
                traci.simulationStep()
                step += 1

    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="run with sumo-gui for visual playback")
    args = parser.parse_args()
    run(gui=args.gui)
