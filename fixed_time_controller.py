"""Fixed-time SUMO baseline controller for comparison with the adaptive version."""

import argparse
import sys
import traci

TLS_ID = "center"
PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2
NS_GREEN = 30
EW_GREEN = 30
MAX_STEPS = 600


def run(gui=False, tripinfo_output="fixed_tripinfo.xml", max_steps=MAX_STEPS):
    sumo_binary = "sumo-gui" if gui else "sumo"
    traci.start([
    sumo_binary,
    "-c", "intersection.sumocfg",
    "--seed", "46",
    "--tripinfo-output", tripinfo_output
])

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
            for _ in range(NS_GREEN):
                if traci.simulation.getMinExpectedNumber() <= 0 or step >= max_steps:
                    break
                traci.simulationStep()
                step += 1

            if traci.simulation.getMinExpectedNumber() <= 0 or step >= max_steps:
                break

            # Let SUMO's existing NS yellow phase run for its configured 3 seconds.
            for _ in range(3):
                traci.simulationStep()
                step += 1

            traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
            for _ in range(EW_GREEN):
                if traci.simulation.getMinExpectedNumber() <= 0 or step >= max_steps:
                    break
                traci.simulationStep()
                step += 1

            if traci.simulation.getMinExpectedNumber() <= 0 or step >= max_steps:
                break

            # EW yellow.
            for _ in range(3):
                traci.simulationStep()
                step += 1

    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tripinfo-output", default="fixed_tripinfo.xml")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    run(gui=args.gui, tripinfo_output=args.tripinfo_output, max_steps=args.max_steps)
