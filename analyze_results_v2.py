"""Analyze a SUMO tripinfo XML file.

Examples:
    python analyze_results_v2.py --file adaptive_tripinfo.xml
    python analyze_results_v2.py --file fixed_tripinfo.xml
"""

import argparse
import statistics
import xml.etree.ElementTree as ET


def analyze(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    wait_times = []
    durations = []
    time_losses = []

    for trip in root.findall("tripinfo"):
        wait_times.append(float(trip.get("waitingTime", 0.0)))
        durations.append(float(trip.get("duration", 0.0)))
        time_losses.append(float(trip.get("timeLoss", 0.0)))

    if not wait_times:
        print("No completed trips found in:", filename)
        return

    print(f"Results: {filename}")
    print(f"Total vehicles completed: {len(wait_times)}")
    print(f"Average waiting time:     {statistics.mean(wait_times):.2f} s")
    print(f"Median waiting time:      {statistics.median(wait_times):.2f} s")
    print(f"Max waiting time:         {max(wait_times):.2f} s")
    print(f"Average trip duration:    {statistics.mean(durations):.2f} s")
    print(f"Average time lost:        {statistics.mean(time_losses):.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="adaptive_tripinfo.xml", help="tripinfo XML file")
    args = parser.parse_args()
    analyze(args.file)
