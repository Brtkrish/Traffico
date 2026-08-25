"""
Parses tripinfo.xml (produced automatically after running adaptive_controller.py,
since intersection.sumocfg now has <tripinfo-output value="tripinfo.xml"/>)
and prints summary metrics you can quote as results.

Run AFTER adaptive_controller.py finishes:
    python3 analyze_results.py
"""

import xml.etree.ElementTree as ET
import statistics

tree = ET.parse("tripinfo.xml")
root = tree.getroot()

wait_times = []
durations = []
time_losses = []

for trip in root.findall("tripinfo"):
    wait_times.append(float(trip.get("waitingTime")))
    durations.append(float(trip.get("duration")))
    time_losses.append(float(trip.get("timeLoss")))

n = len(wait_times)
print(f"Total vehicles completed: {n}")
print(f"Average waiting time:     {statistics.mean(wait_times):.2f} s")
print(f"Max waiting time:         {max(wait_times):.2f} s")
print(f"Average trip duration:    {statistics.mean(durations):.2f} s")
print(f"Average time lost:        {statistics.mean(time_losses):.2f} s")
