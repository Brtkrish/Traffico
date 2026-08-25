# 🚦 Trafico
## Explainable Adaptive Traffic-Signal Control with YOLOv8 + SUMO

Trafico is an end-to-end traffic-signal research prototype that combines **computer vision, adaptive control, and microscopic traffic simulation**.

The project is built around a simple engineering question:

> **Can an adaptive traffic controller make a four-leg intersection handle changing traffic demand better than a conventional fixed-time controller?**

The answer is evaluated in two layers:

- **YOLOv8 + Streamlit** demonstrates perception and live adaptive decision-making from traffic-camera footage.
- **SUMO** provides the controlled environment used to measure whether the adaptive policy actually improves traffic performance.

This separation is deliberate: the camera videos are pre-recorded, so they can demonstrate detection and control decisions, while SUMO provides the quantitative closed-loop performance evaluation.

---

# ✨ Why this project is interesting

A fixed-time signal assumes that traffic demand is reasonably predictable:

```text
NS → 30 s green
EW → 30 s green
NS → 30 s green
EW → 30 s green
...
```

That wastes green time when one approach is nearly empty and the other is congested.

Trafico instead estimates current demand and adapts the phase:

```text
                 Traffic cameras
                        │
                        ▼
                  YOLOv8 detection
                        │
                        ▼
             Vehicle count + tracking
                        │
                        ▼
          Persistent queue estimation
                        │
                        ▼
              Adaptive NS / EW score
                        │
                        ▼
             Green / Yellow / All-Red
                        │
                        ▼
               Next signal phase
```

The controller also includes **fairness protection** so a quieter side cannot wait indefinitely.

---

# 🧠 System architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    COMPUTER VISION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  North video ─┐                                             │
│  South video ─┼──► YOLOv8 ─► ROI filtering ─► tracking     │
│  East video  ─┤                         │                   │
│  West video  ─┘                         ▼                   │
│                               stopped / queue estimates    │
│                                                             │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     CONTROL LAYER                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NS demand = vehicles + queue pressure + red waiting       │
│  EW demand = vehicles + queue pressure + red waiting       │
│                                                             │
│  v4.1 adds:                                                │
│    • minimum green                           │
│    • maximum green                           │
│    • queue-clearing logic                    │
│    • switching hysteresis                    │
│    • starvation protection                   │
│                                                             │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       Streamlit dashboard                 SUMO / TraCI
                │                               │
                ▼                               ▼
       live explanation of             closed-loop performance
       what the controller sees       and objective comparison
```

---

# 🚥 Signal-control model

The final validated SUMO controller is **Adaptive v4.1**.

The intersection uses two safe signal groups:

```text
NS phase
  North + South GREEN
  East + West RED

EW phase
  East + West GREEN
  North + South RED
```

This is intentional.

The simulated intersection has **one incoming lane per approach**, so splitting left turns into separate protected phases created additional yellow/clearance overhead without improving the tested results. Experimental protected-left versions were evaluated and rejected.

## Demand scoring

The controller combines three ideas:

### 1. Current demand

More vehicles increase the priority of a phase.

### 2. Queue pressure

Stopped vehicles are more urgent than vehicles that are already moving.

### 3. Fairness / starvation protection

A phase that remains red accumulates waiting pressure and eventually receives priority.

Conceptually:

```text
phase_score =
    vehicle_demand
    + queue_pressure
    + red_wait_pressure
```

The selected phase receives a green duration bounded by minimum and maximum limits.

---

# 🛡️ Why v4.1 instead of simply "pick the busiest side"

Early experiments showed that a naive adaptive controller can switch too aggressively.

So v4.1 introduced **phase stability / hysteresis**:

```text
Small demand difference
        ↓
     HOLD

Large persistent difference
        ↓
     SWITCH
```

It also evaluates whether the current phase is still clearing its queue.

This avoids a controller that repeatedly behaves like:

```text
NS → EW → NS → EW → NS → EW
```

because of small fluctuations in detected demand.

---

# 👁️ Computer-vision pipeline

The Streamlit application uses four pre-recorded camera feeds:

```text
north.mp4
south.mp4
east.mp4
west.mp4
```

Each camera has a calibrated polygonal ROI.

Only detections whose center falls inside the ROI contribute to the approach demand.

## Detection classes

YOLOv8 is configured to detect:

- cars
- motorcycles
- buses
- trucks

## Tracking

A lightweight centroid tracker associates detections between processed frames.

Each track can be classified as:

```text
MOVE
STOP
QUEUE
```

The queue estimate is deliberately conservative:

```text
YOLO detection
      ↓
stable track
      ↓
low motion confirmed
      ↓
persistent stop confirmed
      ↓
queue member
```

This is more robust than treating every detected vehicle as a queue.

---

# 🎛️ Decision smoothing for video noise

Real camera detections fluctuate.

A vehicle can disappear for a frame or two, and detection confidence can change.

The camera controller therefore does not immediately switch phases because of one low-count observation.

Instead:

```text
Low demand observation #1 → HOLD
Low demand observation #2 → HOLD
Low demand observation #3 → CONFIRM
                         → SWITCH
```

A short post-switch hold also prevents immediate reversals.

This makes the camera-side behavior more stable and easier to interpret.

---

# 🖥️ Streamlit dashboard

The live dashboard exposes:

- active NS/EW phase;
- countdown timer;
- North / South / East / West vehicle counts;
- estimated queues;
- stopped vehicles;
- per-camera tracking overlays;
- phase-switch logs;
- demand history / analytics.

Example decision log:

```text
NS → EW
NS = 8 vehicles
EW = 25 vehicles
reason = PERSISTENT_OPPOSING_DEMAND
```

That makes the adaptive decision explainable instead of presenting the controller as a black box.

---

# 🧪 SUMO validation methodology

The camera system demonstrates perception and adaptive decisions.

**SUMO is the performance experiment.**

The validation compares:

```text
Fixed-time controller
        VS
Adaptive v4.1
```

under the same:

- network;
- route demand;
- simulation duration;
- random seed.

## Reproducible seeds

Five seeds were used:

```text
42
43
44
45
46
```

A seed determines the reproducible random traffic realization in SUMO.

For every seed:

```text
Seed 42 ─┬─ Fixed-time
         └─ Adaptive v4.1

Seed 43 ─┬─ Fixed-time
         └─ Adaptive v4.1

...
```

This paired design makes the comparison substantially fairer than comparing two unrelated traffic runs.

---

# 📊 Final five-seed results

## Mean across seeds 42–46

| Metric | Fixed-time | Adaptive v4.1 | Improvement |
|---|---:|---:|---:|
| Vehicles completed | 218.40 | **292.20** | **+33.8%** |
| Average waiting time | 35.67 s | **32.15 s** | **−9.9%** |
| Median waiting time | **22.50 s** | 24.50 s | +8.9% |
| Maximum waiting time | 268.20 s | **113.80 s** | **−57.6%** |
| Average trip duration | **87.40 s** | 102.01 s | +16.7% |
| Average time lost | **57.08 s** | 71.65 s | +25.5% |

## What the results actually show

The adaptive controller achieved:

### ✅ Higher throughput

```text
218.4 → 292.2 vehicles
```

approximately **33.8% more completed vehicles**.

### ✅ Lower average waiting

```text
35.67 s → 32.15 s
```

approximately **9.9% lower average waiting**.

### ✅ Much lower worst-case waiting

```text
268.2 s → 113.8 s
```

approximately **57.6% lower maximum waiting**.

### ⚠️ Important trade-off

Average trip duration and time loss increased.

That means the adaptive controller is **not universally better on every metric**.

A defensible interpretation is:

> Adaptive v4.1 improved throughput and reduced average and worst-case waiting under the tested scenarios, but this came with a travel-efficiency trade-off reflected in higher average trip duration and time loss.

That trade-off is an important part of the result rather than something to hide.

---

# 🔬 Experiments that were tested and rejected

The project was developed iteratively rather than stopping at the first working controller.

## v4

Established the successful two-phase adaptive baseline.

## v5

Attempted to reduce phase switching.

Result: travel efficiency improved on one controlled test, but throughput and waiting performance deteriorated.

**Rejected.**

## v6

Added movement-aware straight / left / right information to the demand score.

Result: movement information alone did not improve the overall performance.

**Rejected.**

## v7

Added protected left-turn phases.

Result: the additional phase/clearance overhead performed dramatically worse for this one-lane-per-approach intersection.

**Rejected.**

## v4.2

Added stronger anti-switching restrictions.

Result: reduced switching too aggressively and allowed queues to build.

**Rejected.**

This experimentation led to v4.1 being retained as the best validated trade-off.

---

# 🧩 Final project structure

Recommended final structure:

```text
Trafico/
│
├── traffic_controller_v41_tracking_v3.py
├── adaptive_controller_v4_1.py
├── fixed_time_controller.py
├── analyze_results_v2.py
├── coordinate_finder.py
│
├── intersection.net.xml
├── intersection.sumocfg
├── intersection_v2.rou.xml
│
├── north.mp4
├── south.mp4
├── east.mp4
├── west.mp4
│
├── yolov8n.pt
├── README.md
│
└── results/
    ├── fixed_seed42.xml
    ├── fixed_seed43.xml
    ├── fixed_seed44.xml
    ├── fixed_seed45.xml
    ├── fixed_seed46.xml
    ├── adaptive_v41_seed42.xml
    ├── adaptive_v41_seed43.xml
    ├── adaptive_v41_seed44.xml
    ├── adaptive_v41_seed45.xml
    ├── adaptive_v41_seed46.xml
    └── Trafico_Final_Validation.xlsx
```

Experimental implementations should be moved to an `archive/` directory rather than kept in the main project root.

---

# ▶️ Running the project

## Camera / Streamlit

Install dependencies:

```powershell
pip install streamlit opencv-python numpy pandas ultralytics
```

Run:

```powershell
streamlit run traffic_controller_v41_tracking_v3.py
```

Use:

```text
Automated Adaptive (AI)
```

and enable:

```text
Run Junction
```

---

## SUMO fixed-time baseline

```powershell
python fixed_time_controller.py --gui --tripinfo-output fixed_seed42.xml --max-steps 600
```

---

## SUMO Adaptive v4.1

```powershell
python adaptive_controller_v4_1.py --gui --tripinfo-output adaptive_v41_seed42.xml --seed 42
```

Analyze:

```powershell
python analyze_results_v2.py --file adaptive_v41_seed42.xml
```

Change the seed to repeat the experiment:

```powershell
python adaptive_controller_v4_1.py --gui --tripinfo-output adaptive_v41_seed43.xml --seed 43
```

---

# ⚙️ Configuration

The camera controller exposes parameters including:

| Parameter | Purpose |
|---|---|
| Confidence threshold | Minimum YOLO confidence used for detections |
| Base green time | Minimum demand-driven green allocation |
| Extra seconds / vehicle | Green extension based on demand |
| Maximum green | Prevents one phase from monopolizing the intersection |
| Yellow time | Clearance between phases |
| All-red buffer | Safety clearance interval |
| Minimum green lock | Prevents unrealistically short greens |
| Frame skip | Controls inference frequency |

ROI coordinates are camera-specific and can be recalibrated with:

```text
coordinate_finder.py
```

---

# ⚠️ Limitations

This is a **research and demonstration system**, not a certified traffic controller.

Important limitations include:

- camera feeds are pre-recorded;
- ROI calibration depends on camera viewpoint;
- lightweight tracking can fail under heavy occlusion;
- queue estimation is heuristic;
- pixel-motion based speed estimation is camera-dependent;
- the SUMO intersection is a simplified single-intersection model;
- the results apply to the tested network and demand rather than all real intersections;
- real deployment would require safety-certified signal hardware, robust tracking, calibrated cameras, and fail-safe control logic.

---

# 🚀 Future research directions

Natural next steps include:

### Better perception
- stronger multi-object tracking;
- camera calibration and perspective normalization;
- more robust occlusion handling;
- real-time live camera feeds.

### Better traffic understanding
- per-vehicle waiting time;
- turn-movement classification;
- lane-level demand;
- arrival-rate prediction.

### Better control
- model-predictive signal control;
- learned demand forecasting;
- multi-intersection coordination;
- dedicated turn lanes with protected-turn phases.

### Better validation
- larger route networks;
- more traffic scenarios;
- peak/off-peak demand profiles;
- repeated stochastic experiments;
- comparisons against additional signal-control baselines.

---

# 📌 Project conclusion

Trafico demonstrates a complete adaptive traffic-control pipeline:

```text
Perception
   ↓
Tracking
   ↓
Queue estimation
   ↓
Explainable adaptive control
   ↓
Simulation validation
```

The final experiments show that Adaptive v4.1 can improve **throughput and waiting-time performance** relative to the fixed-time baseline under the tested SUMO scenarios, while also revealing a meaningful **travel-time trade-off**.

That makes the project more than a vehicle-counting demo: it is a complete prototype connecting **computer vision → control decisions → quantitative traffic simulation**.
