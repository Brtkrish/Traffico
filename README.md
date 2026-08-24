# 🚥 4-Way Adaptive Signal Controller

A Streamlit dashboard that watches four traffic-camera video feeds, uses YOLOv8 to count vehicles inside a per-lane region of interest (ROI), and drives a live green/yellow/all-red state machine — automatically prioritizing the busiest or longest-waiting lane, or letting an operator lock a lane manually.


---

## ✨ Features

- **Real-time vehicle detection** — YOLOv8n (`ultralytics`) detects cars, motorcycles, buses, and trucks per frame.
- **Polygon ROI counting** — each lane has its own quadrilateral zone; only vehicles whose center falls inside it are counted, so vehicles on other roads or in the distance don't skew the signal.
- **Adaptive state machine** — `GREEN → YELLOW → ALL_RED → GREEN …` per lane, with:
  - Green time scaled to vehicle count (`base + count × seconds/vehicle`, capped at a max).
  - A **starvation score** (`vehicle count + wait_time × 0.5`) so a quiet lane that's been waiting a long time still gets picked over a marginally busier one.
  - A **minimum green lock** so the AI can't cut a phase unrealistically short.
- **Manual override mode** — pin the green light to a specific lane for operator control or testing.
- **Live dashboard**
  - Control-tower header bar (clock, uptime, update rate, active lane, run/pause badge).
  - Animated top-down intersection diagram with per-lane signal lights and live counts.
  - Countdown ring for the current phase.
  - Per-lane status cards and annotated video feeds with ROI overlays.
  - **Analytics tab** — rolling vehicle-count history chart + summary metrics.
  - **Logs tab** — timestamped record of every lane switch.
- **Live-tunable controls** — confidence threshold, timing parameters, and frame-skip rate all take effect immediately (no restart needed) via `st.fragment`.
- **Start/Stop control** — the app doesn't touch your video files or GPU until you flip it on.

---

## 🧰 Tech Stack

| Component        | Purpose                                  |
|-------------------|-------------------------------------------|
| Streamlit         | Web dashboard / UI                        |
| Ultralytics YOLOv8 | Vehicle detection (`yolov8n.pt`)          |
| OpenCV (`cv2`)    | Video I/O, ROI point-in-polygon test, drawing overlays |
| NumPy             | Polygon coordinate arrays                 |
| Pandas            | Analytics history → chart data            |

---

## 📦 Prerequisites

- Python 3.9+
- A GPU is optional but recommended for smoother real-time inference (CPU works fine at a lower frame rate / with a higher "process every Nth frame" setting).
- Four video files representing each approach to the junction.

Install dependencies:

```bash
pip install streamlit opencv-python numpy pandas ultralytics
```

> The first run will auto-download `yolov8n.pt` (~6 MB) via Ultralytics.

---

## 🎥 Video Setup

Place four video files in the **same folder as the script**, named:

```
north.mp4
south.mp4
east.mp4
west.mp4
```

Each video should be a fixed camera angle looking at one approach of the intersection. Videos loop automatically when they reach the end.

### ROI Calibration

The four lane detection zones are hardcoded as pixel-coordinate polygons at the top of the script:

```python
roi_polygons = {
    "North": np.array([[643, 390], [460, 694], [870, 705], [693, 394]], np.int32),
    "South": np.array([[236, 714], [895, 237], [1066, 237], [1270, 706]], np.int32),
    "East":  np.array([[251, 711], [758, 382], [1097, 364], [1171, 710]], np.int32),
    "West":  np.array([[44, 493], [463, 231], [720, 229], [951, 520]], np.int32),
}
```

⚠️ **These coordinates are tuned for a specific frame resolution.** If your videos are a different resolution (or you swap footage), you'll need to re-draw these polygons.

#### Recalibrating with `coordinate_finder.py`

The project includes a small helper script, `coordinate_finder.py`, for this exact purpose — it opens a video frame and lets you click four points to trace a lane's ROI, printing the pixel coordinates to the terminal as you go.

1. Open `coordinate_finder.py` and point it at the lane you're calibrating:
   ```python
   cap = cv2.VideoCapture("west.mp4")  # swap for north.mp4 / south.mp4 / east.mp4
   ```
2. Run it:
   ```bash
   python coordinate_finder.py
   ```
3. A window titled **"Click 4 points of your lane"** opens on the first frame. Click the **four corners of that lane's detection zone**, in order around the perimeter (don't skip across the shape — go corner to corner, either clockwise or counter-clockwise) so the resulting polygon is valid and non-self-intersecting.
4. Each click prints a `[x, y]` pair to the terminal and drops a green dot on the frame so you can see what you've traced. After the 4th click, press any key to close the window.
5. Copy the four printed pairs into the corresponding lane entry in `roi_polygons` inside `traffic_controller.py`:
   ```python
   "West": np.array([[44, 493], [463, 231], [720, 229], [951, 520]], np.int32),
   ```
6. Repeat steps 1–5 for the other three lanes (`north.mp4`, `south.mp4`, `east.mp4`).

> Tip: calibrate against a frame where traffic is light, so you can clearly see the lane markings/edges you're tracing without vehicles in the way.

---

## ▶️ Running the App

```bash
streamlit run traffic_controller.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

1. In the sidebar, choose **Automated Adaptive (AI)** or **Manual Lane Override**.
2. Adjust confidence threshold / timing parameters / frame-skip as needed — changes apply live.
3. Flip **▶️ Run Junction** on to start processing.
4. Watch the **Live View** tab for real-time state, or check **Analytics** / **Logs** for history.

---

## ⚙️ Configuration Reference

| Control                     | Default | Description |
|------------------------------|---------|-------------|
| Model Confidence Threshold   | 0.40    | Minimum YOLO detection confidence to count a vehicle |
| Base Green Time              | 8.0 s   | Green duration with zero detected vehicles |
| Extra Seconds / Vehicle      | 1.5 s   | Added per vehicle detected in the ROI at switch-in |
| Max Green Time               | 35.0 s  | Hard cap on green duration |
| Yellow Time                  | 3.0 s   | Fixed clearance phase |
| All-Red Buffer               | 2.0 s   | Fixed all-directions-red safety phase |
| Minimum Green Lock           | 5.0 s   | Floor below which the AI cannot cut a green phase short |
| Process every Nth frame      | 1       | Run inference every frame (1) or skip frames to save compute |

---

## 🧠 How Lane Selection Works (Automated Mode)

At the end of each green phase, every *other* lane gets a score:

```
score(lane) = current_vehicle_count(lane) + 0.5 × seconds_waited(lane)
```

The lane with the highest score becomes the next green. This balances two goals:

- **Throughput** — busier lanes get priority.
- **Fairness** — a lane that's been sitting red for a long time will eventually out-score a busier one, preventing starvation.

---

## 📁 Project Structure

```
.
├── traffic_controller.py   # Main Streamlit app
├── coordinate_finder.py     # Helper: click 4 points on a frame to get ROI polygon coords
├── north.mp4                # Video feed — North approach
├── south.mp4                # Video feed — South approach
├── east.mp4                 # Video feed — East approach
├── west.mp4                 # Video feed — West approach
├── yolov8n.pt                # YOLOv8 weights (auto-downloaded on first run if missing)
└── README.md
```

---

## 🛠️ Troubleshooting

| Issue | Likely Cause / Fix |
|-------|---------------------|
| "Could not open these video files" error on load | Video files aren't in the same folder as the script, or are named incorrectly |
| Vehicles not being counted | ROI polygon doesn't match your video's resolution/framing — recalibrate coordinates |
| Sliders/mode changes don't seem to do anything | Make sure **Run Junction** is toggled on — some effects only appear on the next detection/state tick |
| Low frame rate / laggy UI | Increase **Process every Nth frame**, lower Streamlit's fragment refresh isn't configurable directly but reducing detection frequency helps most |
| SVG/diagram shows as raw text instead of rendering | Fixed in current version — HTML/SVG strings are stripped of leading whitespace before being passed to `st.markdown` (a Markdown parser quirk where indented lines are treated as code blocks) |

---

## ⚠️ Notes & Limitations

- This is a research/demo tool, not a certified traffic-control system — timing logic and detection are illustrative, not safety-rated.
- Detection quality depends entirely on camera angle, lighting, and ROI accuracy.
- `yolov8n` is the smallest/fastest YOLOv8 variant; swap in `yolov8s.pt` or larger for better accuracy at the cost of speed.

---