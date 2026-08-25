import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1. Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic 4-Way Traffic Controller",
    page_icon="🚥",
    layout="wide",
)

LANES = ["North", "South", "East", "West"]
PHASES = ["NS", "EW"]
PHASE_LANES = {
    "NS": ("North", "South"),
    "EW": ("East", "West"),
}


def _html(raw: str) -> str:
    """Strip per-line leading whitespace so Streamlit's Markdown renderer
    never mistakes indented HTML/SVG for a fenced code block."""
    return "\n".join(line.strip() for line in raw.strip().splitlines())

# ---------------------------------------------------------------------------
# 2. Theme: full dark mode + fonts + reusable components (rings, header, etc)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0e1117;
        --panel: #161b22;
        --panel-2: #1c2129;
        --border: #262c36;
        --text: #e6edf3;
        --muted: #8b949e;
        --green: #2ea043;
        --yellow: #d29922;
        --red: #f85149;
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; color: var(--text); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    .stMetric, .stJson, code { font-family: 'JetBrains Mono', monospace !important; }

    [data-testid="stAppViewContainer"] { background: var(--bg); }
    [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    /* ---- Top control-tower header bar ---- */
    .hdr-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, var(--panel), var(--panel-2));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 22px;
        margin-bottom: 18px;
    }
    .hdr-title { font-size: 1.15rem; font-weight: 700; display:flex; align-items:center; gap:10px; }
    .hdr-stats { display: flex; gap: 26px; font-family: 'JetBrains Mono', monospace; }
    .hdr-stat { text-align: center; }
    .hdr-stat .v { font-size: 1.0rem; font-weight: 600; }
    .hdr-stat .l { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-run { background: rgba(46,160,67,0.18); color: var(--green); border: 1px solid var(--green); }
    .badge-stop { background: rgba(248,81,73,0.18); color: var(--red); border: 1px solid var(--red); }

    /* ---- Lane status cards ---- */
    .lane-card { border-radius: 14px; padding: 16px 18px; margin-bottom: 8px; color: #fff; border: 1px solid var(--border); }
    .lane-card h4 { margin: 0 0 6px 0; font-size: 0.9rem; font-weight: 600; opacity: 0.9; }
    .lane-card .big { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; line-height: 1.1; }
    .lane-card .sub { font-size: 0.78rem; opacity: 0.85; margin-top: 4px; }
    .lane-green  { background: linear-gradient(135deg,#1b7a3d,#0f5a2a); }
    .lane-yellow { background: linear-gradient(135deg,#c99a12,#a67c0a); }
    .lane-red    { background: linear-gradient(135deg,#8a2020,#5c1414); }

    /* ---- Countdown ring ---- */
    .ring-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .ring-caption { margin-top:8px; font-size:0.8rem; color: var(--muted); text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("Multi-stream YOLOv8 junction control — YOLO counting + adaptive NS/EW phase control, v4.1-style stability and starvation protection.")

# ---------------------------------------------------------------------------
# 3. Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Control & Selection Settings")

run_toggle = st.sidebar.toggle("▶️ Run Junction", value=False,
                                help="Start/stop processing. Streams stay paused until you turn this on.")

control_mode = st.sidebar.radio("Traffic Signal Mode", ["Automated Adaptive (AI)", "Manual Lane Override"])

manual_selected_phase = "NS"
if control_mode == "Manual Lane Override":
    manual_selected_phase = st.sidebar.selectbox(
        "Select Active Green Phase",
        PHASES,
        format_func=lambda p: "North + South (NS)" if p == "NS" else "East + West (EW)",
    )

conf_threshold = st.sidebar.slider("Model Confidence Threshold", 0.1, 1.0, 0.40, 0.05)

with st.sidebar.expander("Timing Parameters"):
    BASE_GREEN_TIME = st.slider("Base Green Time (s)", 4.0, 20.0, 8.0, 0.5)
    SEC_PER_VEHICLE = st.slider("Extra Seconds / Vehicle", 0.5, 4.0, 1.5, 0.1)
    MAX_GREEN_TIME = st.slider("Max Green Time (s)", 15.0, 60.0, 35.0, 1.0)
    YELLOW_TIME = st.slider("Yellow Time (s)", 1.0, 6.0, 3.0, 0.5)
    ALL_RED_TIME = st.slider("All-Red Buffer (s)", 0.5, 5.0, 2.0, 0.5)
    MIN_GREEN_TIME = st.slider("Minimum Green Lock (s)", 2.0, 15.0, 5.0, 0.5,
                                help="Prevents the AI from cutting a green phase unrealistically short.")

frame_skip = st.sidebar.slider("Process every Nth frame", 1, 5, 1,
                                help="Higher = lighter on GPU/CPU, less smooth detection.")

with st.sidebar.expander("Adaptive Decision Smoothing"):
    LOW_DEMAND_CONFIRMATIONS = st.slider(
        "Low-demand confirmations",
        1, 6, 3, 1,
        help="Consecutive control checks required before a low-demand early switch."
    )
    MIN_PHASE_HOLD_AFTER_SWITCH = st.slider(
        "Minimum phase hold after switch (s)",
        0.0, 15.0, 5.0, 0.5,
        help="Prevents an immediate reversal after a phase change."
    )

if st.sidebar.button("🔄 Reset State"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

roi_polygons = {
    "North": np.array([[643, 390], [460, 694], [870, 705], [693, 394]], np.int32),
    "South": np.array([[236, 714], [895, 237], [1066, 237], [1270, 706]], np.int32),
    "East": np.array([[251, 711], [758, 382], [1097, 364], [1171, 710]], np.int32),
    "West": np.array([[44, 493], [463, 231], [720, 229], [951, 520]], np.int32),
}

video_files = {"North": "north.mp4", "South": "south.mp4", "East": "east.mp4", "West": "west.mp4"}

# Lightweight centroid tracking/queue estimation.
TRACK_MAX_DISTANCE_PX = 90
TRACK_MAX_MISSED = 8

# A vehicle is considered stopped only after several consecutive low-motion
# observations. This is intentionally conservative for perspective-heavy
# stock traffic footage.
STOP_SPEED_PX_PER_SEC = 18.0
STOP_CONFIRMATIONS = 4
MOVING_CONFIRMATIONS = 2
MIN_TRACK_AGE = 3

TRACKING_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------------------------------------------------------------------------
# 4. Persistent State
# ---------------------------------------------------------------------------
def init_state():
    st.session_state.caps = {}
    missing = []
    for direction, path in video_files.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            missing.append(path)
        st.session_state.caps[direction] = cap
    st.session_state.missing_files = missing

    st.session_state.active_phase = "NS"
    st.session_state.target_next_phase = "NS"
    st.session_state.current_phase = "GREEN"
    st.session_state.allocated_green_duration = BASE_GREEN_TIME
    st.session_state.phase_start_time = time.time()
    st.session_state.red_wait_times = {p: 0.0 for p in PHASES}
    st.session_state.current_start_queue = 0
    st.session_state.previous_queue = 0
    st.session_state.reassess_elapsed = 0.0
    st.session_state.current_phase_reason = "INITIAL"
    st.session_state.low_demand_confirmations = {p: 0 for p in PHASES}
    st.session_state.phase_hold_elapsed = 0.0
    st.session_state.last_tick_time = time.time()
    st.session_state.frame_counter = 0
    st.session_state.last_frames = {}
    st.session_state.last_counts = {l: 0 for l in LANES}

    # Per-camera lightweight centroid trackers.
    st.session_state.next_track_id = {l: 1 for l in LANES}
    st.session_state.tracks = {l: {} for l in LANES}
    st.session_state.track_metrics = {
        l: {"vehicles": 0, "stopped": 0, "avg_speed_px_s": 0.0, "queue": 0}
        for l in LANES
    }
    st.session_state.video_fps = {l: 30.0 for l in LANES}
    st.session_state.last_processed_time = {l: None for l in LANES}

    st.session_state.app_start_time = time.time()
    st.session_state.history = []       # list of dicts: {t, North, South, East, West}
    st.session_state.event_log = []     # list of strings, newest first
    st.session_state.fps_ema = 0.0


if "caps" not in st.session_state:
    init_state()

if st.session_state.get("missing_files"):
    st.error("Could not open these video files (check they exist next to the script): "
              + ", ".join(st.session_state.missing_files))
    st.stop()

header_slot = st.empty()

# ---------------------------------------------------------------------------
# 5. Tabs
# ---------------------------------------------------------------------------
tab_live, tab_analytics, tab_logs = st.tabs(["🎥 Live View", "📊 Analytics", "🧾 Logs"])

with tab_live:
    top_col1, top_col2 = st.columns([1, 2])
    ring_slot = top_col1.empty()
    diagram_slot = top_col2.empty()

    st.markdown("#### Junction Signal Status")
    sig_cols = st.columns(4)
    sig_slots = {name: col.empty() for name, col in zip(LANES, sig_cols)}

    st.markdown("---")
    g_col1, g_col2 = st.columns(2)
    view_slots = {"North": g_col1.empty(), "South": g_col2.empty(), "East": g_col1.empty(), "West": g_col2.empty()}

with tab_analytics:
    st.markdown("#### Vehicle Count History (per lane, ROI ingest)")
    chart_slot = st.empty()
    st.markdown("#### Snapshot Stats")
    metric_slot = st.empty()

with tab_logs:
    st.markdown("#### Event Log")
    st.caption("Most recent lane switches and phase changes appear at the top.")
    log_slot = st.empty()

# ---------------------------------------------------------------------------
# 6. Reusable render helpers
# ---------------------------------------------------------------------------
def make_ring(pct, color, big_text, caption):
    pct = max(0.0, min(100.0, pct))
    return _html(f"""
    <div class="ring-wrap">
      <div style="position:relative;width:130px;height:130px;border-radius:50%;
                  background:conic-gradient({color} {pct}%, #262c36 {pct}% 100%);
                  display:flex;align-items:center;justify-content:center;">
        <div style="width:98px;height:98px;border-radius:50%;background:#0e1117;
                    display:flex;align-items:center;justify-content:center;">
          <span style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:#e6edf3;">{big_text}</span>
        </div>
      </div>
      <div class="ring-caption">{caption}</div>
    </div>
    """)


def make_intersection_svg(active_phase, phase, lane_counts):
    def color_for(lane):
        if lane in PHASE_LANES[active_phase]:
            return {"GREEN": "#2ea043", "YELLOW": "#d29922", "ALL_RED": "#f85149"}[phase]
        return "#f85149"

    positions = {"North": (160, 40), "South": (160, 280), "East": (280, 160), "West": (40, 160)}
    blink = 'begin="0s"'
    steady = 'begin="indefinite"'
    lights_parts = []

    for lane, (x, y) in positions.items():
        c = color_for(lane)
        cnt = lane_counts.get(lane, 0)
        is_active = lane in PHASE_LANES[active_phase]
        anim_begin = blink if (is_active and phase != "GREEN") else steady
        label_y = y - 26 if y < 160 else y + 40

        lights_parts.append(_html(f"""
            <circle cx="{x}" cy="{y}" r="16" fill="{c}" stroke="#0e1117" stroke-width="3">
              <animate attributeName="opacity" values="1;0.55;1" dur="1.4s" repeatCount="indefinite" {anim_begin} />
            </circle>
            <text x="{x}" y="{y+4}" text-anchor="middle" font-size="12" font-weight="700"
                  font-family="JetBrains Mono, monospace" fill="#0e1117">{cnt}</text>
            <text x="{x}" y="{label_y}" text-anchor="middle" font-size="13"
                  font-family="Space Grotesk, sans-serif" fill="#8b949e">{lane}</text>
        """))

    lights = " ".join(lights_parts)

    return _html(f"""
    <svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:320px;display:block;margin:0 auto;">
      <rect x="0" y="0" width="320" height="320" fill="#0e1117" rx="14"/>
      <rect x="0" y="130" width="320" height="60" fill="#2a2f38"/>
      <rect x="130" y="0" width="60" height="320" fill="#2a2f38"/>
      <line x1="0" y1="160" x2="320" y2="160" stroke="#4b5261" stroke-width="2" stroke-dasharray="8,8"/>
      <line x1="160" y1="0" x2="160" y2="320" stroke="#4b5261" stroke-width="2" stroke-dasharray="8,8"/>
      {lights}
    </svg>
    """)


# ---------------------------------------------------------------------------
# 6b. Lightweight per-camera vehicle tracking
# ---------------------------------------------------------------------------

def update_centroid_tracks(direction, detections, now_ts):
    """
    Lightweight centroid tracker with hysteresis for STOP/MOVE classification.

    A track must have multiple consecutive low-speed observations before it is
    considered stopped. This prevents a single noisy YOLO frame from turning a
    moving vehicle into a queue member.
    """
    tracks = st.session_state.tracks[direction]
    next_id = st.session_state.next_track_id[direction]

    previous_ts = st.session_state.last_processed_time[direction]
    fps = max(st.session_state.video_fps.get(direction, 30.0), 1.0)

    # Use real wall-clock spacing when available, but keep a sensible lower bound.
    dt = 1.0 / fps
    if previous_ts is not None:
        dt = max(0.03, now_ts - previous_ts)

    unmatched_track_ids = set(tracks.keys())

    # Greedy nearest-neighbor matching.
    for det in detections:
        best_id = None
        best_dist = float("inf")

        for tid in unmatched_track_ids:
            tr = tracks[tid]
            dx = det["cx"] - tr["cx"]
            dy = det["cy"] - tr["cy"]
            dist = (dx * dx + dy * dy) ** 0.5

            # Use a slightly wider tolerance for already-established tracks.
            allowed_dist = TRACK_MAX_DISTANCE_PX + min(tr["age"] * 4.0, 40.0)

            if dist < best_dist and dist <= allowed_dist:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = next_id
            next_id += 1

            tracks[best_id] = {
                "cx": det["cx"],
                "cy": det["cy"],
                "speed": 0.0,
                "age": 1,
                "missed": 0,
                "stopped": False,
                "low_speed_count": 0,
                "moving_count": 0,
                "last_seen": now_ts,
            }
        else:
            tr = tracks[best_id]
            dx = det["cx"] - tr["cx"]
            dy = det["cy"] - tr["cy"]
            pixel_speed = ((dx * dx + dy * dy) ** 0.5) / dt

            # Exponential smoothing.
            tr["speed"] = 0.70 * tr["speed"] + 0.30 * pixel_speed
            tr["age"] += 1
            tr["missed"] = 0
            tr["last_seen"] = now_ts
            tr["cx"] = det["cx"]
            tr["cy"] = det["cy"]

            if tr["age"] >= MIN_TRACK_AGE:
                if tr["speed"] <= STOP_SPEED_PX_PER_SEC:
                    tr["low_speed_count"] += 1
                    tr["moving_count"] = 0

                    if tr["low_speed_count"] >= STOP_CONFIRMATIONS:
                        tr["stopped"] = True
                else:
                    tr["moving_count"] += 1
                    tr["low_speed_count"] = 0

                    if tr["moving_count"] >= MOVING_CONFIRMATIONS:
                        tr["stopped"] = False

        unmatched_track_ids.discard(best_id)

        tr = tracks[best_id]
        det["track_id"] = best_id
        det["speed_px_s"] = tr["speed"]
        det["stopped"] = bool(tr["stopped"])
        det["track_age"] = tr["age"]

    # Age unmatched tracks. Keep them briefly in case YOLO misses a frame.
    for tid in list(unmatched_track_ids):
        tracks[tid]["missed"] += 1
        if tracks[tid]["missed"] > TRACK_MAX_MISSED:
            del tracks[tid]

    active = [
        tr for tr in tracks.values()
        if tr["missed"] == 0
    ]

    # Only mature, visible tracks count toward the queue estimate.
    mature = [
        tr for tr in active
        if tr["age"] >= MIN_TRACK_AGE
    ]

    stopped = [
        tr for tr in mature
        if tr["stopped"]
    ]

    avg_speed = (
        sum(tr["speed"] for tr in mature) / len(mature)
        if mature else 0.0
    )

    # Important: queue estimate is now conservative.
    # It cannot exceed the number of mature active tracks.
    queue_estimate = min(len(stopped), len(mature))

    st.session_state.track_metrics[direction] = {
        "vehicles": len(active),
        "mature_vehicles": len(mature),
        "stopped": len(stopped),
        "avg_speed_px_s": avg_speed,
        "queue": queue_estimate,
    }

    st.session_state.last_processed_time[direction] = now_ts

    return detections


def phase_tracking_metrics():
    """Return NS/EW tracking metrics for adaptive decisions."""
    ns = {
        "vehicles": (
            st.session_state.track_metrics["North"]["vehicles"]
            + st.session_state.track_metrics["South"]["vehicles"]
        ),
        "stopped": (
            st.session_state.track_metrics["North"]["stopped"]
            + st.session_state.track_metrics["South"]["stopped"]
        ),
        "queue": (
            st.session_state.track_metrics["North"]["queue"]
            + st.session_state.track_metrics["South"]["queue"]
        ),
    }

    ew = {
        "vehicles": (
            st.session_state.track_metrics["East"]["vehicles"]
            + st.session_state.track_metrics["West"]["vehicles"]
        ),
        "stopped": (
            st.session_state.track_metrics["East"]["stopped"]
            + st.session_state.track_metrics["West"]["stopped"]
        ),
        "queue": (
            st.session_state.track_metrics["East"]["queue"]
            + st.session_state.track_metrics["West"]["queue"]
        ),
    }

    return ns, ew


# ---------------------------------------------------------------------------
# 7. Main processing fragment
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.15)
def run_junction():
    now = time.time()
    dt = now - st.session_state.last_tick_time
    st.session_state.last_tick_time = now

    if dt > 0:
        inst_fps = 1.0 / dt
        st.session_state.fps_ema = (0.85 * st.session_state.fps_ema) + (0.15 * inst_fps) if st.session_state.fps_ema else inst_fps

    uptime = int(now - st.session_state.app_start_time)
    uptime_str = f"{uptime // 60:02d}:{uptime % 60:02d}"
    clock_str = time.strftime("%H:%M:%S")
    run_badge = '<span class="badge badge-run">● RUNNING</span>' if run_toggle else '<span class="badge badge-stop">● PAUSED</span>'

    header_slot.markdown(
        _html(f"""
        <div class="hdr-bar">
          <div class="hdr-title">🚥 4-Way Adaptive Signal Controller &nbsp; {run_badge}</div>
          <div class="hdr-stats">
            <div class="hdr-stat"><div class="v">{clock_str}</div><div class="l">Local Time</div></div>
            <div class="hdr-stat"><div class="v">{uptime_str}</div><div class="l">Uptime</div></div>
            <div class="hdr-stat"><div class="v">{st.session_state.fps_ema:.1f}</div><div class="l">Update Hz</div></div>
            <div class="hdr-stat"><div class="v">{st.session_state.active_phase}</div><div class="l">Active Phase</div></div>
          </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    if not run_toggle:
        with tab_live:
            ring_slot.info("Paused")
        return

    phase_elapsed = now - st.session_state.phase_start_time
    st.session_state.frame_counter += 1
    do_inference = (st.session_state.frame_counter % frame_skip == 0)

    lane_counts = dict(st.session_state.last_counts)
    frames = dict(st.session_state.last_frames)

    if do_inference:
        for direction, cap in st.session_state.caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if frame is None:
                continue

            roi = roi_polygons[direction]
            results = model(frame, classes=[2, 3, 5, 7], conf=conf_threshold, verbose=False)

            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
                confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                        detections.append({
                            "cx": cx,
                            "cy": cy,
                            "bbox": (x1, y1, x2, y2),
                            "cls": int(classes[i]) if len(classes) > i else -1,
                            "conf": float(confs[i]) if len(confs) > i else 0.0,
                        })

            tracked = update_centroid_tracks(direction, detections, time.time())

            annotated = frame.copy()
            cv2.polylines(
                annotated,
                [roi],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )

            for det in tracked:
                x1, y1, x2, y2 = det["bbox"]
                color = (0, 255, 0) if not det["stopped"] else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                status = "STOP" if det["stopped"] else "MOVE"
                label = (
                    f"ID {det['track_id']} {status} "
                    f"{det['speed_px_s']:.0f}px/s"
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            lane_counts[direction] = len(tracked)
            frames[direction] = annotated

        st.session_state.last_counts = lane_counts
        st.session_state.last_frames = frames

        st.session_state.history.append({"t": clock_str, **lane_counts})
        if len(st.session_state.history) > 120:
            st.session_state.history.pop(0)

    # v4.1-style phase wait accounting with conservative tracking-derived queues:
    # the phase that is red accrues waiting pressure continuously.
    for p in PHASES:
        if p != st.session_state.active_phase:
            st.session_state.red_wait_times[p] += dt

    st.session_state.reassess_elapsed += dt
    st.session_state.phase_hold_elapsed += dt

    # ---- v4.1 phase state machine ----
    phase = st.session_state.current_phase
    if phase == "GREEN":
        elapsed_ok = (
            phase_elapsed >= MIN_GREEN_TIME
            and st.session_state.reassess_elapsed >= 5.0
        )

        if elapsed_ok:
            ns_phase, ew_phase = phase_tracking_metrics()
            ns_count = ns_phase["vehicles"]
            ew_count = ew_phase["vehicles"]

            ns_queue = ns_phase["queue"]
            ew_queue = ew_phase["queue"]

            # Tracking-aware score:
            # active vehicles + stopped vehicles (queue pressure) + red wait.
            ns_score = (
                ns_count * 0.35
                + ns_phase["stopped"] * 1.75
                + st.session_state.red_wait_times["NS"] * 0.55
            )
            ew_score = (
                ew_count * 0.35
                + ew_phase["stopped"] * 1.75
                + st.session_state.red_wait_times["EW"] * 0.55
            )

            current_phase = st.session_state.active_phase
            current_phase_data = ns_phase if current_phase == "NS" else ew_phase
            opposite_phase = "EW" if current_phase == "NS" else "NS"
            opposite_phase_data = ew_phase if current_phase == "NS" else ns_phase

            current_count = current_phase_data["vehicles"]
            opposite_count = opposite_phase_data["vehicles"]
            current_queue = current_phase_data["queue"]
            opposite_queue = opposite_phase_data["queue"]

            start_queue = st.session_state.current_start_queue
            previous_queue = st.session_state.previous_queue
            queue_reduction = start_queue - current_queue if start_queue > 0 else 0
            current_score = ns_score if current_phase == "NS" else ew_score
            opposite_score = ew_score if current_phase == "NS" else ns_score

            switch = False
            reason = "HOLD"

            # Smooth noisy YOLO counts:
            # one low-demand observation is not enough to trigger a switch.
            current_low = current_count <= 2
            if current_low:
                st.session_state.low_demand_confirmations[current_phase] += 1
            else:
                st.session_state.low_demand_confirmations[current_phase] = 0

            # Reset the opposite confirmation when it clearly has traffic.
            if opposite_count > 2:
                st.session_state.low_demand_confirmations[opposite_phase] = 0

            # Hard starvation protection.
            if (
                st.session_state.red_wait_times[opposite_phase] >= 60.0
                and phase_elapsed >= MIN_GREEN_TIME
            ):
                switch = True
                reason = "STARVATION"

            # Newly selected phases get a short stability lock.
            elif st.session_state.phase_hold_elapsed < MIN_PHASE_HOLD_AFTER_SWITCH:
                reason = "POST_SWITCH_LOCK"

            # Respect maximum green.
            elif phase_elapsed >= MAX_GREEN_TIME:
                switch = True
                reason = "MAX_GREEN"

            # Continue if the current phase is demonstrably clearing its
            # detected queue and the opposite phase is not substantially better.
            elif (
                start_queue > 0
                and queue_reduction / max(start_queue, 1) >= 0.15
                and current_queue > 0
                and opposite_score <= current_score + 9.0
            ):
                reason = "CURRENT_CLEARING"

            # New rule: low demand must persist across multiple checks.
            elif (
                st.session_state.low_demand_confirmations[current_phase]
                >= LOW_DEMAND_CONFIRMATIONS
                and opposite_queue > 0
            ):
                switch = True
                reason = "CONFIRMED_LOW_DEMAND"

            # Otherwise require a meaningful opposing advantage.
            elif (
                opposite_count > 0
                and opposite_score > current_score + 9.0
            ):
                switch = True
                reason = "PERSISTENT_OPPOSING_DEMAND"

            if switch:
                st.session_state.target_next_phase = opposite_phase
                st.session_state.current_phase_reason = reason
                st.session_state.current_phase_start_queue = start_queue

                st.session_state.current_phase = "YELLOW"
                st.session_state.phase_start_time = now
                st.session_state.reassess_elapsed = 0.0

            else:
                # Refresh the green target from the latest observed demand.
                current_v_count = current_count
                wait_bonus = st.session_state.red_wait_times[current_phase]
                refreshed = min(
                    MAX_GREEN_TIME,
                    max(
                        MIN_GREEN_TIME,
                        BASE_GREEN_TIME
                        + current_v_count * SEC_PER_VEHICLE
                        + wait_bonus * 0.02,
                    ),
                )
                st.session_state.allocated_green_duration = max(
                    st.session_state.allocated_green_duration,
                    phase_elapsed + refreshed,
                )
                st.session_state.allocated_green_duration = min(
                    st.session_state.allocated_green_duration,
                    MAX_GREEN_TIME,
                )
                st.session_state.previous_queue = current_count
                st.session_state.reassess_elapsed = 0.0

    elif phase == "YELLOW":
        if phase_elapsed >= YELLOW_TIME:
            st.session_state.current_phase = "ALL_RED"
            st.session_state.phase_start_time = now

    elif phase == "ALL_RED":
        if phase_elapsed >= ALL_RED_TIME:
            prev_phase = st.session_state.active_phase
            st.session_state.active_phase = st.session_state.target_next_phase
            st.session_state.red_wait_times[st.session_state.active_phase] = 0.0

            # Reset smoothing state after a real phase transition.
            st.session_state.phase_hold_elapsed = 0.0
            st.session_state.low_demand_confirmations = {p: 0 for p in PHASES}

            v_count = (
                lane_counts["North"] + lane_counts["South"]
                if st.session_state.active_phase == "NS"
                else lane_counts["East"] + lane_counts["West"]
            )

            st.session_state.allocated_green_duration = min(
                BASE_GREEN_TIME + (v_count * SEC_PER_VEHICLE),
                MAX_GREEN_TIME,
            )

            st.session_state.current_phase = "GREEN"
            st.session_state.phase_start_time = now
            # Start queue is now based on tracked stopped vehicles, not raw
            # detections alone.
            _, ew_phase_now = phase_tracking_metrics()
            ns_phase_now, _ = phase_tracking_metrics()
            current_phase_data = ns_phase_now if st.session_state.active_phase == "NS" else ew_phase_now
            st.session_state.current_start_queue = current_phase_data["queue"]
            st.session_state.previous_queue = current_phase_data["queue"]
            st.session_state.reassess_elapsed = 0.0

            reason = st.session_state.get("current_phase_reason", "SWITCH")

            st.session_state.event_log.insert(
                0,
                f"{clock_str} — {prev_phase} → {st.session_state.active_phase} "
                f"(green {int(st.session_state.allocated_green_duration)}s, "
                f"NS {lane_counts['North'] + lane_counts['South']}, "
                f"EW {lane_counts['East'] + lane_counts['West']}, "
                f"reason {reason})"
            )

            if len(st.session_state.event_log) > 50:
                st.session_state.event_log.pop()

    # Keep the initial phase queue proxy populated.
    if st.session_state.current_start_queue == 0:
        ns_phase_init, ew_phase_init = phase_tracking_metrics()
        init_data = ns_phase_init if st.session_state.active_phase == "NS" else ew_phase_init
        st.session_state.current_start_queue = init_data["queue"]
        st.session_state.previous_queue = init_data["queue"]

    # ---- Render ----
    phase = st.session_state.current_phase
    phase_elapsed = now - st.session_state.phase_start_time
    target_dur = {"GREEN": st.session_state.allocated_green_duration, "YELLOW": YELLOW_TIME, "ALL_RED": ALL_RED_TIME}[phase]
    t_rem = max(0.0, target_dur - phase_elapsed)
    pct_remaining = (t_rem / target_dur * 100) if target_dur > 0 else 0
    ring_color = {"GREEN": "#2ea043", "YELLOW": "#d29922", "ALL_RED": "#f85149"}[phase]

    with tab_live:
        ring_slot.markdown(
            make_ring(
                pct_remaining,
                ring_color,
                f"{int(t_rem)}s",
                f"{st.session_state.active_phase} — {phase} | "
                f"{st.session_state.get('current_phase_reason', 'HOLD')}"
            ),
            unsafe_allow_html=True,
        )
        diagram_slot.markdown(make_intersection_svg(st.session_state.active_phase, phase, lane_counts), unsafe_allow_html=True)

        tracking_cols = st.columns(4)
        for idx, direction in enumerate(LANES):
            info = st.session_state.track_metrics.get(direction, {})
            tracking_cols[idx].metric(
                f"{direction} Queue",
                int(info.get("queue", 0)),
                help="Vehicles repeatedly confirmed as stopped inside this approach ROI.",
            )
            tracking_cols[idx].caption(
                f"Tracked {int(info.get('vehicles', 0))} • "
                f"Mature {int(info.get('mature_vehicles', 0))} • "
                f"Stopped {int(info.get('stopped', 0))}"
            )

        for direction, slot in sig_slots.items():
            count = lane_counts.get(direction, 0)
            track_info = st.session_state.track_metrics.get(
                direction,
                {
                    "stopped": 0,
                    "queue": 0,
                    "mature_vehicles": 0,
                    "avg_speed_px_s": 0.0,
                },
            )
            phase_for_direction = "NS" if direction in ("North", "South") else "EW"
            wait_s = int(st.session_state.red_wait_times[phase_for_direction])
            active = phase_for_direction == st.session_state.active_phase

            if active and phase == "GREEN":
                css, label, sub = (
                    "lane-green",
                    f"🟢 ACTIVE — {count} vehicles",
                    f"{int(t_rem)}s left of {int(st.session_state.allocated_green_duration)}s"
                    f" • queue {track_info['queue']} • stopped {track_info['stopped']}",
                )
            elif active and phase == "YELLOW":
                css, label, sub = "lane-yellow", "🟡 YELLOW", f"{int(t_rem)}s"
            elif active:
                css, label, sub = "lane-red", "🔴 ALL-RED", "clearing"
            else:
                css, label, sub = (
                    "lane-red",
                    f"🔴 {count} vehicles",
                    f"waiting {wait_s}s • queue {track_info['queue']} • stopped {track_info['stopped']}",
                )

            slot.markdown(
                _html(
                    f"""<div class="lane-card {css}">
                        <h4>{direction} APPROACH</h4>
                        <div class="big">{label}</div>
                        <div class="sub">{sub}</div>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

        for direction, slot in view_slots.items():
            if direction not in frames:
                continue
            frame_disp = frames[direction].copy()
            phase_for_direction = "NS" if direction in ("North", "South") else "EW"
            active = phase_for_direction == st.session_state.active_phase
            wait_s = int(st.session_state.red_wait_times[phase_for_direction])

            if active and phase == "GREEN":
                txt, col = f"{direction}: GREEN ({lane_counts[direction]} cars) | {int(t_rem)}s left", (0, 255, 0)
            elif active and phase == "YELLOW":
                txt, col = f"{direction}: YELLOW | {int(t_rem)}s", (255, 255, 0)
            elif active:
                txt, col = f"{direction}: ALL-RED CLEARING", (255, 80, 80)
            else:
                txt, col = f"{direction}: RED | phase wait {wait_s}s", (255, 0, 0)
            cv2.putText(frame_disp, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            slot.image(frame_disp, caption=f"{direction} ROI Feed", width="stretch")

    with tab_analytics:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history).set_index("t")
            chart_slot.line_chart(df)
        else:
            chart_slot.info("Waiting for detection data…")

        totals = {l: sum(h.get(l, 0) for h in st.session_state.history) for l in LANES}
        avg_wait = sum(st.session_state.red_wait_times.values()) / len(PHASES)
        with metric_slot.container():
            mcols = st.columns(5)
            for i, l in enumerate(LANES):
                mcols[i].metric(f"{l} samples sum", totals[l])
            mcols[4].metric("Avg red wait (s)", f"{avg_wait:.1f}")

    with tab_logs:
        if st.session_state.event_log:
            log_slot.markdown(
                "\n\n".join(f"`{entry}`" for entry in st.session_state.event_log)
            )
        else:
            log_slot.info("No lane switches yet.")


run_junction()