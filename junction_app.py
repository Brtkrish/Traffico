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

st.caption("Multi-stream YOLOv8 junction control — polygon ROI detection, dynamic timing, starvation protection.")

# ---------------------------------------------------------------------------
# 3. Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Control & Selection Settings")

run_toggle = st.sidebar.toggle("▶️ Run Junction", value=False,
                                help="Start/stop processing. Streams stay paused until you turn this on.")

control_mode = st.sidebar.radio("Traffic Signal Mode", ["Automated Adaptive (AI)", "Manual Lane Override"])

manual_selected_lane = "North"
if control_mode == "Manual Lane Override":
    manual_selected_lane = st.sidebar.selectbox("Select Active Green Lane", LANES)

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

    st.session_state.active_lane = "North"
    st.session_state.target_next_lane = "North"
    st.session_state.current_phase = "GREEN"
    st.session_state.allocated_green_duration = BASE_GREEN_TIME
    st.session_state.phase_start_time = time.time()
    st.session_state.red_wait_times = {l: 0.0 for l in LANES}
    st.session_state.last_tick_time = time.time()
    st.session_state.frame_counter = 0
    st.session_state.last_frames = {}
    st.session_state.last_counts = {l: 0 for l in LANES}
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


def make_intersection_svg(active_lane, phase, lane_counts):
    def color_for(lane):
        if lane != active_lane:
            return "#f85149"
        return {"GREEN": "#2ea043", "YELLOW": "#d29922", "ALL_RED": "#f85149"}[phase]

    positions = {"North": (160, 40), "South": (160, 280), "East": (280, 160), "West": (40, 160)}
    blink = 'begin="0s"'
    steady = 'begin="indefinite"'
    lights_parts = []
    for lane, (x, y) in positions.items():
        c = color_for(lane)
        cnt = lane_counts.get(lane, 0)
        anim_begin = blink if (lane == active_lane and phase != "GREEN") else steady
        label_y = y - 26 if y < 160 else y + 40
        lights_parts.append(_html(f"""
            <circle cx="{x}" cy="{y}" r="16" fill="{c}" stroke="#0e1117" stroke-width="3">
              <animate attributeName="opacity" values="1;0.55;1" dur="1.4s" repeatCount="indefinite" {anim_begin} />
            </circle>
            <text x="{x}" y="{y+4}" text-anchor="middle" font-size="12" font-weight="700" font-family="JetBrains Mono, monospace" fill="#0e1117">{cnt}</text>
            <text x="{x}" y="{label_y}" text-anchor="middle" font-size="13" font-family="Space Grotesk, sans-serif" fill="#8b949e">{lane}</text>
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
            <div class="hdr-stat"><div class="v">{st.session_state.active_lane}</div><div class="l">Active Lane</div></div>
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

            roi_count = 0
            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                        roi_count += 1
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            annotated = results[0].plot()
            cv2.polylines(annotated, [roi], isClosed=True, color=(0, 255, 255), thickness=2)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            lane_counts[direction] = roi_count
            frames[direction] = annotated

        st.session_state.last_counts = lane_counts
        st.session_state.last_frames = frames

        st.session_state.history.append({"t": clock_str, **lane_counts})
        if len(st.session_state.history) > 120:
            st.session_state.history.pop(0)

    for lane in LANES:
        if lane != st.session_state.active_lane:
            st.session_state.red_wait_times[lane] += dt

    # ---- State machine ----
    phase = st.session_state.current_phase
    if phase == "GREEN":
        if phase_elapsed >= st.session_state.allocated_green_duration and phase_elapsed >= MIN_GREEN_TIME:
            if control_mode == "Manual Lane Override":
                st.session_state.target_next_lane = manual_selected_lane
            else:
                scores = {l: c + (st.session_state.red_wait_times[l] * 0.5)
                          for l, c in lane_counts.items() if l != st.session_state.active_lane}
                st.session_state.target_next_lane = max(scores, key=scores.get) if scores else st.session_state.active_lane
            st.session_state.current_phase = "YELLOW"
            st.session_state.phase_start_time = now

    elif phase == "YELLOW":
        if phase_elapsed >= YELLOW_TIME:
            st.session_state.current_phase = "ALL_RED"
            st.session_state.phase_start_time = now

    elif phase == "ALL_RED":
        if phase_elapsed >= ALL_RED_TIME:
            prev_lane = st.session_state.active_lane
            st.session_state.active_lane = st.session_state.target_next_lane
            st.session_state.red_wait_times[st.session_state.active_lane] = 0.0
            v_count = lane_counts.get(st.session_state.active_lane, 0)
            st.session_state.allocated_green_duration = min(BASE_GREEN_TIME + (v_count * SEC_PER_VEHICLE), MAX_GREEN_TIME)
            st.session_state.current_phase = "GREEN"
            st.session_state.phase_start_time = now

            st.session_state.event_log.insert(0,
                f"{clock_str} — {prev_lane} → {st.session_state.active_lane} "
                f"(green alloc {int(st.session_state.allocated_green_duration)}s, {v_count} vehicles)")
            if len(st.session_state.event_log) > 50:
                st.session_state.event_log.pop()

    # ---- Render ----
    phase = st.session_state.current_phase
    phase_elapsed = now - st.session_state.phase_start_time
    target_dur = {"GREEN": st.session_state.allocated_green_duration, "YELLOW": YELLOW_TIME, "ALL_RED": ALL_RED_TIME}[phase]
    t_rem = max(0.0, target_dur - phase_elapsed)
    pct_remaining = (t_rem / target_dur * 100) if target_dur > 0 else 0
    ring_color = {"GREEN": "#2ea043", "YELLOW": "#d29922", "ALL_RED": "#f85149"}[phase]

    with tab_live:
        ring_slot.markdown(
            make_ring(pct_remaining, ring_color, f"{int(t_rem)}s", f"{st.session_state.active_lane} — {phase}"),
            unsafe_allow_html=True,
        )
        diagram_slot.markdown(make_intersection_svg(st.session_state.active_lane, phase, lane_counts), unsafe_allow_html=True)

        for direction, slot in sig_slots.items():
            count = lane_counts.get(direction, 0)
            wait_s = int(st.session_state.red_wait_times[direction])
            active = direction == st.session_state.active_lane

            if active and phase == "GREEN":
                css, label, sub = "lane-green", f"🟢 ACTIVE — {count} vehicles", f"{int(t_rem)}s left of {int(st.session_state.allocated_green_duration)}s"
            elif active and phase == "YELLOW":
                css, label, sub = "lane-yellow", "🟡 YELLOW", f"{int(t_rem)}s"
            elif active:
                css, label, sub = "lane-red", "🔴 ALL-RED", "clearing"
            else:
                css, label, sub = "lane-red", f"🔴 {count} vehicles", f"waiting {wait_s}s"

            slot.markdown(
                _html(f"""<div class="lane-card {css}"><h4>{direction} LANE</h4>
                        <div class="big">{label}</div><div class="sub">{sub}</div></div>"""),
                unsafe_allow_html=True,
            )

        for direction, slot in view_slots.items():
            if direction not in frames:
                continue
            frame_disp = frames[direction].copy()
            active = direction == st.session_state.active_lane
            if active and phase == "GREEN":
                txt, col = f"{direction}: GREEN ({lane_counts[direction]} cars) | {int(t_rem)}s left", (0, 255, 0)
            elif active and phase == "YELLOW":
                txt, col = f"{direction}: YELLOW | {int(t_rem)}s", (255, 255, 0)
            elif active:
                txt, col = f"{direction}: ALL-RED CLEARING", (255, 80, 80)
            else:
                txt, col = f"{direction}: RED | waiting {int(st.session_state.red_wait_times[direction])}s", (255, 0, 0)
            cv2.putText(frame_disp, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            slot.image(frame_disp, caption=f"{direction} ROI Feed", width="stretch")

    with tab_analytics:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history).set_index("t")
            chart_slot.line_chart(df)
        else:
            chart_slot.info("Waiting for detection data…")

        totals = {l: sum(h.get(l, 0) for h in st.session_state.history) for l in LANES}
        avg_wait = sum(st.session_state.red_wait_times.values()) / len(LANES)
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