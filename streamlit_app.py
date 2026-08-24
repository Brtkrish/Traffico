from collections import defaultdict
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Traffic Control System",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Smart Traffic Control & Adaptive Signal System")
st.caption(
    "Real-time vehicle tracking, flow estimation, gridlock detection, and adaptive signal timing."
)

# 2. Sidebar Controls
st.sidebar.header("⚙️ System Configuration")
conf_threshold = st.sidebar.slider(
    "Model Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.40,
    step=0.05,
)
video_source = st.sidebar.text_input("Video File Name", "traffic1.mp4")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# 3. Create Dashboard Metric Cards
col1, col2, col3, col4 = st.columns(4)
metric_count = col1.empty()
metric_signal = col2.empty()
metric_density = col3.empty()
metric_stuck = col4.empty()

st_frame = st.empty()

# 4. Define ROI Polygon (Your tuned coordinates)
roi_points = np.array(
    [[922, 509], [109, 463], [436, 251], [732, 250]], np.int32
)

# Position history dictionary to track vehicle movement: {track_id: [(x1, y1), (x2, y2), ...]}
track_history = defaultdict(lambda: [])
STATIONARY_THRESHOLD_PIXELS = 10  # Maximum pixel drift to consider a car stuck
STATIONARY_FRAMES = 30  # Number of frames (~1-2 secs) to detect gridlock

# Open Video Stream
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    st.error(
        f"Unable to open video file '{video_source}'. Ensure it is in your project root folder."
    )

# 5. Process Video Feed with BYTETrack Multi-Object Tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.info("End of video stream reached.")
        break

    # Run YOLOv8 Tracking (BYTETrack engine)
    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        conf=conf_threshold,
        verbose=False,
    )

    roi_vehicle_count = 0
    stuck_vehicle_count = 0

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Point-in-polygon test
            if (
                cv2.pointPolygonTest(roi_points, (center_x, center_y), False)
                >= 0
            ):
                roi_vehicle_count += 1

                # Update tracking history for displacement checks
                history = track_history[track_id]
                history.append((center_x, center_y))
                if len(history) > STATIONARY_FRAMES:
                    history.pop(0)

                # Gridlock Check: Calculate movement over frame window
                is_stuck = False
                if len(history) == STATIONARY_FRAMES:
                    start_pt = np.array(history[0])
                    end_pt = np.array(history[-1])
                    distance_moved = np.linalg.norm(end_pt - start_pt)

                    if distance_moved < STATIONARY_THRESHOLD_PIXELS:
                        is_stuck = True
                        stuck_vehicle_count += 1

                # Visualization: Red dot for stuck vehicles, Green dot for moving
                dot_color = (0, 0, 255) if is_stuck else (0, 255, 0)
                cv2.circle(frame, (center_x, center_y), 6, dot_color, -1)

                # Draw track ID label
                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    dot_color,
                    2,
                )

    annotated_frame = results[0].plot()

    # Draw Yellow ROI Polygon
    cv2.polylines(
        annotated_frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=3
    )

    # 6. Adaptive Signal Logic with Gridlock Override
    if stuck_vehicle_count > 0:
        signal_text = "PRIORITY GREEN (35s)"
        status_label = f"GRIDLOCK ALERT ({stuck_vehicle_count} Stuck)"
        density_status = "Jammed"
    elif roi_vehicle_count < 5:
        signal_text = "GREEN (10s)"
        status_label = "Low Traffic"
        density_status = "Light"
    elif 5 <= roi_vehicle_count < 10:
        signal_text = "GREEN (20s)"
        status_label = "Moderate Traffic"
        density_status = "Medium"
    else:
        signal_text = "EXTENDED GREEN (30s)"
        status_label = "Heavy Traffic Congestion"
        density_status = "High"

    # Update Dashboard Metrics
    metric_count.metric("Active Vehicles", f"{roi_vehicle_count}")
    metric_signal.metric(
        "Signal Duration", signal_text, delta=status_label
    )
    metric_density.metric("Traffic Density", density_status)
    metric_stuck.metric("Stationary Vehicles", f"{stuck_vehicle_count}")

    # Render Frame
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st_frame.image(
        annotated_frame,
        caption="Live Vehicle Tracking & Gridlock Monitor",
        use_container_width=True,
    )

cap.release()