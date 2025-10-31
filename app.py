# app.py
import streamlit as st
import tempfile
from PIL import Image
import io
import os
import time
import threading

st.set_page_config(page_title="SafetyVision AI (YOLOv5 Optimized)", layout="wide")

# ---------------------- YOLOv5 Loader ----------------------
@st.cache_resource
def try_load_yolov5_model(custom_weights_path: str = None):
    """Load YOLOv5 model safely via torch.hub or custom weights."""
    import torch
    try:
        if custom_weights_path:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=custom_weights_path, force_reload=False)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        model.to("cpu")
        return model
    except Exception as e:
        raise RuntimeError(
            "❌ YOLOv5 could not be loaded.\n\n"
            "Fix this by running:\n"
            "   pip install git+https://github.com/ultralytics/yolov5.git\n"
            "   pip install torch torchvision\n\n"
            f"Error details: {e}"
        )

# ---------------------- Styling / Header ----------------------
st.markdown("""
    <style>
    .header { background-color: rgba(30,30,30,0.9); padding: 1rem 2rem; border-bottom: 1px solid #444; color: white; }
    .section-card { background-color: #1f1f1f; border: 1px solid #333; padding: 1rem; border-radius: 8px; color: #fff; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="font-size:28px;">🛡️</div>
    <div>
      <div style="font-weight:700; font-size:18px;">SafetyVision AI</div>
      <div style="font-size:12px; color:gray;">PPE Detection & Monitoring (YOLOv5 Optimized)</div>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <div style="padding:6px 10px; border-radius:14px; background:rgba(22,163,74,0.08); border:1px solid rgba(22,163,74,0.2); color:#16a34a;">🟢 System Active</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------
st.sidebar.header("⚙️ Settings")

uploaded_weights = st.sidebar.file_uploader("Upload YOLOv5 .pt weights (optional)", type=["pt"])
custom_weights_path = None
if uploaded_weights:
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    t.write(uploaded_weights.read())
    t.flush()
    t.close()
    custom_weights_path = t.name
    st.sidebar.success("✅ Custom model uploaded.")

use_custom = st.sidebar.checkbox("Use uploaded weights", value=False)
use_local_cam = st.sidebar.checkbox("Use local webcam (OpenCV)", value=False)
frame_interval_ms = st.sidebar.slider("Frame interval (ms)", 50, 1000, 200, 50)
show_fps = st.sidebar.checkbox("Show FPS", value=True)

# ---------------------- Load Model ----------------------
try:
    model = try_load_yolov5_model(custom_weights_path if use_custom else None)
    st.sidebar.success("✅ Model loaded successfully.")
except Exception as e:
    st.sidebar.error("Model load failed. Follow on-screen instructions.")
    st.stop()

# ---------------------- Stats ----------------------
st.title("📊 System Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Cameras Active", "1", "+0")
col2.metric("Detections Today", "0", "+0")
col3.metric("Compliance Rate", "—", "—")
st.divider()

# ---------------------- PPE Detection (Image) ----------------------
st.subheader("🎥 PPE Detector (Image)")
st.write("Upload or capture an image for YOLOv5 PPE detection.")

camera_feed = st.camera_input("Capture Image")
uploaded_img = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"])

image_bytes = None
if camera_feed:
    image_bytes = camera_feed.getvalue()
elif uploaded_img:
    image_bytes = uploaded_img.read()

def run_detection(model, image_path):
    results = model(image_path)
    results.render()
    import numpy as np
    annotated = results.ims[0]
    if annotated.dtype != "uint8":
        annotated = (255 * np.clip(annotated, 0, 1)).astype("uint8")
    pil_img = Image.fromarray(annotated)
    df = results.pandas().xyxy[0]
    return pil_img, df

if image_bytes:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(image_bytes)
    tmp.flush()
    tmp.close()

    try:
        annotated_img, df = run_detection(model, tmp.name)
        st.image(annotated_img, caption="YOLOv5 PPE Detection", use_column_width=True)
        if not df.empty:
            st.dataframe(df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]])
    except Exception as e:
        st.error(f"❌ Inference failed: {e}")

# ---------------------- Optimized Live Stream ----------------------
st.subheader("📡 Live Webcam Stream (Optimized)")
st.write("Run this locally to stream from your webcam in real-time with YOLOv5 detection.")

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = False

start_button = st.button("▶ Start Live Stream") if use_local_cam else None
stop_button = st.button("⏹ Stop Stream") if use_local_cam else None

live_placeholder = st.empty()
info_placeholder = st.empty()

def live_stream_loop(model, frame_interval_ms, show_fps):
    """Optimized YOLOv5 live stream loop with frame skipping & resizing."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        info_placeholder.error("❌ Could not open webcam.")
        st.session_state.streaming = False
        return

    frame_skip = 2       # process every 3rd frame
    resize_width = 640   # resize width for faster inference
    processed = 0
    displayed = 0
    start_time = time.time()

    while st.session_state.streaming and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            info_placeholder.warning("⚠️ Failed to read frame.")
            break

        # Skip frames for speed
        if processed % frame_skip != 0:
            processed += 1
            continue
        processed += 1

        # Resize for performance
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = model(rgb_frame, size=resize_width)
            results.render()
            annotated = results.ims[0]
            if annotated.dtype != "uint8":
                annotated = (255 * np.clip(annotated, 0, 1)).astype("uint8")
            pil_img = Image.fromarray(annotated)
        except Exception as e:
            info_placeholder.error(f"Inference error: {e}")
            break

        # Display annotated frame
        live_placeholder.image(pil_img, use_column_width=True)
        displayed += 1

        # Compute FPS
        elapsed = time.time() - start_time
        fps = displayed / elapsed if elapsed > 0 else 0
        if show_fps:
            info_placeholder.markdown(f"**Live FPS:** {fps:.1f} | Processed Frames: {processed}")
        else:
            info_placeholder.markdown("Live stream running...")

        time.sleep(frame_interval_ms / 1000.0)

        if st.session_state.stop_signal:
            break

    cap.release()
    st.session_state.streaming = False
    st.session_state.stop_signal = False
    info_placeholder.info("✅ Stream stopped.")

# Start / Stop Stream
if use_local_cam:
    if start_button:
        if not st.session_state.streaming:
            st.session_state.streaming = True
            st.session_state.stop_signal = False
            t = threading.Thread(target=live_stream_loop, args=(model, frame_interval_ms, show_fps), daemon=True)
            t.start()
        else:
            st.warning("⚠️ Stream already running.")
    if stop_button:
        if st.session_state.streaming:
            st.session_state.stop_signal = True
            st.session_state.streaming = False
            info_placeholder.info("Stopping stream...")
        else:
            st.info("No active stream.")
else:
    st.info("Enable webcam in sidebar to use live detection (local only).")

# ---------------------- Info Cards ----------------------
st.subheader("💡 System Features")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="section-card"><h4>🧠 YOLOv5 Detection</h4><p>Fast, optimized object detection for PPE identification.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="section-card"><h4>💬 AI Assistant</h4><p>Smart conversational safety support (simulated).</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="section-card"><h4>📡 Live Monitoring</h4><p>Continuous PPE detection via optimized webcam feed.</p></div>', unsafe_allow_html=True)
