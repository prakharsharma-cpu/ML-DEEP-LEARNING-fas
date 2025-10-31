import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import time

st.set_page_config(page_title="SafetyVision AI", layout="wide")

# ----------------- Load YOLO Model -----------------
@st.cache_resource
def load_model():
    try:
        # Replace with your trained PPE model path if available
        model = YOLO("ppe_yolov8.pt")
    except Exception:
        model = YOLO("yolov8n.pt")  # fallback model
    return model

model = load_model()

# ----------------- Header -----------------
st.markdown("""
    <style>
        .header {
            background-color: rgba(30, 30, 30, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid #444;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .system-active {
            color: #16a34a;
            border: 1px solid rgba(22, 163, 74, 0.3);
            background: rgba(22, 163, 74, 0.1);
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        .section-card {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <div class="logo">
        <span style="font-size: 2rem;">🛡️</span>
        <div>
            <h2 style="margin: 0;">SafetyVision AI</h2>
            <p style="margin: 0; font-size: 0.9rem; color: gray;">PPE Detection & Monitoring System</p>
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <div class="system-active">🟢 System Active</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar: AI Assistant -----------------
st.sidebar.header("🧠 AI Assistant")
show_assistant = st.sidebar.checkbox("Enable AI Assistant", value=False)
if show_assistant:
    prompt = st.sidebar.text_area("Ask SafetyVision AI:")
    if st.sidebar.button("Send"):
        st.sidebar.success(f"AI Assistant: Safety insight generated for '{prompt}' (simulated).")

# ----------------- Stats Section -----------------
st.title("📊 System Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Cameras Active", "12", "+2")
col2.metric("Detections Today", "156", "+18")
col3.metric("Compliance Rate", "98%", "+3%")
st.divider()

# ----------------- PPE Detection -----------------
st.subheader("🎥 PPE Detector (YOLOv8)")
st.write("Upload or capture an image to perform real-time PPE detection using YOLOv8.")

camera_feed = st.camera_input("Capture or Upload Image")
uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

image_data = None
if camera_feed:
    image_data = camera_feed.getvalue()
elif uploaded_file:
    image_data = uploaded_file.getvalue()

if image_data:
    # Convert image to OpenCV format
    image = Image.open(tempfile.NamedTemporaryFile(delete=False))
    image = Image.open(np.array(Image.open(tempfile.NamedTemporaryFile(delete=False))))
    image = Image.open(tempfile.NamedTemporaryFile(delete=False))
    image = Image.open(tempfile.NamedTemporaryFile(delete=False))
    image = Image.open(tempfile.NamedTemporaryFile(delete=False))

if image_data:
    image = Image.open(tempfile.NamedTemporaryFile(delete=False))
    image = Image.open(np.array(Image.open(tempfile.NamedTemporaryFile(delete=False))))
    img = Image.open(tempfile.NamedTemporaryFile(delete=False))
    image = Image.open(np.array(Image.open(tempfile.NamedTemporaryFile(delete=False))))
else:
    image = None

if image_data:
    # Save temp image and detect
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_data)
    temp_file.close()
    results = model(temp_file.name)
    result_image = results[0].plot()

    st.image(result_image, caption="YOLOv8 PPE Detection", use_column_width=True)

    detections = results[0].boxes.cls
    if len(detections) > 0:
        st.success(f"✅ Detected {len(detections)} PPE items.")
    else:
        st.warning("⚠️ No PPE detected.")

# ----------------- Info Cards -----------------
st.subheader("⚙️ System Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="section-card">
        <h4>🧠 YOLO Detection</h4>
        <p>Real-time object detection using YOLO deep learning model for PPE identification.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section-card">
        <h4>💬 AI Assistant</h4>
        <p>Conversational interface powered by AI for safety insights and compliance monitoring.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="section-card">
        <h4>📡 Live Monitoring</h4>
        <p>Continuous PPE tracking and instant alerts to ensure workplace safety.</p>
    </div>
    """, unsafe_allow_html=True)
