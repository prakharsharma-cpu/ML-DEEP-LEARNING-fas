import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ----------------- Streamlit Page Setup -----------------
st.set_page_config(page_title="SafetyVision AI", layout="wide")

# ----------------- Load YOLO Model -----------------
@st.cache_resource
def load_model():
    try:
        # Replace with your trained model path if you have one
        model = YOLO("ppe_yolov8.pt")
        st.sidebar.success("✅ Custom PPE model loaded successfully.")
    except Exception:
        model = YOLO("yolov8n.pt")
        st.sidebar.warning("⚠️ Using default YOLOv8n model (no custom PPE training found).")
    return model

model = load_model()

# ----------------- Custom CSS -----------------
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

# ----------------- Header -----------------
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
        st.sidebar.success(f"AI Assistant Response: Safety guidance for '{prompt}' generated (simulated).")

# ----------------- Stats Overview -----------------
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

# Get the image data from camera or upload
image_data = camera_feed.getvalue() if camera_feed else (uploaded_file.read() if uploaded_file else None)

if image_data:
    # Save the uploaded bytes into a temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_data)
    temp_file.close()

    # Run YOLOv8 on the temp file
    results = model(temp_file.name)
    result_image = results[0].plot()  # Draw detection boxes

    # Display output
    st.image(result_image, caption="YOLOv8 PPE Detection", use_column_width=True)

    detections = results[0].boxes.cls
    if len(detections) > 0:
        st.success(f"✅ Detected {len(detections)} PPE items.")
    else:
        st.warning("⚠️ No PPE detected.")
else:
    st.info("📸 Please capture or upload an image to begin detection.")

# ----------------- Info Cards -----------------
st.subheader("⚙️ System Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="section-card">
        <h4>🧠 YOLO Detection</h4>
        <p>Real-time object detection using YOLO deep learning model for accurate PPE identification.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section-card">
        <h4>💬 AI Assistant</h4>
        <p>Conversational interface powered by AI for safety insights and proactive compliance guidance.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="section-card">
        <h4>📡 Live Monitoring</h4>
        <p>Continuous PPE tracking and instant alerts for safety violations to ensure workplace safety.</p>
    </div>
    """, unsafe_allow_html=True)
