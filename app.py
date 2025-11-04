# app_yolov5.py
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Try importing torch
try:
    import torch
except Exception:
    st.error(
        "❌ PyTorch not installed. Install it first:\n\n"
        "`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`"
    )
    st.stop()

# ------------------------
# Streamlit App Setup
# ------------------------
st.set_page_config(page_title="🦺 SmartPPE - YOLOv5 Detection", layout="centered")
st.title("🦺 SmartPPE — YOLOv5 PPE Detection")
st.markdown("Upload an image and detect PPE (helmet, vest, mask, etc.) using YOLOv5.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("YOLOv5 Model Path (or use 'yolov5s')", value="best.pt")
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.01)
max_det = st.sidebar.number_input("Max Detections per Image", min_value=1, max_value=300, value=100)
use_gpu = st.sidebar.checkbox("Use GPU (CUDA)", value=False)
img_size = st.sidebar.selectbox("Inference Image Size", [320, 416, 640, 960], index=2)

# ------------------------
# Load YOLOv5 Model
# ------------------------
@st.cache_resource(ttl=60 * 60)
def load_yolov5_model(path: str, device: str = "cpu"):
    """Loads YOLOv5 model via torch.hub."""
    try:
        if Path(path).exists() and Path(path).suffix == ".pt":
            model = torch.hub.load("ultralytics/yolov5", "custom", path, force_reload=False)
        else:
            model = torch.hub.load("ultralytics/yolov5", path, pretrained=True)
        model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"⚠️ Failed to load YOLOv5 model: {e}")

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Choose device
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if use_gpu and device == "cpu":
        st.warning("⚠️ GPU requested but not available — running on CPU.")

    # Load model
    try:
        model = load_yolov5_model(model_path, device)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Apply thresholds
    try:
        model.conf = conf_thres
        model.max_det = int(max_det)
    except Exception:
        pass

    # Run inference
    with st.spinner("🔍 Detecting PPE..."):
        results = model(np.array(image), size=img_size)
        preds = results.xyxy[0].cpu().numpy() if len(results.xyxy) else np.empty((0, 6))

    if preds.shape[0] == 0:
        st.info("No detections above threshold.")
        st.stop()

    names = getattr(model, "names", {i: str(i) for i in range(100)})

    # Annotate image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    COLORS = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 128, 255), (255, 0, 255)]
    detections = []

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(preds):
        x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
        label = names.get(cls, f"class_{cls}")
        conf_f = float(conf)
        color = COLORS[i % len(COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {conf_f:.2f}"
        tw, th = draw.textsize(text, font)
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), text, fill="white", font=font)

        # PPE recommendations
        ll = label.lower()
        if "helmet" in ll or "hardhat" in ll:
            rec = "✅ Helmet detected"
        elif "vest" in ll or "hi-vis" in ll:
            rec = "✅ Safety vest detected"
        elif "mask" in ll or "respirator" in ll:
            rec = "✅ Mask detected"
        elif "glove" in ll:
            rec = "✅ Gloves detected"
        else:
            rec = "⚠️ Unknown object"

        detections.append({
            "Label": label,
            "Confidence": round(conf_f, 3),
            "BBox": [x1, y1, x2, y2],
            "Recommendation": rec
        })

    # Display annotated image
    st.subheader("🖼️ Annotated Image")
    st.image(annotated, use_container_width=True)

    # Detection table
    st.subheader("📋 Detected Objects")
    st.table(detections)

    # Download annotated image
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("📥 Download Annotated Image", data=buf, file_name="annotated.png", mime="image/png")

else:
    st.info("⬆️ Upload an image to start detection.")
