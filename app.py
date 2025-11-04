# app.py
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Try to import ultralytics; show helpful message if missing.
try:
    from ultralytics import YOLO
except Exception as e:
    st.error(
        "Failed to import ultralytics. Install with:\n\n"
        "`pip install ultralytics` \n\nThen restart Streamlit."
    )
    st.stop()

st.set_page_config(page_title="SmartPPE - PPE Detection", layout="centered")

st.title("🦺 SmartPPE — Computer Vision PPE Detection")
st.markdown(
    "Upload an image and the YOLO model will detect PPE items (helmet, vest, mask, etc.)."
)

# Sidebar: model selection and threshold
st.sidebar.header("Model & Settings")
model_path = st.sidebar.text_input(
    "YOLO model path (local .pt or pretrained)", value="best.pt"
)
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
max_det = st.sidebar.number_input("Max detections", min_value=1, max_value=200, value=50)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)

# Load model (safe cached load)
@st.cache_resource(ttl=60 * 60)  # cache for an hour
def load_model(path: str, device_gpu: bool):
    device = 0 if device_gpu else "cpu"
    try:
        model = YOLO(path)
        # set device if supported by ultralytics
        try:
            model.to(device)
        except Exception:
            pass
        return model
    except Exception as e:
        raise RuntimeError(f"Unable to load model from '{path}': {e}")

# Upload image
uploaded_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    # Attempt to load model
    try:
        model = load_model(model_path, use_gpu)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Run inference
    with st.spinner("Running model inference..."):
        # ultralytics can accept PIL images, file paths, or numpy arrays.
        # Use model.predict with sensible kwargs.
        results = model.predict(
            source=np.array(image),
            conf=conf_thres,
            max_det=max_det,
            verbose=False,
        )

    if len(results) == 0:
        st.warning("No results returned by model.")
        st.stop()

    res = results[0]

    # If no boxes
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        st.info("No PPE detected above the confidence threshold.")
        st.stop()

    # Prepare annotation
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a reasonable font; fallback if not found
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    detections = []
    # ultralytics Boxes: .xyxy, .cls, .conf - but depending on version attributes may differ.
    # Convert to numpy-friendly structure:
    try:
        xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)
    except Exception:
        # Fallback for attribute names
        try:
            xyxy = np.array([b.xyxy[0].cpu().numpy() for b in boxes])
            cls_ids = np.array([int(b.cls[0]) for b in boxes])
            confs = np.array([float(b.conf[0]) for b in boxes])
        except Exception:
            st.error("Unexpected model output format. Update ultralytics or check model.")
            st.stop()

    # Map model class indices to human-readable names if available
    names = getattr(model, "names", None)
    if names is None:
        # default placeholder
        names = {i: f"class_{i}" for i in range(max(cls_ids) + 1)}

    # PPE-specific bin/recommendation mapping (customize as needed)
    def ppe_recommendation(label: str):
        label_lower = label.lower()
        if "helmet" in label_lower or "hardhat" in label_lower:
            return "🟢 Helmet detected — Good (Head protection)"
        if "vest" in label_lower or "hi-vis" in label_lower or "vest" in label_lower:
            return "🟢 Safety Vest detected — Good (Visibility)"
        if "mask" in label_lower or "respirator" in label_lower:
            return "🟢 Mask detected — Good (Respiratory protection)"
        if "glove" in label_lower:
            return "🟢 Gloves detected — Good (Hand protection)"
        # default
        return "⚪️ No specific PPE recommendation available for this label"

    # Colors for boxes (choose visually distinct)
    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 165, 0),
        (255, 255, 0),
        (128, 0, 128),
    ]

    for i, (box, cls_id, conf) in enumerate(zip(xyxy, cls_ids, confs)):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = names.get(int(cls_id), str(int(cls_id)))
        conf_f = float(conf)
        color = COLORS[i % len(COLORS)]
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Label background
        text = f"{label} {conf_f:.2f}"
        text_size = draw.textsize(text, font=font)
        text_bg = [x1, y1 - text_size[1] - 6, x1 + text_size[0] + 6, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 3, y1 - text_size[1] - 3), text, fill=(255, 255, 255), font=font)

        detections.append(
            {
                "label": label,
                "confidence": round(conf_f, 4),
                "bbox": [x1, y1, x2, y2],
                "recommendation": ppe_recommendation(label),
            }
        )

    # Display annotated image
    st.markdown("### Annotated image")
    st.image(annotated, use_container_width=True)

    # Show detections table
    st.markdown("### Detections")
    st.table(
        [
            {
                "Label": d["label"],
                "Confidence": d["confidence"],
                "BBox (x1,y1,x2,y2)": str(d["bbox"]),
                "Recommendation": d["recommendation"],
            }
            for d in detections
        ]
    )

    # Download annotated image
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")

    # Summary
    st.markdown("### Summary")
    labels_present = list({d["label"] for d in detections})
    st.write(f"Detected: **{', '.join(labels_present)}**")
    st.write(f"Model: `{model_path}` — confidence threshold: {conf_thres}")

else:
    st.info("Upload an image to get started.")
