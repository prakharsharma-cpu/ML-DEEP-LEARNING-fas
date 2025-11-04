# app_yolov5.py
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Optional: import torch (required for torch.hub)
try:
    import torch
except Exception:
    st.error(
        "PyTorch is not installed. Install it first (see https://pytorch.org/) or run:\n\n"
        "`pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118` (example for CUDA 11.8)\n\n"
        "Then restart Streamlit."
    )
    st.stop()

st.set_page_config(page_title="SmartPPE - YOLOv5 PPE Detection", layout="centered")
st.title("🦺 SmartPPE — YOLOv5 PPE Detection")
st.markdown("Upload an image and a YOLOv5 model will detect PPE items (helmet, vest, mask, etc.).")

# Sidebar settings
st.sidebar.header("Model & Settings")
model_path = st.sidebar.text_input("YOLOv5 model path (local .pt or use 'yolov5s')", value="best.pt")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
max_det = st.sidebar.number_input("Max detections (per image)", min_value=1, max_value=300, value=100)
use_gpu = st.sidebar.checkbox("Use GPU if available (cuda)", value=False)
img_size = st.sidebar.selectbox("Inference image size", options=[320, 416, 640, 960], index=2)

# Helper: load YOLOv5 model via torch.hub
@st.cache_resource(ttl=60 * 60)
def load_yolov5_model(path: str, device: str = "cpu"):
    """
    Loads a YOLOv5 model via torch.hub.
    - path: 'yolov5s' for pretrained small, or path to custom .pt weights.
    - device: 'cpu' or 'cuda'
    """
    try:
        # Use ultralytics/yolov5 hub implementation
        # 'custom' loads custom weights when path points to a .pt
        if Path(path).exists() and Path(path).suffix == ".pt":
            model = torch.hub.load("ultralytics/yolov5", "custom", path, force_reload=False)
        else:
            # allow model names like 'yolov5s', 'yolov5m', etc.
            model = torch.hub.load("ultralytics/yolov5", path, pretrained=True)
        model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv5 model ({path}): {e}")

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # choose device
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if use_gpu and device == "cpu":
        st.warning("GPU requested but not available — running on CPU.")

    # load model
    try:
        model = load_yolov5_model(model_path, device=device)
    except Exception as e:
        st.error(str(e))
        st.markdown(
            "If you don't have a custom `best.pt`, try `yolov5s` in the model path field (pretrained)."
        )
        st.stop()

    # set model confidence threshold and max det
    # YOLOv5 models expose .conf, .max_det attributes
    try:
        model.conf = conf_thres  # confidence threshold
        model.max_det = int(max_det)
    except Exception:
        # not fatal; continue
        pass

    # Run inference
    with st.spinner("Running YOLOv5 inference..."):
        # Convert PIL -> numpy array (BGR/ RGB handled by model)
        img_np = np.array(image)
        # model expects either path, list of images, or numpy array
        results = model(img_np, size=img_size)  # returns a Results object

    # Extract detections
    # results.xyxy is a list (per image) of tensor Nx6: x1,y1,x2,y2,conf,class
    try:
        preds = results.xyxy[0].cpu().numpy()  # shape (N,6)
    except Exception:
        preds = np.empty((0, 6))

    if preds.shape[0] == 0:
        st.info("No detections above the confidence threshold.")
        st.stop()

    # model.names maps class indices to labels
    names = getattr(model, "names", None)
    if names is None:
        # fallback
        names = {i: str(i) for i in range(100)}

    # Prepare annotated image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    detections = []
    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 165, 0),
        (255, 255, 0),
        (128, 0, 128),
    ]

    for i, row in enumerate(preds):
        x1, y1, x2, y2, conf, cls = row
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        label = names.get(cls, f"class_{cls}")
        conf_f = float(conf)

        color = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {conf_f:.2f}"
        text_size = draw.textsize(text, font=font)
        text_bg = [x1, y1 - text_size[1] - 6, x1 + text_size[0] + 6, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 3, y1 - text_size[1] - 3), text, fill=(255, 255, 255), font=font)

        # simple PPE recommendation mapping (customize to your labels)
        ll = label.lower()
        if "helmet" in ll or "hardhat" in ll:
            rec = "Helmet — OK"
        elif "vest" in ll or "hi-vis" in ll:
            rec = "High-visibility vest — OK"
        elif "mask" in ll or "respirator" in ll:
            rec = "Mask — OK"
        elif "glove" in ll:
            rec = "Gloves — OK"
        else:
            rec = "No specific PPE recommendation"

        detections.append(
            {
                "label": label,
                "confidence": round(conf_f, 4),
                "bbox": [x1, y1, x2, y2],
                "recommendation": rec,
            }
        )

    # Show annotated
    st.markdown("### Annotated image")
    st.image(annotated, use_container_width=True)

    # Detections table
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
