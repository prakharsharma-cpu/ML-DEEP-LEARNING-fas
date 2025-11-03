import streamlit as st
from PIL import Image
import torch
import tempfile
import numpy as np

st.set_page_config(page_title="PPE Detection App", layout="centered")
st.title("🦺 PPE Detection (YOLOv5 Streamlit Cloud App)")
st.write("Upload an image and detect PPE items using YOLOv5. Works directly on Streamlit Cloud.")

@st.cache_resource
def load_yolov5(weights_path=None):
    try:
        if weights_path:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def run_inference(model, image):
    try:
        results = model(image, size=640)
        try:
            results.render()
        except Exception:
            pass
        img_out = image
        if hasattr(results, 'imgs') and results.imgs:
            raw = results.imgs[0]
            if isinstance(raw, np.ndarray):
                img_out = Image.fromarray(raw.astype('uint8'))
        detections = []
        try:
            df = results.pandas().xyxy[0]
            detections = df.to_dict(orient='records')
        except Exception:
            detections = []
        return img_out, detections
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return image, []

uploaded_weights = st.sidebar.file_uploader("Upload custom YOLOv5 weights (.pt)", type=['pt'])
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

weights_path = None
if uploaded_weights is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp:
            temp.write(uploaded_weights.read())
            weights_path = temp.name
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded weights: {e}")

model = load_yolov5(weights_path)

if model is None:
    st.error('Model is not available. Please upload valid weights or check internet connection.')
    st.stop()

if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert('RGB')
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        st.stop()
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect PPE'):
        with st.spinner('Detecting PPE items...'):
            out_image, detections = run_inference(model, image)
        st.image(out_image, caption='Detections', use_column_width=True)

        if detections:
            st.subheader('Detections:')
            for d in detections:
                name = d.get('name') or str(d.get('class', 'unknown'))
                conf = float(d.get('confidence', 0))
                st.write(f"- {name} ({conf:.2f})")
        else:
            st.info('No PPE detected with sufficient confidence.')
else:
    st.info('Upload an image to begin PPE detection.')

st.markdown('---')
st.caption('YOLOv5 PPE Detection App — Streamlit Cloud Compatible. Requires only: streamlit, torch, torchvision, Pillow, opencv-python-headless.')
