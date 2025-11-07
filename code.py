# ✅ Updated SmartPPE: Full-Person PPE Compliance Detection (Colab)
# - Detects person only if a FULL PPE kit is worn.r
# - Green = FULL PPE
# - Yellow = PARTIALLY PPE
# - Red = NO PPE
# - All PPE items included: helmet, vest, mask, goggles, gloves, boots, coverall

!pip install ultralytics pillow matplotlib opencv-python

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from google.colab import files
import os

# ✅ Load YOLOv8x (or your custom PPE model)
model = YOLO('yolov8x.pt')
print("✅ Model Loaded:", model.model.yaml.get('name', 'yolov8x'))

# ✅ FULL PPE MASTER LIST
PPE_CATEGORIES = {
    "helmet": ["helmet", "hardhat"],
    "vest": ["vest", "safety vest", "hi-vis"],
    "mask": ["mask", "face mask", "surgical mask"],
    "goggles": ["goggles", "safety goggles", "glasses"],
    "gloves": ["glove", "gloves"],
    "boots": ["boots", "shoe", "safety boot"],
    "coverall": ["coverall", "bodysuit", "ppe suit"]
}

# ✅ Person label
PERSON_LABELS = ["person"]

# Map predicted labels to PPE keys

def label_to_ppe_key(label):
    ll = label.lower()
    for key, options in PPE_CATEGORIES.items():
        if any(v.lower() in ll for v in options):
            return key
    return None

# IoU helper

def iou(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (A[2] - A[0]) * (A[3] - A[1])
    areaB = (B[2] - B[0]) * (B[3] - B[1])
    return inter / (areaA + areaB - inter + 1e-9)

# ✅ FULL PPE REQUIRED LIST
REQUIRED_PPE = [
    "helmet", "vest", "mask", "goggles", "gloves", "boots", "coverall"
]

# ✅ Analyze image for full PPE

def analyze_image(path, conf=0.35, imgsz=1280):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = model.predict(rgb, conf=conf, imgsz=imgsz)[0]
    names = result.names

    detections = []
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0].cpu().numpy())
        cid = int(b.cls[0].cpu().numpy())
        label = names[cid]
        detections.append({
            "box": [x1, y1, x2, y2],
            "score": conf,
            "label": label
        })

    persons = [d for d in detections if d["label"].lower() in PERSON_LABELS]
    ppe_items = [d for d in detections if d["label"].lower() not in PERSON_LABELS]

    report = []

    for idx, p in enumerate(persons):
        boxP = p["box"]
        found = set()

        for od in ppe_items:
            key = label_to_ppe_key(od["label"])
            if key and iou(boxP, od["box"]) > 0.12:
                found.add(key)

        missing = [r for r in REQUIRED_PPE if r not in found]

        # ✅ Status color rules
        if len(found) == 0:
            status = "RED"      # No PPE
        elif len(found) < len(REQUIRED_PPE):
            status = "YELLOW"   # Partially PPE
        else:
            status = "GREEN"    # Fully PPE

        report.append({
            "person_id": idx+1,
            "present": list(found),
            "missing": missing,
            "status": status
        })

        # ✅ Draw colored box
        color = (0,255,0) if status=="GREEN" else (0,255,255) if status=="YELLOW" else (0,0,255)
        x1,y1,x2,y2 = map(int, boxP)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
        cv2.putText(img, f"P{idx+1} {status}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_name = f"annotated_{os.path.basename(path)}"
    cv2.imwrite(out_name, img)
    print("✅ Saved:", out_name)

    return img, report

# ✅ Upload
print("Upload images with people wearing PPE:")
files_uploaded = files.upload()

for f in files_uploaded:
    print("\n🔍 Processing:", f)
    img, rep = analyze_image(f)

    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print("📌 REPORT:")
    for r in rep:
        print(r)
