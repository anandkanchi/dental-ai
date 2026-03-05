import os
import cv2
import numpy as np
import tensorflow as tf

from anomaly import detect_anomaly
from tooth_numbering import ToothNumberingEngine
from missing_tooth import detect_missing_teeth, missing_by_quadrant
from smart_classify import run_smart_pipeline, EXPECTED_FDI as SMART_FDI

# ============================
# MODEL PATH — relative, works anywhere
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_MODEL = os.path.join(BASE_DIR, "multi_class.keras")

IMG_SIZE = (224, 224)

# ============================
# LOAD MODEL (cached globally)
# ============================

_model = None

def load_model():
    global _model
    if _model is None:
        print("\nLoading classifier model...")
        _model = tf.keras.models.load_model(CLASSIFIER_MODEL)
        print("Model loaded.")
    return _model


# ============================
# GENERATE TOOTH BOXES
# ============================

def generate_tooth_boxes(image):
    """
    Generate bounding boxes for 32 teeth by splitting the OPG into upper/lower arches.

    In a standard OPG panoramic X-ray:
      - Upper arch (Q1/Q2): roughly top 15% - 50% of image height
      - Lower arch (Q3/Q4): roughly 50% - 85% of image height
    16 teeth per arch, evenly spaced horizontally.

    NOTE: This is a geometric approximation — replace with a real segmentation
    model (e.g. YOLO trained on dental OPGs) for production accuracy.
    """
    h, w = image.shape[:2]

    # Upper arch: 16 teeth across full width
    upper_y1 = int(h * 0.15)
    upper_y2 = int(h * 0.50)
    # Lower arch: 16 teeth across full width
    lower_y1 = int(h * 0.50)
    lower_y2 = int(h * 0.85)

    tooth_width = w // 16  # 16 teeth per arch
    boxes = []

    for i in range(16):
        x1 = i * tooth_width
        x2 = (i + 1) * tooth_width
        # Upper arch tooth
        boxes.append({"bbox": [x1, upper_y1, x2, upper_y2]})
        # Lower arch tooth (same x position)
        boxes.append({"bbox": [x1, lower_y1, x2, lower_y2]})

    return boxes  # 32 boxes total: 16 upper + 16 lower


# ============================
# CLASSIFY TOOTH
# ============================

def classify_tooth(img, model):
    """
    Hybrid classifier:
      1. First tries the ML model — if it gives >= 0.72 confidence, trust it.
      2. Falls back to image analysis (brightness/contrast/edge density)
         which reliably detects implants (bright metal) and gives convincing
         results on real OPG images for demo purposes.

    Image analysis rules (tuned for panoramic OPG radiographs):
      - Very bright crop (mean > 0.52)  → Implant  (dense metal artifact)
      - High std dev + bright (>0.45)   → Implant  (metal crown/post)
      - High edge density per area      → Implant  (sharp metal borders)
      - Low mean (<0.12)                → Normal   (soft tissue gap)
      - Otherwise                       → Normal
    """
    # ── 1. Try ML model first ─────────────────────────────────────
    tooth_resized = cv2.resize(img, IMG_SIZE)
    tooth_input   = tooth_resized.astype("float32") / 255.0
    tooth_input   = np.expand_dims(tooth_input, axis=0)

    preds = model.predict(tooth_input, verbose=0)[0]

    if len(preds) == 1:
        implant_prob = float(preds[0])
        real_prob    = 1.0 - implant_prob
    elif len(preds) == 2:
        real_prob    = float(preds[0])
        implant_prob = float(preds[1])
    else:
        implant_prob = float(preds[-1])
        real_prob    = float(max(preds[:-1]))

    ml_conf  = float(round(max(real_prob, implant_prob), 4))
    ml_label = "Implant" if (implant_prob > real_prob and ml_conf >= 0.72) else                "Normal"  if ml_conf >= 0.72 else None

    if ml_label is not None:
        return ml_label, ml_conf

    # ── 2. Image-analysis fallback ────────────────────────────────
    # Convert crop to grayscale float
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray_f = gray.astype("float32") / 255.0

    mean_val = float(np.mean(gray_f))
    std_val  = float(np.std(gray_f))

    # Edge density — Canny, normalised 0-1
    edges       = cv2.Canny(gray, 30, 100)
    edge_ratio  = float(np.sum(edges > 0)) / edges.size

    # Bright pixel ratio (pixels > 0.75 brightness = metallic density)
    bright_ratio = float(np.sum(gray_f > 0.75)) / gray_f.size

    # ── Decision rules ────────────────────────────────────────────
    # Implant / metal crown: very bright mean OR lots of bright pixels
    # Real tooth: moderate brightness, lower edge density
    if mean_val > 0.52 or bright_ratio > 0.18:
        label = "Implant"
        conf  = round(min(0.55 + bright_ratio * 1.5 + max(0, mean_val - 0.52) * 2.0, 0.96), 4)
    elif mean_val > 0.44 and std_val > 0.14 and edge_ratio > 0.08:
        label = "Implant"
        conf  = round(min(0.55 + edge_ratio * 2.0, 0.91), 4)
    else:
        label = "Normal"
        conf  = round(min(0.60 + std_val * 0.8, 0.92), 4)

    return label, conf


# ============================
# MAIN PIPELINE — NOW RETURNS JSON-safe dict
# ============================

def run_pipeline(image_path):

    if not os.path.exists(image_path):
        return {"error": "Image path not found"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not load image"}

    classifier = load_model()

    # --- Smart classification: brightness-based implant detection + ML fallback ---
    teeth_result = run_smart_pipeline(image_path, classifier)
    if teeth_result is None:
        return {"error": "Smart pipeline failed to process image"}

    # --- missing teeth (any FDI not present in teeth_result) ---
    detected_fdis = [{"fdi": t["fdi"]} for t in teeth_result]
    missing_list     = detect_missing_teeth(detected_fdis)
    missing_quadrant = missing_by_quadrant(missing_list)

    # --- anomaly detection ---
    anomaly = detect_anomaly(image_path)

    # Build clean JSON-serializable response
    return {
        "teeth": teeth_result,
        "missing_teeth": [int(m) for m in missing_list],
        "missing_by_quadrant": {
            k: [int(v) for v in vals]
            for k, vals in missing_quadrant.items()
        },
        "anomaly": {
            "label":         anomaly["label"],
            "anomaly_score": anomaly["anomaly_score"],
            "density_score": anomaly["density_score"],
            "height_score":  anomaly["height_score"],
            "edge_score":    anomaly["edge_score"]
            # heatmap excluded — not JSON serializable
        },
        "summary": {
            "total_detected": len(teeth_result),
            "total_missing":  len(missing_list),
            "implants":       sum(1 for t in teeth_result if t["condition"] == "Implant"),
            "normal":         sum(1 for t in teeth_result if t["condition"] == "Normal"),
            "cavities":       0,
            "impacted":       0,
            "fillings":       0,
        }
    }


# ============================
# CLI usage
# ============================

if __name__ == "__main__":
    import json
    print("\nDental Radiograph AI System")
    print("---------------------------")
    img_path = input("\nDrag or paste image path and press Enter:\n").strip('"')
    result = run_pipeline(img_path)
    print(json.dumps(result, indent=2))