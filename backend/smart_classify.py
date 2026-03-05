"""
Smart OPG tooth classification using full-image bright-region detection.
Finds actual bright (metallic) regions in the X-ray and maps them to FDI numbers.
"""
import cv2
import numpy as np


# Valid FDI layout — left to right as they appear in OPG image
# Upper arch: image-left is patient-right (Q1), image-right is patient-left (Q2)
FDI_UPPER = [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28]
FDI_LOWER = [48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]

EXPECTED_FDI = set(FDI_UPPER + FDI_LOWER)


def find_implants_by_brightness(image_path):
    """
    Detect implants using bright-region analysis on the full OPG image.
    Returns list of (fdi, confidence) tuples for detected implants.

    Implants appear as the brightest objects in a dental X-ray (dense metal).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape

    # ── Preprocessing ──────────────────────────────────────────────
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    norm    = blurred.astype('float32') / 255.0

    # ── Focus on tooth arch region (ignore skull/spine) ────────────
    # Typical OPG: arches occupy roughly y: 25%-85%, x: 5%-95%
    roi_y1 = int(h * 0.25)
    roi_y2 = int(h * 0.85)
    roi_x1 = int(w * 0.05)
    roi_x2 = int(w * 0.95)
    roi = norm[roi_y1:roi_y2, roi_x1:roi_x2]

    # ── Find very bright blobs (implants/metal crowns) ─────────────
    # Use adaptive threshold: anything brighter than mean + 1.8*std
    mean_v = np.mean(roi)
    std_v  = np.std(roi)
    bright_thresh = min(mean_v + 1.8 * std_v, 0.85)

    bright_mask = (roi > bright_thresh).astype(np.uint8) * 255

    # Morphological close to merge nearby bright pixels into blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed,      cv2.MORPH_OPEN,  kernel)

    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    implant_regions = []
    min_area = (h * w) * 0.0005   # at least 0.05% of image = real structure

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # Convert back to full-image coordinates
        cx = (x + bw // 2) + roi_x1
        cy = (y + bh // 2) + roi_y1

        # Calculate brightness of this region
        region = norm[y + roi_y1 : y + roi_y1 + bh, x + roi_x1 : x + roi_x1 + bw]
        region_mean = float(np.mean(region))

        implant_regions.append({
            'cx': cx, 'cy': cy,
            'area': area,
            'brightness': region_mean,
            'bbox_full': [x + roi_x1, y + roi_y1, x + roi_x1 + bw, y + roi_y1 + bh]
        })

    # ── Map bright blobs to FDI numbers ────────────────────────────
    arch_mid_y = int(h * 0.52)   # approximate centre line between arches

    detected_implants = []
    for reg in implant_regions:
        cx, cy = reg['cx'], reg['cy']

        # Which arch?
        if cy < arch_mid_y:
            fdi_row = FDI_UPPER
        else:
            fdi_row = FDI_LOWER

        # Map x position to FDI index (0 = leftmost = image-left)
        x_frac  = (cx - roi_x1) / (roi_x2 - roi_x1)
        x_frac  = max(0.0, min(1.0, x_frac))
        fdi_idx = int(round(x_frac * (len(fdi_row) - 1)))
        fdi     = fdi_row[fdi_idx]

        # Confidence = scaled brightness
        conf = round(min(0.55 + (reg['brightness'] - bright_thresh) * 4.0, 0.97), 4)

        detected_implants.append((fdi, conf))

    return detected_implants


def run_smart_pipeline(image_path, model):
    """
    Full classification pipeline using smart implant detection.
    Returns all 32 teeth with conditions, using brightness analysis for implants
    and the ML model as a secondary signal.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]

    # Step 1: Detect actual implants by brightness
    implant_hits = find_implants_by_brightness(image_path)
    implant_fdi_map = {fdi: conf for fdi, conf in implant_hits}

    # Step 2: Build per-FDI results
    # Generate geometric boxes (for bbox coordinates only, not for classification)
    upper_y1 = int(h * 0.15)
    upper_y2 = int(h * 0.50)
    lower_y1 = int(h * 0.50)
    lower_y2 = int(h * 0.85)
    tooth_width = w // 16

    # Build bbox map per FDI
    bbox_map = {}
    for i, fdi in enumerate(FDI_UPPER):
        x1 = (i % 16) * tooth_width
        bbox_map[fdi] = [x1, upper_y1, x1 + tooth_width, upper_y2]
    for i, fdi in enumerate(FDI_LOWER):
        x1 = (i % 16) * tooth_width
        bbox_map[fdi] = [x1, lower_y1, x1 + tooth_width, lower_y2]

    # Step 3: Classify each tooth
    IMG_SIZE = (224, 224)
    teeth_result = []

    for fdi in sorted(EXPECTED_FDI):
        bbox = bbox_map.get(fdi, [0, 0, 50, 50])
        x1, y1, x2, y2 = bbox
        crop = img_bgr[y1:y2, x1:x2]

        if fdi in implant_fdi_map:
            # Bright region detected → Implant
            condition = 'Implant'
            confidence = implant_fdi_map[fdi]
        elif crop.size > 0 and model is not None:
            # Try ML model
            tooth_r = cv2.resize(crop, IMG_SIZE).astype('float32') / 255.0
            preds   = model.predict(np.expand_dims(tooth_r, 0), verbose=0)[0]
            if len(preds) >= 2:
                implant_p = float(preds[-1])
                real_p    = float(preds[0])
            else:
                implant_p = float(preds[0])
                real_p    = 1.0 - implant_p
            ml_conf = max(implant_p, real_p)
            if ml_conf >= 0.72 and implant_p > real_p:
                condition, confidence = 'Implant', round(ml_conf, 4)
            else:
                condition, confidence = 'Normal', round(min(0.62 + real_p * 0.3, 0.94), 4)
        else:
            condition, confidence = 'Normal', 0.65

        teeth_result.append({
            'fdi':        int(fdi),
            'condition':  condition,
            'confidence': float(confidence),
            'bbox':       [int(v) for v in bbox]
        })

    return teeth_result
