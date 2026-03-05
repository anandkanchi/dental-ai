import numpy as np
import cv2

# ============================
# Expected FDI adult dentition
# ============================

EXPECTED_FDI = [
    11,12,13,14,15,16,17,18,
    21,22,23,24,25,26,27,28,
    31,32,33,34,35,36,37,38,
    41,42,43,44,45,46,47,48
]


# ============================
# Tooth Numbering Engine
# ============================

class ToothNumberingEngine:

    def compute_centers(self, teeth):

        for t in teeth:

            x1, y1, x2, y2 = t["bbox"]

            t["center_x"] = (x1 + x2) / 2
            t["center_y"] = (y1 + y2) / 2

        return teeth


    def split_upper_lower(self, teeth):

        y_vals = [t["center_y"] for t in teeth]

        mid_y = np.mean(y_vals)

        upper = []
        lower = []

        for t in teeth:

            if t["center_y"] < mid_y:
                upper.append(t)

            else:
                lower.append(t)

        return upper, lower


    def assign_upper(self, upper):

        upper_sorted = sorted(upper, key=lambda x: x["center_x"])

        mid = len(upper_sorted) // 2

        left = upper_sorted[:mid]
        right = upper_sorted[mid:]

        # Upper Left (21–28)

        for i, t in enumerate(left):

            t["fdi"] = 21 + i


        # Upper Right (18–11)

        for i, t in enumerate(reversed(right)):

            t["fdi"] = 11 + i


        return upper_sorted


    def assign_lower(self, lower):

        lower_sorted = sorted(lower, key=lambda x: x["center_x"])

        mid = len(lower_sorted) // 2

        left = lower_sorted[:mid]
        right = lower_sorted[mid:]


        # Lower Left (31–38)

        for i, t in enumerate(left):

            t["fdi"] = 31 + i


        # Lower Right (48–41)

        for i, t in enumerate(reversed(right)):

            t["fdi"] = 41 + i


        return lower_sorted


    def run(self, teeth):

        teeth = self.compute_centers(teeth)

        upper, lower = self.split_upper_lower(teeth)

        upper = self.assign_upper(upper)
        lower = self.assign_lower(lower)

        return upper + lower


# ============================
# Missing Tooth Detection
# ============================

def detect_missing_teeth(numbered_teeth):

    detected = {t["fdi"] for t in numbered_teeth}

    missing = sorted(set(EXPECTED_FDI) - detected)

    return missing


# ============================
# Missing Teeth by Quadrant
# ============================

def missing_by_quadrant(missing):

    result = {
        "upper_right": [],
        "upper_left": [],
        "lower_left": [],
        "lower_right": []
    }

    for t in missing:

        if 11 <= t <= 18:
            result["upper_right"].append(t)

        elif 21 <= t <= 28:
            result["upper_left"].append(t)

        elif 31 <= t <= 38:
            result["lower_left"].append(t)

        elif 41 <= t <= 48:
            result["lower_right"].append(t)

    return result


# ============================
# Impacted Tooth Extraction
# ============================

def get_impacted_teeth(teeth):

    impacted = []

    for t in teeth:

        if t.get("class") == "impacted":

            impacted.append(t["fdi"])

    return impacted


# ============================
# Visualization
# ============================

def draw_results(image, teeth):

    img = image.copy()

    for t in teeth:

        x1, y1, x2, y2 = t["bbox"]

        cx = int(t["center_x"])
        cy = int(t["center_y"])

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            img,
            str(t["fdi"]),
            (cx,cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    return img


# ============================
# Example Pipeline
# ============================

def run_pipeline(image_path, detected_boxes):

    """
    detected_boxes example:

    [
        {"bbox":[x1,y1,x2,y2]},
        {"bbox":[x1,y1,x2,y2]},
    ]
    """

    image = cv2.imread(image_path)

    engine = ToothNumberingEngine()

    numbered_teeth = engine.run(detected_boxes)

    missing = detect_missing_teeth(numbered_teeth)

    quadrant_missing = missing_by_quadrant(missing)

    impacted = get_impacted_teeth(numbered_teeth)

    vis = draw_results(image, numbered_teeth)

    return {

        "numbered_teeth": numbered_teeth,
        "missing_teeth": missing,
        "missing_by_quadrant": quadrant_missing,
        "impacted_teeth": impacted,
        "visualization": vis

    }