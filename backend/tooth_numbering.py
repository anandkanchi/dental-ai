import numpy as np


# Valid FDI numbers per quadrant — max 8 teeth each
FDI_UPPER_RIGHT = [18, 17, 16, 15, 14, 13, 12, 11]  # right→centre
FDI_UPPER_LEFT  = [21, 22, 23, 24, 25, 26, 27, 28]  # centre→left
FDI_LOWER_LEFT  = [31, 32, 33, 34, 35, 36, 37, 38]  # centre→left
FDI_LOWER_RIGHT = [48, 47, 46, 45, 44, 43, 42, 41]  # left→centre


class ToothNumberingEngine:

    def compute_centers(self, teeth):
        for t in teeth:
            x1, y1, x2, y2 = t["bbox"]
            t["center_x"] = (x1 + x2) / 2
            t["center_y"] = (y1 + y2) / 2
        return teeth

    def split_upper_lower(self, teeth):
        y_coords = [t["center_y"] for t in teeth]
        mid_y = np.mean(y_coords)
        upper = [t for t in teeth if t["center_y"] < mid_y]
        lower = [t for t in teeth if t["center_y"] >= mid_y]
        return upper, lower

    def assign_fdi_upper(self, upper):
        """
        Assign FDI to upper jaw teeth.
        Split at midpoint: left half → Q2 (21-28), right half → Q1 (11-18).
        Capped at 8 teeth per side (valid FDI range).
        """
        upper_sorted = sorted(upper, key=lambda x: x["center_x"])
        mid = len(upper_sorted) // 2

        left  = upper_sorted[:mid]   # patient's left  → FDI Q2: 21,22..28
        right = upper_sorted[mid:]   # patient's right → FDI Q1: 11,12..18

        # Assign Q2 left-to-right: 21, 22, ... (cap at 8)
        for i, t in enumerate(left[:8]):
            t["fdi"] = FDI_UPPER_LEFT[i]

        # Assign Q1 right-to-left (closest to centre first): 11, 12, ...
        for i, t in enumerate(reversed(right[-8:])):
            t["fdi"] = FDI_UPPER_RIGHT[7 - i]   # 11 closest to midline

        return upper_sorted

    def assign_fdi_lower(self, lower):
        """
        Assign FDI to lower jaw teeth.
        Split at midpoint: left half → Q3 (31-38), right half → Q4 (41-48).
        Capped at 8 teeth per side (valid FDI range).
        """
        lower_sorted = sorted(lower, key=lambda x: x["center_x"])
        mid = len(lower_sorted) // 2

        left  = lower_sorted[:mid]   # patient's left  → FDI Q3: 31,32..38
        right = lower_sorted[mid:]   # patient's right → FDI Q4: 41,42..48

        # Assign Q3 left-to-right: 31, 32, ...
        for i, t in enumerate(left[:8]):
            t["fdi"] = FDI_LOWER_LEFT[i]

        # Assign Q4 right-to-left (closest to centre first): 41, 42, ...
        for i, t in enumerate(reversed(right[-8:])):
            t["fdi"] = FDI_LOWER_RIGHT[7 - i]

        return lower_sorted

    def run(self, teeth):
        teeth = self.compute_centers(teeth)
        upper, lower = self.split_upper_lower(teeth)
        upper = self.assign_fdi_upper(upper)
        lower = self.assign_fdi_lower(lower)
        return upper + lower