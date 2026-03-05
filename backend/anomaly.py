# anomaly_detection_algorithm.py

import cv2
import numpy as np


IMG_SIZE = (224,224)


# ---------------------------------
# Preprocess Radiograph
# ---------------------------------

def preprocess(img):

    img = cv2.resize(img, IMG_SIZE)

    # reduce noise
    img = cv2.GaussianBlur(img,(5,5),0)

    # improve contrast
    img = cv2.equalizeHist(img)

    img = img.astype("float32")/255.0

    return img


# ---------------------------------
# Bone Density Analysis
# ---------------------------------

def analyze_bone_density(img):

    # bone should appear bright
    mean_intensity = np.mean(img)

    dark_regions = img < (mean_intensity * 0.75)

    density_score = np.sum(dark_regions)/dark_regions.size

    return density_score, dark_regions.astype("float32")


# ---------------------------------
# Bone Crest Height Estimation
# ---------------------------------

def bone_height_loss(img):

    edges = cv2.Canny((img*255).astype("uint8"),40,120)

    h, w = edges.shape

    crest_positions = []

    for col in range(w):

        column = edges[:,col]

        points = np.where(column>0)[0]

        if len(points)>0:

            crest_positions.append(points[0])

    if len(crest_positions)==0:
        return 0

    crest_positions = np.array(crest_positions)

    variation = np.std(crest_positions)

    height_loss_score = variation / h

    return height_loss_score


# ---------------------------------
# Edge Irregularity
# ---------------------------------

def edge_irregularity(img):

    edges = cv2.Canny((img*255).astype("uint8"),50,150)

    # Canny outputs 0 or 255 (not 0 or 1), so divide by 255 to get true 0-1 ratio
    edge_density = np.sum(edges) / (edges.size * 255.0)

    return edge_density


# ---------------------------------
# Main Detection Function
# ---------------------------------

def detect_anomaly(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = preprocess(img)


    density_score, density_map = analyze_bone_density(img)

    height_score = bone_height_loss(img)

    edge_score = edge_irregularity(img)


    # weighted anomaly score
    anomaly_score = (
        0.5 * density_score +
        0.3 * height_score +
        0.2 * edge_score
    )


    # classification
    if anomaly_score < 0.12:
        label = "Normal Bone"

    elif anomaly_score < 0.25:
        label = "Bone Loss"

    else:
        label = "Peri-implantitis"


    # create heatmap
    heatmap = cv2.applyColorMap(
        (density_map*255).astype("uint8"),
        cv2.COLORMAP_JET
    )


    return {
        "anomaly_score": round(float(anomaly_score),5),
        "density_score": round(float(density_score),5),
        "height_score": round(float(height_score),5),
        "edge_score": round(float(edge_score),5),
        "label": label,
        "heatmap": heatmap
    }