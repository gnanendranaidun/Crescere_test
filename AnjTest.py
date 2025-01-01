from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import math
from PIL import Image

# Flask App
app = Flask(__name__)

# Constants
SCALING_FACTOR = 0.178343949
depth_map = None
rightco = leftco = []

# Utility Functions
def euclidean_distance(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def two_points_distance(a, b, roi):
    x, y, w, h = roi
    lx, ly = a
    rx, ry = b
    lx_pixel = int(lx * w)
    rx_pixel = int(rx * w)
    ly_pixel = int(ly * h)
    ry_pixel = int(ry * h)
    return euclidean_distance(lx, rx, ly, ry, depth_map[ly_pixel, lx_pixel], depth_map[ry_pixel, rx_pixel])

def get_depth_map(image, token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf", use_auth_token=token)
        model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf", use_auth_token=token)
        encoding = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return prediction.cpu().numpy()
    except Exception as e:
        print(f"Error in depth map generation: {e}")
        return None

# Armspan Calculation Function
def estimate_person_armspan(image, roi, scaling_factor, rightcoordinates, leftcoordinates, depth_map):
    Rlength = Shoulderlength = Llength = 0
    Shoulderlength += two_points_distance(leftcoordinates[0], rightcoordinates[0], roi)
    for i in range(len(leftcoordinates) - 1):
        Llength += two_points_distance(leftcoordinates[i], leftcoordinates[i + 1], roi)
    for i in range(len(rightcoordinates) - 1):
        Rlength += two_points_distance(rightcoordinates[i], rightcoordinates[i + 1], roi)
    return (Shoulderlength + Rlength + Llength) * scaling_factor

# Flask Endpoint for Processing Image
@app.route('/process_image', methods=['POST'])
def process_image():
    global depth_map, rightco, leftco

    # Get the image from the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Load Image
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load YOLO model
    model = YOLO("yolov10n.pt")

    # Detect persons
    results = model(image)
    person_class_id = 0
    person_detected = False
    roi = None

    for result in results[0].boxes.data:
        class_id = int(result[5])
        if class_id == person_class_id:
            person_detected = True
            x1, y1, x2, y2 = map(int, result[:4])
            roi = (x1, y1, x2 - x1, y2 - y1)
            roi_image = image[y1:y2, x1:x2]
            roi_pil = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
            depth_map = get_depth_map(roi_pil)
            if depth_map is None:
                return jsonify({"error": "Failed to compute depth map"}), 500
            rightco = [(0.9, 0.1), (0.8, 0.2)]  # Dummy coordinates; replace with actual pose coordinates
            leftco = [(0.1, 0.9), (0.2, 0.8)]  # Dummy coordinates; replace with actual pose coordinates
            armspan = estimate_person_armspan(image, roi, SCALING_FACTOR, rightco, leftco, depth_map)
            return jsonify({"armspan": armspan}), 200

    if not person_detected:
        return jsonify({"error": "No person detected"}), 404

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
