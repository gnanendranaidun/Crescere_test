from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image

# Replace 'your_huggingface_token' with your actual Hugging Face token if required
huggingface_token = 'YOUR_HUGGINGFACE_TOKEN'

# Fixed scaling factor
SCALING_FACTOR = 0.95
depth_map = None
rightco = leftco = []
def two_points_distance(a,b,roi):
    x, y, w, h = roi
    w-=1
    h-=1
    lx,ly = a
    rx,ry = b
    lx_pixel = int(lx*w)
    rx_pixel = int(rx*w)
    ly_pixel = int(ly*h)
    ry_pixel = int(ry*h)
    return euclidean_distance(lx,rx,ly,ry,depth_map[ly_pixel,lx_pixel],depth_map[ry_pixel,rx_pixel])
def euclidean_distance(x1,x2,y1,y2,z1,z2):
    return abs(z1-z2)
def get_depth_map(image, token=None):
    try:
        print("Loading image processor and model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf", use_auth_token=token)
        model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf", use_auth_token=token)

        print("Preparing image for the model...")
        encoding = image_processor(images=image, return_tensors="pt").to(device)
        print(f"Image tensor shape: {encoding['pixel_values'].shape}")

        print("Performing forward pass to get predicted depth...")
        with torch.no_grad():
            outputs = model(**encoding)
            predicted_depth = outputs.predicted_depth
            print(f"Predicted depth shape: {predicted_depth.shape}")

        print("Interpolating to original size...")
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map_np = prediction.cpu().numpy()
        print(f"Depth map shape: {depth_map_np.shape}")
        print(f"Depth map (sample values): {depth_map_np[::100, ::100]}")  # Print every 100th value
        print(f"Depth map min: {depth_map_np.min()}, max: {depth_map_np.max()}")

        return depth_map_np
    except Exception as e:
        print(f"Error in get_depth_map: {e}")
        return None

def estimate_person_armspan(image, roi, scaling_factor, rightcoordinates,leftcoordinates, depth_map,token=None):
    Rlength = Shoulderlength = Llength=0
    Shoulderlength += two_points_distance(leftcoordinates[0],rightcoordinates[0],roi)
    Llength+=two_points_distance(leftcoordinates[0],leftcoordinates[1],roi)
    Llength+=two_points_distance(leftcoordinates[1],leftcoordinates[2],roi)
    Llength+=two_points_distance(leftcoordinates[2],leftcoordinates[3],roi)
    Rlength+=two_points_distance(rightcoordinates[0],rightcoordinates[1],roi)
    Rlength+=two_points_distance(rightcoordinates[1],rightcoordinates[2],roi)
    Rlength+=two_points_distance(rightcoordinates[2],rightcoordinates[3],roi)
    print("right arm length",Rlength)
    print("left arm length",Llength)
    return (Shoulderlength+Rlength+Llength)*scaling_factor

def estimate_closest_distance(image, roi, scaling_factor, depth_map,token=None):
    try:
        # Extract ROI from the image

        if depth_map is None:
            print("No depth map returned.")
            return 0,0

        # Ensure no negative depth values
        depth_map[depth_map < 0] = 0

        # Find the minimum depth value in the depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        print(f"Depth map min: {depth_min}, max: {depth_max}")

        # Apply the fixed scaling factor to the minimum depth value
        closest_distance = depth_min * scaling_factor
        farthest_distance = depth_max * scaling_factor
        print(f"Estimated closest distance to object: {closest_distance} meters")
        print(f"Estimated farthest distance to object: {farthest_distance} meters")

        # Convert depth map to distances using the scaling factor
        distances = depth_map * scaling_factor

        # Print some values from the distances map for debugging
        print(f"Distances (sample values): {distances[::100, ::100]}")  # Print every 100th value

        # Display the depth map and distances map for ROI
        plt.subplot(1, 2, 1)
        plt.imshow(depth_map, cmap='gray')
        plt.colorbar(label='Depth')
        plt.title('Depth Map (ROI)')

        plt.subplot(1, 2, 2)
        plt.imshow(distances, cmap='inferno')
        plt.colorbar(label='Distance (meters)')
        plt.title('Distance Map (ROI)')
        plt.show()

        return closest_distance, farthest_distance
    except Exception as e:
        print(f"Error in estimate_closest_distance: {e}")
    return None, None

def estimate_person_height(image, roi, scaling_factor, depth_map,token=None):
    try:
        if depth_map is None:
            print("No depth map returned.")
            return 0
        x, y, w, h = roi
        # Ensure no negative depth values
        depth_map[depth_map < 0] = 0
        # Find the depth at the top and bottom of the bounding box
        top_depth = depth_map[0, w // 2]  # Depth at the top (center of the top edge)
        bottom_depth = depth_map[h-1, w // 2]  # Depth at the bottom (center of the bottom edge)

        print(f"Top depth: {top_depth}, Bottom depth: {bottom_depth}")

        # Calculate the height in pixels (the height of the bounding box in the image)
        height_in_pixels = h

        # Calculate the actual height in meters (using scaling factor)
        height_in_meters=euclidean_distance(0,h-1,w//2,w//2,bottom_depth,top_depth)*scaling_factor

        print(f"Estimated height of person: {height_in_meters:.2f} meters")

        return height_in_meters

    except Exception as e:
        print(f"Error in estimate_person_height: {e}")
        return None

def find(path):
# Load the YOLOv8 model
    global depth_map
    global SCALING_FACTOR
    global rightco
    global leftco
    model = YOLO("yolov10n.pt")  # Load a pretrained YOLOv8 model

# Load an image
    image_path = path # Replace with the path to your image
    image = cv2.imread(image_path)

# Run inference
    results = model(image)

# Filter results to show only persons
    person_class_id = 0  # COCO class ID for person

# Initialize a flag to check if any person is detected
    person_detected = False
    roi = None

# Loop through results and filter for persons
    for result in results[0].boxes.data:
        class_id = int(result[5])
        if class_id == person_class_id:
            person_detected = True
            x1, y1, x2, y2 = map(int, result[:4])
            #Region of Interest
            #cv2.imshow("roi",image[y1:y2, x1:x2])
            from trial_Int_DPT_POSE import pose_extimation
            rightco,leftco=pose_extimation(image[y1:y2, x1:x2])
            from trial_silhouette import find_outline1
            find_outline1(image[y1:y2, x1:x2])
            roi = (x1, y1, x2 - x1, y2 - y1) 
            confidence = result[4].item()
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, f"Person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # If persons are detected, display the image with bounding boxes
    if person_detected:
        # Display the image
        # image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read the image at {image_path}. Please check the path.")
            exit()

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Persons')
        plt.axis('off')
        plt.show()

        # Perform depth estimation on the ROI
        if roi:
            print("Starting depth estimation on the ROI...")
            x, y, w, h = roi
            print("this is roi",roi)
            roi_image = image[y:y+h, x:x+w]
            roi_pil = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
            depth_map = get_depth_map(roi_pil)
            closest_distance, farthest_distance = estimate_closest_distance(image, roi, SCALING_FACTOR, token=huggingface_token,depth_map=depth_map)
            if closest_distance is not None:
                print(f"Estimated closest distance to object: {closest_distance} meters")
                print(f"Estimated farthest distance to object: {farthest_distance} meters")
            height = estimate_person_height(image, roi, SCALING_FACTOR, token=huggingface_token,depth_map=depth_map)
            armspan = estimate_person_armspan(image,roi, SCALING_FACTOR,leftcoordinates=leftco,rightcoordinates=rightco,token= huggingface_token,depth_map=depth_map)
            print(f"HEIGHT IS : {height*100} Cms for SCALING FACTOR : {SCALING_FACTOR} and roi {roi}")
    else:
        print("No person detected in the image.")

# find("/Users/gnanendranaidun/Documents/projects/EL_SEM3/TRIAL_pose/Height-Detection/Ref_image9.jpg")
# print("next")
find("/Users/gnanendranaidun/Documents/projects/EL_SEM3/TRIAL_pose/Height-Detection/Ref_image5.jpg")




    