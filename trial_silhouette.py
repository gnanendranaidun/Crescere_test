# import cv2
# def find_outline(image):
#     # Load the image

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Canny edge detection
#     edges = cv2.Canny(gray, threshold1=50, threshold2=150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours on a black background
#     outline = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

#     # Display the outline
#     cv2.imshow("Outline", outline)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# import cv2
# import numpy as np
# def find_outline1(image):
#     # Load the image


#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply edge detection (Canny Edge Detection)
#     edges = cv2.Canny(gray, 50, 150)

#     # Create a blank (black) image with the same dimensions as the original
#     outline_image = np.zeros_like(image)

#     # Draw the edges (outline) on the blank image in white
#     outline_image[edges > 0] = (255, 255, 255)  # White outline

#     # Save the new outline image
#     cv2.imwrite('outline_only.png', outline_image)

#     # Display the new outline image
#     cv2.imshow("Outline", outline_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
import cv2
import numpy as np
import mediapipe as mp

def find_outline1(image):
    # Initialize Mediapipe segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Convert the image to RGB (Mediapipe expects RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply Mediapipe Selfie Segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentation:
        results = segmentation.process(image_rgb)

    # Create a binary mask where the person is segmented
    # The segmentation mask returns values between 0 and 1, so we threshold it to get a binary mask (0 or 255)
    mask = (results.segmentation_mask > 0.2).astype(np.uint8) * 255  # Binary mask for the person

    # Find contours of the mask (this will give us the outline of the person)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image with the same dimensions as the original, but with transparency (RGBA)
    outline_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # Draw the contours (outline) on the blank image in green (or any other color you'd like)
    cv2.drawContours(outline_image, contours, -1, (0, 255, 0, 255), thickness=2)  # Green outline with full opacity

    # Save the outline as a separate PNG with transparency
    cv2.imwrite('outline_only.png', outline_image)

    # Display the result (outline of the person)
    cv2.imshow('Outline Only', outline_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()