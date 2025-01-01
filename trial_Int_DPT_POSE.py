import cv2
import mediapipe as mp
import pyttsx3
import time
import math
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()
def pose_extimation(image):
    # cv2.imshow("roi",image)
    img_rgb= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(image,result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        h, w, c = image.shape
        right = []
        left = []
        for id ,co in enumerate(result.pose_landmarks.landmark):
            print(co)
            if(id == 11):
                left.append((co.x,co.y))
            elif(id == 12):
                right.append((co.x,co.y))
            elif(id == 13):
                left.append((co.x,co.y))
            elif(id == 14):
                right.append((co.x,co.y))
            elif(id == 15):
                left.append((co.x,co.y))
            elif(id == 16):
                right.append((co.x,co.y))
            elif(id == 19):
                left.append((co.x,co.y))
            elif(id == 20):
                right.append((co.x,co.y))
        cv2.imshow("results",image)
        return (right,left)
'''
19. left_index
15. left_wrist
13. left_elbow
11. left_shoulder
20. right_index
16. right_wrist
14. right_elbow
12. right_shoulder
'''

# ARM SPAN?
# import cv2
# import mediapipe as mp
# import math

# mpPose = mp.solutions.pose
# mpDraw = mp.solutions.drawing_utils
# pose = mpPose.Pose()

# def calculate_distance(point1, point2):
#     """Calculate Euclidean distance between two 3D points."""
#     return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

# def pose_estimation_and_armspan(image):
#     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     result = pose.process(img_rgb)
#     arm_span = 0

#     if result.pose_landmarks:
#         mpDraw.draw_landmarks(image, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
#         h, w, c = image.shape

#         # Landmarks for left and right arm
#         landmarks = result.pose_landmarks.landmark
#         left_arm_points = [11, 13, 15, 19]  # Shoulder, Elbow, Wrist, Index
#         right_arm_points = [12, 14, 16, 20]  # Shoulder, Elbow, Wrist, Index

#         # Extract 3D coordinates (x, y, z)
#         left_coords = [(landmarks[pt].x * w, landmarks[pt].y * h, landmarks[pt].z) for pt in left_arm_points]
#         right_coords = [(landmarks[pt].x * w, landmarks[pt].y * h, landmarks[pt].z) for pt in right_arm_points]

#         # Calculate distances along each arm
#         left_arm_length = sum(calculate_distance(left_coords[i], left_coords[i+1]) for i in range(len(left_coords) - 1))
#         right_arm_length = sum(calculate_distance(right_coords[i], right_coords[i+1]) for i in range(len(right_coords) - 1))

#         # Total arm span
#         arm_span = left_arm_length + right_arm_length

#     return arm_span

# # Example usage:
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     arm_span = pose_estimation_and_armspan(frame)
#     cv2.putText(frame, f"Arm Span: {arm_span:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow("Pose Estimation", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
