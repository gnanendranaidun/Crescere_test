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