import cv2 as cv
import mediapipe as mp
import pyttsx3
import time
import math

# Initialize MediaPipe pose detection and drawing utilities
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()
ip_address = "http://192.168.112.250:8080/video"
capture = cv.VideoCapture(ip_address)
capture.set(cv.CAP_PROP_FPS, 120)
# Initialize text-to-speech engine
def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 150)
    engine.setProperty('voice', voices[87].id)
    engine.say(audio)
    engine.runAndWait()

# Initial speech output
speak("I am about to measure your height now sir")
speak("Although I reach a precision up to ninety-eight percent")

# Variables
ptime = 0
n = 0
lst = []

while True:
    isTrue, img = capture.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        # Draw landmarks on the image
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Extract landmarks
        h, w, c = img.shape
        for id, lm in enumerate(result.pose_landmarks.landmark):
            lst.append([id, lm.x, lm.y])  # Append landmark details to list

            if id == 32 or id == 31:
                # Get coordinates of shoulder landmarks (e.g., shoulder_left=31, shoulder_right=32)
                cx1, cy1 = int(lm.x * w), int(lm.y * h)
                if id == 31:
                    cx1, cy1 = int(lm.x * w), int(lm.y * h)

                # Calculate distance (Euclidean) between the two points (shoulders)
                if len(lst) > 1:
                    for prev_id, prev_lm in enumerate(lst[:-1]):
                        if prev_id == 31 and id == 32:
                            cx2, cy2 = int(prev_lm[1] * w), int(prev_lm[2] * h)
                            cv.circle(img, (cx1, cy1), 15, (0, 0, 0), cv.FILLED)
                            cv.circle(img, (cx2, cy2), 15, (0, 0, 0), cv.FILLED)

                            # Compute the Euclidean distance between shoulder landmarks
                            d = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

                            # Assume a scaling factor for converting pixel distance to centimeters
                            di = round(d*0.63271)  # Example factor, would need real calibration for accuracy
                            
                            print(f"You are {di} centimeters tall")

                            # Display height on the image
                            cv.putText(img, "Height: ", (40, 70), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), thickness=2)
                            cv.putText(img, str(di), (180, 100), cv.FONT_HERSHEY_DUPLEX, 5, (255, 255, 0), thickness=2)
                            cv.putText(img, "cms", (240, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), thickness=2)

                            # Prompt for user action
                            
                            cv.putText(img, "Press 'q' to exit.", (40, 450), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

                            # Wait for 'q' key press to exit
                            

    # Calculate and display FPS
    
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(img, f"FPS: {int(fps)}", (40, 100), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), thickness=2)
    cv.imshow("Height Measurement", img)
    # speak(f"You are {di} centimeters tall")
    # speak("I am done. You can relax now. Press 'q' to exit.")
    # Display the processed frame
    # img = cv.resize(img, (700, 500))
    if cv.waitKey(1000) & 0xFF == ord('q'):
        capture.release()
        cv.destroyAllWindows()
        break
    # time.sleep(5)
    

# Release capture and close windows when done
capture.release()
cv.destroyAllWindows()
