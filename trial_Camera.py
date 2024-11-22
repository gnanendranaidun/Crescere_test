import cv2

# Replace this with your phone's IP stream URL from the IP Webcam app
ip_address = "http://192.168.108.143:8080/video"  # Example IP; replace with yours

# Create a VideoCapture object
capture = cv2.VideoCapture(ip_address)

# capture.set(cv2.CAP_PROP_FPS, 120)  # Example: Limit to 15 FPS

if not capture.isOpened():
    print("Error: Unable to access the stream. Check the IP address and network connection.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    
    if not ret:
        print("Failed to grab a frame. Exiting...")
        break

    # Display the frame
    cv2.imshow("Phone Camera Stream", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
capture.release()
cv2.destroyAllWindows()
