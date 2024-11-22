from ultralytics import YOLO
import cv2
from PIL import Image
model = YOLO("yolov10n.pt")
source = '/Users/gnanendranaidun/Documents/projects/EL_SEM3/TRIAL_pose/Height-Detection/Tripod_img_1.jpg'
# model.predict(source=source, save=True)
image = cv2.imread(source)
results = model(image)
person_class = 0
# print(results[0].boxes.data)
for result in results[0].boxes.data:
    # if result[5]==person_class:
    #print("person")
    x1, y1, x2, y2 =map(int, result[:4])
    roi_image=image[y1:y2,x1:x2]
    cv2.imwrite("/Users/gnanendranaidun/Documents/projects/EL_SEM3/TRIAL_pose/Height-Detection/images/height.png",roi_image)
    #print(roi_image)
    # roi_pil = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    # cv2.imshow("roi pil",roi_pil)
    print(f"Height is {(y2-y1)*0.178343949}")
    #print(result)