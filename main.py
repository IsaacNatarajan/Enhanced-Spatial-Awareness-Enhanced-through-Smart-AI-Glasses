from ultralytics import YOLO
import cv2
import math
import pyttsx3

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)  

model = YOLO("yolo-Weights/yolo11n.pt")

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

spoken_objects = set()

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)            
            confidence = math.ceil((box.conf[0] * 100)) / 100          
            cls = int(box.cls[0])
            object_name = classNames[cls]            
            print(f"Detected: {object_name}, Confidence: {confidence}")            
            label = f"{object_name}: {confidence * 100:.1f}%"
            org = (x1, y1 - 10)  
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, label, org, font, fontScale, color, thickness)
            if object_name not in spoken_objects:
                spoken_objects.add(object_name)
                engine.say(f"{object_name} detected")
                engine.runAndWait()
    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
