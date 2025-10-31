from ultralytics import YOLO
import cv2
import cvzone
import time

# Function to find a working camera
def find_working_camera(preferred_index=1, max_index=5):
    print(f"🎯 Trying to open webcam at index {preferred_index}...")
    cap = cv2.VideoCapture(preferred_index, cv2.CAP_DSHOW)  # Use DirectShow
    cap.set(3, 1280)  # Safe width
    cap.set(4, 640)  # Safe height
    if cap.isOpened():
        print(f"✅ Webcam found at index {preferred_index}")
        return cap

    print("⚠ Preferred webcam not accessible. Scanning other indexes...")
    for i in range(max_index):
        if i == preferred_index:
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use correct index
        cap.set(3, 640)
        cap.set(4, 480)
        if cap.isOpened():
            print(f"✅ Fallback webcam found at index {i}")
            return cap
    print("❌ No accessible webcam found.")
    return None

# Try to open camera index 0 first
cap = find_working_camera()

if not cap:
    exit("❌ Exiting: No camera available.")

    #img = cv2.flip(img, 0)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Use the model's built-in class labels
classNames = model.model.names  # ✅ Use model.model.names

prev_frame_time = 0

while True:
    new_frame_time = time.time()

    success, img = cap.read()
    if not success:
        print("❌ Failed to read from camera.")
        break

    img = cv2.flip(img, 1)  # Flip for front camera use

    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # Convert tensor coordinates to int
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)

            conf = round(float(box.conf.item()), 2)
            cls = int(box.cls.item())
            class_name = classNames.get(cls, f"Class {cls}")

            cvzone.putTextRect(img, f'{class_name} {conf}', (x1, y1 - 10), scale=1, thickness=1)

    fps = 1 / (new_frame_time - prev_frame_time + 1e-5)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (50, 50), scale=2, thickness=2)

    cv2.imshow("YOLOv8 Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
