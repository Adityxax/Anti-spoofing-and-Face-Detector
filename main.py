from ultralytics import YOLO
import cv2
import cvzone

def find_camera(preferred=0, max_index=5, camWidth=640, camHeight=480):
    """
    Tries to find a working camera by iterating through indices.
    """
    for idx in range(max_index):
        current_idx = preferred if idx == 0 else (idx if idx <= preferred else idx)
        if idx > 0 and current_idx == preferred: continue
        
        print(f"🔍 Testing camera index {current_idx}...")
        cap = cv2.VideoCapture(current_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
        
        if cap.isOpened():
            success, img = cap.read()
            if success:
                print(f"✅ Working camera found at index {current_idx}")
                return cap
            else:
                print(f"⚠️ Index {current_idx} is open but failed to provide frames. Releasing...")
                cap.release()
                
    print("❌ No working camera found. Please check your connections or other apps.")
    return None

def main():
    # Load the trained model. Update the path if necessary (e.g. 'runs/detect/face_detector_run/weights/best.pt')
    model = YOLO("models/yolov8n.pt")  # Change to best.pt once trained

    cap = find_camera()
    if cap is None:
        return


    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence
                conf = round(float(box.conf[0]), 2)
                # Class Name
                cls = int(box.cls[0])
                className = model.names[cls]

                cvzone.putTextRect(img, f'{className} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2.imshow("Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
