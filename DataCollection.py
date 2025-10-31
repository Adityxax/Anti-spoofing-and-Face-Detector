import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import os
from time import time

# ==== Settings ====
classID = 1  # 0: fake, 1: real
outputFolderPath = 'Datasets/DataCollect'
os.makedirs(outputFolderPath, exist_ok=True)
confidence = 0.8
save = True
blurThreshold = 35
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 64
debug = False
flipImage = False  # Flip camera if mirrored

# ==== Find Available Camera ====
def find_camera(preferred=0, max_index=5):
    print(f"🔍 Trying camera index {preferred}...")
    cap = cv2.VideoCapture(preferred)
    cap.set(3, camWidth)
    cap.set(4, camHeight)
    if cap.isOpened():
        print(f"✅ Using camera index {preferred}")
        return cap
    for i in range(max_index):
        if i == preferred:
            continue
        cap = cv2.VideoCapture(i)
        cap.set(3, camWidth)
        cap.set(4, camHeight)
        if cap.isOpened():
            print(f"⚠️ Fallback to camera index {i}")
            return cap
    print("❌ No camera found.")
    return None

cap = find_camera()
if cap is None:
    exit()

detector = FaceDetector()

# ==== Main Loop ====
while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to read frame.")
        continue

    if flipImage:
        img = cv2.flip(img, 1)

    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        print("✅ Face Detected")

        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]
            print(f"🟡 Score: {round(score, 4)}")

            if score < confidence:
                continue

            # Apply offset
            offsetW = int((offsetPercentageW / 100) * w)
            offsetH = int((offsetPercentageH / 100) * h)

            x = max(x - offsetW, 0)
            y = max(y - offsetH * 3, 0)
            w = min(w + offsetW * 2, camWidth - x)
            h = min(h + int(offsetH * 3.5), camHeight - y)

            imgFace = img[y:y + h, x:x + w]
            if imgFace.size == 0:
                continue

            cv2.imshow("imgFace", imgFace)
            blurvalue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
            isClear = blurvalue > blurThreshold
            listBlur.append(isClear)

            # Normalize values
            ih, iw, _ = img.shape
            xc, yc = x + w / 2, y + h / 2
            xcn = round(xc / iw, floatingPoint)
            ycn = round(yc / ih, floatingPoint)
            wn = round(w / iw, floatingPoint)
            hn = round(h / ih, floatingPoint)

            # Clamp to 1.0
            xcn, ycn, wn, hn = map(lambda v: min(v, 1), [xcn, ycn, wn, hn])

            print(f"🔹 Normalized - xcn:{xcn}, ycn:{ycn}, wn:{wn}, hn:{hn}")

            listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn}\n')

            # Draw bounding box and info
            cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}%  Blur: {blurvalue}', (x, y - 20), scale=2, thickness=2)

        if save and listBlur and all(listBlur):
            timestamp = str(time()).replace('.', '')
            filename = f"{outputFolderPath}/{timestamp}"
            cv2.imwrite(f"{filename}.jpg", img)

            with open(f"{filename}.txt", 'w') as f:
                f.writelines(listInfo)

            print(f"📸 Saved: {filename}.jpg")
            print(f"📝 Labels: {listInfo}")
        else:
            print("⚠️ Not saved due to blur or no face.")
    else:
        print("❌ No Face Detected")

    cv2.imshow("Image", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== Cleanup ====
cap.release()
cv2.destroyAllWindows()
