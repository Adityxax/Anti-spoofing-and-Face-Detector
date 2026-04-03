from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # Load the model
    model.train(
        data="Datasets/SplitData/data.yaml",
        epochs=300,
        imgsz=640,
        name="face_detector_run"
    )

if __name__ == "__main__":
    main()
