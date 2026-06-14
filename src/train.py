from ultralytics import YOLO
def run_training():
    model = YOLO("yolov8n-pose.pt")
    model.train(
        data="/content/skylark-gcp-detection/data.yaml",
        epochs=30,
        imgsz=1280,
        batch=8,
        device="0",
        project="/content/drive/MyDrive/GCP_YOLO_Project",
        name="gcp_keypoint_pose_run"
    )
if __name__ == '__main__':
    run_training()
