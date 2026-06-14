import os
import cv2
from ultralytics import YOLO

def run_sample_inference():
    model_path = "/content/skylark-gcp-detection/weights/best.pt"
    if not os.path.exists(model_path):
        return
    model = YOLO(model_path)
    val_img_dir = "/content/yolo_pose_data/images/val"
    out_dir = "/content/skylark-gcp-detection/outputs/sample_predictions"
    
    if os.path.exists(val_img_dir):
        sample_imgs = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg','.png'))][:5]
        for idx, img_path in enumerate(sample_imgs):
            res = model(img_path, imgsz=1280)
            res[0].save(os.path.join(out_dir, f"sample_prediction_{idx}.jpg"))
    print("[INFERENCE] Saved visual sample verification slates.")

if __name__ == '__main__':
    run_sample_inference()
