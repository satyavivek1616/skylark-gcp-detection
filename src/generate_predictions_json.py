import os
import json
from ultralytics import YOLO

def build_predictions_json():
    model_path = "/content/skylark-gcp-detection/weights/best.pt"
    if not os.path.exists(model_path):
        print("[ERROR] best.pt not found inside weights/")
        return
    model = YOLO(model_path)
    val_img_dir = "/content/yolo_pose_data/images/val"
    
    predictions_output = {}
    CLASS_REV_MAP = {0: "L-Shape", 1: "Square", 2: "Cross"}
    
    if os.path.exists(val_img_dir):
        for img_name in os.listdir(val_img_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(val_img_dir, img_name)
                results = model(full_path, imgsz=1280, verbose=False)
                result = results[0]
                
                # Re-map flat filename back to the original relative JSON key structure
                orig_key = img_name.replace("_", "/", 3) 
                
                if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
                    box = result.boxes[0]
                    cls_id = int(box.cls[0].item())
                    pred_shape = CLASS_REV_MAP.get(cls_id, "L-Shape")
                    
                    kp = result.keypoints.xy[0]
                    if len(kp) > 0:
                        pred_x = float(kp[0][0].item())
                        pred_y = float(kp[0][1].item())
                    else:
                        pred_x, pred_y = 0.0, 0.0
                        
                    predictions_output[orig_key] = {
                        "verified_shape": pred_shape,
                        "mark": {"x": round(pred_x, 2), "y": round(pred_y, 2)}
                    }
                    
    out_file_path = "/content/skylark-gcp-detection/predictions/predictions.json"
    with open(out_file_path, 'w') as f:
        json.dump(predictions_output, f, indent=4)
    print(f"[SUCCESS] Exported standardized submission: {out_file_path}")

if __name__ == '__main__':
    build_predictions_json()
