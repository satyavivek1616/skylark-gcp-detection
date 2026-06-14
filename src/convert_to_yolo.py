import os
import json
import shutil
from sklearn.model_selection import train_test_split

def prepare_yolo_pose_data():
    BASE_POSE_DIR = "/content/yolo_pose_data"
    for split in ['train', 'val']:
        os.makedirs(os.path.join(BASE_POSE_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(BASE_POSE_DIR, "labels", split), exist_ok=True)
        
    JSON_FILE_PATH = "/content/drive/MyDrive/GCP_YOLO_Project/gcp_marks.json"
    IMAGE_FOLDERS = ["/content/data1", "/content/data2"]
    
    disk_file_map = {}
    for folder in IMAGE_FOLDERS:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        disk_file_map[file] = os.path.join(root, file)
                        
    with open(JSON_FILE_PATH, 'r') as f:
        annotations = json.load(f)
        
    parsed_records = []
    CLASS_MAP = {"Cross": 0, "Square": 1, "L-Shape": 2}
    IMG_WIDTH, IMG_HEIGHT, BOX_SIZE = 4096, 2730, 80
    
    for rel_path, meta in annotations.items():
        base_name = os.path.basename(rel_path)
        if base_name in disk_file_map and meta.get("verified_shape") in CLASS_MAP and "mark" in meta:
            src_path = disk_file_map[base_name]
            cid = CLASS_MAP[meta["verified_shape"]]
            x, y = meta["mark"]["x"], meta["mark"]["y"]
            
            pose_line = f"{cid} {x/IMG_WIDTH:.6f} {y/IMG_HEIGHT:.6f} {BOX_SIZE/IMG_WIDTH:.6f} {BOX_SIZE/IMG_HEIGHT:.6f} {x/IMG_WIDTH:.6f} {y/IMG_HEIGHT:.6f} 2\n"
            parsed_records.append({"src": src_path, "rel": rel_path, "line": pose_line})
            
    train_rec, val_rec = train_test_split(parsed_records, test_size=0.2, random_state=42)
    
    def move_split(records, name):
        for item in records:
            flat = item["rel"].replace("/", "_")
            with open(os.path.join(BASE_POSE_DIR, "labels", name, f"{os.path.splitext(flat)[0]}.txt"), "w") as f:
                f.write(item["line"])
            shutil.copy(item["src"], os.path.join(BASE_POSE_DIR, "images", name, flat))
            
    move_split(train_rec, "train")
    move_split(val_rec, "val")
    
    yaml_text = "path: /content/yolo_pose_data\ntrain: images/train\nval: images/val\nkpt_shape: [1, 3]\nnames:\n  0: L-Shape\n  1: Square\n  2: Cross"
    with open("/content/skylark-gcp-detection/data.yaml", "w") as f:
        f.write(yaml_text)
    print("[CONVERT] Staging ready. data.yaml written.")

if __name__ == '__main__':
    prepare_yolo_pose_data()
