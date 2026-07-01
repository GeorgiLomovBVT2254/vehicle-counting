# scripts/train_retinanet.py (ИСПРАВЛЕННЫЙ)
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
import json
import time

print("=" * 50)
print("🚗 RETINANET - ТЕСТИРОВАНИЕ НА ДАННЫХ UA-DETRAC")
print("=" * 50)

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Устройство: {device}")

# Параметры
BATCH_SIZE = 2
CONFIDENCE_THRESHOLD = 0.5

DATA_DIR = "data"
os.makedirs("models/retinanet", exist_ok=True)

# Датасет
class UADETRACDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.images_dir = os.path.join(data_dir, split, 'images')
        self.labels_dir = os.path.join(data_dir, split, 'labels')
        
        self.images = [f for f in os.listdir(self.images_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"📁 Загружено {len(self.images)} изображений для {split}")
        
        self.annotations_json = self._create_coco_annotations()
    
    def _create_coco_annotations(self):
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "car"},
                {"id": 2, "name": "bus"},
                {"id": 3, "name": "van"},
                {"id": 4, "name": "others"}
            ]
        }
        
        annotation_id = 1
        
        for img_idx, img_name in enumerate(self.images):
            img_path = os.path.join(self.images_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            
            coco_data["images"].append({
                "id": img_idx,
                "file_name": img_name,
                "width": width,
                "height": height
            })
            
            txt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            txt_path = os.path.join(self.labels_dir, txt_name)
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            box_width = w * width
                            box_height = h * height
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": img_idx,
                                "category_id": class_id + 1,
                                "bbox": [x, y, box_width, box_height],
                                "area": box_width * box_height,
                                "iscrowd": 0
                            })
                            annotation_id += 1
        
        json_path = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)
        
        print(f"   ✅ JSON сохранён: {json_path} (аннотаций: {annotation_id - 1})")
        return json_path
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            return torch.zeros((3, 224, 224)), {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.int64)}
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        image = image / 255.0
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        with open(self.annotations_json, 'r') as f:
            coco_data = json.load(f)
        
        boxes = []
        labels = []
        
        img_id = None
        for img_info in coco_data["images"]:
            if img_info["file_name"] == img_name:
                img_id = img_info["id"]
                break
        
        if img_id is not None:
            for ann in coco_data["annotations"]:
                if ann["image_id"] == img_id:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"])
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        return image, target

# Загрузка данных
print("\n📁 ЗАГРУЗКА ДАННЫХ...")
train_dataset = UADETRACDataset(DATA_DIR, 'train')
val_dataset = UADETRACDataset(DATA_DIR, 'val')

def collate_fn(batch):
    return tuple(zip(*batch))

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

# Загрузка предобученной модели RetinaNet
print("\n📦 ЗАГРУЗКА МОДЕЛИ RETINANET...")
weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
model = retinanet_resnet50_fpn(weights=weights)
model.eval()
model.to(device)

print(f"✅ Модель RetinaNet загружена (предобучена на COCO)")
print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Классы COCO для транспортных средств
VEHICLE_CLASSES = {3: 'car', 6: 'bus', 8: 'truck'}
VEHICLE_IDS = [3, 6, 8]  # car, bus, truck

# Тестирование модели
print("\n📊 ТЕСТИРОВАНИЕ RETINANET НА ВАЛИДАЦИОННЫХ ДАННЫХ...")

total_vehicles_found = 0
total_inference_time = 0
num_batches = 0
total_detections = 0

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        
        # Инференс
        start_time = time.time()
        predictions = model(images)
        batch_time = (time.time() - start_time) * 1000
        total_inference_time += batch_time
        num_batches += 1
        
        # Подсчёт обнаруженных транспортных средств
        for pred in predictions:
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()
            
            # Фильтруем по уверенности
            mask = pred_scores > CONFIDENCE_THRESHOLD
            
            # Фильтруем по классам (car=3, bus=6, truck=8)
            # Используем обычный цикл вместо .isin()
            vehicle_mask = torch.zeros_like(mask, dtype=torch.bool)
            for i, label in enumerate(pred_labels):
                if label.item() in VEHICLE_IDS:
                    vehicle_mask[i] = True
            
            # Объединяем маски
            final_mask = mask & vehicle_mask
            vehicles_in_image = final_mask.sum().item()
            total_vehicles_found += vehicles_in_image
            total_detections += len(pred_boxes)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"   Обработано {batch_idx + 1}/{len(val_loader)} батчей")

avg_inference_time = total_inference_time / num_batches if num_batches > 0 else 0

print("\n" + "=" * 50)
print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ RETINANET:")
print(f"   Всего обнаружено транспортных средств: {total_vehicles_found}")
print(f"   Всего детекций (все классы): {total_detections}")
print(f"   Среднее время инференса: {avg_inference_time:.1f} мс")
print("=" * 50)
print("✅ RETINANET тестирование завершено!")