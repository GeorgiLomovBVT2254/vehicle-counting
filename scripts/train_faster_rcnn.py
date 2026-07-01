# scripts/train_faster_rcnn.py
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
import json
import time

print("=" * 50)
print("🚗 FASTER R-CNN - ТЕСТИРОВАНИЕ НА ДАННЫХ UA-DETRAC")
print("=" * 50)

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Устройство: {device}")

# Параметры
BATCH_SIZE = 4
NUM_CLASSES = 5  # 4 класса + background (car, bus, van, others, background)
CONFIDENCE_THRESHOLD = 0.5

DATA_DIR = "data"
os.makedirs("models/faster_rcnn", exist_ok=True)

# Классы
CLASSES = ['car', 'bus', 'van', 'others']
CLASS_MAPPING = {'car': 1, 'bus': 2, 'van': 3, 'others': 4}

# Датасет для Faster R-CNN (нужен COCO-формат JSON)
class UADETRACDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.images_dir = os.path.join(data_dir, split, 'images')
        self.labels_dir = os.path.join(data_dir, split, 'labels')
        
        self.images = [f for f in os.listdir(self.images_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"📁 Загружено {len(self.images)} изображений для {split}")
        
        # Создаём JSON аннотации в формате COCO
        self.annotations_json = self._create_coco_annotations()
    
    def _create_coco_annotations(self):
        """Конвертирует YOLO аннотации в COCO JSON формат"""
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
            # Информация об изображении
            img_path = os.path.join(self.images_dir, img_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            
            coco_data["images"].append({
                "id": img_idx,
                "file_name": img_name,
                "width": width,
                "height": height
            })
            
            # Загрузка YOLO аннотаций
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
                            
                            # Конвертация в x, y, width, height (COCO формат)
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            box_width = w * width
                            box_height = h * height
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": img_idx,
                                "category_id": class_id + 1,  # +1 для background
                                "bbox": [x, y, box_width, box_height],
                                "area": box_width * box_height,
                                "iscrowd": 0
                            })
                            annotation_id += 1
        
        # Сохраняем JSON
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Нормализация
        image = image / 255.0
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        # Загрузка аннотаций из JSON
        with open(self.annotations_json, 'r') as f:
            coco_data = json.load(f)
        
        boxes = []
        labels = []
        
        # Ищем аннотации для этого изображения
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

print(f"\n✅ Загружено {len(train_loader)} батчей для train")
print(f"✅ Загружено {len(val_loader)} батчей для val")

# Загрузка предобученной модели Faster R-CNN
print("\n📦 ЗАГРУЗКА МОДЕЛИ FASTER R-CNN...")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

# Адаптация под количество классов
in_features = model.roi_heads.box_predictor.cls_score.in_features
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.to(device)
model.eval()

print(f"✅ Модель Faster R-CNN загружена (предобучена на COCO)")
print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Тестирование модели на валидационных данных
print("\n📊 ТЕСТИРОВАНИЕ FASTER R-CNN НА ВАЛИДАЦИОННЫХ ДАННЫХ...")

total_boxes = 0
correct_detections = 0
total_inference_time = 0
num_batches = 0

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        
        # Инференс с замером времени
        start_time = time.time()
        predictions = model(images)
        batch_time = (time.time() - start_time) * 1000
        total_inference_time += batch_time
        num_batches += 1
        
        # Оценка
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()
            
            true_boxes = target['boxes']
            true_labels = target['labels']
            
            # Фильтруем по уверенности
            mask = pred_scores > CONFIDENCE_THRESHOLD
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            
            total_boxes += len(true_boxes)
            
            # Проверяем совпадения (IoU > 0.5)
            if len(pred_boxes) > 0 and len(true_boxes) > 0:
                from torchvision.ops import box_iou
                for true_box in true_boxes:
                    ious = box_iou(pred_boxes, true_box.unsqueeze(0))
                    if len(ious) > 0 and ious.max() > 0.5:
                        correct_detections += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"   Обработано {batch_idx + 1}/{len(val_loader)} батчей")

avg_inference_time = total_inference_time / num_batches if num_batches > 0 else 0

print("\n" + "=" * 50)
if total_boxes > 0:
    accuracy = (correct_detections / total_boxes) * 100
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ FASTER R-CNN:")
    print(f"   Всего объектов в валидации: {total_boxes}")
    print(f"   Правильно обнаружено: {correct_detections}")
    print(f"   Точность детекции: {accuracy:.1f}%")
    print(f"   Среднее время инференса: {avg_inference_time:.1f} мс")
else:
    print(f"⚠️ Нет аннотаций для оценки")

print("=" * 50)
print("✅ FASTER R-CNN тестирование завершено!")

# Сохраняем модель
torch.save(model.state_dict(), "models/faster_rcnn/best.pth")
print(f"📁 Модель сохранена в: models/faster_rcnn/best.pth")