# scripts/train_ssd.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.ops import box_iou
import cv2
import os
import numpy as np

print("=" * 50)
print("🚗 SSD - ТЕСТИРОВАНИЕ ПРЕДОБУЧЕННОЙ МОДЕЛИ")
print("=" * 50)

# Параметры
BATCH_SIZE = 2
NUM_CLASSES = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Устройство: {device}")

DATA_DIR = "data"

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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Парсинг YOLO аннотации
        txt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        txt_path = os.path.join(self.labels_dir, txt_name)
        
        boxes = []
        labels = []
        
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
                        
                        x1 = (x_center - w/2) * width
                        y1 = (y_center - h/2) * height
                        x2 = (x_center + w/2) * width
                        y2 = (y_center + h/2) * height
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id + 1)
        
        # Изменение размера
        image = cv2.resize(image, (320, 320))
        
        # Масштабирование boxes
        scale_x = 320 / width
        scale_y = 320 / height
        boxes = np.array(boxes)
        if len(boxes) > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
        
        # Нормализация
        image = image / 255.0
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        return image, target

# Функция для объединения батчей
def collate_fn(batch):
    return tuple(zip(*batch))

# Загрузка данных
print("\n📁 ЗАГРУЗКА ДАННЫХ...")
train_dataset = UADETRACDataset(DATA_DIR, 'train')
val_dataset = UADETRACDataset(DATA_DIR, 'val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)  # <--- ЭТО БЫЛО ПРОПУЩЕНО

print(f"\n✅ Загружено {len(train_loader)} батчей для train")
print(f"✅ Загружено {len(val_loader)} батчей для val")

# Загрузка модели SSD
print("\n📦 ЗАГРУЗКА МОДЕЛИ SSD...")
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()
model.to(device)

print(f"✅ Модель SSD загружена (предобучена на COCO)")
print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Тестирование модели на валидационных данных
print("\n📊 ТЕСТИРОВАНИЕ SSD НА ВАЛИДАЦИОННЫХ ДАННЫХ...")

total_boxes = 0
correct_detections = 0

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        
        # Инференс
        predictions = model(images)
        
        # Оценка
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()
            
            true_boxes = target['boxes']
            true_labels = target['labels']
            
            # Фильтруем по уверенности
            mask = pred_scores > 0.5
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            
            total_boxes += len(true_boxes)
            
            # Проверяем совпадения
            if len(pred_boxes) > 0 and len(true_boxes) > 0:
                for true_box in true_boxes:
                    ious = box_iou(pred_boxes, true_box.unsqueeze(0))
                    if len(ious) > 0 and ious.max() > 0.5:
                        correct_detections += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"   Обработано {batch_idx + 1}/{len(val_loader)} батчей")

print("\n" + "=" * 50)
if total_boxes > 0:
    accuracy = (correct_detections / total_boxes) * 100
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ SSD:")
    print(f"   Всего объектов в валидации: {total_boxes}")
    print(f"   Правильно обнаружено: {correct_detections}")
    print(f"   Точность детекции: {accuracy:.1f}%")
else:
    print(f"⚠️ Нет аннотаций для оценки")

print("=" * 50)
print("✅ SSD тестирование завершено!")