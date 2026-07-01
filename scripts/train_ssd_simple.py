# scripts/train_ssd_simple.py
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import cv2
import os
import numpy as np
from PIL import Image

print("=" * 50)
print("🚗 SSD - ТЕСТИРОВАНИЕ НА ДАННЫХ UA-DETRAC")
print("=" * 50)

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Устройство: {device}")

# Загрузка предобученной модели SSD (уже обучена на COCO)
print("\n📦 Загрузка предобученной модели SSD...")
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model = model.to(device)
model.eval()

print(f"✅ Модель SSD загружена (предобучена на COCO)")
print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Классы COCO (интересующие нас транспортные средства)
COCO_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck'
}

# Тестирование на изображениях из вашего датасета
def test_on_image(image_path, model, device):
    # Загрузка изображения
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Предобработка
    img_resized = cv2.resize(img_rgb, (320, 320))
    img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        predictions = model(img_tensor)
    
    return predictions, img

# Тестирование на нескольких изображениях
test_images_dir = "data/test/images"
if os.path.exists(test_images_dir):
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]
    print(f"\n📸 Найдено {len(test_images)} тестовых изображений")
    
    # Проверяем первые 5 изображений
    vehicles_found = []
    for img_name in test_images[:5]:
        img_path = os.path.join(test_images_dir, img_name)
        predictions, img = test_on_image(img_path, model, device)
        
        # Подсчёт транспортных средств
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        vehicles = []
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5 and label in COCO_CLASSES:
                vehicles.append((COCO_CLASSES[label], score))
        
        vehicles_found.append(len(vehicles))
        print(f"   {img_name}: {len(vehicles)} транспортных средств")
        
        # Визуализация (опционально)
        if len(vehicles) > 0:
            img_with_boxes = img.copy()
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5 and label in COCO_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{COCO_CLASSES[label]}: {score:.2f}", 
                               (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Сохраняем результат
            os.makedirs("results/ssd_test", exist_ok=True)
            cv2.imwrite(f"results/ssd_test/ssd_{img_name}", img_with_boxes)
            print(f"      → Результат сохранён в results/ssd_test/")
    
    total_vehicles = sum(vehicles_found)
    print(f"\n📊 Всего обнаружено транспортных средств: {total_vehicles}")
else:
    print(f"⚠️ Папка {test_images_dir} не найдена")

print("\n" + "=" * 50)
print("✅ SSD тестирование завершено!")
print("📁 Результаты в папке: results/ssd_test/")
print("=" * 50)