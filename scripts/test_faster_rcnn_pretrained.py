# scripts/test_faster_rcnn_pretrained.py
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
import os
import time
from PIL import Image

print("=" * 50)
print("🚗 FASTER R-CNN (ПРЕДОБУЧЕННАЯ НА COCO)")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Устройство: {device}")

# Загрузка модели БЕЗ изменения головы
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
model.to(device)

print(f"✅ Модель загружена (80 классов COCO)")
print(f"📊 Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Трансформации для изображений
transform = weights.transforms()

# Классы COCO для транспортных средств
COCO_VEHICLES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck'
}

# Тестирование на тестовых изображениях
test_dir = "data/test/images"
if os.path.exists(test_dir):
    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:10]
    
    print(f"\n📸 Тестирование на {len(images)} изображениях...")
    print("-" * 50)
    
    total_vehicles = 0
    total_time = 0
    
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        
        # Загрузка изображения через PIL (как требует transform)
        img_pil = Image.open(img_path).convert('RGB')
        
        # Предобработка (теперь правильно)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Инференс
        start = time.time()
        with torch.no_grad():
            predictions = model(img_tensor)
        inference_time = (time.time() - start) * 1000
        total_time += inference_time
        
        # Анализ результатов
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Подсчёт автомобилей (класс 3 = car, 6 = bus, 8 = truck)
        vehicles = []
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5 and label in [3, 6, 8]:
                vehicles.append((label, score))
        
        total_vehicles += len(vehicles)
        
        print(f"   {img_name}: найдено {len(vehicles)} авто/грузовиков (время {inference_time:.0f} мс)")
        
        # Сохраняем результат с отрисовкой (используем OpenCV для визуализации)
        img_cv = cv2.imread(img_path)
        if len(vehicles) > 0:
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5 and label in [3, 6, 8]:
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if label == 3 else (0, 255, 255)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_cv, f"{COCO_VEHICLES.get(label, '?')}: {score:.2f}",
                               (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            os.makedirs("results/faster_rcnn", exist_ok=True)
            cv2.imwrite(f"results/faster_rcnn/{img_name}", img_cv)
    
    avg_time = total_time / len(images) if images else 0
    
    print("-" * 50)
    print(f"\n📊 ИТОГО:")
    print(f"   Всего обнаружено: {total_vehicles} транспортных средств")
    print(f"   Среднее время: {avg_time:.0f} мс на изображение")
    print(f"   Результаты сохранены в: results/faster_rcnn/")

print("\n✅ Тест завершён!")