# scripts/compare_yolo_ssd.py
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import os  
import cv2
import torch
import time


print("=" * 50)
print("🚗 СРАВНЕНИЕ YOLOv8 vs SSD")
print("=" * 50)

# Загрузка моделей
print("\n📦 Загрузка YOLOv8...")
yolo_model = YOLO('yolov8n.pt')

print("📦 Загрузка SSD...")
ssd_weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
ssd_model = ssdlite320_mobilenet_v3_large(weights=ssd_weights)
ssd_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssd_model = ssd_model.to(device)

# Тестовое изображение
test_img_path = "data/test/images"
if os.path.exists(test_img_path):
    images = [f for f in os.listdir(test_img_path) if f.endswith('.jpg')]
    if images:
        img_path = os.path.join(test_img_path, images[0])
        img = cv2.imread(img_path)
        
        print(f"\n📸 Тестовое изображение: {images[0]}")
        
        # YOLO тест
        print("\n🔍 YOLOv8:")
        start = time.time()
        yolo_results = yolo_model(img)
        yolo_time = (time.time() - start) * 1000
        
        cars_yolo = 0
        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
                    if int(box.cls[0]) == 2:
                        cars_yolo += 1
        
        print(f"   Автомобилей: {cars_yolo}")
        print(f"   Время: {yolo_time:.1f} мс")
        
        # SSD тест
        print("\n🔍 SSD:")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (320, 320))
        img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        start = time.time()
        with torch.no_grad():
            ssd_results = ssd_model(img_tensor)
        ssd_time = (time.time() - start) * 1000
        
        # Подсчёт автомобилей (класс 2 в COCO)
        pred = ssd_results[0]
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        cars_ssd = sum(1 for label, score in zip(labels, scores) if label == 2 and score > 0.5)
        print(f"   Автомобилей: {cars_ssd}")
        print(f"   Время: {ssd_time:.1f} мс")
        
        # Итоговое сравнение
        print("\n" + "=" * 50)
        print("📊 ИТОГОВОЕ СРАВНЕНИЕ:")
        print(f"   YOLOv8: {cars_yolo} машин, {yolo_time:.1f} мс")
        print(f"   SSD:    {cars_ssd} машин, {ssd_time:.1f} мс")
        print("=" * 50)

print("\n✅ Сравнение завершено!")