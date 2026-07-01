# scripts/train_yolo.py
from ultralytics import YOLO
import os

print("=" * 50)
print("🚗 ОБУЧЕНИЕ YOLOv8 НА ДАННЫХ UA-DETRAC")
print("=" * 50)

# Загрузка предобученной модели
model = YOLO('yolov8n.pt')

# Путь к конфигу
data_yaml = 'data.yaml'

# Проверка наличия data.yaml
if not os.path.exists(data_yaml):
    print(f"❌ Файл {data_yaml} не найден!")
    exit()

print(f"✅ Конфиг найден: {data_yaml}")
print("\n🏁 НАЧАЛО ОБУЧЕНИЯ...")
print("=" * 50)

# Обучение модели
results = model.train(
    data=data_yaml,      # путь к конфигу
    epochs=30,           # количество эпох
    imgsz=640,           # размер изображения
    batch=8,             # размер батча (уменьшите если мало памяти)
    device='cpu',        # используем CPU (если нет GPU)
    workers=0,           # для Windows
    project='models/yolov8',
    name='exp',
    verbose=True
)

print("\n" + "=" * 50)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"📁 Результаты сохранены в: models/yolov8/exp")
print("=" * 50)