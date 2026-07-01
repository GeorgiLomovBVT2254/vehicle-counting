# scripts/run_detection.py
import cv2
from ultralytics import YOLO
import os
import time
import ssl

# ОБХОД SSL ОШИБКИ (если понадобится при загрузке)
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 50)
print("🚗 CARmodel - Детекция автомобилей на видео")
print("=" * 50)

# ПУТЬ К МОДЕЛИ
model_path = "yolov8n.pt"

# Проверка модели
if not os.path.exists(model_path):
    print(f"❌ Модель не найдена: {model_path}")
    print("📁 Положите yolov8n.pt в папку CARmodel")
    exit()
else:
    print(f"✅ Модель загружена: {model_path}")

# Загрузка модели
print("📦 Загрузка модели YOLOv8...")
model = YOLO(model_path)

# ПУТЬ К ВИДЕО (выберите одно)
video_path = "data/raw/Teat1CAR.mp4"  # или Teat2CAR.mp4

print(f"🔍 Поиск видео: {video_path}")

if not os.path.exists(video_path):
    print(f"❌ Видео не найдено!")
    print("\n📁 Доступные видео:")
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(('.mp4', '.webm', '.avi')):
                print(f"   - {f}")
    exit()

# Открываем видео
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n🎬 Видео: {video_path}")
print(f"📐 Размер: {width}x{height}")
print(f"🎞️ FPS: {fps}")
print(f"📊 Всего кадров: {total_frames}")
print("=" * 50)

# Сохранение результата
os.makedirs("results", exist_ok=True)
out = cv2.VideoWriter(
    "results/output_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_count = 0
total_cars = 0
start_time = time.time()

print("🚗 Обработка видео... (может занять несколько минут)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Детекция на кадре
    results = model(frame)
    
    # Подсчёт машин на кадре
    cars_in_frame = 0
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if int(box.cls[0]) == 2:  # класс 'car'
                    cars_in_frame += 1
                    total_cars += 1
    
    # Отрисовка
    annotated_frame = results[0].plot()
    
    # Добавляем текст
    cv2.putText(annotated_frame, f"Cars in frame: {cars_in_frame}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total cars: {total_cars}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(annotated_frame)
    
    # Прогресс
    if frame_count % 50 == 0 or frame_count == total_frames:
        progress = (frame_count / total_frames) * 100
        elapsed = time.time() - start_time
        fps_processing = frame_count / elapsed
        print(f"📹 {frame_count}/{total_frames} ({progress:.1f}%) | Авто: {total_cars} | {fps_processing:.1f} fps")

cap.release()
out.release()

end_time = time.time()
processing_time = end_time - start_time

print("\n" + "=" * 50)
print("✅ ОБРАБОТКА ЗАВЕРШЕНА!")
print(f"📁 Результат: results/output_video.mp4")
print(f"🚗 Всего обнаружено автомобилей: {total_cars}")
print(f"⏱️ Время обработки: {processing_time:.1f} сек")
print(f"📊 Средний FPS обработки: {frame_count / processing_time:.1f}")
print("=" * 50)