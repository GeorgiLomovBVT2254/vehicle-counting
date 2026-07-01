# scripts/track_and_count.py
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
import torch

print("=" * 60)
print("🚗 ТРЕКИНГ И ПОДСЧЁТ АВТОМОБИЛЕЙ")
print("=" * 60)

# ========== КОНФИГУРАЦИЯ ==========
# ИСПРАВЛЕННЫЙ ПУТЬ (с r перед строкой)
VIDEO_PATH = r"C:\Users\georg\Desktop\Практика_Ломов\CARmodel\data\raw\TESTCAR_1.mp4"
MODEL_PATH = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.30
LINE_POSITION = 0.6

# ========== ПРОВЕРКА ВИДЕО ==========
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Видео не найдено: {VIDEO_PATH}")
    # Показываем доступные видео
    raw_dir = r"C:\Users\georg\Desktop\Практика_Ломов\CARmodel\data\raw"
    if os.path.exists(raw_dir):
        print("\n📁 Доступные видео:")
        for f in os.listdir(raw_dir):
            if f.endswith(('.mp4', '.webm', '.avi')):
                print(f"   - {f}")
    exit()

print(f"🎬 Видео: {VIDEO_PATH}")

# ========== ПРОВЕРКА GPU ==========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📱 Устройство: {device.upper()}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ========== ЗАГРУЗКА МОДЕЛИ ==========
print("\n📦 Загрузка модели YOLOv8s...")
model = YOLO(MODEL_PATH)
print(f"✅ Модель загружена")

# ========== ОТКРЫВАЕМ ВИДЕО ==========
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

line_y = int(height * LINE_POSITION)

print(f"\n📐 Размер видео: {width}x{height}")
print(f"🎞️ FPS: {fps}")
print(f"📊 Всего кадров: {total_frames}")
print(f"📏 Линия подсчёта Y: {line_y}")

# ========== ПРОВЕРКА НА ПЕРВОМ КАДРЕ ==========
print("\n🔍 Проверка на первом кадре...")
ret, first_frame = cap.read()
if not ret:
    print("❌ Не удалось прочитать видео")
    exit()

results = model(first_frame, verbose=False)

cars_on_first_frame = 0
for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            if int(box.cls[0]) == 2 and float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                cars_on_first_frame += 1

print(f"   🚗 Автомобилей на первом кадре: {cars_on_first_frame}")

if cars_on_first_frame == 0:
    print("\n⚠️ ВНИМАНИЕ: На первом кадре нет автомобилей!")
    print("   Возможно, они появятся позже. Продолжаем...")

# Сохраняем первый кадр для отчёта
os.makedirs("results", exist_ok=True)
annotated = results[0].plot()
cv2.imwrite("results/debug_frame.jpg", annotated)
print("✅ Сохранён кадр: results/debug_frame.jpg")

# ========== ПОДГОТОВКА К ОБРАБОТКЕ ==========
out = cv2.VideoWriter(
    "results/tracked_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# ========== ТРЕКЕР ==========
tracked_objects = {}
next_id = 1
counted_ids = set()
total_count = 0
frame_count = 0
max_distance = 80

print("\n🚗 Начинаем обработку...")
print("=" * 60)

start_time = time.time()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Детекция
    results = model(frame, verbose=False)
    
    # Сбор обнаруженных машин
    current_cars = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if int(box.cls[0]) == 2 and float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_cars.append([cx, cy, x1, y1, x2, y2, float(box.conf[0])])
    
    # Обновление треков
    updated_tracks = {}
    used_ids = set()
    
    for car in current_cars:
        cx, cy, x1, y1, x2, y2, conf = car
        found = False
        
        for obj_id, track in tracked_objects.items():
            if obj_id in used_ids:
                continue
            
            last_cx, last_cy = track[-1][:2]
            dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            
            if dist < max_distance:
                updated_tracks[obj_id] = track + [[cx, cy, x1, y1, x2, y2, conf]]
                used_ids.add(obj_id)
                found = True
                
                # Проверка пересечения линии
                if len(track) >= 2 and obj_id not in counted_ids:
                    prev_y = track[-1][1]
                    if prev_y <= line_y < cy:
                        counted_ids.add(obj_id)
                        total_count += 1
                        print(f"   🚗 Машина #{obj_id} пересекла линию! Всего: {total_count}")
                break
        
        if not found:
            updated_tracks[next_id] = [[cx, cy, x1, y1, x2, y2, conf]]
            next_id += 1
    
    tracked_objects = updated_tracks
    
    # ========== ВИЗУАЛИЗАЦИЯ ==========
    for obj_id, track in tracked_objects.items():
        if len(track) > 0:
            cx, cy, x1, y1, x2, y2, conf = track[-1]
            
            color = (0, 255, 0) if obj_id not in counted_ids else (0, 0, 255)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{obj_id}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if len(track) > 3:
                pts = np.array([[p[0], p[1]] for p in track[-15:]], np.int32)
                cv2.polylines(frame, [pts], False, (255, 255, 0), 1)
    
    # Линия
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
    cv2.putText(frame, "LINE", (width//2-30, line_y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Счётчик
    cv2.putText(frame, f"TOTAL: {total_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"TRACKS: {len(tracked_objects)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    out.write(frame)
    
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"📹 {frame_count}/{total_frames} ({progress:.1f}%) | Машин: {total_count} | Треков: {len(tracked_objects)}")

cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time

print("\n" + "=" * 60)
print("✅ ОБРАБОТКА ЗАВЕРШЕНА!")
print(f"📁 Результат: results/tracked_video.mp4")
print(f"🚗 Всего пересекло линию: {total_count} автомобилей")
print(f"📊 Всего уникальных треков: {next_id - 1}")
print(f"⏱️ Время обработки: {total_time:.1f} сек")
print("=" * 60)