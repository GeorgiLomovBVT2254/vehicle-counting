# scripts/check_video.py
import cv2
import os

VIDEO_PATH = r"C:\Users\georg\Desktop\Практика_Ломов\CARmodel\data\raw\Teat1CAR.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🎬 Информация о видео:")
print(f"   Размер: {width}x{height}")
print(f"   FPS: {fps}")
print(f"   Кадров: {total_frames}")

# Сохраняем первый кадр для проверки
ret, frame = cap.read()
if ret:
    cv2.imwrite("check_frame.jpg", frame)
    print(f"✅ Сохранён кадр: check_frame.jpg")
    print("📸 Откройте его, чтобы увидеть, что на видео")

cap.release()