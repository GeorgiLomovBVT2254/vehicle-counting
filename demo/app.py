# demo/app.py
import streamlit as st
import cv2
import os
import tempfile
import time
import numpy as np
from ultralytics import YOLO
import torch
import json
import datetime
import pandas as pd

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(
    page_title="CARmodel - Подсчёт автомобилей",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# ========== ЗАГОЛОВОК ==========
st.title("🚗 CARmodel - Система подсчёта автомобилей")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin: 0; font-size: 16px;'>
        📹 Загрузите видео с дорожным движением, и система автоматически 
        обнаружит автомобили и подсчитает их количество.
        </p>
    </div>
""", unsafe_allow_html=True)

# ========== БОКОВАЯ ПАНЕЛЬ ==========
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели
    model_option = st.selectbox(
        "Выберите модель:",
        ["YOLOv8n (nano) - быстрая", "YOLOv8s (small) - точная"],
        index=1,
        help="YOLOv8s точнее, но медленнее. YOLOv8n быстрее, но менее точная."
    )
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:",
        min_value=0.1,
        max_value=0.9,
        value=0.40,
        step=0.05,
        help="Чем выше порог, тем меньше ложных срабатываний."
    )
    
    # Положение линии
    line_position = st.slider(
        "Положение линии подсчёта:",
        min_value=0.1,
        max_value=0.9,
        value=0.60,
        step=0.05,
        help="Процент от высоты видео (0 = верх, 1 = низ)."
    )
    
    st.markdown("---")
    st.subheader("🎬 Настройки обработки")
    
    # Пропуск кадров
    frame_skip = st.slider(
        "Пропуск кадров:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="1 = каждый кадр, 5 = каждый 5-й кадр (быстрее для длинных видео)"
    )
    
    # Максимальное количество кадров
    max_frames = st.number_input(
        "Максимум кадров:",
        min_value=100,
        max_value=50000,
        value=15000,
        step=100,
        help="Ограничение на количество обрабатываемых кадров"
    )
    
    st.markdown("---")
    
    # Кнопка показа истории
    if st.button("📊 Показать историю", use_container_width=True):
        st.session_state.show_history = not st.session_state.show_history
    
    st.markdown("---")
    st.caption("🔧 Разработано в рамках практики МТУСИ")
    st.caption("📅 2026 г.")

# ========== ИСТОРИЯ ЗАПУСКОВ ==========
if st.session_state.show_history:
    st.markdown("---")
    st.subheader("📋 История запусков")
    
    history_path = "results/history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except:
                history = []
        
        if history:
            df = pd.DataFrame(history)
            columns = ["timestamp", "video_name", "model", "vehicles_count", "processing_time_sec", "avg_fps", "device"]
            available_columns = [col for col in columns if col in df.columns]
            
            st.dataframe(df[available_columns], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Скачать историю (JSON)",
                    data=json.dumps(history, indent=2, ensure_ascii=False),
                    file_name="history.json",
                    mime="application/json"
                )
            with col2:
                if st.button("🔄 Обновить историю"):
                    st.rerun()
        else:
            st.info("📭 История пока пуста. Запустите обработку видео.")
    else:
        st.info("📭 История пока пуста. Запустите обработку видео.")

# ========== ОСНОВНАЯ ОБЛАСТЬ ==========
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Загрузите видеофайл",
        type=['mp4', 'avi', 'mov', 'webm'],
        help="Поддерживаются форматы: MP4, AVI, MOV, WebM"
    )

with col2:
    st.info("""
    **📋 Требования к видео:**
    - Формат: MP4, AVI, MOV, WebM
    - Рекомендуемое разрешение: 720p или выше
    - Длительность: до 10 минут
    """)

# ========== ОБРАБОТКА ВИДЕО ==========
if uploaded_file is not None:
    st.markdown("---")
    st.header("📊 Обработка видео")
    
    # Сохраняем загруженное видео во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name
    
    # Определяем модель
    model_name = "yolov8s.pt" if "YOLOv8s" in model_option else "yolov8n.pt"
    
    # Кнопка запуска
    if st.button("🚀 Запустить обработку", type="primary"):
        
        # Прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Загрузка модели
            status_text.info("📦 Загрузка модели...")
            model = YOLO(model_name)
            
            # Открываем видео
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0:
                fps = 30
            
            # Применяем лимит кадров
            if max_frames > 0 and max_frames < total_frames:
                total_frames = max_frames
            
            # Параметры для трекинга
            line_y = int(height * line_position)
            tracked_objects = {}
            next_id = 1
            counted_ids = set()
            total_count = 0
            frame_count = 0
            processed_frames = 0
            max_distance = 80
            
            # Создаём выходное видео
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            status_text.info("🚗 Начинаем обработку...")
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Пропускаем кадры
                if frame_count % frame_skip != 0:
                    continue
                
                processed_frames += 1
                
                # Детекция
                results = model(frame, verbose=False)
                
                # Сбор обнаруженных машин
                current_cars = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            if int(box.cls[0]) == 2 and float(box.conf[0]) > confidence_threshold:
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
                            
                            if len(track) >= 2 and obj_id not in counted_ids:
                                prev_y = track[-1][1]
                                if prev_y <= line_y < cy:
                                    counted_ids.add(obj_id)
                                    total_count += 1
                            break
                    
                    if not found:
                        updated_tracks[next_id] = [[cx, cy, x1, y1, x2, y2, conf]]
                        next_id += 1
                
                tracked_objects = updated_tracks
                
                # Визуализация
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
                
                # Линия подсчёта
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
                cv2.putText(frame, "LINE", (width//2-30, line_y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Счётчик
                cv2.putText(frame, f"TOTAL: {total_count}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, f"TRACKS: {len(tracked_objects)}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                out.write(frame)
                
                # Обновление прогресса
                if processed_frames % 10 == 0:
                    if total_frames > 0:
                        progress = min(processed_frames / (total_frames / frame_skip), 1.0)
                        progress_bar.progress(progress)
                        
                        elapsed = time.time() - start_time
                        fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                        
                        status_text.info(f"📹 Обработано {processed_frames} кадров | Найдено машин: {total_count} | {fps_processing:.1f} FPS")
            
            cap.release()
            out.release()
            
            end_time = time.time()
            processing_time = end_time - start_time
            avg_fps = processed_frames / processing_time if processing_time > 0 else 0
            
            progress_bar.progress(1.0)
            status_text.success(f"✅ Обработка завершена! Время: {processing_time:.1f} сек")
            
            # ========== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ==========
            st.markdown("---")
            st.header("📊 Результаты")
            
            # Метрики
            col_res1, col_res2, col_res3, col_res4, col_res5 = st.columns(5)
            
            with col_res1:
                st.metric("🚗 Всего машин", total_count)
            
            with col_res2:
                st.metric("🎞️ Кадров", processed_frames)
            
            with col_res3:
                st.metric("⏱️ Время", f"{processing_time:.1f} сек")
            
            with col_res4:
                st.metric("📊 Скорость", f"{avg_fps:.1f} FPS")
            
            with col_res5:
                st.metric("📌 Треков", next_id - 1)
            
            # Отображение видео
            st.subheader("🎬 Результат обработки")
            
            # Читаем выходной файл для отображения
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            
            st.video(video_bytes)
            
            # Кнопка скачивания
            st.download_button(
                label="📥 Скачать обработанное видео",
                data=video_bytes,
                file_name="tracked_video_result.mp4",
                mime="video/mp4"
            )
            
            # ========== СОХРАНЕНИЕ В JSON ==========
            result_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "video_name": uploaded_file.name,
                "model": model_name,
                "confidence_threshold": confidence_threshold,
                "line_position": line_position,
                "frame_skip": frame_skip,
                "total_frames": processed_frames,
                "vehicles_count": total_count,
                "processing_time_sec": round(processing_time, 1),
                "avg_fps": round(avg_fps, 1),
                "tracks_created": next_id - 1,
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "video_resolution": f"{width}x{height}",
                "video_fps": fps
            }
            
            # Сохраняем в историю
            os.makedirs("results", exist_ok=True)
            history_path = "results/history.json"
            
            history = []
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    try:
                        history = json.load(f)
                    except:
                        history = []
            
            history.append(result_data)
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            # ========== ДЕТАЛЬНАЯ СТАТИСТИКА ==========
            with st.expander("📋 Детальная статистика"):
                st.json(result_data)
            
            # Очистка временных файлов
            try:
                os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass
            
        except Exception as e:
            st.error(f"❌ Ошибка при обработке: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Очистка временных файлов
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if 'output_path' in locals() and os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass
    
    else:
        st.info("👆 Нажмите кнопку 'Запустить обработку' для начала")

else:
    # Показываем пример
    st.markdown("---")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.markdown("""
        **📋 Как это работает:**
        1. Загрузите видео
        2. Настройте параметры
        3. Нажмите "Запустить обработку"
        4. Скачайте результат
        """)
    
    with col_ex2:
        st.markdown("""
        **🎯 Что делает система:**
        - Обнаруживает автомобили (YOLOv8)
        - Отслеживает их движение (SORT)
        - Подсчитывает пересечение линии
        - Сохраняет результат
        """)
    
    with col_ex3:
        st.markdown("""
        **📊 Результаты:**
        - Видео с рамками и ID
        - Счётчик автомобилей
        - Статистика обработки
        - История запусков
        """)
    
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **💡 Совет:** Для лучшего результата используйте видео с:
        - Хорошим освещением
        - Контрастными автомобилями
        - Движением в одном направлении
        """)
    
    with col_info2:
        st.warning("""
        **⚠️ Примечание:** Обработка длительных видео (более 5 минут) 
        может занять значительное время. Используйте настройку "Пропуск кадров" 
        для ускорения.
        """)
    
    st.caption("💡 Поддерживаются видеоформаты: MP4, AVI, MOV, WebM")
    