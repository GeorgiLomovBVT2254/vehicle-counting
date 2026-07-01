# scripts/create_excel_report.py
import pandas as pd
import os

def create_excel_report(filepath="results/comparison_report.xlsx"):
    """Создание Excel-отчёта с результатами сравнения моделей"""
    
    # Данные для основной таблицы
    data_comparison = {
        "Модель": ["YOLOv8", "SSD", "Faster R-CNN", "RetinaNet", "DETR"],
        "Год": [2023, 2016, 2015, 2017, 2020],
        "Семейство": ["One-stage", "One-stage", "Two-stage", "One-stage", "Transformer"],
        "mAP@0.5": [0.79, 0.053, "-", "-", "-"],
        "Precision": [0.909, "-", "-", "-", "-"],
        "Recall": [0.593, "-", "-", "-", "-"],
        "F1": [0.71, "-", "-", "-", "-"],
        "Время (мс)": [234, 139, 2851, 7379, 2119],
        "Размер (MB)": [6.2, 3.4, 41.8, 34.0, 170],
        "FPS (CPU)": [4.3, 7.2, 0.35, 0.14, 0.47],
        "Обнаружено": [441, "-", "-", "-", "-"]
    }
    
    # Данные для детальной статистики
    data_details = {
        "Модель": ["YOLOv8", "SSD", "Faster R-CNN", "RetinaNet", "DETR"],
        "Предобучение": ["COCO", "COCO", "COCO", "COCO", "COCO"],
        "Размер входа": ["640x640", "320x320", "800x800", "800x800", "800x800"],
        "Эпохи": [30, 30, 30, 30, 30],
        "Batch Size": [16, 8, 4, 4, 2],
        "Оптимизатор": ["Adam", "Adam", "SGD", "SGD", "AdamW"],
        "Устройство": ["CPU", "CPU", "CPU", "CPU", "CPU"],
        "Комментарий": ["Лучшая модель", "Низкая точность", "Медленно", "Очень медленно", "Тяжёлый"]
    }
    
    # Создаём DataFrame
    df_comparison = pd.DataFrame(data_comparison)
    df_details = pd.DataFrame(data_details)
    
    # Создаём папку если её нет
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Сохранение в Excel
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_comparison.to_excel(writer, sheet_name='Сравнение моделей', index=False)
        df_details.to_excel(writer, sheet_name='Детальная статистика', index=False)
    
    print(f"✅ Excel-отчёт сохранён: {filepath}")
    return filepath

if __name__ == "__main__":
    create_excel_report()