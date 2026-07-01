# scripts/create_charts.py
import matplotlib.pyplot as plt
import numpy as np
import os

def create_comparison_charts():
    """Создание графиков для сравнения моделей"""
    
    # Создаём папку для графиков
    os.makedirs("report/images", exist_ok=True)
    
    # Данные
    models = ['YOLOv8', 'SSD', 'Faster R-CNN', 'RetinaNet', 'DETR']
    colors = ['#2ECC71', '#F1C40F', '#3498DB', '#E74C3C', '#9B59B6']
    
    # Метрики
    mAP = [0.79, 0.053, 0, 0, 0]  # mAP@0.5 (для остальных нет данных)
    time_ms = [234, 139, 2851, 7379, 2119]
    size_mb = [6.2, 3.4, 41.8, 34.0, 170]
    fps = [4.3, 7.2, 0.35, 0.14, 0.47]
    detected = [441, 0, 367, 561, 946]
    
    # ============================================================
    # ГРАФИК 1: Сравнение времени инференса
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(models, time_ms, color=colors)
    ax1.set_ylabel('Время (мс)', fontsize=12)
    ax1.set_xlabel('Модель', fontsize=12)
    ax1.set_title('Сравнение времени инференса моделей\n(чем меньше, тем лучше)', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars1, time_ms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{value} мс', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('report/images/chart_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ График 1: Сравнение времени инференса")
    
    # ============================================================
    # ГРАФИК 2: Сравнение размера модели
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.bar(models, size_mb, color=colors)
    ax2.set_ylabel('Размер (MB)', fontsize=12)
    ax2.set_xlabel('Модель', fontsize=12)
    ax2.set_title('Сравнение размера моделей\n(чем меньше, тем лучше)', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, value in zip(bars2, size_mb):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{value} MB', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('report/images/chart_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ График 2: Сравнение размера модели")
    
    # ============================================================
    # ГРАФИК 3: Сравнение FPS
    # ============================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars3 = ax3.bar(models, fps, color=colors)
    ax3.set_ylabel('FPS (кадров/сек)', fontsize=12)
    ax3.set_xlabel('Модель', fontsize=12)
    ax3.set_title('Сравнение производительности (FPS)\n(чем больше, тем лучше)', fontsize=14)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, value in zip(bars3, fps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{value} FPS', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('report/images/chart_fps_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ График 3: Сравнение FPS")
    
    # ============================================================
    # ГРАФИК 4: Сравнение обнаруженных объектов
    # ============================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    bars4 = ax4.bar(models, detected, color=colors)
    ax4.set_ylabel('Обнаружено объектов', fontsize=12)
    ax4.set_xlabel('Модель', fontsize=12)
    ax4.set_title('Сравнение количества обнаруженных объектов\n(чем больше, тем лучше)', fontsize=14)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, value in zip(bars4, detected):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{value}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('report/images/chart_detected_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ График 4: Сравнение обнаруженных объектов")
    
    # ============================================================
    # ГРАФИК 5: Сравнение mAP (только для моделей с данными)
    # ============================================================
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    models_mAP = ['YOLOv8', 'SSD']
    mAP_values = [0.79, 0.053]
    colors_mAP = ['#2ECC71', '#F1C40F']
    
    bars5 = ax5.bar(models_mAP, mAP_values, color=colors_mAP)
    ax5.set_ylabel('mAP@0.5', fontsize=12)
    ax5.set_xlabel('Модель', fontsize=12)
    ax5.set_title('Сравнение точности детекции (mAP@0.5)\n(чем больше, тем лучше)', fontsize=14)
    ax5.set_ylim(0, 1)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, value in zip(bars5, mAP_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('report/images/chart_map_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ График 5: Сравнение mAP")
    
    # ============================================================
    # ГРАФИК 6: Сводный график (радарная диаграмма) для YOLOv8
    # ============================================================
    # Для сводной оценки нормализуем метрики (0-1)
    # Нормализация: время (чем меньше, тем лучше) -> 1 - (value - min)/(max - min)
    
    print("\n📊 Все графики сохранены в папку: report/images/")
    print("   - chart_time_comparison.png")
    print("   - chart_size_comparison.png")
    print("   - chart_fps_comparison.png")
    print("   - chart_detected_comparison.png")
    print("   - chart_map_comparison.png")

if __name__ == "__main__":
    create_comparison_charts()