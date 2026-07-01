# scripts/prepare_data.py
import os
import shutil
import random

print("=" * 50)
print("📦 ПОДГОТОВКА ДАННЫХ UA-DETRAC")
print("=" * 50)

# ПУТИ - ИСПРАВЛЕННЫЕ
SOURCE_DIR = r"C:\Users\georg\Desktop\Практика_Ломов\CARmodel\UA_DETRAC_CUSTOM"
TARGET_DIR = r"C:\Users\georg\Desktop\Практика_Ломов\CARmodel\data"

print(f"🔍 Ищем датасет в: {SOURCE_DIR}")

# Проверяем, существует ли папка с датасетом
if not os.path.exists(SOURCE_DIR):
    print(f"❌ Папка с датасетом не найдена: {SOURCE_DIR}")
    print("📁 Проверьте, что папка UA_DETRAC_CUSTOM существует")
    exit()

# Определяем, как называется папка с изображениями
IMAGES_DIR = None
if os.path.exists(os.path.join(SOURCE_DIR, "sequences")):
    IMAGES_DIR = "sequences"
elif os.path.exists(os.path.join(SOURCE_DIR, "images")):
    IMAGES_DIR = "images"

if IMAGES_DIR is None:
    print("❌ Не найдены папки 'images' или 'sequences'")
    print("📁 Содержимое UA_DETRAC_CUSTOM:")
    for f in os.listdir(SOURCE_DIR):
        print(f"   - {f}")
    exit()

print(f"✅ Найдена папка с изображениями: {IMAGES_DIR}")

# Список видеопоследовательностей
sequences = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034"]

# Собираем все файлы
all_files = []
for seq in sequences:
    img_dir = os.path.join(SOURCE_DIR, IMAGES_DIR, seq)
    ann_dir = os.path.join(SOURCE_DIR, "annotations", seq)
    
    if not os.path.exists(img_dir):
        print(f"⚠️ Папка не найдена: {img_dir}")
        continue
        
    if not os.path.exists(ann_dir):
        print(f"⚠️ Папка с аннотациями не найдена: {ann_dir}")
        continue
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    for img in images:
        txt = img.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(os.path.join(ann_dir, txt)):
            all_files.append((seq, img, txt))

print(f"\n✅ Найдено {len(all_files)} изображений с разметкой")

if len(all_files) == 0:
    print("❌ Не найдено ни одного изображения!")
    print("📁 Проверьте структуру папок UA_DETRAC_CUSTOM")
    exit()

# Перемешиваем
random.seed(42)
random.shuffle(all_files)

# Разбиваем на train/val/test (60/20/20)
train_ratio = 0.6
val_ratio = 0.2

train_split = int(len(all_files) * train_ratio)
val_split = int(len(all_files) * (train_ratio + val_ratio))

train_files = all_files[:train_split]
val_files = all_files[train_split:val_split]
test_files = all_files[val_split:]

print(f"\n📊 Train: {len(train_files)} изображений")
print(f"📊 Val:   {len(val_files)} изображений")
print(f"📊 Test:  {len(test_files)} изображений")

# Функция копирования
def copy_files(file_list, split_name):
    count = 0
    for seq, img_name, txt_name in file_list:
        # Копируем изображение
        src_img = os.path.join(SOURCE_DIR, IMAGES_DIR, seq, img_name)
        dst_img = os.path.join(TARGET_DIR, split_name, "images", img_name)
        shutil.copy2(src_img, dst_img)
        
        # Копируем аннотацию
        src_txt = os.path.join(SOURCE_DIR, "annotations", seq, txt_name)
        dst_txt = os.path.join(TARGET_DIR, split_name, "labels", txt_name)
        shutil.copy2(src_txt, dst_txt)
        count += 1
        
        if count % 20 == 0:
            print(f"   {split_name}: скопировано {count}/{len(file_list)}")
    
    print(f"   ✅ {split_name}: готово {count} файлов")

# Очищаем целевые папки
for split in ["train", "val", "test"]:
    images_dir = os.path.join(TARGET_DIR, split, "images")
    labels_dir = os.path.join(TARGET_DIR, split, "labels")
    
    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            os.remove(os.path.join(images_dir, f))
    if os.path.exists(labels_dir):
        for f in os.listdir(labels_dir):
            os.remove(os.path.join(labels_dir, f))
    
    print(f"\n📁 Очищена папка: {split}")

print("\n🚀 Копирование файлов...")
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("\n" + "=" * 50)
print("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
print("=" * 50)
print(f"📁 Train images: {len(os.listdir(TARGET_DIR + '/train/images'))}")
print(f"📁 Train labels: {len(os.listdir(TARGET_DIR + '/train/labels'))}")
print(f"📁 Val images:   {len(os.listdir(TARGET_DIR + '/val/images'))}")
print(f"📁 Test images:  {len(os.listdir(TARGET_DIR + '/test/images'))}")
print("=" * 50)