import torch
import ultralytics
import cv2
import streamlit as st
import pandas as pd
import matplotlib
import seaborn as sns

print("=" * 50)
print("✅ ПРОВЕРКА УСТАНОВКИ")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"YOLO (Ultralytics): {ultralytics.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
print("=" * 50)
print("✅ ВСЕ БИБЛИОТЕКИ ГОТОВЫ К РАБОТЕ!")
print("=" * 50)
