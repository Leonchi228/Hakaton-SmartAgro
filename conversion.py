import torch
import torch.nn as nn
from torchvision import models
import os
import shutil

# ищем пути
input_base = '/kaggle/input'
pth_dir = None
data_dir = None

for root, dirs, files in os.walk(input_base):
    if 'saved_models' in root and not pth_dir:
        pth_dir = root
    if 'sorted_plants' in root and not data_dir:
        data_dir = root

tflite_dir = '/kaggle/working/mobile_models'
os.makedirs(tflite_dir, exist_ok=True)

print(f" Путь к моделям: {pth_dir}")
print(f" Путь к данным: {data_dir}")

# конвертация
for model_file in sorted(os.listdir(pth_dir)):
    if not model_file.endswith('.pth'): continue
    
    # Извлекаем имя растения
    plant_name = model_file.split('_')[0] 
    print(f"\n Обработка: {plant_name}")
    
    try:
        # Считаем реальное колво классов из папок
        plant_path = os.path.join(data_dir, plant_name)
        num_classes = len([d for d in os.listdir(plant_path) if os.path.isdir(os.path.join(plant_path, d))])
        
        # Загрузка модели
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(pth_dir, model_file), map_location='cpu'))
        model.eval()

        # конвертация
        example_input = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example_input)
        
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_model = optimize_for_mobile(traced_script_module)
        
        # сохраняем фомат ptl
        save_path = os.path.join(tflite_dir, f"{plant_name}.ptl")
        optimized_model._save_for_lite_interpreter(save_path)
        
        # сохраняем болезни в файл
        classes = sorted(os.listdir(plant_path))
        with open(os.path.join(tflite_dir, f"{plant_name}_labels.txt"), "w") as f:
            f.write("\n".join(classes))
            
        print(f" Успешно: {plant_name}.ptl")
        
    except Exception as e:
        print(f" Пропуск {plant_name}: {e}")

# архивируем
shutil.make_archive('/kaggle/working/PLANT_EXPERTS_MOBILE', 'zip', tflite_dir)
print("\n ГОТОВО! Файл PLANT_EXPERTS_MOBILE.zip ждет в /working")
