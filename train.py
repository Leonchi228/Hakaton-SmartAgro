import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

#сортируем данные
source_dir = '/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'
dataset_dir = '/kaggle/working/sorted_plants'
models_dir = '/kaggle/working/saved_models'

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("1. Начинаем сортировку датасета...")
for folder_name in os.listdir(source_dir):
    full_path = os.path.join(source_dir, folder_name)
    if not os.path.isdir(full_path) or "___" not in folder_name:
        continue
    
    # имя растения
    plant_name = folder_name.split("___")[0].split("(")[0].strip()
    
    plant_target_dir = os.path.join(dataset_dir, plant_name, folder_name)
    os.makedirs(plant_target_dir, exist_ok=True)
    
    for img in os.listdir(full_path):
        shutil.copy2(os.path.join(full_path, img), os.path.join(plant_target_dir, img))

print(f"Сортировка завершена! Растения: {os.listdir(dataset_dir)}\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Обучение будет проходить на: {device}\n")

# трансформация фото
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# обучения
def train_plant_expert(plant_name, data_path, epochs=3):
    print(f"--- Запуск обучения эксперта для: {plant_name.upper()} ---")
    
    
    full_dataset = datasets.ImageFolder(data_path, transform=data_transforms)
    num_classes = len(full_dataset.classes)
    
    
    if num_classes < 2:
        print(f"Пропуск {plant_name}: найдено менее 2 классов болезней.\n")
        return None
    
    # разделяем данные
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    
    model = models.mobilenet_v3_small(weights='DEFAULT')
    # замена последнего слоя на количество болезней
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # обучения
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # валидация
        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        epoch_acc = running_corrects.double() / val_size
        
        # сохранение лучшего результата
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(models_dir, f"{plant_name}_expert.pth"))
            
    print(f" {plant_name} обучен! Лучшая точность: {best_acc:.2%}")
    print(f"сохранены классы: {full_dataset.classes}\n")
    return best_acc.item()


EPOCHS_PER_PLANT = 12 
results = {}

for plant_folder in os.listdir(dataset_dir):
    plant_path = os.path.join(dataset_dir, plant_folder)
    accuracy = train_plant_expert(plant_folder, plant_path, epochs=EPOCHS_PER_PLANT)
    if accuracy is not None:
        results[plant_folder] = accuracy

# Вывод итоговой таблицы

print(" ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ КУЛЬТУРАМ ")

for plant, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"Растение: {plant.ljust(15)} | Точность: {acc:.2%}")
print(f"Все файлы весов сохранены в папке: {models_dir}")
