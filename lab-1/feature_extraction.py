import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,)
import matplotlib.pyplot as plt

# 1. Основные настройки
DATA_DIRECTORY = "./data"

BATCH_SIZE = 32 # количество изображений для обработки за один шаг
NUMBER_OF_EPOCHS = 2 # количество эпох
LEARNING_RATE = 0.001 
NUMBER_OF_CLASSES = 10
RANDOM_SEED = 42
FINE_TUNING_EPOCHS = 2
FINE_TUNING_LEARNING_RATE = 0.00001

torch.manual_seed(RANDOM_SEED)
# Названия классов STL-10
CLASS_NAMES = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck",]

# выбор устройства для вычислений
device = torch.device("mps")

# Преобразования обучающих изображений
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),])

# Преобразования тестовых изображений
evaluation_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),])

# Загрузка STL-10
train_dataset = datasets.STL10(root=DATA_DIRECTORY, split="train", download=True, transform=train_transform)
test_dataset = datasets.STL10(root=DATA_DIRECTORY, split="test", download=True, transform=evaluation_transform)

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Загрузка предобученной ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Замораживание весов базовой сети
for parameter in model.parameters():
    parameter.requires_grad = False

# Замена выходного слоя
number_of_features = model.fc.in_features
model.fc = nn.Linear(number_of_features, NUMBER_OF_CLASSES)

# Перенос модели на MPS
model = model.to(device)

# Количество обучаемых параметров
trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

# Функция ошибки и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Обучение выходного слоя
for epoch in range(NUMBER_OF_EPOCHS):
    model.eval()
    model.fc.train()

    total_loss = 0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_images += labels.size(0)

    epoch_loss = total_loss / total_images
    epoch_accuracy = correct_predictions / total_images

    print(f"Эпоха {epoch + 1}/{NUMBER_OF_EPOCHS}")
    print(f"Ошибка: {epoch_loss:.4f}")
    print(f"Точность: {epoch_accuracy:.4f}")

# Получение прогнозов на тестовых данных
model.eval()
true_labels = []
predicted_labels = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions.cpu().numpy())

# Вычисление метрик Feature Extraction
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

feature_extraction_accuracy = accuracy
feature_extraction_precision = precision
feature_extraction_recall = recall

print("\n#### Feature Extraction ####")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")

# Построение матрицы ошибок
matrix = confusion_matrix(true_labels, predicted_labels, labels=range(NUMBER_OF_CLASSES))
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
figure, axis = plt.subplots(figsize=(10, 8))
display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)

plt.title("Confusion Matrix — Feature Extraction")
plt.tight_layout()
plt.savefig("confusion_matrix_feature_extraction.png", dpi=300)
plt.show()

# Размораживание верхнего этапа ResNet18
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Новый оптимизатор для layer4 и fc
fine_tuning_optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), 
                                         lr=FINE_TUNING_LEARNING_RATE)

for epoch in range(FINE_TUNING_EPOCHS):
    # Замороженные слои остаются в режиме оценки
    model.eval()

    # Режим обучения включается только для layer4 и fc
    model.layer4.train()
    model.fc.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        fine_tuning_optimizer.zero_grad()  # Очистка градиентов layer4 и fc
        outputs = model(images) # Прямой проход
        loss = criterion(outputs, labels) # Вычисление ошибки
        loss.backward() # Вычисление градиентов
        fine_tuning_optimizer.step() # Изменение параметров layer4 и fc
        running_loss += loss.item() * images.size(0) # Суммирование ошибки
        predictions = outputs.argmax(dim=1) # Получение предсказанных классов
        correct_predictions += (predictions == labels).sum().item() # Подсчёт правильных предсказаний
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_dataset) # Средняя ошибка за эпоху
    epoch_accuracy = correct_predictions / total_predictions # Точность за эпоху

    print(f"Эпоха Fine-tuning {epoch + 1}/{FINE_TUNING_EPOCHS}")
    print(f"Ошибка: {epoch_loss:.4f}")
    print(f"Точность: {epoch_accuracy:.4f}")

# Проверка прогнозов после Fine-tuning
model.eval()

fine_tuning_true_labels = []
fine_tuning_predicted_labels = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        fine_tuning_true_labels.extend(labels.tolist())
        fine_tuning_predicted_labels.extend(predictions.cpu().tolist())

# Метрики Fine-tuning
fine_tuning_accuracy = accuracy_score(fine_tuning_true_labels, fine_tuning_predicted_labels)
fine_tuning_precision = precision_score(fine_tuning_true_labels, fine_tuning_predicted_labels, average="weighted", zero_division=0)
fine_tuning_recall = recall_score(fine_tuning_true_labels, fine_tuning_predicted_labels, average="weighted", zero_division=0)

print("\n#### Fine-tuning ####")
print("Accuracy:", round(fine_tuning_accuracy * 100, 2), "%")
print("Precision:", round(fine_tuning_precision * 100, 2), "%")
print("Recall:", round(fine_tuning_recall * 100, 2), "%")

# Матрица ошибок Fine-tuning
fine_tuning_matrix = confusion_matrix(fine_tuning_true_labels, fine_tuning_predicted_labels, labels=range(NUMBER_OF_CLASSES))

display = ConfusionMatrixDisplay(confusion_matrix=fine_tuning_matrix, display_labels=CLASS_NAMES)
display.plot(cmap="Blues", xticks_rotation=45, values_format="d")

plt.title("Confusion Matrix — Fine-tuning")
plt.tight_layout()
plt.savefig("confusion_matrix_fine_tuning.png")
plt.show()