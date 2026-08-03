import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Устройство: {device}")

df = pd.read_csv("data/metadata.csv")
images = np.load("data/images.npy")

CLASSES = sorted(df["dx"].unique())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
labels = df["dx"].map(CLASS_TO_IDX).values

print(f"Классы: {CLASSES}")
print(df["dx"].value_counts())

train_idx, val_idx = train_test_split(
    np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SkinLesionDataset(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images = images[indices]
        self.labels = labels[indices]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.labels[idx]
        return image, label


train_ds = SkinLesionDataset(images, labels, train_idx, train_transform)
val_ds = SkinLesionDataset(images, labels, val_idx, val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

class_weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images_batch, labels_batch in train_loader:
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(images_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images_batch, labels_batch in val_loader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            outputs = model(images_batch)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    val_acc = correct / total
    print(f"Эпоха {epoch+1}/{EPOCHS} — train loss: {train_loss/len(train_loader):.4f} — val accuracy: {val_acc:.2%}")

torch.save({"model_state": model.state_dict(), "classes": CLASSES}, "model.pth")
print("Модель сохранена в model.pth")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images_batch, labels_batch in val_loader:
        images_batch = images_batch.to(device)
        outputs = model(images_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

print("\nОтчёт по классам:")
print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))
print("Матрица ошибок (строки — истина, столбцы — предсказание):")
print(confusion_matrix(all_labels, all_preds))
