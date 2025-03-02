import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from collections import Counter

# 데이터 경로 및 하이퍼파라미터
data_dir = r"data\data-augmented"
batch_size = 32
learning_rate = 0.0001  # 학습률
num_epochs = 5
num_classes = 8  # 전체 클래스 수
target_classes = ["glass", "vinyl"]  # 집중할 클래스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# 데이터셋 로드
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms["train"])

# Train/Validation Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 클래스별 데이터 수 확인
class_counts = Counter([label for _, label in train_dataset])
print(f"Class counts: {class_counts}")

# 클래스 가중치 계산 (glass와 vinyl 클래스에 가중치 부여)
total_samples = sum(class_counts.values())
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
for cls in target_classes:
    class_weights[full_dataset.class_to_idx[cls]] *= 2.0  # 타겟 클래스에 추가 가중치 부여
class_weights_tensor = torch.tensor(class_weights).to(device)
print(f"Class weights: {class_weights}")

# 모델 정의 및 로드
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 출력 크기를 전체 클래스 수로 설정
model = model.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 클래스 가중치 적용
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler()

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {running_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Accuracy: {100 * correct / total:.2f}%")

# 모델 저장
torch.save(model.state_dict(), "resnet50_trash_classifier.pth")
print("Updated model saved as 'resnet50_trash_classifier.pth'.")
