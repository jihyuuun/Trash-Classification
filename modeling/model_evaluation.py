import os
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 데이터 경로 및 하이퍼파라미터
data_dir = r"data\data-augmented"
batch_size = 32

# 데이터 전처리
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 로드
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# 클래스 이름 로드 및 확인
with open("class_names.json", "r") as f:
    class_names = json.load(f)
print(f"Loaded class names: {class_names}")
print(f"Dataset classes: {dataset.classes}")

# Validation 데이터셋 분리 및 DataLoader 생성
val_size = int(0.2 * len(dataset))  # 검증 데이터셋 크기 (20%)
train_size = len(dataset) - val_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])  # 검증 데이터만 추출
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Validation dataset size: {len(val_dataset)}")  # 검증 데이터셋 크기 확인

# 모델 정의 및 로드
model = models.resnet50(pretrained=False)  # ResNet50 모델
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # Fully Connected Layer 크기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 가중치 로드
try:
    state_dict = torch.load("resnet50_trash_classifier.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")
    exit()

model = model.to(device)
model.eval()

# Validation 및 평가 함수 정의
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 실제 데이터에 존재하는 고유 레이블
    unique_labels = sorted(set(all_labels))

    # 정확도 계산
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # 혼동 행렬 계산
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
    print("Confusion Matrix:\n", cm)

    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[class_names[i] for i in unique_labels],
                yticklabels=[class_names[i] for i in unique_labels])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    report = classification_report(all_labels, all_predictions, target_names=[class_names[i] for i in unique_labels],
                                    labels=unique_labels)
    print("Classification Report:\n", report)

    return accuracy, cm, report

# 모델 평가
evaluate_model(model, val_loader, device, class_names)
