import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# 클래스 이름 로드
class_names_file = "class_names.json"
if not os.path.exists(class_names_file):
    raise FileNotFoundError(f"{class_names_file} not found. Ensure it exists in the same directory.")

with open(class_names_file, "r") as f:
    class_names = json.load(f)

# 모델 로드
model = models.resnet50(pretrained=False)  # 동일한 모델 사용
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # 클래스 수에 따라 출력 크기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("resnet50_trash_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 미리 이미지 크기를 줄임
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 테스트 이미지 경로 (하드코딩된 경로 사용)
image_path = r"data\test_paper_6.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The file at {image_path} was not found. Please provide a valid path.")


# 이미지 로드 및 전처리
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    raise ValueError(f"Unable to load image at {image_path}. Ensure it is a valid image file. Error: {e}")

input_tensor = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

# 예측 수행
with torch.no_grad():
    with torch.cuda.amp.autocast():  # Autocast로 추론 속도 개선
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

# 결과 출력
predicted_label = class_names[predicted_class.item()]
print(f"Predicted class: {predicted_label}")
