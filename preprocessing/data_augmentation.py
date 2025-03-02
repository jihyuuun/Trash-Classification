import os
from torchvision import transforms
from PIL import Image

# 데이터 경로 설정
data_dir = r"data\data-augmented_1"
output_dir = r"data\data-augmented"

# 증강할 클래스 설정
classes_to_augment = ["trash", "vinyl"]
num_augmentations = 5  # 각 이미지당 생성할 증강 이미지 수

# 증강 전처리
augmentation_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# 증강 데이터 생성 함수
def augment_class_data(class_name, num_augmentations):
    input_class_dir = os.path.join(data_dir, class_name)
    output_class_dir = os.path.join(output_dir, class_name)

    # 출력 디렉토리 생성
    os.makedirs(output_class_dir, exist_ok=True)

    # 모든 이미지에 대해 증강 수행
    for img_name in os.listdir(input_class_dir):
        img_path = os.path.join(input_class_dir, img_name)
        try:
            # 이미지 로드
            img = Image.open(img_path).convert("RGB")

            # 원본 이미지 저장
            img.save(os.path.join(output_class_dir, f"original_{img_name}"))

            # 증강 이미지 생성
            for i in range(num_augmentations):
                augmented_img = augmentation_transforms(img)
                augmented_img_pil = transforms.ToPILImage()(augmented_img)  # Tensor -> PIL 변환
                augmented_img_pil.save(os.path.join(output_class_dir, f"augmented_{i}_{img_name}"))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# 증강 데이터 생성 실행
for class_name in classes_to_augment:
    print(f"Augmenting data for class: {class_name}")
    augment_class_data(class_name, num_augmentations)

print("Data augmentation completed. Augmented data saved to:", output_dir)
