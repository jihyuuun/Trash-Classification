# Trash-Classification
CNN을 활용한 재활용품 분류 시스템

📅 **기간**: Sep 2024 – Dec 2024

## 📖 프로젝트 개요 

본 프로젝트는 **CNN(Convolutional Neural Network)**을 활용하여 다양한 **재활용 쓰레기를 자동으로 분류**하는 시스템을 개발하는 것을 목표로 합니다.

## 🔍 프로젝트 필요성  

- 분리수거가 제대로 이루어지지 않아 선별장에서 많은 인력이 직접 분류함
- 사고 위험이 높고, 선별되지 않은 재활용 쓰레기는 소각 및 매립됨
- 딥러닝을 활용한 기존 선행 연구들은 많았지만 플라스틱, 비닐, 스티로폼, 종이,유리, 일반쓰레기로 세부적으로 나누어 각각 분류하는 연구는 찾아보기 어려움

## 🛠 기술 스택  

### **프로그래밍 언어**  
- **Python**: 모델 구현 및 데이터 처리  

### **데이터 전처리**  
- **pandas**: 데이터 분석 및 테이블 변환  
- **numpy**: 행렬 연산 및 수치 계산  
- **albumentations**: 이미지 데이터 증강(Augmentation)  
- **Pillow (PIL)**: 이미지 리사이징 및 변환  

### **딥러닝 프레임워크**
- **PyTorch**: CNN 모델 구현 및 학습  
- **torchvision**: ResNet 모델 로드 및 이미지 변환  

### **모델 구조 및 학습**
- **ResNet18, ResNet50**: 사전 학습된 CNN 모델 활용  
- **torch.nn (PyTorch)**: 손실 함수 및 활성화 함수 사용  
- **Adam Optimizer**: 최적화 알고리즘 적용  

### **성능 평가 & 시각화**
- **scikit-learn (sklearn.metrics)**: 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-score 평가  
- **seaborn & matplotlib**: 혼동 행렬(Confusion Matrix) 시각화  

### **실행 코드 및 배포**
- **Python 스크립트** (`model_training.py`, `model_testing.py`, `model_evaluation.py`)  
- **CLI 실행 지원**: `python model_training.py` 로 모델 학습 가능

## 📂 프로젝트 구조  

```
data/
├── after/
│   ├── cardboard             # 박스 사진
│   ├── glass                 # 유리병 사진
│   ├── metal                 # 철 캔 사진
│   ├── paper                 # 종이 사진
│   ├── plastic               # 플라스틱 사진
│   ├── styrofoam             # 스티로폼 사진
│   ├── trash                 # 일반쓰레기 사진
│   ├── vinyl                 # 비닐 사진
modeling/
├── class_names.json          # 클래스 이름
├── model_evaluation.py       # 모델 검증 코드
├── model_testing.py          # 모델 평가 코드
├── model_training.py         # 모델 학습 코드
preprocessing/
├── data_augmentation.py      # 데이터 증강 코드
├── resize.py                 # 이미지 크기 조정 코드
README.md                     # 프로젝트 설명
```

## 📊 데이터 설명

### **데이터 출처**  
- TrashNet Dataset (garythug/trashnet)  
- 직접 수집한 이미지
- 클래스 세분화: 기존 `trash` 클래스를 `styrofoam`, `vinyl`로 분리
  
![image](https://github.com/user-attachments/assets/5a1488bc-4335-47e9-8baf-9e6d4b240283)

### **데이터 전처리**  
- 이미지 크기 조정 (모든 이미지 514 x 384로 리사이징)  
- 데이터 정규화 및 증강 적용

## 🎯 모델 학습

### **1️⃣ ResNet18 적용**

- **모델링**
  - Pretrained Model: ResNet18 (사전 학습된 모델 활용)
  - Loss Function: CrossEntropyLoss
  - Optimizer: Adam
  - Batch Size: 32
  - Learning Rate: 0.001
  - Epochs: 10
  - Output Nodes: 8개 클래스에 맞게 Fully Connected Layer 수정
- **실험 과정**
  - Fine-tuning 적용: 사전 학습된 모델을 기반으로 일부 층을 동결하고 학습을 진행함
  - 데이터 증강: 좌우 반전, ±15도 회전, 밝기, 대비, 채도 변화 적용
  - 클래스별 가중치 부여: 데이터 불균형 문제 해결을 위해 클래스별 가중치를 CrossEntropyLoss에 반영
  - 혼동 행렬 분석: 특정 클래스(Trash, Vinyl)에서 낮은 성능이 나타남 → 추가 데이터 증강 필요
  - 추가적인 데이터 증강 수행: Trash 및 Vinyl 클래스 데이터 5배 증강 후 재학습
  - 클래스별 가중치 재조정: 재증강된 데이터를 반영하여 클래스별 가중치를 다시 조정
- **실험 결과**
  - 클래스별 Precision, Recall, F1-score 개선됨
  - Trash 클래스의 Recall이 1.00으로 상승했지만, 일부 클래스에서 성능 저하 발생
  - Plastic 클래스가 Glass 클래스로 혼동되는 문제가 남아 있음
  - 전반적인 모델 성능은 개선되었으나, 추가적인 개선이 필요하여 ResNet50을 실험하기로 결정
 
### **2️⃣ ResNet50 적용**

- **모델링**
  - Pretrained Model: ResNet50 (사전 학습된 모델 활용)
  - Loss Function: CrossEntropyLoss
  - Optimizer: Adam
  - Batch Size: 64
  - Learning Rate: 0.001
  - Epochs: 5
  - Sampling Fraction: 0.3
  - Output Nodes: 8개 클래스에 맞게 Fully Connected Layer 수정
- **실험 과정**
  - 초기 학습 시 손실 값이 불안정하거나 계산 오류 발생 → Learning Rate 감소 필요
  - Learning Rate 0.0001로 감소: 모델 출력값에 NaN이 포함되는 문제 해결
  - Sampling Fraction 1.0으로 증가: 학습 데이터 수를 늘려 모델 성능 개선
  - Fine-tuning 및 Frozen Layers 적용: 초기 층을 동결하고 고수준 층만 학습하여 학습 시간 단축 및 과적합 방지
  - 클래스별 가중치 조정: 클래스 불균형 문제 해결을 위해 가중치 적용
- **실험 결과**
  - Train Loss 및 Validation Loss가 지속적으로 감소하며 학습 안정화
  - Validation Accuracy: 5번째 Epoch에서 98.39% 도달
  - Precision, Recall, F1-score가 모두 0.99 이상으로 매우 높은 성능 기록
  - 모델 성능이 이전보다 크게 개선되었으며, 모든 클래스에서 높은 정확도를 보임
  - 테스트 데이터에서도 모든 클래스가 정확히 분류됨 → 최종 모델로 선택
