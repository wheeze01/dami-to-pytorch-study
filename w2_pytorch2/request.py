import torch  # PyTorch 모듈
import torch.nn as nn  # PyTorch 신경망 구성 모듈
import torch.optim as optim  # PyTorch 최적화 도구
from torch.utils.data import DataLoader  # 데이터 로딩을 위한 유틸리티
import torchvision  # PyTorch에서 제공하는 컴퓨터 비전 관련 라이브러리
import torchvision.transforms as transforms  # 데이터 변환 도구
import time  # 시간 측정을 위한 모듈

"""
    PyTorch 기반의 분산 데이터 병렬 처리(DDP)를 활용한 CIFAR-10 이미지 분류 모델 학습 코드입니다.
    DDP 적용 후 학습 시간 비교가 목표입니다.
    DDP 예제 코드는 https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html 을 참조하세요.
"""

# CNN 모델 정의
class SimpleCNN(nn.Module):  # nn.Module을 상속받아 사용자 정의 CNN 모델 생성
    """
    간단한 CNN 모델 정의
    - Conv2D 계층 3개와 MaxPooling, Dropout, Fully Connected 계층으로 구성됨
    """
    def __init__(self):  # 생성자 함수
        super(SimpleCNN, self).__init__()  # 부모 클래스 초기화
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 첫 번째 Convolutional Layer. 3 size input, 32x32 size output, 3x3 kernel size
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 두 번째 Convolutional Layer. 32 size input, 64x64 size output, 3x3 kernel size
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)  # 세 번째 Convolutional Layer. 64 size input, 64x64 size output, 3x3 kernel size
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling Layer
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # Fully Connected Layer 1
        self.fc2 = nn.Linear(512, 10)  # Fully Connected Layer 2 (최종 출력)
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.dropout = nn.Dropout(0.25)  # Dropout Layer (과적합 방지)

    def forward(self, x):  # 순전파 함수
        """
        순전파 함수 정의
        """
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPooling
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPooling
        x = self.pool(self.relu(self.conv3(x)))  # Conv3 -> ReLU -> MaxPooling
        x = x.view(-1, 64 * 4 * 4)  # Flatten (2D -> 1D)
        x = self.dropout(self.relu(self.fc1(x)))  # FC1 -> ReLU -> Dropout
        x = self.fc2(x)  # FC2 (최종 출력)
        return x  # 최종 결과 반환

# DDP 환경 설정
def setup(rank, world_size):  # 분산 학습 초기화
    """
    DDP 초기화 설정
    - rank: 프로세스 순서
    - world_size: 총 프로세스 수
    """
    pass  # 나중에 구현 필요

def cleanup():  # 분산 학습 종료
    """
    DDP 종료
    """
    pass  # 나중에 구현 필요

def run_demo(demo_fn, world_size):  # 분산 학습 실행 데모 함수
    """
    DDP 데모 실행
    """
    pass  # 나중에 구현 필요

# 데이터 로드 함수
def load_data(batch_size):  # 학습/테스트 데이터셋 로드
    """
    CIFAR-10 데이터셋 로드 및 DataLoader 생성
    - 데이터 증강 적용 (학습 데이터만)
    - 정규화 수행
    """
    # 학습 데이터 변환: 데이터 증강 및 정규화
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 랜덤으로 32x32 크기로 자르기
        transforms.RandomHorizontalFlip(),  # 랜덤으로 좌우 반전
        transforms.ToTensor(),  # 텐서 변환 : 픽셀값(0~255) -> 실수형(0.001)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 정규화(총 데이터의 평균값, 총 데이터의 표준분포값)
    ])
    
    # 테스트 데이터 변환: 정규화만 적용
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 텐서 변환
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 정규화
    ])

    # 학습 데이터셋 로드
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    # 테스트 데이터셋 로드
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 학습 DataLoader 생성
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 배치 크기
        shuffle=True,  # 데이터 순서 섞기
        num_workers=4,  # 병렬 처리 워커 수
        pin_memory=True  # 고정 메모리 사용 (GPU 성능 최적화)
    )
    
    # 테스트 DataLoader 생성
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 배치 크기
        shuffle=False,  # 데이터 순서 섞지 않음
        num_workers=4,  # 병렬 처리 워커 수
        pin_memory=True  # 고정 메모리 사용
    )
    
    return train_loader, test_loader  # 학습 및 테스트 로더 반환

# 학습 함수
def train_epoch(model, train_loader, criterion, optimizer, epoch, device):  
    """
    한 에폭 동안 학습 수행
    - Loss와 Accuracy 계산
    """
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0  # 누적 손실 초기화
    correct = 0  # 올바른 예측 개수 초기화
    total = 0  # 총 데이터 개수 초기화
    
    for batch_idx, (images, labels) in enumerate(train_loader):  # DataLoader에서 데이터 배치 반복
        images, labels = images.to(device), labels.to(device)  # 데이터를 GPU로 전송
        
        optimizer.zero_grad()  # 이전 그래디언트 초기화
        outputs = model(images)  # 모델 예측
        loss = criterion(outputs, labels)  # 손실 함수 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 업데이트
        
        running_loss += loss.item()  # 손실 누적
        _, predicted = outputs.max(1)  # 가장 높은 점수의 클래스를 예측값으로 선택
        total += labels.size(0)  # 총 데이터 개수 증가
        correct += predicted.eq(labels).sum().item()  # 맞춘 개수 증가
        
        # 진행 상황 출력
        if batch_idx % 100 == 0:  # 매 100번째 배치마다 출력
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.3f}%')
    
    return running_loss / len(train_loader), 100. * correct / total  # 평균 손실 및 정확도 반환


def validate(model, test_loader, criterion, device):
    """
    검증 수행
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

#DistributedDataParallel 적용하기!( main = demo_basic())
def main():
    """
    메인 학습 함수
    """
    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 하이퍼파라미터 설정
    batch_size = 256
    num_epochs = 100
    learning_rate = 0.1
    
    # 데이터 로드
    train_loader, test_loader = load_data(batch_size)
    
    # 모델 설정
    model = SimpleCNN().to(device)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 결과 저장을 위한 리스트
    epoch_times = []
    
    # 학습 루프
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        scheduler.step()
        
        print(f'\nEpoch: {epoch}')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
        print(f'Epoch Time: {epoch_time:.2f} seconds\n')
        
    # 평균 에폭 시간 계산
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f'\nAverage epoch time: {avg_epoch_time:.2f} seconds')
    

if __name__ == "__main__":
    main()