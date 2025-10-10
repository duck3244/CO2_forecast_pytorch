# PyTorch를 활용한 CO2 농도 예측 프로젝트

하이브리드 모델과 앙상블 기법을 활용한 포괄적인 딥러닝 CO2 농도 예측 시스템입니다.

## 🌟 주요 기능

- **다양한 모델 아키텍처**: LSTM, Transformer, 하이브리드 모델
- **앙상블 학습**: 여러 모델을 결합하여 향상된 성능 구현
- **포괄적인 평가**: 다양한 지표와 시각화 도구 제공
- **자동 데이터 파이프라인**: 자동 데이터 다운로드 및 전처리
- **설정 기반 시스템**: YAML 파일을 통한 간편한 파라미터 조정
- **확장 가능한 설계**: 새로운 모델과 기능 추가 용이

## 🏗️ 프로젝트 구조

```
CO2_forecast_pytorch/
├── configs/
│   └── config.yaml          # 설정 파일
├── src/
│   ├── data/
│   │   └── data_loader.py    # 데이터 로딩 및 전처리
│   ├── models/
│   │   └── models.py         # 모델 아키텍처들
│   ├── training/
│   │   ├── trainer.py        # 훈련 로직
│   │   └── metrics.py        # 평가 지표
│   └── evaluation/
│   │   └── evaluator.py      # 모델 평가 및 시각화
│   └── utils/
│       ├── visualization.py
├── examples/
│   └── quick_start.py        # 빠른 시작 예제
├── data/                     # 데이터 디렉토리 (자동 생성)
├── models/                   # 저장된 모델들 (자동 생성)
├── logs/                     # 훈련 로그 (자동 생성)
├── plots/                    # 생성된 그래프들 (자동 생성)
├── main.py                   # 메인 실행 스크립트
├── requirements.txt          # 의존성 패키지
└── README.md                 # 이 파일
```

## 🚀 시작하기

### 1. 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정

`configs/config.yaml` 파일에서 다음 항목들을 조정할 수 있습니다:
- 모델 파라미터
- 훈련 설정
- 데이터 전처리 옵션
- 평가 지표

### 3. 실행

```bash
# 모든 모델 훈련 및 평가
python main.py --mode all

# 특정 모델만 훈련
python main.py --mode train --model lstm

# 훈련된 모델 평가
python main.py --mode evaluate

# 앙상블 모델 훈련
python main.py --mode train --model ensemble

# 빠른 시작 예제 실행
python examples/quick_start.py
```

## 📊 모델 소개

### LSTM 모델
- 양방향 LSTM 레이어
- 드롭아웃을 통한 정규화
- 완전연결 출력 레이어
- 순차적 의존성 처리에 특화

### Transformer 모델
- 멀티헤드 어텐션 메커니즘
- 위치 인코딩
- 레이어 정규화
- 장기 의존성 포착에 우수

### 하이브리드 모델
- LSTM과 Transformer 결합
- 피처 융합 레이어
- 두 아키텍처의 장점 활용
- 향상된 성능 기대

### 앙상블 모델
- 개별 모델들의 가중 조합
- 학습 가능하거나 고정된 가중치
- 향상된 견고성과 정확도

## 📈 데이터

프로젝트는 NOAA Mauna Loa Observatory의 CO2 데이터를 자동으로 다운로드합니다:
- 1958년부터 현재까지의 월별 CO2 측정값
- 자동 전처리 및 피처 엔지니어링
- 훈련/검증/테스트 데이터 분할

### 생성되는 피처들:
- 계절성 피처 (sin/cos 인코딩)
- 트렌드 피처
- 지연 피처 (lag features)
- 이동평균

## 🔧 설정 옵션

### 데이터 설정
```yaml
data:
  sequence_length: 24      # 입력 시퀀스 길이 (월 단위)
  forecast_horizon: 12     # 예측 기간 (월 단위)
  train_ratio: 0.8        # 훈련 데이터 비율
  val_ratio: 0.1          # 검증 데이터 비율
```

### 모델 설정
```yaml
models:
  lstm:
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
  
  transformer:
    d_model: 64
    nhead: 8
    num_layers: 4
```

### 훈련 설정
```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 200
  patience: 20            # 조기 종료
```

## 📊 평가 지표

- **MSE**: 평균 제곱 오차
- **RMSE**: 평균 제곱근 오차
- **MAE**: 평균 절대 오차
- **MAPE**: 평균 절대 백분율 오차
- **R²**: 결정 계수

## 📈 시각화

프로젝트는 다음과 같은 포괄적인 시각화를 제공합니다:

1. **훈련 과정**: 손실 곡선과 지표 변화
2. **예측 vs 실제값**: 시계열 그래프와 산점도
3. **잔차 분석**: 오차 분포와 패턴
4. **모델 비교**: 여러 모델의 성능 비교
5. **예측 그래프**: 미래 예측과 신뢰 구간

## 🛠️ 고급 사용법

### 새로운 모델 추가

`src/models/models.py`에서 새 모델을 구현하고:
1. 모델 클래스 정의
2. `create_model` 함수에 추가
3. 설정 파일 업데이트

### 피처 엔지니어링 확장

`src/data/data_loader.py`에서 피처를 확장할 수 있습니다:
- 경제 지표 추가
- 기상 데이터 포함
- 사용자 정의 변환 구현

### 하이퍼파라미터 튜닝

설정 파일을 통해 다음을 실험할 수 있습니다:
- 학습률
- 모델 아키텍처
- 정규화 파라미터
- 훈련 스케줄

## 📋 요구사항

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- Requests

---

*실제 성능은 데이터와 설정에 따라 달라질 수 있습니다.