# PyTorch를 활용한 CO2 농도 예측 프로젝트

하이브리드 모델과 앙상블 기법을 활용한 포괄적인 딥러닝 CO2 농도 예측 시스템입니다.

## 🌟 주요 기능

- **다양한 모델 아키텍처**: LSTM, Transformer, 하이브리드 모델
- **앙상블 학습**: 여러 모델을 결합하여 향상된 성능 구현
- **포괄적인 평가**: 다양한 지표와 시각화 도구 제공
- **자동 데이터 파이프라인**: 자동 데이터 다운로드 및 전처리
- **설정 기반 시스템**: YAML 파일을 통한 간편한 파라미터 조정
- **확장 가능한 설계**: 새로운 모델과 기능 추가 용이
- **대화형 시각화**: Plotly를 활용한 인터랙티브 그래프 지원

## 🏗️ 프로젝트 구조

```
CO2_forecast_pytorch/
├── configs/
│   └── config.yaml          # 설정 파일
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py    # 데이터 로딩 및 전처리
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py         # 모델 아키텍처들
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # 훈련 로직
│   │   └── metrics.py        # 평가 지표
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py      # 모델 평가 및 시각화
│   └── utils/
│       ├── __init__.py
│       └── visualization.py  # 시각화 유틸리티
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

#### 데이터 설정
```yaml
data:
  url: "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
  sequence_length: 24    # 2년치 월별 데이터
  forecast_horizon: 12   # 1년 앞 예측
  train_ratio: 0.8       # 훈련 데이터 비율
  val_ratio: 0.1         # 검증 데이터 비율
  test_ratio: 0.1        # 테스트 데이터 비율
```

#### 전처리 설정
```yaml
preprocessing:
  normalize: true
  add_seasonal_features: true
  add_trend_features: true
  add_lag_features: true
  lag_periods: [1, 12, 24]
```

#### 모델 설정
```yaml
models:
  lstm:
    input_size: 9
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    bidirectional: false
  
  transformer:
    d_model: 64
    nhead: 8
    num_layers: 4
    dropout: 0.1
    
  hybrid:
    lstm_hidden: 32
    transformer_d_model: 32
    transformer_layers: 2
    fusion_hidden: 64
```

#### 훈련 설정
```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 200
  patience: 20           # 조기 종료
  weight_decay: 0.00001
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 10
  scheduler_factor: 0.5
```

#### 앙상블 설정
```yaml
ensemble:
  models: ["lstm", "transformer", "hybrid"]
  weights: [0.3, 0.3, 0.4]  # 가중치 (학습 가능)
```

### 3. 실행

```bash
# 모든 모델 훈련 및 평가
python main.py --mode all

# 특정 모델만 훈련
python main.py --mode train --model lstm
python main.py --mode train --model transformer
python main.py --mode train --model hybrid
python main.py --mode train --model ensemble

# 훈련된 모델 평가만 수행
python main.py --mode evaluate

# 특정 모델만 평가
python main.py --mode evaluate --model lstm

# 사용자 정의 설정 파일 사용
python main.py --config custom_config.yaml
```

## 📊 모델 소개

### 1. LSTM 모델
**특징:**
- 양방향 LSTM 레이어 (선택 가능)
- 다층 구조 (기본 2층)
- 드롭아웃을 통한 정규화
- 완전연결 출력 레이어
- **적합한 경우**: 순차적 의존성이 강한 데이터, 중단기 패턴 포착

**아키텍처:**
```
Input → LSTM Layer 1 → LSTM Layer 2 → Dense → ReLU → Dropout → Output
```

### 2. Transformer 모델
**특징:**
- 멀티헤드 어텐션 메커니즘
- 위치 인코딩 (Positional Encoding)
- 레이어 정규화
- 병렬 처리 가능
- **적합한 경우**: 장기 의존성 포착, 복잡한 패턴 학습

**아키텍처:**
```
Input → Projection → Positional Encoding → Transformer Encoder → Dense → Output
```

### 3. 하이브리드 모델
**특징:**
- LSTM과 Transformer 결합
- 피처 융합 레이어
- 두 아키텍처의 장점 활용
- **적합한 경우**: 순차적 + 장기 의존성 모두 중요한 경우

**아키텍처:**
```
Input → [LSTM Branch] → Features ↘
                                   → Fusion → Dense → Output
Input → [Transformer Branch] → Features ↗
```

### 4. 앙상블 모델
**특징:**
- 개별 모델들의 가중 조합
- 학습 가능한 가중치
- 향상된 견고성과 정확도
- **적합한 경우**: 최고 성능이 필요한 프로덕션 환경

**앙상블 전략:**
- 각 개별 모델 훈련
- 가중치 파인튜닝
- Softmax를 통한 가중치 정규화

## 📈 데이터

### 데이터 소스
프로젝트는 NOAA Mauna Loa Observatory의 CO2 데이터를 자동으로 다운로드합니다:
- **기간**: 1958년 3월부터 현재까지
- **빈도**: 월별 측정값
- **단위**: ppm (parts per million)
- **자동 처리**: 결측치 처리 및 피처 생성

### 생성되는 피처
1. **계절성 피처**:
   - `month`: 월 정보 (1-12)
   - `month_sin`: sin 변환 월 (주기성 포착)
   - `month_cos`: cos 변환 월 (주기성 포착)

2. **트렌드 피처**:
   - `trend`: 시간 인덱스
   - `trend_normalized`: 정규화된 트렌드

3. **지연 피처** (Lag Features):
   - `co2_lag_1`: 1개월 이전 값
   - `co2_lag_12`: 12개월 이전 값 (1년 전)
   - `co2_lag_24`: 24개월 이전 값 (2년 전)

### 데이터 분할
- **훈련**: 80%
- **검증**: 10%
- **테스트**: 10%

## 🔧 고급 기능

### 1. 조기 종료 (Early Stopping)
```python
training:
  patience: 20  # 20 에포크 동안 개선 없으면 중단
```

### 2. 학습률 스케줄링
```python
training:
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 10
  scheduler_factor: 0.5  # 학습률을 절반으로 감소
```

### 3. 그래디언트 클리핑
```python
# trainer.py에서 자동 적용
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. 가중치 정규화
```python
training:
  weight_decay: 0.00001  # L2 정규화
```

## 📊 평가 지표

프로젝트는 다음과 같은 포괄적인 평가 지표를 제공합니다:

| 지표 | 설명 | 해석 |
|------|------|------|
| **MSE** | 평균 제곱 오차 | 낮을수록 좋음 |
| **RMSE** | 평균 제곱근 오차 | 원래 단위로 해석 가능 |
| **MAE** | 평균 절대 오차 | 이상치에 덜 민감 |
| **MAPE** | 평균 절대 백분율 오차 | 상대적 오차 (%) |
| **R²** | 결정 계수 | 1에 가까울수록 좋음 (최대 1.0) |

## 📈 시각화

### 1. 훈련 과정 시각화
- 손실 곡선 (Training/Validation Loss)
- 스무딩된 손실 곡선
- 에포크별 변화 추이

### 2. 예측 결과 시각화
- 시계열 그래프 (예측 vs 실제)
- 산점도 (Actual vs Predicted)
- 잔차 플롯 (Residuals Plot)
- 잔차 분포 히스토그램

### 3. 모델 비교 시각화
- 지표별 바 차트
- 예측값 비교 그래프
- 성능 비교 테이블

### 4. 데이터 탐색 시각화
- 시계열 분해 (Trend, Seasonal, Residual)
- 월별 박스플롯
- 성장률 그래프
- 분포 히스토그램

### 5. 인터랙티브 시각화 (Plotly)
- 확대/축소 가능한 예측 그래프
- 호버 정보 표시
- HTML 파일로 저장 가능

## 🛠️ 확장 가이드

### 새로운 모델 추가

1. `src/models/models.py`에 새 모델 클래스 정의:
```python
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 모델 정의
    
    def forward(self, x):
        # 순전파 로직
        return output
```

2. `create_model` 함수에 추가:
```python
def create_model(model_type, config):
    if model_type == 'custom':
        return CustomModel(config)
    # ...
```

3. `configs/config.yaml`에 설정 추가:
```yaml
models:
  custom:
    param1: value1
    param2: value2
```

### 피처 엔지니어링 확장

`src/data/data_loader.py`의 `create_features` 메서드 수정:
```python
def create_features(self, df):
    df = df.copy()
    
    # 새로운 피처 추가
    df['custom_feature'] = calculate_custom_feature(df)
    
    return df
```

### 새로운 평가 지표 추가

`src/training/metrics.py`에 추가:
```python
def custom_metric(y_true, y_pred):
    # 계산 로직
    return score

def calculate_metrics(y_true, y_pred):
    metrics = {
        # 기존 지표들...
        'Custom': custom_metric(y_true, y_pred)
    }
    return metrics
```

## 📋 의존성 패키지

주요 패키지:
- **PyTorch** (≥2.0.0): 딥러닝 프레임워크
- **NumPy** (≥1.21.0): 수치 계산
- **Pandas** (≥1.3.0): 데이터 처리
- **Scikit-learn** (≥1.0.0): 전처리 및 평가
- **Matplotlib** (≥3.5.0): 정적 시각화
- **Seaborn** (≥0.11.0): 통계 시각화
- **Plotly** (≥5.0.0): 인터랙티브 시각화
- **Statsmodels** (≥0.13.0): 시계열 분석
- **PyYAML** (≥6.0): 설정 파일 처리
- **tqdm** (≥4.62.0): 진행률 표시
- **Requests** (≥2.25.0): 데이터 다운로드
- **TensorBoard** (≥2.8.0): 실험 추적

## 🎯 모범 사례

### 1. 하이퍼파라미터 튜닝
```yaml
# 낮은 학습률로 시작
learning_rate: 0.001

# 충분한 patience 설정
patience: 20

# 적절한 배치 크기
batch_size: 32  # GPU 메모리에 맞게 조정
```

### 2. 과적합 방지
```yaml
# 드롭아웃 사용
dropout: 0.2

# 가중치 감쇠
weight_decay: 0.00001

# 조기 종료
patience: 20
```

### 3. 시퀀스 길이 선택
```yaml
# 계절성 패턴 포착: 최소 12개월
# 장기 트렌드 포착: 24개월 이상 권장
sequence_length: 24
```

## 🚨 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
batch_size: 16  # 또는 8

# 또는 CPU 사용
device: cpu
```

### 데이터 다운로드 실패
```python
# data_loader.py에서 자동으로 샘플 데이터 생성
# 또는 수동으로 데이터 파일 배치:
# data/co2-ppm-mauna-loa-*.csv
```

### 모델 학습이 불안정할 때
```yaml
# 학습률 낮추기
learning_rate: 0.0001

# 그래디언트 클리핑 확인 (trainer.py에 기본 적용)
# 배치 크기 늘리기
batch_size: 64
```

## 📚 추가 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [NOAA CO2 데이터](https://gml.noaa.gov/ccgg/trends/)

---

*실제 성능은 데이터와 설정에 따라 달라질 수 있습니다.*