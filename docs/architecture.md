# CO2 Forecast — 시스템 아키텍처

NOAA Mauna Loa CO2 월별 데이터를 기반으로 LSTM / Transformer / Hybrid / Ensemble 딥러닝 모델을 학습·평가·예측하는 풀스택 애플리케이션입니다. PyTorch 학습 코어 위에 FastAPI 서비스 레이어와 React(Vite) SPA를 올린 3계층 구조입니다.

---

## 1. 상위 구성도

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (SPA)                              │
│   React 18 + React Router + Recharts + Tailwind (Vite dev :5173)  │
│   ├─ pages/ (Dashboard, Predictions, Comparison, Training)        │
│   ├─ api/client.ts  ─ fetch wrapper                               │
│   └─ hooks/useSSE.ts ─ EventSource wrapper                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (JSON) + SSE
                             │ dev: /api → 127.0.0.1:8000 (Vite proxy)
                             │ prod: 같은 오리진 (/api/*, /assets/*)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              FastAPI (backend/app.py, uvicorn :8000)              │
│                                                                   │
│  api/routers/                                                     │
│   ├─ datasets.py        GET  /api/datasets/co2                    │
│   ├─ models.py          GET  /api/models                          │
│   ├─ predictions.py     POST /api/predictions                     │
│   ├─ evaluations.py     POST /api/evaluations                     │
│   └─ training.py        POST /api/training/jobs                   │
│                         GET  /api/training/jobs[/{id}[/events]]   │
│                         DEL  /api/training/jobs/{id}              │
│                                                                   │
│  api/state.py       JobRegistry (in-memory) + TrainingJob + SSE   │
│  api/schemas.py     Pydantic I/O                                  │
│  api/config.py      configs/config.yaml 로더                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 함수 호출
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Services Layer                           │
│                                                                   │
│  services/dataset_cache.py   NOAA 다운로드 + .data_cache/ 캐시      │
│  services/model_registry.py  체크포인트 경로/메타 + device 정보       │
│  services/inference_service.py  모델 LRU(1) 캐시 + predict()       │
│  services/training_service.py   ThreadPool(1) + 진행 콜백 → SSE    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Core (PyTorch) — backend/src/                  │
│                                                                   │
│  data/data_loader.py    CO2DataLoader · CO2Dataset(Dataset)       │
│                         · detrend / yoy_diff / lag / seasonal     │
│  models/models.py       CO2LSTM · CO2Transformer · CO2HybridModel │
│                         · EnsembleModel · create_model / ensemble │
│  training/trainer.py    Trainer · EnsembleTrainer                 │
│  training/metrics.py    MSE/RMSE/MAE/MAPE/R2                      │
│  evaluation/evaluator.py ModelEvaluator · MultiModelComparison    │
│  utils/visualization.py  plotly/matplotlib 플롯                     │
│                                                                   │
│  backend/main.py         CLI (--mode train|evaluate|all)          │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 파일 I/O
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Persistence / Filesystem                    │
│   backend/models/{lstm,transformer,hybrid,ensemble}_model.pth     │
│   backend/.data_cache/co2_mm_mlo.csv                              │
│   backend/plots/*.png        backend/logs/                        │
└──────────────────────────────────────────────────────────────────┘
                             ▲
                             │ 최초 요청 시 download
                             │
                     NOAA Mauna Loa CO2 text feed
                     gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt
```

---

## 2. 계층 책임

| 계층 | 위치 | 역할 |
|---|---|---|
| **Presentation** | `frontend/src/` | SPA. 라우팅, 차트, SSE 구독, 사용자 입력 폼 |
| **API** | `backend/api/` | HTTP/SSE 엔드포인트, 입력 검증(Pydantic), 에러 매핑 |
| **Service** | `backend/services/` | 유스케이스 오케스트레이션: 학습 잡 큐, 추론 캐시, 데이터 캐시, 체크포인트 |
| **Core** | `backend/src/` | 순수 PyTorch 학습/평가 로직. 프레임워크/HTTP 비의존 |
| **CLI** | `backend/main.py` | API를 거치지 않고 Core를 직접 구동하는 스크립트 경로 |
| **Persistence** | `backend/models/`, `backend/.data_cache/` | 체크포인트 `.pth`, 원천 데이터 CSV 캐시 |

핵심 원칙: **Core는 하부(API, 서비스)를 모른다.** 서비스는 Core를 호출하고, API는 서비스만 호출한다 — CLI와 웹 서비스가 동일한 Core를 공유할 수 있는 이유입니다.

---

## 3. 주요 플로우

### 3.1 데이터셋 조회 — `GET /api/datasets/co2`

```
Frontend ── fetch /api/datasets/co2 ──▶ datasets router
                                         │
                                         ▼
                                  dataset_cache.get_co2_dataframe()
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                     CACHE_FILE 존재?         NOAA HTTP GET
                              │                     │
                              ▼                     ▼
                         CSV 로드              parse + CSV 저장
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                                  CO2Dataset 스키마로 반환
```

- 최초 요청 1회만 NOAA에 접근, 이후는 `backend/.data_cache/co2_mm_mlo.csv` 캐시 사용.
- `?force_refresh=true`로 강제 갱신 가능.

### 3.2 추론 — `POST /api/predictions`

```
POST { model: "lstm" }
        │
        ▼
predictions router ─ run_in_threadpool ─▶ inference_service.predict(name)
                                             │
                                             ├─ _get_model_cached(name)
                                             │    └─ LRU(size=1) → 이전 모델 evict
                                             │       → torch.load(checkpoint_path)
                                             ├─ _get_data_loader()  (lazy CO2DataLoader)
                                             └─ ModelEvaluator(test_loader).evaluate()
                                                 └─ inverse_transform_co2 → 원단위 복원
                                             │
                                             ▼
                                      PredictionResult (dates, actual, predicted, metrics)
        ▼
PredictionResponse (JSON)
```

- 모델 캐시 크기 1 — GPU 메모리 보호. 다른 모델 요청 시 기존 모델 해제 후 로드.
- 학습 잡이 시작되면 `training_service`가 `inference_service.unload_all()`을 호출해 충돌 방지.

### 3.3 학습 — `POST /api/training/jobs` + SSE

```
POST { model, overrides }
        │
        ▼
training router ─ registry.create(job) ─▶ JobRegistry (in-memory)
                ─ training_service.submit(job) ─▶ ThreadPoolExecutor(max_workers=1)
                ─ 202 { job_id, status: "queued" }
                                                    │
                                                    ▼ (백그라운드 스레드)
                                           _run_job(job)
                                            ├─ inference_service.unload_all()
                                            ├─ CO2DataLoader.prepare_data()
                                            ├─ Trainer(...).train(
                                            │       progress_callback=cb,
                                            │       cancel_flag=job.cancel_event)
                                            │    · 에폭마다 cb() → job.push("progress", …)
                                            │    · cancel_event 체크 → "cancelled"
                                            ├─ trainer.save_model(checkpoint_path)
                                            └─ job.mark_done() + job.push("completed"|"error")

Frontend ── EventSource(/api/training/jobs/{id}/events) ─▶
  SSE 스트림: log / progress / completed / error / done / heartbeat
  · Condition 기반 대기 — 새 이벤트 도착 시 즉시 flush
  · Last-Event-ID 헤더로 재연결 복구
  · 15초 heartbeat로 프록시 타임아웃 방지
```

- **단일 워커**: 동시 학습 금지 → GPU 메모리 경합 제거, SSE 로그 직렬화.
- **취소**: `DELETE /api/training/jobs/{id}` → `cancel_event.set()` → Trainer가 다음 에폭 경계에서 체크 후 종료.
- **휘발성**: `JobRegistry`는 프로세스 메모리. 서버 재시작 시 잡 히스토리 유실 (MVP 의도).

### 3.4 다모델 비교 — `POST /api/evaluations`

순차적으로 각 모델에 대해 `inference_service.predict`를 호출한 뒤, 동일한 `dates`/`actual` 축을 공유하는 `EvaluationResponse`를 조립하고 R² 최상위 모델을 `best_by_r2`로 선택합니다.

---

## 4. 데이터 파이프라인 (Core)

```
NOAA .txt  ─▶  DataFrame[date, co2]
               │
               ▼
       (옵션) detrend(degree=2)  또는  yoy_diff  ← 상호 배타
               │
               ▼
       create_features:
         · month, month_sin, month_cos
         · trend, trend_normalized
         · co2_lag_{1,12,24}
               │
               ▼
       StandardScaler (train 통계로 fit, 전체 transform)
               │
               ▼
       sliding window → (X: [N, L=24, F=9], y: [N, H=12])
               │
               ▼
       train/val/test = 0.8 / 0.1 / 0.1 (시간 순)
               │
               ▼
       torch.utils.data.DataLoader (batch_size=32)
```

역변환 시 `inverse_transform_co2(..., split=...)`가 스케일·detrend·yoy_diff를 순서대로 되돌려 원단위(ppm)로 복원합니다.

---

## 5. 모델 카탈로그

| 모델 | 계층 | 입력 shape | 출력 shape | 특징 |
|---|---|---|---|---|
| `CO2LSTM` | LSTM → Dense | `(B, 24, 9)` | `(B, 12)` | `num_layers=2`, 선택적 bidirectional |
| `CO2Transformer` | InputProj → PositionalEnc → EncoderLayer×N | 〃 | 〃 | `d_model=64, nhead=8, num_layers=4` |
| `CO2HybridModel` | [LSTM] + [Transformer] → concat → Fusion MLP | 〃 | 〃 | 마지막 타임스텝 피처 결합 |
| `EnsembleModel` | 세 모델 출력 stack → softmax(weights)로 가중합 | 〃 | 〃 | weights는 `nn.Parameter` (학습 가능) |

팩토리: `create_model(type, config)`, `create_ensemble(config)`.

---

## 6. 배포 형태

- **Dev**: Vite(:5173)와 FastAPI(:8000)를 별도 구동, Vite가 `/api`를 8000으로 프록시 (`make dev-backend` + `make dev-frontend`).
- **Prod (단일 포트)**: `make serve` → `vite build` 산출물을 `backend/static/`에 배치, FastAPI가 `/assets/*`와 SPA fallback(`{full_path}`)을 함께 서빙. `/api/*`가 항상 우선.
- **타입 동기화**: `make types` 또는 `npm run types:gen`으로 FastAPI `openapi.json` → `frontend/src/api/schema.d.ts` 재생성.

---

## 7. 관측·제약

- **장애 격리**: 예측/학습 예외는 각 라우터에서 `HTTPException`으로 매핑 (404 / 500 / 502).
- **GPU 메모리**: 학습 시작 전 추론 캐시 비우기, 추론 모델 캐시 크기 1 — 단일 GPU 가정.
- **동시성**: 학습 워커 1개, `threading.Lock` + `Condition`으로 이벤트 큐 보호, FastAPI는 `run_in_threadpool`로 블로킹 추론 우회.
- **설정**: `backend/configs/config.yaml` 단일 파일 (`api.config.load_config()`가 `lru_cache`로 1회 파싱).
