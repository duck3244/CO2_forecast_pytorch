# CO2 Forecast — UML 다이어그램

Mermaid 문법으로 작성된 UML 다이어그램 모음입니다. GitHub/GitLab/VSCode Markdown Preview에서 바로 렌더링됩니다.

---

## 1. 클래스 다이어그램 — Core (PyTorch 모델 & 학습)

```mermaid
classDiagram
    class nn_Module {
        <<PyTorch>>
        +forward(x)
    }

    class CO2LSTM {
        -input_size: int
        -hidden_size: int
        -num_layers: int
        -dropout: float
        -forecast_horizon: int
        -lstm: nn.LSTM
        -fc: nn.Sequential
        +forward(x) Tensor
    }

    class CO2Transformer {
        -d_model: int
        -nhead: int
        -num_layers: int
        -input_projection: nn.Linear
        -pos_encoding: PositionalEncoding
        -transformer: nn.TransformerEncoder
        -fc: nn.Sequential
        +forward(x) Tensor
    }

    class CO2HybridModel {
        -lstm: nn.LSTM
        -input_projection: nn.Linear
        -pos_encoding: PositionalEncoding
        -transformer: nn.TransformerEncoder
        -fusion: nn.Sequential
        +forward(x) Tensor
    }

    class EnsembleModel {
        -models: nn.ModuleList
        -weights: Parameter|Buffer
        +forward(x) Tensor
    }

    class PositionalEncoding {
        -pe: Tensor
        +forward(x) Tensor
    }

    nn_Module <|-- CO2LSTM
    nn_Module <|-- CO2Transformer
    nn_Module <|-- CO2HybridModel
    nn_Module <|-- EnsembleModel
    nn_Module <|-- PositionalEncoding
    EnsembleModel o-- "1..*" nn_Module : wraps
    CO2Transformer *-- PositionalEncoding
    CO2HybridModel *-- PositionalEncoding

    class CO2Dataset {
        -sequences: FloatTensor
        -targets: FloatTensor
        +__len__() int
        +__getitem__(idx) tuple
    }

    class CO2DataLoader {
        -config: dict
        -scaler: StandardScaler
        -feature_scalers: dict
        -trend_series: Series
        -_target_dates: dict
        +download_data() DataFrame
        +create_features(df) DataFrame
        +prepare_data() tuple
        +inverse_transform_co2(arr, split) ndarray
    }

    CO2DataLoader ..> CO2Dataset : creates

    class Trainer {
        -model: nn.Module
        -train_loader: DataLoader
        -val_loader: DataLoader
        -optimizer: Adam
        -scheduler: ReduceLROnPlateau
        -best_val_loss: float
        +train_epoch() float
        +validate() tuple
        +train(progress_callback, cancel_flag) dict
        +save_model(path) void
    }

    class EnsembleTrainer {
        -ensemble_model: EnsembleModel
        -already_trained: bool
        +fine_tune_ensemble() void
    }

    Trainer o-- nn_Module
    EnsembleTrainer o-- EnsembleModel
    EnsembleTrainer --|> Trainer : (similar role)

    class ModelEvaluator {
        -model: nn.Module
        -data_loader: DataLoader
        -scaler: CO2DataLoader
        -split: str
        +predict() tuple
        +evaluate() dict
        +plot_predictions(results, path, title) void
    }

    class MultiModelComparison {
        -results: list
        -names: list
        +compare_metrics() DataFrame
        +plot_comparison(path) void
        +plot_predictions_comparison(path) void
    }

    ModelEvaluator o-- nn_Module
    ModelEvaluator o-- CO2DataLoader : inverse_transform
    MultiModelComparison o-- "many" ModelEvaluator
```

---

## 2. 클래스 다이어그램 — API / Service 계층

```mermaid
classDiagram
    class FastAPI {
        <<framework>>
    }

    class app {
        +include_router(...)
        +GET /api/health
    }
    FastAPI <|-- app

    class APIRouter_datasets {
        +GET /api/datasets/co2
    }
    class APIRouter_models {
        +GET /api/models
    }
    class APIRouter_predictions {
        +POST /api/predictions
    }
    class APIRouter_evaluations {
        +POST /api/evaluations
    }
    class APIRouter_training {
        +POST /api/training/jobs
        +GET  /api/training/jobs
        +GET  /api/training/jobs/:id
        +DELETE /api/training/jobs/:id
        +GET  /api/training/jobs/:id/events (SSE)
    }

    app o-- APIRouter_datasets
    app o-- APIRouter_models
    app o-- APIRouter_predictions
    app o-- APIRouter_evaluations
    app o-- APIRouter_training

    class JobEvent {
        +id: int
        +ts: float
        +type: str
        +data: dict
    }

    class TrainingJob {
        +id: str
        +model: str
        +overrides: dict
        +status: str
        +created_at: float
        +started_at: float
        +finished_at: float
        +events: List~JobEvent~
        +cancel_event: threading.Event
        +push(type, data) void
        +mark_running() void
        +mark_done(status, error, reason) void
        +events_since(last_id) list
        +wait_for_change(last_id, timeout) void
        +is_terminal() bool
        +to_snapshot() dict
    }

    class JobRegistry {
        -_jobs: Dict~str,TrainingJob~
        +create(model, overrides) TrainingJob
        +get(id) TrainingJob
        +list() List~TrainingJob~
    }

    JobRegistry o-- "many" TrainingJob
    TrainingJob o-- "many" JobEvent

    class training_service {
        <<module>>
        -_executor: ThreadPoolExecutor(1)
        +submit(job) Future
        -_run_job(job) void
        -_apply_overrides(cfg, ov) dict
    }

    class inference_service {
        <<module>>
        -_model_cache: Dict (LRU-1)
        -_data_loader: CO2DataLoader
        +predict(name) PredictionResult
        +unload_all() void
    }

    class model_registry {
        <<module>>
        +KNOWN_MODELS: tuple
        +checkpoint_path(name) Path
        +list_models() List~ModelInfo~
        +device_info() tuple
    }

    class dataset_cache {
        <<module>>
        +get_co2_dataframe(force_refresh) tuple
    }

    APIRouter_training ..> JobRegistry : uses
    APIRouter_training ..> training_service : submit
    APIRouter_predictions ..> inference_service : predict
    APIRouter_evaluations ..> inference_service : predict
    APIRouter_models ..> model_registry : list/device
    APIRouter_datasets ..> dataset_cache : get_co2_dataframe
    training_service ..> inference_service : unload_all
    training_service ..> TrainingJob : push/mark
    training_service ..> Trainer : train
    inference_service ..> ModelEvaluator : evaluate
    inference_service ..> model_registry : checkpoint_path
```

---

## 3. 클래스 다이어그램 — Pydantic I/O 스키마

```mermaid
classDiagram
    class BaseModel {
        <<pydantic>>
    }

    class CO2Dataset {
        +source: str
        +n_records: int
        +start_date: str
        +end_date: str
        +dates: List~str~
        +values: List~float~
    }

    class ModelInfo {
        +name: str
        +trained: bool
        +file_size_bytes: int?
        +saved_at: str?
        +checkpoint_path: str?
    }
    class ModelsResponse {
        +models: List~ModelInfo~
        +device: str
        +device_name: str?
    }
    ModelsResponse o-- "many" ModelInfo

    class PredictionRequest {
        +model: str
    }
    class PredictionResponse {
        +model_name: str
        +horizon: int
        +n_sequences: int
        +dates: List~str~
        +actual: List~float~
        +predicted: List~float~
        +metrics: Dict
    }

    class EvaluationRequest {
        +models: List~str~
    }
    class PerModelEvaluation {
        +model_name: str
        +predicted: List~float~
        +metrics: Dict
    }
    class EvaluationResponse {
        +horizon: int
        +n_sequences: int
        +dates: List~str~
        +actual: List~float~
        +results: List~PerModelEvaluation~
        +best_by_r2: str?
    }
    EvaluationResponse o-- "many" PerModelEvaluation

    class TrainingOverrides {
        +epochs: int?
        +learning_rate: float?
        +batch_size: int?
        +patience: int?
        +weight_decay: float?
    }
    class TrainingJobRequest {
        +model: str
        +overrides: TrainingOverrides?
    }
    class TrainingJobSnapshot {
        +id: str
        +model: str
        +overrides: TrainingOverrides
        +status: str
        +created_at: float
        +started_at: float?
        +finished_at: float?
        +stopped_reason: str?
        +error: str?
        +n_events: int
    }
    class TrainingJobsResponse {
        +jobs: List~TrainingJobSnapshot~
    }
    TrainingJobRequest o-- TrainingOverrides
    TrainingJobSnapshot o-- TrainingOverrides
    TrainingJobsResponse o-- "many" TrainingJobSnapshot

    BaseModel <|-- CO2Dataset
    BaseModel <|-- ModelInfo
    BaseModel <|-- ModelsResponse
    BaseModel <|-- PredictionRequest
    BaseModel <|-- PredictionResponse
    BaseModel <|-- EvaluationRequest
    BaseModel <|-- PerModelEvaluation
    BaseModel <|-- EvaluationResponse
    BaseModel <|-- TrainingOverrides
    BaseModel <|-- TrainingJobRequest
    BaseModel <|-- TrainingJobSnapshot
    BaseModel <|-- TrainingJobsResponse
```

---

## 4. 시퀀스 다이어그램 — 추론 (`POST /api/predictions`)

```mermaid
sequenceDiagram
    actor User as Browser
    participant FE as Frontend (client.ts)
    participant R as predictions router
    participant IS as inference_service
    participant MR as model_registry
    participant DL as CO2DataLoader
    participant ME as ModelEvaluator
    participant FS as Filesystem

    User->>FE: select model → click "Predict"
    FE->>R: POST /api/predictions (model)
    R->>R: validate model ∈ KNOWN_MODELS
    R->>IS: run_in_threadpool(predict, name)
    IS->>IS: _get_model_cached(name)
    alt cache miss
        IS->>IS: evict previous model
        IS->>MR: checkpoint_path(name)
        MR-->>IS: Path
        IS->>FS: torch.load(path)
        FS-->>IS: state_dict + config
        IS->>IS: load_state_dict → to(device).eval()
    end
    IS->>DL: _get_data_loader() / _get_test_loader()
    Note over DL: lazy prepare_data() on first call
    IS->>ME: ModelEvaluator(model, test_loader, scaler=DL).evaluate()
    ME->>ME: predict() — no_grad loop
    ME->>DL: inverse_transform_co2(pred, split="test")
    DL-->>ME: 원단위 복원
    ME-->>IS: metrics + predictions + targets
    IS-->>R: PredictionResult
    R-->>FE: PredictionResponse (JSON)
    FE-->>User: 차트 갱신
```

---

## 5. 시퀀스 다이어그램 — 학습 + SSE (`POST /api/training/jobs`)

```mermaid
sequenceDiagram
    actor User as Browser
    participant FE as Training page
    participant RT as training router
    participant Reg as JobRegistry
    participant TS as training_service
    participant Exec as ThreadPool(1)
    participant IS as inference_service
    participant TR as Trainer
    participant FS as Filesystem
    participant SSE as EventSource

    User->>FE: submit form (model, overrides)
    FE->>RT: POST /api/training/jobs
    RT->>Reg: create(model, overrides)
    Reg-->>RT: TrainingJob (queued)
    RT->>TS: submit(job)
    TS->>Exec: _run_job(job)
    RT-->>FE: 202 — job_id + status="queued"

    FE->>SSE: new EventSource(/api/training/jobs/:id/events)
    SSE->>RT: GET .../events  (Last-Event-ID?)
    RT->>Reg: get(id)
    Reg-->>RT: job

    par Background training
        Exec->>IS: unload_all()  (GPU 회수)
        Exec->>TR: Trainer(...).train(cb, cancel_flag)
        loop each epoch
            TR->>TR: train_epoch() / validate()
            TR->>Exec: cb(epoch, losses, metrics)
            Exec->>Reg: job.push progress payload
            Reg->>Reg: Condition.notify_all()
            alt cancel_flag.is_set()
                TR-->>Exec: return stopped_reason="cancelled"
            end
        end
        Exec->>FS: trainer.save_model(checkpoint_path)
        Exec->>IS: unload_all()  (캐시 무효화)
        Exec->>Reg: job.push completed payload
        Exec->>Reg: job.mark_done("completed")
    and SSE streaming
        loop until terminal
            RT->>Reg: events_since(cursor)
            Reg-->>RT: new events
            RT-->>SSE: event progress|log|completed ...
            SSE-->>FE: onMessage → setLossHistory / setLogs
            RT->>Reg: wait_for_change(cursor, 0.3s)
            opt idle > 15s
                RT-->>SSE: event heartbeat
            end
        end
        RT-->>SSE: event done — status + reason + error
        SSE-->>FE: onDone → close connection
    end

    opt User cancels
        User->>FE: click "Cancel"
        FE->>RT: DELETE /api/training/jobs/:id
        RT->>Reg: get(id) → cancel_event.set()
        RT-->>FE: cancelled=true
    end
```

---

## 6. 시퀀스 다이어그램 — 다모델 비교 (`POST /api/evaluations`)

```mermaid
sequenceDiagram
    actor User
    participant FE as Comparison page
    participant RE as evaluations router
    participant IS as inference_service

    User->>FE: pick ["lstm","hybrid","ensemble"]
    FE->>RE: POST /api/evaluations (models)
    RE->>RE: validate ⊂ KNOWN_MODELS
    loop for each name in models
        RE->>IS: predict(name)
        IS-->>RE: PredictionResult
    end
    RE->>RE: pick best_by_r2 = argmax(metrics.R2)
    RE-->>FE: EvaluationResponse — dates, actual, results[], best_by_r2
    FE-->>User: overlay chart + metrics table
```

---

## 7. 컴포넌트 다이어그램 (배포 관점)

```mermaid
flowchart LR
    subgraph Browser
      SPA[React SPA<br/>Dashboard · Predictions · Comparison · Training]
    end

    subgraph Frontend_Build["Vite dev :5173 (dev only)"]
      Vite[Vite dev server<br/>/api proxy → :8000]
    end

    subgraph Backend_Process["uvicorn :8000 (backend/app.py)"]
      direction TB
      FAPI[FastAPI app]
      subgraph Routers
        R1[datasets]
        R2[models]
        R3[predictions]
        R4[evaluations]
        R5[training + SSE]
      end
      subgraph Services
        S1[dataset_cache]
        S2[model_registry]
        S3[inference_service<br/>LRU model cache]
        S4[training_service<br/>ThreadPool 1]
      end
      subgraph Core["PyTorch Core (backend/src/)"]
        C1[CO2DataLoader]
        C2[Models: LSTM/Transformer/Hybrid/Ensemble]
        C3[Trainer · EnsembleTrainer]
        C4[ModelEvaluator]
      end
      Static[(backend/static/<br/>SPA build, prod only)]
    end

    FS[(backend/models/*.pth<br/>backend/.data_cache/*.csv)]
    NOAA[(NOAA gml.noaa.gov)]

    SPA -- dev --> Vite --> FAPI
    SPA -- prod (same origin) --> FAPI
    FAPI --> R1 & R2 & R3 & R4 & R5
    R1 --> S1
    R2 --> S2
    R3 --> S3
    R4 --> S3
    R5 --> S4
    S3 --> C4
    S3 --> C2
    S4 --> C3
    S4 --> C1
    S4 --> C2
    C3 --> FS
    S1 --> FS
    S1 -- first fetch --> NOAA
    S2 --> FS
    FAPI -. prod only .- Static
    Static -. serve SPA .- SPA
```

---

## 8. 상태 다이어그램 — TrainingJob

```mermaid
stateDiagram-v2
    [*] --> queued : registry.create()
    queued --> running : mark_running()<br/>(worker picks up)
    running --> completed : train() 정상 종료<br/>mark_done("completed")
    running --> cancelled : cancel_event.set()<br/>→ stopped_reason="cancelled"
    running --> failed : Exception<br/>mark_done("failed", error)
    completed --> [*]
    cancelled --> [*]
    failed --> [*]

    note right of running
      epoch 단위로
      progress / log 이벤트 push
      (SSE 소비자가 구독)
    end note
    note right of queued
      ThreadPool(max_workers=1)
      → 동시 학습 불가
    end note
```

---

## 9. 프론트엔드 라우트 / 컴포넌트

```mermaid
flowchart TB
    Main[main.tsx] --> App
    App[App.tsx<br/>BrowserRouter + Suspense] --> Layout
    Layout[layout/Layout.tsx] --> Sidebar[layout/Sidebar.tsx]
    Layout --> Outlet
    Outlet --> Dashboard[pages/Dashboard]
    Outlet --> Predictions[pages/Predictions]
    Outlet --> Comparison[pages/Comparison]
    Outlet --> Training[pages/Training]

    Dashboard --> Chart1[charts/TimeSeriesChart]
    Dashboard --> Metrics1[ui/MetricsStrip]
    Predictions --> Chart1
    Predictions --> Select1[ui/ModelSelect]
    Predictions --> Metrics1
    Comparison --> Chart1
    Comparison --> Select1
    Training --> Chart1
    Training --> Select1
    Training --> SSE[hooks/useSSE]

    Dashboard -.uses.-> ApiClient
    Predictions -.uses.-> ApiClient
    Comparison -.uses.-> ApiClient
    Training -.uses.-> ApiClient
    SSE -.EventSource.-> TrainingEvents[/api/training/jobs/:id/events/]
    ApiClient[api/client.ts] --> BackendAPI[/api/*/]
```
