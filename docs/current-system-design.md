# Current System Design

```mermaid
flowchart TD
    subgraph Sources[Data Sources]
        PG[(PostgreSQL)]
        GCS[(Google Cloud Storage)]
        CSV[data/raw/*.csv]
    end

    subgraph Orchestration[Orchestration Layer]
        WF[flyte/full_pipeline_wf.py]
        INGEST[flyte/tasks/ingest.py]
    end

    subgraph Core[Modular Execution Code]
        PRE[junyi_predictor.pipeline.preprocessing]
        FE[junyi_predictor.pipeline.feature_engineering]
        TR[junyi_predictor.pipeline.training]
        STORE[junyi_predictor.storage.gcs]
    end

    subgraph Contracts[Stage Contracts]
        PREOUT[PreprocessStageOutput<br/>log, user, content]
        FEOUT[FeatureStageOutput<br/>log, concept_proficiency, level4_proficiency]
        TROS[TrainingSplit<br/>X_train, y_train, X_test, y_test]
    end

    subgraph Outputs[Outputs]
        METRICS[Model metrics]
        LOCAL[data/output, data/experiment, data/feature_store]
        MODEL[model/]
    end

    PG --> WF
    WF --> PRE
    PRE --> PREOUT
    PREOUT --> FE
    FE --> FEOUT
    FEOUT --> TR
    TR --> TROS
    TROS --> TR
    TR --> METRICS

    GCS --> STORE
    STORE --> INGEST
    INGEST --> LOCAL

    CSV --> PRE
    FE --> LOCAL
    TR --> MODEL

    PRE -. reused by .-> data/create_db.py
    STORE -. reused by .-> INGEST
```

## Notes

- `flyte/full_pipeline_wf.py` is the main end-to-end execution path today.
- The core runtime is now stage-based: preprocessing, feature engineering, then training.
- Each stage exposes an explicit contract to reduce cross-stage coupling.
- GCS access is isolated in `junyi_predictor.storage.gcs`.
- Tests cover stage behavior and stage-to-stage integration boundaries in `tests/`.
