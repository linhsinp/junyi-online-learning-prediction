# Application Flow

```mermaid
flowchart TD
    subgraph Sources[Data Sources]
        PG[(PostgreSQL)]
        GCS[(Google Cloud Storage)]
        CSV[artifacts/data/raw/*.csv]
    end

    subgraph Orchestration[Flyte 2 Tasks]
        PRETASK[preprocess_from_database]
        FULL[full_pipeline]
        GTRAIN[train_from_gcs]
    end

    subgraph Runtime[Modular Stage Code]
        PRE[junyi_predictor.pipeline.preprocessing]
        FE[junyi_predictor.pipeline.feature_engineering]
        TR[junyi_predictor.pipeline.training]
        STORE[junyi_predictor.storage.gcs]
    end

    subgraph Contracts[Stage Contracts]
        PREOUT[PreprocessStageOutput<br/>log, user, content]
        FEOUT[FeatureStageOutput<br/>log, concept_proficiency, level4_proficiency]
        SPLIT[TrainingSplit<br/>X_train, y_train, X_test, y_test]
    end

    subgraph Outputs[Outputs]
        METRICS[Model metrics]
        LOCAL[artifacts/data/output, artifacts/data/experiment, artifacts/data/feature_store]
        MODEL[artifacts/model/]
    end

    PG --> PRETASK
    PG --> FULL
    PRETASK --> PRE
    FULL --> PRE
    CSV --> PRE

    PRE --> PREOUT
    PREOUT --> FE
    FE --> FEOUT
    FEOUT --> TR
    TR --> SPLIT
    SPLIT --> TR

    GCS --> STORE
    STORE --> GTRAIN
    GTRAIN --> TR

    FE --> LOCAL
    TR --> METRICS
    TR --> MODEL
```

## Notes

- `orchestration/flyte_app.py` is the current Flyte 2 entrypoint.
- Runtime code is intentionally split into preprocessing, feature engineering, and training stages.
- GCS access stays in `junyi_predictor.storage.gcs` so storage concerns do not leak into stage logic.
- Dataclass stage contracts are the main seams between modules.
