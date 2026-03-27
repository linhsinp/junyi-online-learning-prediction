# Quality and Delivery

```mermaid
flowchart TD
    subgraph Code[Repository Code]
        APP[junyi_predictor/ and orchestration/]
        TESTS[tests/unit and tests/acceptance]
        INFRA[infra/helm and infra/docker]
    end

    subgraph Local[Local Validation]
        LINT[make lint]
        PYTEST[make test]
        HELMLINT[make helm-lint]
    end

    subgraph CI[GitHub Actions]
        PUSH[push event]
        WORKFLOW[.github/workflows/ci-cd.yml]
        UV[uv sync --all-groups]
        CILINT[make lint]
        CITEST[make test]
    end

    APP --> LINT
    APP --> PYTEST
    INFRA --> HELMLINT
    TESTS --> PYTEST

    PUSH --> WORKFLOW
    WORKFLOW --> UV
    UV --> CILINT
    UV --> CITEST
```

## Notes

- Unit and acceptance coverage are separated under `tests/`.
- CI currently validates Python code on every push with `uv`, `make lint`, and `make test`.
- Helm validation is available locally through `make helm-lint` and complements the Python test flow.
