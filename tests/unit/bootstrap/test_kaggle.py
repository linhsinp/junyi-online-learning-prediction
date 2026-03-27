from __future__ import annotations

from junyi_predictor.bootstrap.kaggle import KAGGLE_DATASET, download_kaggle_data


def test_download_kaggle_data_authenticates_and_downloads(monkeypatch, tmp_path):
    calls: list[tuple[str, str, bool]] = []

    class FakeApi:
        def authenticate(self):
            calls.append(("auth", "", False))

        def dataset_download_files(self, dataset: str, path: str, unzip: bool):
            calls.append((dataset, path, unzip))

    monkeypatch.setattr(
        "junyi_predictor.bootstrap.kaggle._get_kaggle_api_class",
        lambda: FakeApi,
    )

    destination = tmp_path / "raw"
    download_kaggle_data(destination)

    assert destination.exists()
    assert calls == [
        ("auth", "", False),
        (KAGGLE_DATASET, str(destination), True),
    ]
