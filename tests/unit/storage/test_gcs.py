from __future__ import annotations

from unittest.mock import MagicMock

from junyi_predictor.storage.gcs import download_blob, upload_blob, upload_folder


def test_download_blob_uses_bucket_blob_and_download(monkeypatch):
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    monkeypatch.setattr(
        "junyi_predictor.storage.gcs.storage.Client", lambda: mock_client
    )

    download_blob("bucket", "path/file.csv", "/tmp/file.csv")

    mock_client.bucket.assert_called_once_with("bucket")
    mock_bucket.blob.assert_called_once_with("path/file.csv")
    mock_blob.download_to_filename.assert_called_once_with("/tmp/file.csv")


def test_upload_blob_uses_bucket_blob_and_upload(monkeypatch):
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    monkeypatch.setattr(
        "junyi_predictor.storage.gcs.storage.Client", lambda: mock_client
    )

    upload_blob("bucket", "/tmp/file.csv", "remote/file.csv")

    mock_client.bucket.assert_called_once_with("bucket")
    mock_bucket.blob.assert_called_once_with("remote/file.csv")
    mock_blob.upload_from_filename.assert_called_once_with("/tmp/file.csv")


def test_upload_folder_uploads_all_files_with_relative_paths(monkeypatch, tmp_path):
    root = tmp_path / "feature_store"
    nested = root / "nested"
    nested.mkdir(parents=True)
    first = root / "first.txt"
    second = nested / "second.txt"
    first.write_text("a")
    second.write_text("b")

    uploaded: list[tuple[str, str]] = []

    class FakeBlob:
        def __init__(self, path: str):
            self.path = path

        def upload_from_filename(self, filename: str):
            uploaded.append((self.path, filename))

    mock_bucket = MagicMock()
    mock_bucket.blob.side_effect = lambda path: FakeBlob(path)
    mock_client = MagicMock()
    mock_client.get_bucket.return_value = mock_bucket

    monkeypatch.setattr(
        "junyi_predictor.storage.gcs.storage.Client", lambda: mock_client
    )

    upload_folder(local_folder=str(root), target_dir="target")

    assert ("target/first.txt", str(first)) in uploaded
    assert ("target/nested/second.txt", str(second)) in uploaded
