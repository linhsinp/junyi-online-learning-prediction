import os
from unittest.mock import MagicMock

import pytest
from scripts.gcs_utils import download_data_to_tmp


@pytest.fixture
def mock_gcs(monkeypatch):
    # Create a mock blob with a fake name and download method
    mock_blob = MagicMock()
    mock_blob.name = "raw/fake_file.csv"
    mock_blob.download_to_filename = MagicMock()

    # Mock list_blobs to return one fake blob
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.list_blobs.return_value = [mock_blob]
    mock_bucket.list_blobs.return_value = [mock_blob]
    mock_client.bucket.return_value = mock_bucket

    # Patch the GCS client in the module
    monkeypatch.setattr("scripts.gcs_utils.storage.Client", lambda: mock_client)
    monkeypatch.setattr("scripts.gcs_utils.BUCKET_NAME", "fake-bucket")

    return mock_blob


def test_download_data_to_tmp_creates_local_file(mock_gcs):
    # Run the function
    download_data_to_tmp(prefix="raw/")

    # Check if download_to_filename was called with a file in /tmp/data
    expected_path = os.path.join("/tmp/data", "fake_file.csv")
    mock_gcs.download_to_filename.assert_called_once_with(expected_path)

    # Ensure local file directory was created
    assert os.path.isdir(os.path.dirname(expected_path))

    # Cleanup after test
    if os.path.exists("/tmp/data"):
        import shutil
        shutil.rmtree("/tmp/data")
