provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "ml_data_bucket" {
  name                        = var.data_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
}

resource "google_service_account" "ml_sa" {
  account_id   = var.service_account_id
  display_name = "Service account for Junyi ML project"
}

resource "google_project_iam_member" "sa_storage_access" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.ml_sa.email}"
}
