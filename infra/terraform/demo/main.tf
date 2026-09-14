locals {
  name_prefix = "junyi-${var.environment}"
  services = toset([
    "artifactregistry.googleapis.com",
    "container.googleapis.com",
    "iamcredentials.googleapis.com",
    "servicenetworking.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
  ])
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "enabled" {
  for_each           = local.services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "main" {
  name                    = "${local.name_prefix}-network"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.enabled]
}

resource "google_compute_subnetwork" "gke" {
  name          = "${local.name_prefix}-gke"
  region        = var.region
  network       = google_compute_network.main.id
  ip_cidr_range = "10.10.0.0/20"
}

resource "google_compute_global_address" "private_services" {
  name          = "${local.name_prefix}-private-services"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}

resource "google_service_networking_connection" "private_services" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_services.name]
}

resource "google_container_cluster" "main" {
  name             = "${local.name_prefix}-gke"
  location         = var.region
  enable_autopilot = true
  network          = google_compute_network.main.id
  subnetwork       = google_compute_subnetwork.gke.id

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  depends_on = [google_project_service.enabled]
}

resource "google_artifact_registry_repository" "runtime" {
  location      = var.region
  repository_id = "${local.name_prefix}-runtime"
  format        = "DOCKER"
  depends_on    = [google_project_service.enabled]
}

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-${local.name_prefix}-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
}

resource "google_storage_bucket" "data_lake" {
  name                        = "${var.project_id}-${local.name_prefix}-data"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
}

resource "google_sql_database_instance" "postgres" {
  name                = "${local.name_prefix}-postgres"
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = false

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
    }
  }

  depends_on = [google_service_networking_connection.private_services]
}

resource "google_sql_database" "junyi" {
  name     = "junyi"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_database" "flyte" {
  name     = "flyte"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "junyi" {
  name     = "junyi"
  instance = google_sql_database_instance.postgres.name
  password = var.database_password
}

resource "google_sql_user" "flyte" {
  name     = "flyte"
  instance = google_sql_database_instance.postgres.name
  password = var.database_password
}

resource "google_service_account" "flyte_task" {
  account_id   = "${local.name_prefix}-flyte-task"
  display_name = "Junyi Flyte task identity"
}

resource "google_service_account" "flyte_control" {
  account_id   = "${local.name_prefix}-flyte-control"
  display_name = "Flyte control-plane identity"
}

resource "google_storage_bucket_iam_member" "flyte_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.flyte_task.email}"
}

resource "google_storage_bucket_iam_member" "flyte_data_lake" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.flyte_task.email}"
}

resource "google_storage_bucket_iam_member" "flyte_control_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.flyte_control.email}"
}

resource "google_project_iam_member" "flyte_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.flyte_task.email}"
}

resource "google_service_account_iam_member" "task_workload_identity" {
  service_account_id = google_service_account.flyte_task.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[flyte/junyi-flyte-task]"
}

resource "google_service_account_iam_member" "control_workload_identity" {
  service_account_id = google_service_account.flyte_control.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[flyte/junyi-flyte-control]"
}
