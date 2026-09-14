locals {
  name_prefix          = "junyi-${var.environment}"
  flyte_chart_version  = "v2.0.20"
  task_config_map_name = "junyi-task-runtime"
  task_secret_name     = "junyi-task-database"
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

data "google_client_config" "current" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.current.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.current.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
  }
}

resource "kubernetes_namespace_v1" "flyte" {
  metadata { name = "flyte" }
}

resource "kubernetes_resource_quota_v1" "flyte" {
  metadata {
    name      = "junyi-demo-limits"
    namespace = kubernetes_namespace_v1.flyte.metadata[0].name
  }

  spec {
    hard = {
      "requests.cpu"               = "4"
      "requests.memory"            = "12Gi"
      "requests.ephemeral-storage" = "12Gi"
      "pods"                       = "8"
    }
  }
}

resource "kubernetes_service_account_v1" "task" {
  metadata {
    name      = "junyi-flyte-task"
    namespace = kubernetes_namespace_v1.flyte.metadata[0].name
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.flyte_task.email
    }
  }
}

resource "kubernetes_service_account_v1" "control" {
  metadata {
    name      = "junyi-flyte-control"
    namespace = kubernetes_namespace_v1.flyte.metadata[0].name
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.flyte_control.email
    }
  }
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

resource "kubernetes_config_map_v1" "task_runtime" {
  metadata {
    name      = local.task_config_map_name
    namespace = kubernetes_namespace_v1.flyte.metadata[0].name
  }

  data = {
    ARTIFACT_BACKEND      = "gcs"
    ARTIFACT_ROOT         = "runs"
    DATA_LAKE_BACKEND     = "gcs"
    DATA_LAKE_BUCKET      = google_storage_bucket.data_lake.name
    DATA_LAKE_PREFIX      = "data/curated/log_problem"
    DIMENSION_DATA_PREFIX = "data/dimensions"
    GCS_BUCKET            = google_storage_bucket.artifacts.name
    JUNYI_LOG_FORMAT      = "json"
    JUNYI_LOG_LEVEL       = "INFO"
  }
}

resource "kubernetes_secret_v1" "task_database" {
  metadata {
    name      = local.task_secret_name
    namespace = kubernetes_namespace_v1.flyte.metadata[0].name
  }

  data = {
    DATABASE_URL = "postgresql://${google_sql_user.junyi.name}:${urlencode(var.database_password)}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.junyi.name}"
  }
}

resource "kubernetes_job_v1" "seed_dimensions" {
  count = var.runtime_image == "" ? 0 : 1

  metadata {
    name      = "junyi-seed-dimensions"
    namespace = kubernetes_namespace_v1.flyte.metadata[0].name
  }

  spec {
    backoff_limit              = 0
    ttl_seconds_after_finished = 900

    template {
      metadata { labels = { app = "junyi-dimension-seeder" } }
      spec {
        service_account_name = kubernetes_service_account_v1.task.metadata[0].name
        restart_policy       = "Never"

        container {
          name    = "seed-dimensions"
          image   = var.runtime_image
          command = ["uv", "run", "python", "-m", "junyi_predictor.cli", "seed-db-from-gcs"]

          env_from {
            config_map_ref { name = kubernetes_config_map_v1.task_runtime.metadata[0].name }
          }
          env_from {
            secret_ref { name = kubernetes_secret_v1.task_database.metadata[0].name }
          }

          resources {
            requests = {
              cpu                 = "500m"
              memory              = "1Gi"
              "ephemeral-storage" = "1Gi"
            }
          }
        }
      }
    }
  }
}

resource "helm_release" "flyte" {
  name       = "flyte"
  namespace  = kubernetes_namespace_v1.flyte.metadata[0].name
  repository = "https://flyteorg.github.io/flyte"
  chart      = "flyte-binary"
  version    = local.flyte_chart_version
  values     = [file("${path.module}/../../helm/flyte/values-demo.yaml")]

  set {
    name  = "configuration.database.host"
    value = google_sql_database_instance.postgres.private_ip_address
  }

  set {
    name  = "flyte-core-components.runs.database.postgres.host"
    value = google_sql_database_instance.postgres.private_ip_address
  }

  set {
    name  = "flyte-core-components.runs.database.postgres.dbname"
    value = google_sql_database.flyte.name
  }

  set {
    name  = "flyte-core-components.runs.database.postgres.username"
    value = google_sql_user.flyte.name
  }

  set_sensitive {
    name  = "flyte-core-components.runs.database.postgres.password"
    value = var.database_password
  }

  set {
    name  = "flyte-core-components.runs.storagePrefix"
    value = "gs://${google_storage_bucket.artifacts.name}/flyte"
  }

  set {
    name  = "configuration.database.port"
    value = "5432"
  }

  set {
    name  = "configuration.storage.providerConfig.gcs.project"
    value = var.project_id
  }

  set {
    name  = "serviceAccount.create"
    value = "false"
  }

  set {
    name  = "serviceAccount.name"
    value = kubernetes_service_account_v1.control.metadata[0].name
  }

  set {
    name  = "configuration.database.dbname"
    value = google_sql_database.flyte.name
  }

  set {
    name  = "configuration.database.username"
    value = google_sql_user.flyte.name
  }

  set_sensitive {
    name  = "configuration.database.password"
    value = var.database_password
  }

  set {
    name  = "configuration.storage.provider"
    value = "gcs"
  }

  set {
    name  = "configuration.storage.metadataContainer"
    value = google_storage_bucket.artifacts.name
  }

  set {
    name  = "configuration.storage.userDataContainer"
    value = google_storage_bucket.artifacts.name
  }

  depends_on = [
    google_sql_database.flyte,
    google_sql_user.flyte,
    google_service_account_iam_member.task_workload_identity,
    google_service_account_iam_member.control_workload_identity,
    google_storage_bucket_iam_member.flyte_control_artifacts,
  ]
}
