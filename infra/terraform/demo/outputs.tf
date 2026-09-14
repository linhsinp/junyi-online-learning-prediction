output "artifact_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "data_lake_bucket" {
  value = google_storage_bucket.data_lake.name
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.runtime.name
}

output "runtime_image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.runtime.repository_id}/junyi-runtime"
}

output "cluster_name" {
  value = google_container_cluster.main.name
}

output "project_id" {
  value = var.project_id
}

output "region" {
  value = var.region
}

output "task_service_account_email" {
  value = google_service_account.flyte_task.email
}

output "control_service_account_email" {
  value = google_service_account.flyte_control.email
}

output "cloud_sql_private_ip" {
  value = google_sql_database_instance.postgres.private_ip_address
}
