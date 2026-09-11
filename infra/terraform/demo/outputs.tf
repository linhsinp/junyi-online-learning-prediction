output "artifact_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.runtime.name
}

output "cluster_name" {
  value = google_container_cluster.main.name
}

output "cloud_sql_private_ip" {
  value = google_sql_database_instance.postgres.private_ip_address
}
