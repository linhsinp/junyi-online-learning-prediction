output "ml_data_bucket_name" {
  value = google_storage_bucket.ml_data_bucket.name
}

output "service_account_email" {
  value = google_service_account.ml_sa.email
}
