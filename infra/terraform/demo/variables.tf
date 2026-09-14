variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "europe-west3"
}

variable "environment" {
  type    = string
  default = "demo"
}

variable "database_password" {
  type      = string
  sensitive = true
}

variable "runtime_image" {
  description = "Immutable Artifact Registry image URI for the one-off dimension seeder Job."
  type        = string
  default     = ""
}
