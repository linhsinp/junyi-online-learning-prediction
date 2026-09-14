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
