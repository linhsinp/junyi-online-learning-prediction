terraform {
  backend "gcs" {
    bucket = "junyi-ml-project-tf-state"
    prefix = "terraform/state"
  }
}
