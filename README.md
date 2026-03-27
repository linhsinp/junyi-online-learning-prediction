Junyi Online Learning Performance Prediction
=============================================

*This repo is an on-going project and is actively evolving!* 

This project aims to build a prototype tool that is applicable to a Taiwanese online learning platform. A best-performing ML model among other candidates makes batch inference on incoming online learning platform data. It predicts students' online learning performance based on their problem-solving history, which sheds light on ways to help students improve their academic performance.

It demonstrates how to orchestrate data and model execution with Flyte 2 while keeping pipeline logic modular and testable in plain Python. Cloud resource provisioning is managed by Terraform, while Kubernetes workload deployment is packaged with Helm.

Local development is expected to use the `uv`-managed virtual environment only.

The end-to-end ML pipeline includes:

1. Data ingestion - raw files saved to Google Cloud Storage
2. Data preprocessing - processed data in self-hosted postgreSQL database
3. Feature engineering - a feature store in self-hosted postgreSQL database
4. Model training, evaluation and registration
5. Batch inference
6. (Continuous monitoring of data and model performance) - to be continued

Infrastructure is separated by concern under `infra/`:

- `infra/terraform/`: provisions shared cloud resources such as buckets and IAM
- `infra/helm/junyi-predictor/`: deploys runtime workloads to Kubernetes as `Job` or `CronJob`
- `infra/docker/`: builds the container images used by local and cluster workloads

Use the example values files under `infra/helm/junyi-predictor/` as starting points for cluster runs.


Open source dataset on Kaggle: [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy/)
