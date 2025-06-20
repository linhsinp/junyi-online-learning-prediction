Junyi Online Learning Performance Prediction
=============================================

*This repo is an on-going project and is actively evolving!* 

This project aims to build a prototype tool that is applicable to a Taiwanese online learning platform. A best-performing ML model among other candidates makes batch inference on incoming online learning platform data. It predicts students' online learning performance based on their problem-solving history, which sheds light on ways to help students improve their academic performance.

It demonstrates how to deploy ML models with Flyte (kubernetes-native container management tool) to orchestrate data and model workflows. Cloud infrastructure is managed by Terraform (example cloud provider: Google Cloud Platform). 

The end-to-end ML pipeline includes (with workflows in place and under construction):

1. Data ingestion - raw files saved to Google Cloud Storage
2. Data preprocessing - processed data in self-hosted postgreSQL database
3. Feature engineering - a feature store in self-hosted postgreSQL database
4. Model training, evaluation and registration
5. Batch inference
6. (Continuous monitoring of data and model performance) - to be continued


Open source dataset on Kaggle: [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy/)


