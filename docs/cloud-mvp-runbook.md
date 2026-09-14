# Bounded Flyte-on-GKE cloud MVP runbook

This is a manually gated demonstration, not a production deployment. Use a
dedicated billed project with a USD 10 budget alert, allow no more than four
hours for the demonstration, and run `terraform destroy` immediately after
collecting the evidence below.

## Provision and stage inputs

Authenticate with `gcloud auth application-default login`, then set the
following shell variables without placing the database password in a file:

```sh
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=europe-west3
export STATE_BUCKET=YOUR_UNIQUE_STATE_BUCKET
export TF_VAR_project_id="$PROJECT_ID"
export TF_VAR_database_password='use-a-unique-secret-here'
```

Create the state bucket once, then create the destroyable demo foundation. The
first demo apply deliberately omits `runtime_image`; it creates GKE, Cloud SQL,
the GCS buckets, identity, task configuration, and the private Flyte service.

```sh
terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap apply \
  -var project_id="$PROJECT_ID" -var state_bucket_name="$STATE_BUCKET"
terraform -chdir=infra/terraform/demo init \
  -backend-config="bucket=$STATE_BUCKET" -backend-config="prefix=junyi/demo"
terraform -chdir=infra/terraform/demo apply
```

Build and push exactly one `linux/amd64` runtime image. Deploy and seed by
digest, not a mutable tag.

```sh
RUNTIME_REPOSITORY="$(terraform -chdir=infra/terraform/demo output -raw runtime_image_repository)"
GIT_SHA="$(git rev-parse --short HEAD)"
gcloud auth configure-docker "$REGION-docker.pkg.dev"
docker buildx build --platform linux/amd64 --push \
  --file infra/docker/Dockerfile --tag "$RUNTIME_REPOSITORY:$GIT_SHA" .
DIGEST="$(gcloud artifacts docker images describe "$RUNTIME_REPOSITORY:$GIT_SHA" --format='value(image_summary.digest)')"
export RUNTIME_IMAGE="$RUNTIME_REPOSITORY@$DIGEST"

export DATA_LAKE_BUCKET="$(terraform -chdir=infra/terraform/demo output -raw data_lake_bucket)"
make upload-curated-data DATA_LAKE_BUCKET="$DATA_LAKE_BUCKET"
make upload-dimension-data DATA_LAKE_BUCKET="$DATA_LAKE_BUCKET"
terraform -chdir=infra/terraform/demo apply -var "runtime_image=$RUNTIME_IMAGE"
```

Wait for `junyi-seed-dimensions` to complete before registering a workflow:

```sh
gcloud container clusters get-credentials \
  "$(terraform -chdir=infra/terraform/demo output -raw cluster_name)" --region "$REGION"
kubectl -n flyte wait --for=condition=complete job/junyi-seed-dimensions --timeout=20m
```

## Register, run, and verify

Keep the control plane private. In one terminal, port-forward the ClusterIP
service; in another, pass the local endpoint directly to the Flyte CLI and
deploy the image mapping.

```sh
kubectl -n flyte port-forward service/flyte-flyte-binary-http 8090:8090
```

Run the following while the port-forward remains active:

```sh
PYTHONPATH=src uv run flyte --endpoint localhost:8090 --insecure deploy \
  --image "runtime=$RUNTIME_IMAGE" --version "$GIT_SHA" \
  src/junyi_predictor/workflows/training.py pipeline_env
PYTHONPATH=src uv run flyte --endpoint localhost:8090 --insecure run \
  src/junyi_predictor/workflows/training.py training_pipeline \
  --start_date 2019-06-01T00:00:00 --end_date 2019-06-02T00:00:00 \
  --training_run_id cloud-mvp-"$GIT_SHA"
```

Verify that each Junyi pod uses `junyi-flyte-task`, the immutable
`$RUNTIME_IMAGE`, and the requested resources. Confirm a successful Flyte run,
Cloud SQL `processed_log` and `feature_snapshot` rows, and these GCS objects in
the artifact bucket: `runs/cloud-mvp-$GIT_SHA/feature_snapshot.json`, a model
bundle and manifest below `models/cloud-mvp-$GIT_SHA/`, and
`models/approved.json`.

## Teardown

Do not leave the environment running after the evidence is collected:

```sh
terraform -chdir=infra/terraform/demo destroy -var "runtime_image=$RUNTIME_IMAGE"
```

The bootstrap state bucket is intentionally retained for future Terraform state;
destroy it only when no demo state must be kept.
