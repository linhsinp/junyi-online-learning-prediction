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
demo apply creates GKE, Cloud SQL, GCS buckets, and Google identities.
Helm subsequently deploys Kubernetes software and project policy.

```sh
terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap apply \
  -var project_id="$PROJECT_ID" -var state_bucket_name="$STATE_BUCKET"
terraform -chdir=infra/terraform/demo init \
  -backend-config="bucket=$STATE_BUCKET" -backend-config="prefix=junyi/demo"
terraform -chdir=infra/terraform/demo plan
terraform -chdir=infra/terraform/demo apply
```

Build and push exactly one `linux/amd64` runtime image. Deploy and seed by
digest, not a mutable tag.

Before pushing, build and smoke-test the target architecture locally. This
confirms that Flyte and both Junyi packages are importable through the image's
default Python interpreter:

```sh
docker buildx build --platform linux/amd64 --load \
  --file infra/docker/Dockerfile --tag junyi-runtime:preflight .
docker run --rm --platform linux/amd64 junyi-runtime:preflight \
  python -c 'import flyte, junyi_predictor, junyi_observability; print(flyte.__version__)'
```

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
```

Generate Helm values from Terraform outputs. The helper reads the database
password from the environment and writes owner-readable JSON files under the
gitignored and Docker-excluded `artifacts/cloud/` directory. It refuses to
overwrite existing files; archive old values securely before regenerating.
These files and Helm release Secrets contain credentials: do not attach them
to PRs, logs, or verification evidence.

```sh
gcloud container clusters get-credentials \
  "$(terraform -chdir=infra/terraform/demo output -raw cluster_name)" --region "$REGION"
terraform -chdir=infra/terraform/demo output -json | uv run python scripts/cloud_helm_values.py
helm upgrade --install junyi-cloud infra/helm/junyi-cloud \
  --namespace flyte --create-namespace -f artifacts/cloud/junyi.values.json
helm upgrade --install flyte \
  "https://flyteorg.github.io/flyte/flyte-binary-$(tr -d '\n' < infra/helm/flyte/chart-version).tgz" \
  --namespace flyte -f infra/helm/flyte/values-demo.yaml \
  -f artifacts/cloud/flyte.values.json --wait --timeout 15m
helm upgrade junyi-cloud infra/helm/junyi-cloud --namespace flyte \
  -f artifacts/cloud/junyi.values.json \
  --set seeder.enabled=true --set-string "seeder.image=$RUNTIME_IMAGE"
kubectl -n flyte wait --for=condition=complete job/junyi-seed-dimensions --timeout=20m
```

## Register, run, and verify

Before provisioning GKE, run the disposable local backend preflight in
[`docs/flyte-backend-preflight.md`](flyte-backend-preflight.md). It validates
the pinned chart/backend, SDK, and linux/amd64 runtime-image scheduling path;
the cloud run below then validates GKE, Workload Identity, Cloud SQL, and GCS.

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
  --image "runtime=$RUNTIME_IMAGE" \
  src/junyi_predictor/workflows/training.py training_pipeline \
  --start_date 2019-06-01T00:00:00 --end_date 2019-06-02T00:00:00 \
  --training_run_id cloud-mvp-"$GIT_SHA"
```

Verify that each Junyi pod uses `junyi-flyte-task`, the immutable
`$RUNTIME_IMAGE`, and the requested resources. Confirm a successful Flyte run,
Cloud SQL `processed_log` and `feature_snapshot` rows, and these GCS objects in
the artifact bucket: `runs/runs/cloud-mvp-$GIT_SHA/feature_snapshot.json`, a model
bundle and manifest below `runs/models/cloud-mvp-$GIT_SHA/`, and
`runs/models/approved.json`. The initial `runs/` is the configured artifact root.

## Teardown

Do not leave the environment running after the evidence is collected:

```sh
helm uninstall flyte --namespace flyte
helm uninstall junyi-cloud --namespace flyte
terraform -chdir=infra/terraform/demo destroy
```

The bootstrap state bucket is intentionally retained for future Terraform state;
destroy it only when no demo state must be kept.

The seeder is an ordinary Job with no automatic retry or TTL deletion. Keep
its enabled flag and digest unchanged on later upgrades so it does not rerun.
To remove the completed Job, upgrade Junyi with seeder.enabled=false.
Do not re-enable it against populated dimensions: seeding is not idempotent.
On failure, inspect Job logs and recover the database before any manual retry.

## Existing Terraform-managed deployments

This path targets a fresh ephemeral demo. If an earlier revision is deployed,
finish its demo and destroy it using that revision's Terraform configuration
before switching to this ownership split. That destruction removes its data;
retain required evidence first. Do not apply this revision over the old state:
the removed provider configurations are still needed to clean up old objects.
An in-place adoption requires a separate explicit state/Helm ownership migration.
No live state migration is performed by the repository changes.
