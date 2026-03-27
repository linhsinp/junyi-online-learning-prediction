# ADR 0001: Use Kubernetes Jobs for the MVP Before Adopting Full Flyte

## Status

Accepted

## Context

This repository currently runs batch-oriented ML workflows through Kubernetes `Job` and `CronJob` resources. The deployed container executes `flyte run --local orchestration/flyte_app.py ...` inside the pod, which means Kubernetes is the scheduler and Flyte is used as an in-container workflow runner.

The alternative is a full Flyte platform deployment, where Flyte itself runs in Kubernetes as a control plane and schedules task pods for submitted workflows.

The project goal today is to reach a working, testable, production-leaning MVP quickly while keeping the system understandable and maintainable.

## Decision

The MVP will use the current Kubernetes `Job` and `CronJob` model with Helm-based deployment and the existing `flyte run --local` entrypoints.

A full Flyte deployment is deferred to a later phase for a more production-grade demo once the application workflow, deployment shape, and operational needs are stable.

## Rationale

### Benefits of the MVP approach

- faster to deliver and easier to debug
- lower infrastructure and operational overhead
- enough to demonstrate modular pipeline stages, test coverage, CI validation, containerization, and Kubernetes execution
- keeps effort focused on application correctness rather than platform management

### Costs of deferring full Flyte

- no Flyte control plane, UI, or remote workflow orchestration
- fewer built-in production workflow features such as richer task scheduling, retries, and observability
- less representative of a mature ML workflow platform

## Consequences

- The current architecture is appropriate for an MVP and early demos.
- Kubernetes remains the execution and scheduling layer.
- Helm packages runtime jobs, and Terraform provisions shared infrastructure.
- If the system grows into a broader production workflow platform, the next architectural step is to introduce a full Flyte deployment.

## Follow-Up

- keep the current Kubernetes job model for the MVP
- document the target full-Flyte architecture as a future-state design
- evaluate migrating to full Flyte when workflow complexity, team usage, and operational requirements justify it
