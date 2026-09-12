# Logging and progress diagnostics

The training pipeline reports operation starts and completions, durations,
input sizes, feature matrix shapes and estimated allocation sizes, model fit
and evaluation events, and artifact persistence. A long operation emits
`operation.running` every 60 seconds by default. Feature loops also attach
completed/total rows and average throughput; counters update every 1,000 rows
and at completion. Estimates describe individual arrays, not peak process memory.

## Local execution

```sh
make flyte-training-local START_DATE=2019-06-01T00:00:00 END_DATE=2019-06-10T00:00:00
```

The ordinary target shows readable application events on stderr and persists
the same events as JSONL under `artifacts/logs/`. An `execution.started` event
identifies the run and log file before loading training inputs. Files are
flushed per record, so an incomplete run still has diagnostics.

```sh
tail -F artifacts/logs/junyi-<process>-<identifier>.jsonl
```

Use the actual filename printed at startup. Multiple runs can write separate
files simultaneously; filter records by `run_id`. Each task execution has a
fresh `invocation_id`, including retries, and Flyte run/action identifiers are
included when the SDK provides them. A process can host several tasks or runs.

`make flyte-training-local-tui` prints the log directory before launching the
TUI and disables application console output. Follow the newly created JSONL
file in a second terminal. Flyte retains ownership of its own display/logging.

The current date filter is start-inclusive and end-exclusive: June 1 through
June 10 at midnight selects nine days. Logs report the requested interval and
the actual number of filtered events.

## Configuration

| Variable | Default | Behavior |
| --- | --- | --- |
| `JUNYI_LOG_LEVEL` | `INFO` | Standard Python level name; `WARNING` suppresses progress events and reporters. |
| `JUNYI_LOG_FORMAT` | `json` | `json` or readable single-line `text`; ordinary local Make target defaults to `text`. |
| `JUNYI_LOG_INTERVAL_SECONDS` | `60` | Positive finite heartbeat interval in seconds. |
| `JUNYI_LOG_DIR` | Unset | Enables JSONL file persistence; local Make targets default to `artifacts/logs/`. |
| `JUNYI_LOG_CONSOLE` | `1` | Set `0` for file-only output; the TUI target sets this automatically. |

For example:

```sh
JUNYI_LOG_INTERVAL_SECONDS=30 JUNYI_LOG_FORMAT=json make flyte-training-local
```

Local Make targets load `.env` before choosing their logging defaults. An
explicit value in that file therefore overrides an exported shell value.
Bootstrap commands use the same settings and report service `junyi-bootstrap`;
the training tasks report `junyi-training`.

Each process file rotates at 10 MiB and retains three backups (`.1` to `.3`).
Old process files are not automatically deleted. The directory is gitignored.
If file creation or subsequent writes fail, a short warning goes to stderr;
diagnostics continue there even if console output was disabled for the TUI.
With neither a file destination nor console enabled, stderr remains the fallback.

Remote task environments explicitly set JSON/INFO/60-second/stderr defaults
without including local file paths. Change the task environment logging values
when registering a different remote policy; these are distinct from Flyte's
`LOG_LEVEL` and `LOG_FORMAT`. No cloud log collector is provisioned by this change.

## Reading a stalled or failed run

An abbreviated pair of events looks like:

```json
{"event":"operation.started","service":"junyi-training","run_id":"example","stage":"training","operation":"model.fit","model_type":"GradientBoostingClassifier"}
{"event":"operation.running","service":"junyi-training","run_id":"example","stage":"training","operation":"model.fit","elapsed_seconds":60.0}
```

Actual records also include UTC timestamp, severity, logger, message, invocation
ID, and available Flyte identifiers. The innermost active operation reports
heartbeats; enclosing operations are suspended to avoid duplicate progress.
A heartbeat means the reporting thread is alive and the operation has not
returned. It does not prove work is advancing. A native call holding the GIL,
process suspension, or a forced kill can also prevent heartbeat delivery.
Percentages and fit-time predictions are not fabricated for opaque model fits.

`execution.failed` includes the innermost operation, context, exception type,
sanitized message, and traceback frame locations. The original exception is
re-raised for Flyte/CLI handling. The parent workflow reports a summary rather
than another application traceback. Application logs omit SQLAlchemy/Pydantic
input-bearing error messages, database driver messages, and KeyError values.
Other messages receive credential-pattern redaction. Tracebacks exclude source
lines and local variables; structured events never include raw learner rows,
SQL parameters, or complete settings. Treat new message/field additions as a
data-exposure boundary: pattern redaction cannot identify arbitrary personal data.
Flyte and other libraries can independently render the original exception;
their output is outside this application's formatter.

Observable Ctrl-C/cancellation emits `execution.interrupted` and is re-raised.
Repeated interrupts or forced termination may prevent a terminal event. Use
the last persisted operation start/heartbeat and last successful completion to
locate unfinished work; absence of a completion is not itself proof of failure.

Logging does not change retries, data validation policy, numerical algorithms,
or partial-write behavior. Retried tasks can still duplicate database appends.
Use these measurements to scope a separate performance/reliability change.

## Shared package and maintenance

`junyi_observability` depends only on the Python standard library. Applications
explicitly call `configure_logging(LoggingOptions(...))`, selecting their own
logger namespaces. Configuration adds only owned handlers and does not rewrite
Flyte/root handlers. Imports have no handler or thread side effects.
`bind_context(...)` restores prior values across nested calls, exceptions, and
async tasks. Each remote boundary must bind its context afresh.

Future inference/monitoring services can use the same formatters and context
helpers with their own service names and request/check fields. Training
heartbeats, Flyte integration, and validation summaries remain in the predictor.
This repository still executes from `PYTHONPATH=src`; the shared package is not
a separately published distribution. Include both source packages in deployment
bundles and runtime images.

Tests exercise the real local Flyte CLI with synthetic input and a temporary
SQLite database. They observe a heartbeat while execution is blocked, then
release the fixture for success/failure or send SIGINT. No production database
or long training run is needed. The bundle test uses Flyte's installed bundler
in dry-run mode and never uploads code.
