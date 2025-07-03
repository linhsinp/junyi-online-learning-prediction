from flytekit import Email, LaunchPlan, Slack, WorkflowExecutionPhase
from flytekit.core.schedule import CronSchedule
from flytekit.models.common import Annotations, Labels

from flyte.full_pipeline_wf import full_pipeline_wf

comprehensive_lp = LaunchPlan.get_or_create(
    workflow=full_pipeline_wf,
    name="comprehensive_launch_plan",
    schedule=CronSchedule(schedule="0 13 * * *"),  # Daily at 1 PM UTC
    notifications=[
        Email(
            phases=[WorkflowExecutionPhase.FAILED],
            recipients_email=["linhsinp@gmail.com"],
        ),
        Slack(
            phases=[
                WorkflowExecutionPhase.SUCCEEDED,
                WorkflowExecutionPhase.ABORTED,
                WorkflowExecutionPhase.TIMED_OUT,
            ],
            recipients_email=["linhsinp@gmail.com"],
        ),
    ],
    labels=Labels({"env": "development", "team": "data-science"}),
    annotations=Annotations({"description": "Daily model training and evaluation"}),
    max_parallelism=20,
    overwrite_cache=False,
    auto_activate=True,
)
