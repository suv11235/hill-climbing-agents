"""Workflow-level OCR with orchestrator, mutable sub-agents, and hybrid rewards."""

from hillclimb.software.ocr_self_iterate.workflow.config import SubAgentConfig, WorkflowConfig
from hillclimb.software.ocr_self_iterate.workflow.feedback import (
    FeedbackBatch,
    collect_output_feedback,
    propagate_feedback_backward,
)
from hillclimb.software.ocr_self_iterate.workflow.offline_train import OfflineTrainConfig, OfflineWorkflowTrainer
from hillclimb.software.ocr_self_iterate.workflow.orchestrator import OrchestratorAgent
from hillclimb.software.ocr_self_iterate.workflow.registry import WorkflowRegistry
from hillclimb.software.ocr_self_iterate.workflow.rewards import HybridRewardFramework
from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow
from hillclimb.software.ocr_self_iterate.workflow.blackbox import (
    BlackboxRunConfig,
    propagate_via_frontier,
    run_blackbox_workflow,
)
from hillclimb.core.frontier import FrontierProvider, get_frontier

__all__ = [
    "SubAgentConfig",
    "WorkflowConfig",
    "OrchestratorAgent",
    "HybridRewardFramework",
    "run_workflow",
    "collect_output_feedback",
    "propagate_feedback_backward",
    "FeedbackBatch",
    "WorkflowRegistry",
    "OfflineWorkflowTrainer",
    "OfflineTrainConfig",
    "BlackboxRunConfig",
    "run_blackbox_workflow",
    "propagate_via_frontier",
    "get_frontier",
    "FrontierProvider",
]
