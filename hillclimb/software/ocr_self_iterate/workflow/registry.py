from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig


class WorkflowStatus(str, Enum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    STABLE = "stable"
    FROZEN = "frozen"


@dataclass
class WorkflowMetrics:
    mean_accuracy: float
    mean_reward: float
    epoch: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowVersion:
    """A versioned workflow that can mature into a stable production artifact."""

    version_id: str
    config: WorkflowConfig
    status: WorkflowStatus = WorkflowStatus.EXPERIMENTAL
    metrics_history: list[WorkflowMetrics] = field(default_factory=list)
    doc_type: str = "invoice"

    @property
    def latest_accuracy(self) -> float:
        if not self.metrics_history:
            return 0.0
        return self.metrics_history[-1].mean_accuracy

    @property
    def is_mutable(self) -> bool:
        return self.status in (WorkflowStatus.EXPERIMENTAL, WorkflowStatus.CANDIDATE)


def workflow_config_to_dict(config: WorkflowConfig) -> dict[str, Any]:
    return {
        "orchestrator_prompt": config.orchestrator_prompt,
        "orchestrator_params": config.orchestrator_params,
        "agents": {
            name: {
                "name": agent.name,
                "role": agent.role,
                "system_prompt": agent.system_prompt,
                "params": agent.params,
            }
            for name, agent in config.agents.items()
        },
    }


def workflow_config_from_dict(data: dict[str, Any]) -> WorkflowConfig:
    from hillclimb.software.ocr_self_iterate.workflow.config import SubAgentConfig

    agents = {
        name: SubAgentConfig(
            name=agent["name"],
            role=agent["role"],
            system_prompt=agent["system_prompt"],
            params=agent.get("params", {}),
        )
        for name, agent in data.get("agents", {}).items()
    }
    return WorkflowConfig(
        orchestrator_prompt=data["orchestrator_prompt"],
        orchestrator_params=data.get("orchestrator_params", {}),
        agents=agents,
    )


@dataclass
class WorkflowRegistry:
    """
    Tracks workflow versions and promotes them to stable when metrics plateau.

    Stable workflows stop receiving backward updates unless explicitly forked.
    """

    versions: dict[str, WorkflowVersion] = field(default_factory=dict)
    stable_version_id: str | None = None
    promote_accuracy: float = 0.95
    promote_epochs: int = 2
    freeze_accuracy: float = 0.98
    max_regression: float = 0.02

    def register(
        self,
        config: WorkflowConfig,
        metrics: WorkflowMetrics,
        *,
        version_id: str | None = None,
        status: WorkflowStatus = WorkflowStatus.EXPERIMENTAL,
    ) -> WorkflowVersion:
        vid = version_id or f"v{len(self.versions) + 1}_{int(time.time())}"
        if vid in self.versions:
            version = self.versions[vid]
            version.metrics_history.append(metrics)
        else:
            version = WorkflowVersion(version_id=vid, config=config.copy(), status=status)
            version.metrics_history.append(metrics)
            self.versions[vid] = version
        self._maybe_promote(version)
        return version

    def _maybe_promote(self, version: WorkflowVersion) -> None:
        history = version.metrics_history
        if len(history) < self.promote_epochs:
            return

        recent = history[-self.promote_epochs :]
        if all(m.mean_accuracy >= self.promote_accuracy for m in recent):
            version.status = WorkflowStatus.CANDIDATE

        if len(history) >= self.promote_epochs + 1:
            window = history[-(self.promote_epochs + 1) :]
            accs = [m.mean_accuracy for m in window]
            if min(accs) >= self.freeze_accuracy and (max(accs) - min(accs)) <= self.max_regression:
                version.status = WorkflowStatus.STABLE
                self.stable_version_id = version.version_id

    def freeze_stable(self) -> None:
        if self.stable_version_id and self.stable_version_id in self.versions:
            self.versions[self.stable_version_id].status = WorkflowStatus.FROZEN

    def get_stable(self) -> WorkflowVersion | None:
        if self.stable_version_id:
            return self.versions.get(self.stable_version_id)
        for v in self.versions.values():
            if v.status in (WorkflowStatus.STABLE, WorkflowStatus.FROZEN):
                return v
        return None

    def get_best_candidate(self) -> WorkflowVersion | None:
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda v: v.latest_accuracy)

    def save(self, path: Path) -> None:
        payload = {
            "stable_version_id": self.stable_version_id,
            "promote_accuracy": self.promote_accuracy,
            "promote_epochs": self.promote_epochs,
            "versions": {
                vid: {
                    "version_id": v.version_id,
                    "status": v.status.value,
                    "doc_type": v.doc_type,
                    "config": workflow_config_to_dict(v.config),
                    "metrics_history": [asdict(m) for m in v.metrics_history],
                }
                for vid, v in self.versions.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> WorkflowRegistry:
        data = json.loads(path.read_text())
        registry = cls(
            stable_version_id=data.get("stable_version_id"),
            promote_accuracy=data.get("promote_accuracy", 0.95),
            promote_epochs=data.get("promote_epochs", 2),
        )
        for vid, vdata in data.get("versions", {}).items():
            version = WorkflowVersion(
                version_id=vdata["version_id"],
                config=workflow_config_from_dict(vdata["config"]),
                status=WorkflowStatus(vdata["status"]),
                doc_type=vdata.get("doc_type", "invoice"),
                metrics_history=[
                    WorkflowMetrics(**m) for m in vdata.get("metrics_history", [])
                ],
            )
            registry.versions[vid] = version
        return registry
