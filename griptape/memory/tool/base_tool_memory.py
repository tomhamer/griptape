from __future__ import annotations
from typing import TYPE_CHECKING, Union
from abc import ABC, abstractmethod
from attr import define, field, Factory
from griptape.artifacts import BaseArtifact

if TYPE_CHECKING:
    from griptape.tasks import ActionSubtask


@define
class BaseToolMemory(ABC):
    id: str = field(default=Factory(lambda self: self.__class__.__name__, takes_self=True), kw_only=True)
    namespace_metadata: dict[str, str] = field(factory=dict, kw_only=True)

    def process_output(
            self,
            tool_activity: callable,
            subtask: ActionSubtask,
            artifact: Union[BaseArtifact, list[BaseArtifact]]
    ) -> BaseArtifact:
        return artifact

    @abstractmethod
    def load_artifacts(self, namespace: str) -> list[BaseArtifact]:
        ...
    