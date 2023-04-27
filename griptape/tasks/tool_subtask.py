from __future__ import annotations
import ast
import json
import re
from typing import TYPE_CHECKING, Optional
import schema
from attr import define, field
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from schema import Schema, And, Literal
from griptape.artifacts import TextOutput, ErrorOutput
from griptape.tasks import PromptTask
from griptape.core import BaseTool
from griptape.utils import J2

if TYPE_CHECKING:
    from griptape.artifacts import BaseArtifact
    from griptape.tasks import ToolkitTask


@define
class ToolSubtask(PromptTask):
    THOUGHT_PATTERN = r"^Thought:\s*(.*)$"
    ACTION_PATTERN = r"^Action:\s*({.*})$"
    OUTPUT_PATTERN = r"^Output:\s?([\s\S]*)$"
    INVALID_ACTION_ERROR_MSG = f"invalid action input, try again"
    ACTION_SCHEMA = Schema(
        description="Actions have type, name, method, and optional input value.",
        schema={
            Literal(
                "type",
                description="Action type"
            ): And(str, lambda s: s in ("tool", "middleware")),
            Literal(
                "name",
                description="Action name"
            ): str,
            Literal(
                "method",
                description="Action method"
            ): str,
            schema.Optional(
                Literal(
                    "input",
                    description="Action method input value"
                )
            ): schema.Or(str, list, {object: object})
        }
    )

    parent_task_id: Optional[str] = field(default=None, kw_only=True)
    thought: Optional[str] = field(default=None, kw_only=True)
    action_type: Optional[str] = field(default=None, kw_only=True)
    action_name: Optional[str] = field(default=None, kw_only=True)
    action_method: Optional[str] = field(default=None, kw_only=True)
    action_input: Optional[str] = field(default=None, kw_only=True)

    _tool: Optional[BaseTool] = None

    def attach(self, parent_task: ToolkitTask):
        self.parent_task_id = parent_task.id
        self.structure = parent_task.structure
        self.__init_from_prompt(self.input.value)

    @property
    def task(self) -> Optional[ToolkitTask]:
        return self.structure.find_task(self.parent_task_id)

    @property
    def parents(self) -> list[ToolSubtask]:
        return [self.task.find_subtask(parent_id) for parent_id in self.parent_ids]

    @property
    def children(self) -> list[ToolSubtask]:
        return [self.task.find_subtask(child_id) for child_id in self.child_ids]

    def before_run(self) -> None:
        self.structure.logger.info(f"Subtask {self.id}\n{self.input.value}")

    def run(self) -> BaseArtifact:
        try:
            if self.action_name == "error":
                self.output = ErrorOutput(self.action_input, task=self.task)
            else:
                if self._tool:
                    observation = self.structure.tool_loader.executor.execute(
                        getattr(self._tool, self.action_method),
                        self.action_input.encode()
                    ).decode()
                else:
                    observation = "tool not found"

                self.output = TextOutput(observation)
        except Exception as e:
            self.structure.logger.error(f"Subtask {self.id}\n{e}", exc_info=True)

            self.output = ErrorOutput(str(e), exception=e, task=self.task)
        finally:
            return self.output

    def after_run(self) -> None:
        self.structure.logger.info(f"Subtask {self.id}\nObservation: {self.output.value}")

    def render(self) -> str:
        return J2("prompts/tasks/tool/subtask.j2").render(
            subtask=self
        )

    def to_json(self) -> str:
        json_dict = {}

        if self.action_type:
            json_dict["type"] = self.action_type

        if self.action_name:
            json_dict["name"] = self.action_name

        if self.action_method:
            json_dict["method"] = self.action_method

        if self.action_input:
            json_dict["input"] = self.action_input

        return json.dumps(json_dict)

    def add_child(self, child: ToolSubtask) -> ToolSubtask:
        if child.id not in self.child_ids:
            self.child_ids.append(child.id)

        if self.id not in child.parent_ids:
            child.parent_ids.append(self.id)

        return child

    def add_parent(self, parent: ToolSubtask) -> ToolSubtask:
        if parent.id not in self.parent_ids:
            self.parent_ids.append(parent.id)

        if self.id not in parent.child_ids:
            parent.child_ids.append(self.id)

        return parent

    def __init_from_prompt(self, value: str) -> None:
        thought_matches = re.findall(self.THOUGHT_PATTERN, value, re.MULTILINE)
        action_matches = re.findall(self.ACTION_PATTERN, value, re.MULTILINE)
        output_matches = re.findall(self.OUTPUT_PATTERN, value, re.MULTILINE)

        if self.thought is None and len(thought_matches) > 0:
            self.thought = thought_matches[-1]

        if len(action_matches) > 0:
            try:
                action_object: dict = ast.literal_eval(action_matches[-1])

                validate(
                    instance=action_object,
                    schema=self.ACTION_SCHEMA.schema
                )

                # Load action type; throw exception if the key is not present
                if self.action_type is None:
                    self.action_type = action_object["type"]

                # Load action name; throw exception if the key is not present
                if self.action_name is None:
                    self.action_name = action_object["name"]

                # Load tool method; throw exception if the key is not present
                if self.action_method is None:
                    self.action_method = action_object["method"]

                # Load the tool itself
                if self.action_name:
                    self._tool = self.task.find_tool(self.action_name)

                # Load optional input value; don't throw exceptions if key is not present
                if self.action_input is None:
                    self.action_input = str(action_object["input"]) if "input" in action_object else None

                # Validate input based on tool schema
                if self._tool:
                    validate(
                        instance=self.action_input,
                        schema=self._tool.action_schema(getattr(self._tool, self.action_method))
                    )

            except SyntaxError as e:
                self.structure.logger.error(f"Subtask {self.task.id}\nSyntax error: {e}")

                self.action_name = "error"
                self.action_input = f"syntax error: {e}"
            except ValidationError as e:
                self.structure.logger.error(f"Subtask {self.task.id}\nInvalid action JSON: {e}")

                self.action_name = "error"
                self.action_input = f"JSON validation error: {e}"
            except Exception as e:
                self.structure.logger.error(f"Subtask {self.task.id}\nError parsing tool action: {e}")

                self.action_name = "error"
                self.action_input = f"error: {self.INVALID_ACTION_ERROR_MSG}"
        elif self.output is None and len(output_matches) > 0:
            self.output = TextOutput(output_matches[-1])
