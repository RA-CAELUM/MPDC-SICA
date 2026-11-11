# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import json

from abc import ABC, abstractmethod
from enum import Enum, Flag, auto
from typing import Any, Set, Type
from typing import Optional, ClassVar, Dict
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr
from dataclasses import dataclass

from .llm_types import TokenUsage, Model, ToolCallContent
from .event_types import EventType
from .tool_types import ToolInterface


class AgentStatus(str, Enum):
    """Possible states of an agent execution."""

    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"
    CANCELLED = "cancelled"  # e.g., by overseer or timeout
    INCOMPLETE = "incomplete"  # e.g., needs more information
    DELEGATED = "delegated"  # handed off to another agent


class ArtifactType(str, Enum):
    """Types of artifacts that can be produced by agents."""

    FILE_PATH = "file_path"  # Path to a file created/modified
    FILE_PATCH = "file_patch"  # Diff/patch of file changes
    TEXT = "text"  # Raw text output
    JSON = "json"  # Structured data
    CODE = "code"  # Code snippet
    THOUGHT = "thought"  # Reasoning or analysis
    PLAN = "plan"  # Action plan or strategy


class Artifact(BaseModel):
    """Represents a concrete output from an agent."""

    type: ArtifactType
    content: Any
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        if self.type == ArtifactType.JSON:
            return json.dumps(self.content, indent=2)
        return str(self.content)


class AgentMetrics(BaseModel):
    """Metrics about the agent execution."""

    start_time: datetime
    end_time: Optional[datetime] = None
    token_usage: TokenUsage = TokenUsage()
    # token_count: int = 0  # TODO: deprecate
    cost: float = 0.0
    tool_calls: int = 0
    agent_calls: int = 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration if completed."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class AgentResult(BaseModel):
    """
    Represents the result of an agent execution.

    This is the standardized output format that all agents must produce.
    It includes the execution status, any artifacts produced, a natural language
    summary of what was done, and various metrics about the execution.
    """

    agent_name: str
    status: AgentStatus
    metrics: AgentMetrics
    result: str = Field(description="A string-valued agent result")
    artifacts: list[Artifact] = Field(default_factory=list)
    errors: Optional[str] = None
    warnings: Optional[str] = None
    invocation_id: str = Field(default_factory=lambda: os.urandom(4).hex())
    parent_id: Optional[str] = None  # ID of calling agent if any
    metadata: dict = Field(default_factory=dict)

    def add_artifact(
        self, type: ArtifactType, content: Any, metadata: dict | None = None
    ):
        """Add a new artifact to the result."""
        self.artifacts.append(
            Artifact(type=type, content=content, metadata=metadata or {})
        )

    def __str__(self) -> str:
        """
        Format the agent result for inclusion in agent prompts.
        This needs to be clear and parseable by the LLM.
        """
        parts = [
            "<AGENT_RESULT>",
            f"<STATUS>{self.status}</STATUS>",
            f"<RESULT>\n{self.result}\n</RESULT>",
        ]

        if self.artifacts:
            parts.append("<ARTIFACTS>")
            for artifact in self.artifacts:
                parts.append(
                    f"<ARTIFACT type='{artifact.type}'>{str(artifact)}</ARTIFACT>"
                )
            parts.append("</ARTIFACTS>")

        if self.warnings:
            parts.append(f"<WARNINGS>{self.warnings}</WARNINGS>")

        if self.errors:
            parts.append(f"<ERRORS>{self.errors}</ERRORS>")

        if self.metrics.duration_seconds:
            parts.append(
                f"<METRICS>Completed in {self.metrics.duration_seconds:.2f}s "
                f"using {self.metrics.token_usage.total_tokens} tokens "
                f"(of which {self.metrics.token_usage.cached_prompt_tokens} cached)"
                f"(${self.metrics.cost:.4f})</METRICS>"
            )

        # from ..llm.metering import get_total_cost
        # from ..llm.metering import budget_info
        # # NOTE: we only calculate these once and cache the results. This is
        # # crucial to avoid breaking the KV cache every time this event is
        # # serialised in the agent prefill. Getting this wrong yields a 25% cost
        # # *increase* instead of a 90% cost *decrease* (!)
        # # This is not working without breaking the cache, so we disable this for now.
        # if budget_info.get("cost_budget"):
        #     if "cost_budget_used" not in self.metadata:
        #         cost_used = get_total_cost()
        #         self.metadata["cost_budget_used"] = cost_used / budget_info.get("cost_budget") * 100
        #     cost_budget_used = self.metadata["cost_budget_used"]
        #     parts.append(f"<BUDGET>After agent execution, you have used up {cost_budget_used}% of the cost budget available</BUDGET>")
        # elif budget_info.get("start_time") and budget_info.get("time_budget"):
        #     if "time_budget_used" not in self.metadata:
        #         time_used = time.time() - budget_info.get("start_time")
        #         self.metadata["time_budget_used"] = time_used / budget_info.get("time_budget") * 100
        #     time_budget_used = self.metadata["time_budget_used"]
        #     parts.append(f"<BUDGET>After agent execution, you have used up {time_budget_used}% of the time budget available</BUDGET>")

        parts.append("</AGENT_RESULT>")
        return "\n".join(parts)


class InheritanceFlags(Flag):
    """Flags controlling what to inherit from the parent context"""

    NONE = 0
    OPEN_FILES = auto()
    CORE_DIALOGUE = auto()
    EVENT_BUS = auto()
    WORKDIR = auto()
    LOGDIR = auto()
    ALL = OPEN_FILES | EVENT_BUS | CORE_DIALOGUE | WORKDIR | LOGDIR


@dataclass
class InheritanceConfig:
    """Configuration for context inheritance"""

    flags: InheritanceFlags = InheritanceFlags.LOGDIR | InheritanceFlags.WORKDIR | InheritanceFlags.OPEN_FILES
    event_types: Set[EventType] | None = None


class AgentInterface(BaseModel, ABC):
    """
    Abstract interface for all agents in the system.

    This interface defines the contract that all agent implementations must fulfill.
    It includes class variables for agent configuration, properties for agent state,
    and methods for agent execution and interaction.
    """

    # Required class-level attributes
    AGENT_NAME: ClassVar[str]
    AGENT_DESCRIPTION: ClassVar[str]
    SYSTEM_PROMPT: ClassVar[str]

    # Optional class-level configuration
    AVAILABLE_TOOLS: ClassVar[Set[Type[ToolInterface]]] = set()
    AVAILABLE_AGENTS: ClassVar[Set[Type["AgentInterface"]]] = set()

    HAS_FILEVIEW: ClassVar[bool] = True
    MODEL: ClassVar[Model] = Model.QWEN3
    TEMPERATURE: ClassVar[float] = 0.666
    MAX_ITERATIONS: ClassVar[int] = 500
    INHERITANCE: ClassVar[InheritanceConfig] = InheritanceConfig()

    # Private agent attributes (not arguments)
    _workdir: Path = PrivateAttr()
    _logdir: Path = PrivateAttr()
    _debug_mode: bool = PrivateAttr()
    _local_state: Dict[str, Any] = PrivateAttr()
    _parent_id: str | None = PrivateAttr()
    _id: str = PrivateAttr()
    _available_agents: Set[Type['AgentInterface']] = PrivateAttr()
    _available_agent_names: Set[str] = PrivateAttr()
    _available_tools: Set[Type[ToolInterface]] = PrivateAttr()
    _available_tool_names: Set[str] = PrivateAttr()
    _metrics: AgentMetrics = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def execute(self) -> AgentResult:
        """
        Executes the agent with parameter values provided in the fields.

        This is the main entry point for running an agent. It should handle the
        agent's lifecycle, including initialization, execution loop, and cleanup.

        Returns:
            The result of the agent execution, as an AgentResult
        """
        pass

    @abstractmethod
    async def construct_core_prompt(self) -> str:
        """
        Constructs and returns the agent's 'core prompt'.

        The core prompt contains the specific instructions and context for this
        agent instance, based on its parameters and the task it needs to perform.

        Returns:
            A string containing the core prompt
        """
        pass

    @classmethod
    @abstractmethod
    def generate_examples(cls) -> list[tuple['AgentInterface', AgentResult]]:
        """
        Generate example uses of the agent with their expected outputs.

        These examples are used in documentation and few-shot learning.

        Returns:
            A list of tuples, each containing an agent instance and its expected result
        """
        pass

    @abstractmethod
    async def _handle_tool_call(self, tool_content: ToolCallContent):
        """
        Handle a tool call from the LLM.

        This method processes tool invocations from the LLM, validates arguments,
        and executes the appropriate tool.

        Args:
            tool_content: The tool call content from the LLM
        """
        pass

    @abstractmethod
    async def _handle_agent_call(self, agent_content: ToolCallContent):
        """
        Handle an agent call from the LLM.

        This method processes agent invocations from the LLM, validates arguments,
        and executes the appropriate sub-agent.

        Args:
            agent_content: The agent call content from the LLM
        """
        pass

    @abstractmethod
    async def construct_system_prompt(self) -> str:
        """
        Constructs the system prompt for the agent.

        This method typically combines the agent's SYSTEM_PROMPT with additional
        context like tool documentation.

        Returns:
            A string containing the system prompt
        """
        pass

    @classmethod
    @abstractmethod
    def to_prompt_format(cls, arg_format=None) -> str:
        """
        Convert the agent definition to string format for prompts.

        Args:
            arg_format: The format to use for arguments

        Returns:
            A string containing the agent documentation in prompt format
        """
        pass

    @classmethod
    @abstractmethod
    def to_plain_prompt_format(cls, arg_format=None) -> str:
        """
        Convert the agent definition to a formatted string for the constrained tool use prompt.

        Args:
            arg_format: The format to use for arguments

        Returns:
            A string containing the agent documentation in plain format
        """
        pass

    @classmethod
    @abstractmethod
    async def args_str_to_dict(cls, args_str: str, arg_format=None) -> tuple[Optional[dict], Optional[str]]:
        """
        Convert an argument string to a dictionary.

        Args:
            args_str: The argument string to parse
            arg_format: The format of the arguments

        Returns:
            A tuple containing the parsed arguments and any warnings
        """
        pass
