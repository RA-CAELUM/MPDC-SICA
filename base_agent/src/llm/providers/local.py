# Self-Improving Coding Agent - Local LLM / Qwen3 Provider for MPDC
"""Local OpenAI-compatible LLM provider (vLLM, SGLang, llama.cpp, MPDC Gateway) with Qwen3 <think> support."""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, AsyncGenerator, List, Optional, Type, Union

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..base import (
    Completion,
    CompletionChunk,
    Message,
    TimingInfo,
)
from ...types.llm_types import (
    FCI,
    Model,
    ReasoningContent,
    ReasoningEffort,
    StopReason,
    TextContent,
    TokenUsage,
    ToolCallContent,
    ToolResultContent,
)
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def split_qwen_think_blocks(raw_text: str) -> tuple[str, str]:
    """Split Qwen3 `<think>...</think>` blocks from final assistant output.

    Returns:
        Tuple of `(reasoning_cot, clean_text)`.
    """
    if not raw_text:
        return "", ""
    thoughts: list[str] = []
    for match in _THINK_PATTERN.finditer(raw_text):
        thought = match.group(1).strip()
        if thought:
            thoughts.append(thought)
    cleaned = _THINK_PATTERN.sub("", raw_text).lstrip()
    # Handle unclosed <think> if output truncated before </think>
    if "<think>" in cleaned and "</think>" not in cleaned:
        before, _, after = cleaned.partition("<think>")
        if after.strip():
            thoughts.append(after.strip())
        cleaned = before.strip()
    return "\n\n".join(thoughts), cleaned


def resolve_local_base_url() -> str:
    """Resolve the OpenAI-compatible local endpoint URL (defaults to MPDC Gateway)."""
    explicit = os.environ.get("LOCAL_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    gateway = os.environ.get("MPDC_GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
    if not gateway.endswith("/v1"):
        return f"{gateway}/v1"
    return gateway


class LocalOpenAIProvider(BaseProvider):
    """Provider implementation for local Qwen3 / OpenAI-compatible servers (MPDC Gateway, vLLM, SGLang)."""

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        if client is not None:
            self.client = client
        else:
            base_url = resolve_local_base_url()
            api_key = (
                os.environ.get("LOCAL_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "mpdc-local"
            )
            node_id = os.environ.get("MPDC_NODE_ID", "sica-self-improver")
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={"X-MPDC-Node": node_id},
            )

    def map_stop_reason(self, response: Any) -> tuple[StopReason, Optional[str]]:
        """Map OpenAI/vLLM finish reasons to SICA standard StopReason."""
        raw = getattr(response, "_raw_response", {})
        finish_reason = raw.get("finish_reason") if isinstance(raw, dict) else None
        if not finish_reason and hasattr(response, "choices") and response.choices:
            finish_reason = response.choices[0].finish_reason

        if finish_reason in ("length", "max_tokens"):
            return StopReason.LENGTH, None
        elif finish_reason == "tool_calls":
            return StopReason.TOOL_CALL, None
        elif finish_reason == "error":
            return StopReason.ERROR, None
        else:
            return StopReason.COMPLETE, None

    def pydantic_to_native_tool(self, tool: Type[BaseModel]) -> dict:
        if hasattr(tool, "TOOL_NAME"):
            return {
                "type": "function",
                "function": {
                    "name": tool.TOOL_NAME,
                    "description": getattr(tool, "TOOL_DESCRIPTION", "")[:1024],
                    "parameters": tool.model_json_schema(),
                },
            }
        elif hasattr(tool, "AGENT_NAME"):
            return {
                "type": "function",
                "function": {
                    "name": tool.AGENT_NAME,
                    "description": getattr(tool, "AGENT_DESCRIPTION", "")[:1024],
                    "parameters": tool.model_json_schema(),
                },
            }
        raise ValueError(f"Provided tool {tool} is neither BaseTool nor BaseAgent")

    def _prepare_messages(
        self, messages: list[Message], is_reasoner: bool = False
    ) -> list[dict[str, Any]]:
        """Convert SICA messages to OpenAI chat dictionaries with Qwen3 think/no_think control."""
        oai_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant":
                msg_content = ""
                tool_calls = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        msg_content += block.text
                    elif isinstance(block, ToolCallContent):
                        tool_calls.append(
                            {
                                "id": block.call_id,
                                "type": "function",
                                "function": {
                                    "name": block.tool_name,
                                    "arguments": json.dumps(block.tool_args),
                                },
                            }
                        )
                entry: dict[str, Any] = {"role": "assistant", "content": msg_content}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                oai_messages.append(entry)
            else:
                msg_content = ""
                for block in msg.content:
                    if isinstance(block, TextContent):
                        msg_content += block.text
                    elif isinstance(block, ToolResultContent):
                        if msg_content != "":
                            oai_messages.append(
                                {"role": msg.role, "content": msg_content}
                            )
                            msg_content = ""
                        oai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.call_id,
                                "content": block.content,
                            }
                        )
                if msg_content != "":
                    oai_messages.append({"role": msg.role, "content": msg_content})

        # Apply Qwen3 soft thinking toggle if requested via env (default enabled for fast non-reasoner turns)
        if (
            oai_messages
            and oai_messages[-1]["role"] == "user"
            and isinstance(oai_messages[-1].get("content"), str)
        ):
            user_text = oai_messages[-1]["content"]
            toggle_mode = os.environ.get("QWEN3_THINK_TOGGLE", "auto").lower()
            if toggle_mode == "auto":
                if is_reasoner and "/no_think" not in user_text and "<think>" not in user_text:
                    pass  # Qwen3 thinks by default
                elif not is_reasoner and "/no_think" not in user_text and "/think" not in user_text:
                    oai_messages[-1]["content"] = f"{user_text}\n/no_think"

        return oai_messages

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsage()
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details and getattr(details, "cached_tokens", None):
            cached_tokens = details.cached_tokens or 0
        return TokenUsage(
            uncached_prompt_tokens=max(0, prompt_tokens - cached_tokens),
            cache_write_prompt_tokens=0,
            cached_prompt_tokens=cached_tokens,
            completion_tokens=completion_tokens,
        )

    async def create_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 0.95,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[BaseModel]] | None = None,
        num_completions: int = 1,
    ) -> Completion:
        """Create a chat completion against local Qwen3 / vLLM / MPDC Gateway."""
        start_time = datetime.now()
        api_messages = self._prepare_messages(messages, is_reasoner=model.is_reasoner)
        wire_model = os.environ.get("LOCAL_LLM_MODEL_NAME") or model.id

        # Qwen3 recommended sampling defaults: temperature=0.6, top_p=0.95 for thinking
        effective_temp = 0.6 if model.is_reasoner else temperature
        effective_top_p = 0.95 if model.is_reasoner else top_p

        args: dict[str, Any] = {
            "messages": api_messages,
            "model": wire_model,
            "temperature": effective_temp,
            "top_p": effective_top_p,
            "max_tokens": max_tokens or model.max_output_tokens,
        }

        if stop:
            args["stop"] = [stop] if isinstance(stop, str) else stop

        if available_tools and model.fci == FCI.CONSTRAINED:
            args["tools"] = [self.pydantic_to_native_tool(t) for t in available_tools]

        response = await self.client.chat.completions.create(**args)
        end_time = datetime.now()
        duration = end_time - start_time

        token_usage = self._extract_token_usage(response)
        tokens_per_second = (
            token_usage.completion_tokens / duration.total_seconds()
            if duration.total_seconds() > 0
            else None
        )
        timing_info = TimingInfo(
            start_time=start_time,
            end_time=end_time,
            total_duration=duration,
            first_token_time=end_time,
            time_to_first_token=duration.total_seconds(),
            tokens_per_second=tokens_per_second,
        )

        stop_reason, stop_sequence = self.map_stop_reason(response)
        choice = response.choices[0]
        message = choice.message

        response_content = []

        # 1. Extract explicit reasoning_content (vLLM --enable-reasoning) or inline <think>...</think>
        explicit_reasoning = getattr(message, "reasoning_content", None)
        raw_text = message.content or ""
        inline_reasoning, clean_text = split_qwen_think_blocks(raw_text)

        combined_reasoning = "\n\n".join(
            part for part in (explicit_reasoning, inline_reasoning) if part
        )
        if combined_reasoning:
            response_content.append(ReasoningContent(text=combined_reasoning))

        # 2. Always append TextContent after ReasoningContent so completion.content[-1] is TextContent for XML tool parsing
        if clean_text or not response_content:
            response_content.append(TextContent(text=clean_text))

        # 3. Native tool calls if FCI.CONSTRAINED was used
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                response_content.append(
                    ToolCallContent(
                        call_id=tc.id,
                        tool_name=tc.function.name,
                        tool_args=json.loads(tc.function.arguments),
                        call_type=FCI.CONSTRAINED,
                    )
                )

        return Completion(
            id=getattr(response, "id", f"local-{wire_model}"),
            content=response_content,
            model=model,
            usage=token_usage,
            timing=timing_info,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            raw_response={
                "finish_reason": choice.finish_reason,
                "created": getattr(response, "created", None),
                "model": getattr(response, "model", wire_model),
                "reasoning_content": combined_reasoning,
            },
        )

    async def create_streaming_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 0.95,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[dict] | None = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Create a streaming completion against local Qwen3 / vLLM / MPDC Gateway."""
        start_time = datetime.now()
        first_token_time = None
        api_messages = self._prepare_messages(messages, is_reasoner=model.is_reasoner)
        wire_model = os.environ.get("LOCAL_LLM_MODEL_NAME") or model.id

        args: dict[str, Any] = {
            "messages": api_messages,
            "model": wire_model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or model.max_output_tokens,
            "stream": True,
        }
        if stop:
            args["stop"] = [stop] if isinstance(stop, str) else stop

        accumulated = []
        current_stop = None

        async for chunk in await self.client.chat.completions.create(**args):
            current_time = datetime.now()
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                current_stop, _ = self.map_stop_reason(chunk)
            delta_text = getattr(choice.delta, "content", None)
            if delta_text:
                if first_token_time is None:
                    first_token_time = current_time
                accumulated.append(delta_text)
                timing_info = TimingInfo(
                    start_time=start_time,
                    end_time=current_time,
                    total_duration=current_time - start_time,
                    first_token_time=first_token_time,
                    time_to_first_token=(current_time - start_time).total_seconds(),
                    tokens_per_second=None,
                )
                yield CompletionChunk(
                    id=getattr(chunk, "id", f"local-stream-{wire_model}"),
                    content=delta_text,
                    model=model,
                    is_finished=choice.finish_reason is not None,
                    timing=timing_info,
                    usage=TokenUsage(),
                    stop_reason=current_stop,
                )
