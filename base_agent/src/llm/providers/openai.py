# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""OpenAI-specific LLM provider implementation."""

import json
import tiktoken

from typing import List, Optional, Union, AsyncGenerator, Any, Type
from datetime import datetime
from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.lib import pydantic_function_tool

from ..base import (
    Message,
    Completion,
    CompletionChunk,
    TimingInfo,
)
from .base_provider import BaseProvider
from ...types.llm_types import TokenUsage, Model, StopReason, ReasoningEffort, TextContent, ToolCallContent, ToolResultContent, FCI


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI's models."""

    def __init__(self, client):
        self.client = client
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    def map_stop_reason(self, response: Any) -> tuple[StopReason, Optional[str]]:
        """Map OpenAI finish reasons to our standard format."""
        # Extract finish reason
        raw = getattr(response, "_raw_response", {})
        finish_reason = raw.get("finish_reason")

        if not finish_reason and hasattr(response, "choices") and response.choices:
            finish_reason = response.choices[0].finish_reason

        # Map to standard format
        if finish_reason == "length":
            return StopReason.LENGTH, None
        elif finish_reason == "tool_calls":
            return StopReason.TOOL_CALL, None
        elif finish_reason == "error":
            return StopReason.ERROR, None
        else:  # 'stop' or others
            return StopReason.COMPLETE, None

    def pydantic_to_native_tool(
        self, tool: Type[BaseModel]
    ) -> dict | ChatCompletionToolParam:
        if hasattr(tool, "TOOL_NAME"):
            # Descriptions cannot exceed 1024 characters (!) with OAI
            desc = tool.TOOL_DESCRIPTION[:1024]
            return {
                "type": "function",
                "function": {
                    "name": tool.TOOL_NAME,
                    "description": desc,
                    "parameters": tool.model_json_schema(),
                },
            }
        elif hasattr(tool, "AGENT_NAME"):
            # Descriptions cannot exceed 1024 characters (!) with OAI
            desc = tool.AGENT_DESCRIPTION[:1024]
            return {
                "type": "function",
                "function": {
                    "name": tool.AGENT_NAME,
                    "description": desc,
                    "parameters": tool.model_json_schema(),
                },
            }
        else:
            # Fallback to generic (beta) option from OpenAI
            return pydantic_function_tool(tool)

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        # Prepare the messages from our format to OpenAI
        oai_messages = []
        for msg in messages:
            if msg.role == "assistant":
                # Handle assistant messages and allowed / expected blocks
                msg_content = ""
                tool_calls = []
                for block in msg.content:
                    # We only expect TextContent and ToolCallContent
                    # Reasoning is omitted in OAI, and we don't expect ToolResponseContent here
                    if isinstance(block, TextContent):
                        msg_content += block.text
                    elif isinstance(block, ToolCallContent):
                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=block.call_id,
                                function=Function(
                                    arguments=json.dumps(block.tool_args),
                                    name=block.tool_name,
                                ),
                                type="function",
                            )
                        )
                oai_messages.append(
                    ChatCompletionMessage(
                        content=msg_content,
                        role="assistant",
                        tool_calls=tool_calls if len(tool_calls) else None,
                    )
                )
            else:
                msg_content = ""
                for block in msg.content:
                    if isinstance(block, TextContent):
                        msg_content += block.text
                    elif isinstance(block, ToolResultContent):
                        # Append what we have so far
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

        return oai_messages

    async def create_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[BaseModel] | None = None,
        num_completions: int = 1,
    ) -> Completion:
        """Create a completion using OpenAI's API."""
        start_time = datetime.now()

        # if messages[-1].role == "assistant":
        #     messages[-1].content = messages[-1].content.rstrip()

        api_messages = self._prepare_messages(messages)

        args = {
            "messages": api_messages,
            "model": model.id,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_tokens or model.max_output_tokens,
        }
        if model.is_reasoner:
            args.pop("temperature")
            args.pop("top_p")

        if stop:
            args["stop"] = [stop] if isinstance(stop, str) else stop

        if available_tools and model.fci == FCI.CONSTRAINED:
            args["tools"] = [self.pydantic_to_native_tool(t) for t in available_tools]

        # raw_response = await self.client.chat.completions.with_raw_response.create(
        #     **args
        # )
        # response = raw_response.parse()
        response = await self.client.chat.completions.create(**args)

        end_time = datetime.now()
        duration = end_time - start_time

        choice = response.choices[0]

        # Extract cached tokens information
        cached_tokens = 0
        if hasattr(response.usage, "prompt_tokens_details") and hasattr(
            response.usage.prompt_tokens_details, "cached_tokens"
        ):
            cached_tokens = response.usage.prompt_tokens_details.cached_tokens

        # Create token usage
        token_usage = TokenUsage(
            uncached_prompt_tokens=response.usage.prompt_tokens,
            cache_write_prompt_tokens=(
                (response.usage.prompt_tokens - cached_tokens)
                if cached_tokens > 0
                else response.usage.prompt_tokens
            ),
            cached_prompt_tokens=cached_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # Calculate timing information
        tokens_per_second = (
            token_usage.completion_tokens / duration.total_seconds()
            if duration.total_seconds() > 0
            else None
        )
        timing_info = TimingInfo(
            start_time=start_time,
            end_time=end_time,
            total_duration=duration,
            first_token_time=end_time,  # For non-streaming, we only get all tokens at once
            time_to_first_token=duration.total_seconds(),
            tokens_per_second=tokens_per_second,
        )

        # Map stop reason
        stop_reason, stop_sequence = self.map_stop_reason(response)
        has_tool_calls = False

        response_content = []
        # First add any assistant content
        if choice.message.content:
            response_content.append(TextContent(text=choice.message.content))
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                has_tool_calls = True
                response_content.append(
                    ToolCallContent(
                        call_id=tc.id,
                        tool_name=tc.function.name,
                        tool_args=json.loads(tc.function.arguments),
                        call_type=FCI.CONSTRAINED,
                    )
                )

        completion = Completion(
            id=response.id,
            content=response_content,
            model=model,
            usage=token_usage,
            timing=timing_info,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            raw_response={
                **(
                    {"full_oai_tool_call_message": choice.message}
                    if has_tool_calls
                    else {}
                ),
                "finish_reason": choice.finish_reason,
                "created": response.created,
                "model": response.model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )

        return completion

    async def create_streaming_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[BaseModel] | None = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Create a streaming completion using OpenAI's API."""
        #
        # WARNING: this function is very out of date; see create_completion for
        # a more up-to-date example of how the framework should work.
        #
        start_time = datetime.now()
        first_token_time = None
        token_count = 0

        api_messages = [
            {
                "role": msg.role,
                "content": (
                    msg.content
                    if isinstance(msg.content, str)
                    else "\n".join(msg.content).rstrip()
                ),
                **({"name": msg.name} if msg.name else {}),
            }
            for msg in messages
        ]

        args = {
            "messages": api_messages,
            "model": model.id,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "max_completion_tokens": max_tokens or model.max_output_tokens,
            "stream_options": {"include_usage": True},
        }

        if stop:
            args["stop"] = [stop] if isinstance(stop, str) else stop

        if reasoning_effort:
            args["reasoning_effort"] = reasoning_effort.value

        current_stop = None
        accumulated_content = ""
        usage_info = None

        async for chunk in await self.client.chat.completions.create(**args):
            chunk: ChatCompletionChunk
            current_time = datetime.now()

            if len(chunk.choices) == 0 and chunk.usage:
                # Handle last chunk with usage info

                # Need to fix this...
                usage_info = TokenUsage(
                    uncached_prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    cached_prompt_tokens=0,
                    cache_write_prompt_tokens=0,
                )

                if (
                    chunk.usage.prompt_tokens_details
                    and chunk.usage.prompt_tokens_details.cached_tokens
                ):
                    usage_info.cached_prompt_tokens = (
                        chunk.usage.prompt_tokens_details.cached_tokens
                    )
                    usage_info.cache_write_prompt_tokens = (
                        usage_info.uncached_prompt_tokens
                        - usage_info.cached_prompt_tokens
                    )

                timing_info = TimingInfo(
                    start_time=start_time,
                    end_time=current_time,
                    total_duration=current_time - start_time,
                    first_token_time=first_token_time,
                    time_to_first_token=(
                        (first_token_time - start_time).total_seconds()
                        if first_token_time
                        else None
                    ),
                    tokens_per_second=(
                        usage_info.completion_tokens
                        / (current_time - start_time).total_seconds()
                        if first_token_time
                        else None
                    ),
                )

                yield CompletionChunk(
                    id=chunk.id,
                    content="",
                    model=model,
                    is_finished=True,
                    timing=timing_info,
                    usage=usage_info,
                    stop_reason=current_stop,
                )
            else:
                delta = chunk.choices[0].delta

                # Get stop information when finish reason available
                if chunk.choices[0].finish_reason is not None:
                    stop_reason, _ = self.map_stop_reason(chunk)
                    current_stop = stop_reason

                # Process content
                if delta.content:
                    accumulated_content += delta.content
                    if chunk.usage:
                        token_count = chunk.usage.completion_tokens
                    else:
                        token_count = len(self.encoding.encode(delta.content))
                    if first_token_time is None:
                        first_token_time = current_time

                    timing_info = TimingInfo(
                        start_time=start_time,
                        end_time=current_time,
                        total_duration=current_time - start_time,
                        first_token_time=first_token_time,
                        time_to_first_token=(
                            (first_token_time - start_time).total_seconds()
                            if first_token_time
                            else None
                        ),
                        tokens_per_second=(
                            token_count / (current_time - start_time).total_seconds()
                            if first_token_time
                            else None
                        ),
                    )

                    chunk_completion = CompletionChunk(
                        id=chunk.id,
                        content=delta.content,
                        model=model,
                        is_finished=chunk.choices[0].finish_reason is not None,
                        timing=timing_info,
                        usage=usage_info,
                        stop_reason=current_stop,
                    )

                    if chunk.choices[0].finish_reason is not None:
                        chunk_completion.raw_response = {
                            "finish_reason": chunk.choices[0].finish_reason,
                            "accumulated_content": accumulated_content,
                        }

                    yield chunk_completion
