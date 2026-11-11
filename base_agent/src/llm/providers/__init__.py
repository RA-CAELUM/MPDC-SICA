# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Provider-specific implementations for different LLM services."""

from .local import LocalOpenAIProvider


def __getattr__(name: str):
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider
        return AnthropicProvider
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider
        return OpenAIProvider
    if name == "DeepSeekProvider":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider
    if name == "FireworksProvider":
        from .fireworks import FireworksProvider
        return FireworksProvider
    if name == "GoogleProvider":
        from .google import GoogleProvider
        return GoogleProvider
    if name == "GoogleRESTProvider":
        from .google_rest import GoogleRESTProvider
        return GoogleRESTProvider
    if name == "GoogleOAIProvider":
        from .google_oai import GoogleOAIProvider
        return GoogleOAIProvider
    if name == "VertexProvider":
        from .vertex import VertexProvider
        return VertexProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "FireworksProvider",
    "GoogleProvider",
    "GoogleRESTProvider",
    "GoogleOAIProvider",
    "VertexProvider",
    "LocalOpenAIProvider",
]
