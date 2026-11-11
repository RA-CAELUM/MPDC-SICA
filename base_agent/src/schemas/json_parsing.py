# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for JSON parsing into dicts, guided by Pydantic models.
"""

import re
import json
import logging

from enum import Enum
from typing import Type, Any, Union, get_args, get_origin, Literal
from pydantic import BaseModel

from .representation import get_json_schema_representation

logger = logging.getLogger(__name__)


def convert_json_value(value: Any, target_type: Type, field_info: Any = None) -> Any:
    """Convert a JSON value to the target type using Pydantic field information."""
    if value is None:
        return None

    # Get the origin type (handles Optional/Union types)
    origin_type = get_origin(target_type) or target_type

    # Handle lists first
    if origin_type == list:
        if not isinstance(value, list):
            value = [value]  # Convert single value to list
        item_type = get_args(target_type)[0] if get_args(target_type) else str
        return [convert_json_value(item, item_type) for item in value]

    # Handle dictionaries
    if origin_type == dict:
        if not isinstance(value, dict):
            try:
                # Try parsing string as JSON dict
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                raise ValueError(f"Cannot convert {value} to dictionary")

        key_type, val_type = (
            get_args(target_type) if get_args(target_type) else (str, Any)
        )
        return {
            convert_json_value(k, key_type): convert_json_value(v, val_type)
            for k, v in value.items()
        }

    # Handle Literal types
    if origin_type is Literal:
        literal_values = get_args(target_type)
        if value in literal_values:
            return value
        # Try converting string representation
        str_value = str(value).strip("\"'")
        if str_value in [str(v) for v in literal_values]:
            for v in literal_values:
                if str(v) == str_value:
                    return v
        raise ValueError(f"Value must be one of: {literal_values}")

    # Handle Enums
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        if isinstance(value, str):
            value = value.strip("\"'")
        try:
            if isinstance(value, str):
                return target_type[value]
            return target_type(value)
        except (KeyError, ValueError):
            raise ValueError(f"Value must be one of: {[e.name for e in target_type]}")

    # Handle nested Pydantic models
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        if isinstance(value, dict):
            return type_aware_json_to_dict(value, target_type)[0]
        raise ValueError(f"Cannot convert {value} to {target_type}")

    # Handle basic types
    try:
        if origin_type == str:
            return str(value)
        elif origin_type == int:
            if isinstance(value, str):
                # Handle string numbers and remove any quotes
                value = value.strip("\"'")
            # Handle float strings that are actually integers
            float_val = float(value)
            if float_val.is_integer():
                return int(float_val)
            raise ValueError(f"Cannot convert non-integer value {value} to int")
        elif origin_type == float:
            if isinstance(value, str):
                value = value.strip("\"'")
            return float(value)
        elif origin_type == bool:
            if isinstance(value, str):
                value = value.strip("\"'").lower()
                if value in ("true", "1", "yes", "on", "t"):
                    return True
                if value in ("false", "0", "no", "off", "f"):
                    return False
            return bool(value)
        return value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert '{value}' to {target_type}: {str(e)}")


def type_aware_json_to_dict(
    json_data: Union[str, dict], model_cls: Type[BaseModel]
) -> tuple[dict, str]:
    """
    Convert JSON data to a dictionary using Pydantic model for type guidance.

    Args:
        json_data: Either a JSON string or already parsed dictionary
        model_cls: Pydantic model class defining the expected structure

    Returns:
        Tuple of (converted dict, warnings string or None)
    """
    # Parse JSON if string input
    if isinstance(json_data, str):
        try:
            raw_dict = json.loads(json_data)
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {str(e)}"
    else:
        raw_dict = json_data

    warnings = []
    parsed_dict = {}

    # Get model fields
    model_fields = model_cls.model_fields

    # Process each field according to its type
    for field_name, field_info in model_fields.items():
        # Get field type and default information
        field_type = field_info.annotation
        is_optional = get_origin(field_type) is Union and type(None) in get_args(
            field_type
        )

        # Handle missing fields
        if field_name not in raw_dict:
            if field_info.is_required():
                warnings.append(f"Required field '{field_name}' is missing")
                continue
            # Use default value if available
            if hasattr(field_info, "default") and field_info.default is not None:
                parsed_dict[field_name] = field_info.default
            elif (
                hasattr(field_info, "default_factory")
                and field_info.default_factory is not None
            ):
                parsed_dict[field_name] = field_info.default_factory()
            continue

        # Convert value according to field type
        try:
            raw_value = raw_dict[field_name]
            parsed_dict[field_name] = convert_json_value(
                raw_value, field_type, field_info
            )
        except ValueError as e:
            warnings.append(f"Warning for field '{field_name}': {str(e)}")
            # For optional fields, use None on conversion failure
            if is_optional:
                parsed_dict[field_name] = None

    return parsed_dict, "\n".join(warnings) if warnings else None


async def json_str_to_dict(
    json_str: str, guide_obj: Type[BaseModel], repair_model = None
) -> tuple[dict | None, str | None]:
    """
    Attempt to parse the (possibly malformed) json string to a dict, and return
    that if successful. Will first attempt rules-based fixes, but if a
    target_schema is provided too, then an LLM will be used to attempt to
    repair the schema-only if sufficient information is provided in the initial
    json_str (to avoid hallucinating omitted fields).

    The function also returns any warnings or errors about fixes that needed to
    be made to the json.
    """
    # 1. Attempt to directly load the json strin to a dict
    json_obj, warnings = type_aware_json_to_dict(json_str, guide_obj)
    if json_obj:
        return json_obj, warnings

    # 2. If failed, attempt to repair using json_repair
    try:
        from json_repair import repair_json
        repair_result = repair_json(json_str, return_objects=True, logging=True)
        if isinstance(repair_result, tuple):
            repaired_obj, repair_logs = repair_result
        else:
            raise ValueError("Unrecognized repair json result")

        # Group similar repairs to avoid repetition
        repair_categories = {
            "quotes": [],
            "delimiters": [],
            "structure": [],
            "comments": [],
            "values": [],
            "other": [],
        }

        for log in repair_logs:
            message = log.get("text", "")

            # Categorize the repairs
            if "quote" in message.lower():
                repair_categories["quotes"].append(message)
            elif any(
                word in message.lower() for word in ["delimiter", "comma", "colon"]
            ):
                repair_categories["delimiters"].append(message)
            elif any(
                word in message.lower()
                for word in ["object", "array", "brace", "bracket"]
            ):
                repair_categories["structure"].append(message)
            elif "comment" in message.lower():
                repair_categories["comments"].append(message)
            elif any(
                word in message.lower() for word in ["value", "literal", "string"]
            ):
                repair_categories["values"].append(message)
            else:
                repair_categories["other"].append(message)

        # Build the feedback message
        feedback_parts = [
            f"Error encountered during JSON parsing:\n- {warnings}\n",
            "\nRepairs needed:",
        ]

        for category, messages in repair_categories.items():
            if messages:
                # Remove duplicates while preserving order
                unique_messages = list(dict.fromkeys(messages))
                if len(unique_messages) > 0:
                    feedback_parts.append(f"\n{category.title()}:")
                    for msg in unique_messages:
                        feedback_parts.append(f"- {msg}")

        return repaired_obj, "\n".join(feedback_parts)
    except Exception as e:
        pass

    # 3. If that fails, use an LLM as a last resort
    from ..llm.api import create_completion
    from ..llm.base import Message
    from ..types.llm_types import Model, TextContent
    from ..config import settings
    if repair_model is None:
        repair_model = settings.MODEL

    target_schema = get_json_schema_representation(guide_obj)
    sys_prompt = "You are an AI assistant specializing in reformatting malformed JSON strings into valid JSON that adheres to a specific schema."
    prompt = f"""The following response needs to be fully reformatted into valid JSON according to this schema:
<SCHEMA>
{target_schema}
</SCHEMA>

Original malformed JSON:
<MALFORMED_JSON>
{json_str}
</MALFORMED_JSON>

The original response failed to parse with the following error:
<ERROR_MESSAGE>
{warnings}
</ERROR_MESSAGE>

Your task is a) to identify what errors in the malformed JSON snippet are causing json.loads errors and what fixes are needed, and b) to reformat the malformed json into a complete and valid JSON object that strictly adheres to the provied schema.

You should provide your answer within two sections, the reasoning encapsulated in <FIX_DESCRIPTION>...</FIX_DESCRIPTION> tags, followed by the valid JSON in <JSON> ... </JSON> tags. If there is insufficient information or missing fields in the malformed json snippet, output <JSON>UNABLE_TO_FORMAT</JSON>.

IMPORTANT:
- Re-output the ENTIRE reformatted json snippet between the <JSON>...</JSON> tags, including the correct parts, making sure it inclues ALL required fields in the schema
- Keep your response terse, and minimise explanations or discussions outside the <FIX_DESCRIPTION> and <JSON> tag pairs.
- Ensure all string values are properly escaped and enclosed in double quotes.
- Do not simply make up a JSON adherant response if it isn't strongly supported or requires significant interpretation of the original malformed JSON.

To recap, your response should look like this:

<FIX_DESCRIPTION>
Explain your approach to reformatting the response, including any challenges you encountered and how you addressed them.
</FIX_DESCRIPTION>

<JSON>
Insert the formatted JSON object here, or UNABLE_TO_FORMAT
</JSON>
"""
    response = await create_completion(
        messages=[
            Message(role="system", content=[TextContent(text=sys_prompt)]),
            Message(role="user", content=[TextContent(text=prompt)]),
        ],
        model=repair_model,
    )
    completion = response.content[0]
    assert isinstance(completion, TextContent), "expected text completion"
    content = completion.text

    fix_match = re.search(
        r"<FIX_DESCRIPTION>(.*?)</FIX_DESCRIPTION>", content, re.DOTALL
    )
    json_match = re.search(r"<JSON>(.*?)</JSON>", content, re.DOTALL)

    fix_description = fix_match.group(1).strip() if fix_match else ""
    json_str = json_match.group(1).strip() if json_match else ""

    logger.info(f"JSON formatting fix description: {fix_description}")
    logger.info(f"Corrected JSON string: {json_str}")

    try:
        return type_aware_json_to_dict(json_str, guide_obj)
    except Exception as e:
        return (
            None,
            f"Could not load json, even after applying the following fix: {fix_description}. New error: {str(e)}",
        )
