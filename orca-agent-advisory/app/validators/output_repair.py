import json
from typing import Any, TypeVar

from pydantic import BaseModel


class JsonOutputParseError(ValueError):
    """Raised when an LLM output cannot be parsed into JSON."""


ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_json_object(raw_output: Any) -> dict[str, Any]:
    text = _stringify_output(raw_output)
    repaired = repair_json_text(text)

    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise JsonOutputParseError(f"LLM output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JsonOutputParseError("LLM output must be a JSON object")
    return parsed


def parse_model_output(raw_output: Any, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(parse_json_object(raw_output))


def repair_json_text(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if not cleaned:
        raise JsonOutputParseError("LLM output is empty")

    fenced = _strip_markdown_fence(cleaned)
    if fenced != cleaned:
        return fenced

    first_object = _extract_first_balanced_object(cleaned)
    if first_object is not None:
        return first_object

    return cleaned


def _stringify_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    if hasattr(raw_output, "raw"):
        return str(raw_output.raw)
    return str(raw_output)


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
