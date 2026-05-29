"""Evaluation case loader for local agent checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_tool_names(case: dict[str, Any]) -> list[str]:
    tools = case.get("expected_tools", [])
    if not isinstance(tools, list):
        return []
    return [str(tool) for tool in tools]

