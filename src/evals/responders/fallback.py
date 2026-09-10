"""Shared fallback/retry utilities for responders.

All local responders (astrobridge, base_qwen, base_qwen_text) share the same
two-pass retry pattern: run initial generation -> identify unparsed responses ->
re-prompt with the task's fallback_tag appended -> merge results back.
"""

from __future__ import annotations

from typing import List

from . import ModelResponse


def identify_failed_indices(
    responses: list[ModelResponse],
    task_name: str = "",
) -> list[int]:
    """Return indices where parsing failed (parsed is None or 'UNKNOWN')."""
    failed = []
    for i, r in enumerate(responses):
        if r.parsed is None or r.parsed == "UNKNOWN":
            failed.append(i)
    return failed


def merge_fallback_responses(
    original_responses: list[ModelResponse],
    failed_indices: list[int],
    fallback_raw_texts: list[str],
    fallback_tag: str,
    task: object,
) -> list[ModelResponse]:
    """Re-parse fallback outputs and merge them back into the original response list.

    Args:
        original_responses: The full list of ModelResponse from the first pass.
        failed_indices: Indices that failed parsing on the first pass.
        fallback_raw_texts: Raw model outputs from the fallback pass (same order as failed_indices).
        fallback_tag: The tag that was appended between original and fallback text.
        task: The task object (for default_parse and name).
    """
    task_name = getattr(task, "name", "")

    for idx, fb_text in zip(failed_indices, fallback_raw_texts):
        combined = original_responses[idx].raw_text + fallback_tag + " " + fb_text
        parsed = task.default_parse(combined)

        if parsed is None or parsed == "UNKNOWN":
            parsed = [] if task_name == "emission_lines" else "UNKNOWN"

        original_responses[idx] = ModelResponse(
            parsed=parsed,
            raw_text=combined,
            forced_fallback=True,
        )

    return original_responses

