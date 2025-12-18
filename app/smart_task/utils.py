# app/smart_task/utils.py
import json
import re
import logging
from typing import Optional, Dict, Any
from .models import SmartTaskResponse, SuggestedTaskForm

logger = logging.getLogger(__name__)

def extract_json_from_llm_response(text: str) -> Optional[Dict]:
    if not text:
        return None

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    try:
        cleaned = text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)
    except:
        pass

    rough = re.search(r'\{.*"suggested_tasks"\s*:\s*\[.*\].*\}', text, re.DOTALL)
    if rough:
        try:
            return json.loads(rough.group(0))
        except:
            pass

    return None

# app/smart_task_prioritization/utils.py

TASK_TYPE_MAPPING = {
    "internal meeting": "meeting",
    "internal task": "administrative",
    "demo": "meeting",
    "presentation": "meeting",
    "follow-up": "follow up",
    "followup": "follow up",
    "send email": "email",
    "make a call": "call",
    "schedule call": "call",
    "admin": "administrative",
    "administrative task": "administrative",
}

def _normalize_task_type(task_type: str) -> str:
    task_type_lower = task_type.strip().lower()
    return TASK_TYPE_MAPPING.get(task_type_lower, task_type_lower)

def parse_structured_tasks(response: Any) -> SmartTaskResponse:
    try:
        text = ""
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            part = response.candidates[0].content.parts[0]
            text = getattr(part, "text", "")
        elif isinstance(response, str):
            text = response

        if not text:
            return SmartTaskResponse(suggested_tasks=[], call_has_next_steps=False, summary="No content detected.")

        data = extract_json_from_llm_response(text)
        if not data:
            return SmartTaskResponse(suggested_tasks=[], call_has_next_steps=False, summary="Invalid JSON from AI.")

        raw_tasks = data.get("suggested_tasks", [])
        summary = data.get("summary", "Next steps generated.")

        validated_tasks = []
        for task_data in raw_tasks[:5]:  # Allow up to 5
            if not isinstance(task_data, dict):
                continue

            # AUTO-FIX task_type
            if "task_type" in task_data:
                original = task_data["task_type"]
                normalized = _normalize_task_type(original)
                if normalized in ["call", "email", "meeting", "administrative", "follow up"]:
                    task_data["task_type"] = normalized
                else:
                    # Final fallback
                    task_data["task_type"] = "follow up" if "follow" in normalized else "administrative"

            try:
                task = SuggestedTaskForm(**task_data)
                validated_tasks.append(task)
            except Exception as e:
                logger.warning(f"Task validation failed after fix: {e} | Original: {task_data}")
                continue

        has_tasks = len(validated_tasks) > 0

        return SmartTaskResponse(
            suggested_tasks=validated_tasks,
            call_has_next_steps=has_tasks,
            summary=summary if has_tasks else "No actionable next steps identified."
        )

    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return SmartTaskResponse(suggested_tasks=[], call_has_next_steps=False, summary="Processing failed.")