# app/smart_task/agent.py
from pathlib import Path
from app.llm import analyze_audio_file, analyze_video_file
from app.smart_task.utils import parse_structured_tasks
from typing import Callable, Any

def suggest_next_tasks(file_path: str, mime_type: str, analyze_func=None) -> dict:
    # Auto-pick analyzer based on MIME
    if analyze_func is None:
        if mime_type.startswith("video/") or Path(file_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            analyze_func = analyze_video_file
        else:
            analyze_func = analyze_audio_file

    prompt = """
You are the world's best B2B sales assistant analyzing a sales call (audio or video).

CRITICAL RULE — TASK TYPES MUST BE EXACTLY ONE OF THESE 5 VALUES ONLY:
"call", "email", "meeting", "administrative", "follow up"

DO NOT invent new types like "internal meeting", "internal task", "demo", etc.
→ Use "meeting" for any kind of meeting (internal or external)
→ Use "follow up" for generic follow-ups
→ Use "administrative" for internal tasks

Return ONLY valid JSON. No markdown.

Example:
{
  "suggested_tasks": [
    {
      "task_title": "Schedule demo with Sarah",
      "task_type": "meeting",
      "related_to": "Acme Corp - Q1 Deal",
      "due_date": "10/12/2025",
      "priority": "high",
      "status": "not started",
      "reminder_date": "08/12/2025",
      "created_by": "Sales AI",
      "notes": "She asked for a demo next week."
    },
    {
      "task_title": "Send proposal PDF",
      "task_type": "email",
      "related_to": "Acme Corp - Q1 Deal",
      "due_date": "09/12/2025",
      "priority": "high",
      "status": "not started",
      "notes": "Include pricing tier 3."
    }
  ],
  "call_has_next_steps": true,
  "summary": "Strong interest. Schedule demo and send proposal."
}

If deal dead:
{
  "suggested_tasks": [],
  "call_has_next_steps": false,
  "summary": "No interest shown."
}

Analyze the call now and return JSON only.
"""

    raw_response = analyze_func(file_path, mime_type, prompt)
    parsed_result = parse_structured_tasks(raw_response)
    return parsed_result.dict()