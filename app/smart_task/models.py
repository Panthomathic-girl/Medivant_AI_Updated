# app/smart_task_prioritization/models.py
from pydantic import BaseModel, Field
from typing import List, Optional
from config import TASKTYPE, PriorityType, StatusType
from typing import Literal

class SuggestedTaskForm(BaseModel):
    task_title: str = Field(..., max_length=255, description="Clear, concise task title")
    task_type: TASKTYPE
    related_to: Optional[str] = Field(default=None, description="e.g., Lead: John Doe, Deal: ABC Corp")
    due_date: Optional[str] = Field(default=None, pattern=r"^\d{2}/\d{2}/\d{4}$", description="dd/mm/yyyy")
    priority: PriorityType = "medium"
    status: StatusType = "not started"
    reminder_date: Optional[str] = Field(default=None, pattern=r"^\d{2}/\d{2}/\d{4}$", description="dd/mm/yyyy")
    created_by: Optional[str] = Field(default=None, max_length=255)
    notes: Optional[str] = Field(default="", description="Detailed next steps, context, quotes from call")

class SmartTaskResponse(BaseModel):
    suggested_tasks: List[SuggestedTaskForm] = Field(default=[], description="List of fully structured task forms")
    call_has_next_steps: bool = Field(default=True)
    summary: Optional[str] = Field(default=None, description="Brief summary of call outcome and next steps")