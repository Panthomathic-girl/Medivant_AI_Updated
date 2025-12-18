# app/sentiment_analysis/models.py
from pydantic import BaseModel, Field
from typing import Literal

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="Overall sentiment classification"
    )
    explanation: str = Field(
        ..., description="Reason for the assigned sentiment"
    )
    confidence_score: float = Field(
        ..., description="Confidence score of the sentiment prediction"
    )