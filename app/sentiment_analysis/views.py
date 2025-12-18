# app/sentiment_analysis/views.py
from fastapi import APIRouter, HTTPException
from .models import SentimentRequest, SentimentResponse
from app.sentiment_analysis.agent import analyze_sentiment_agent

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])

@router.post("/", response_model=SentimentResponse)
async def sentiment_llm(req: SentimentRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text is required")

    try:
        result = analyze_sentiment_agent(req.text.strip())
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Sentiment analysis failed: {str(e)}")