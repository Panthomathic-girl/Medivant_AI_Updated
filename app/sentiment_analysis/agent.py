# app/sentiment_analysis/agent.py
from config import settings
from app.sentiment_analysis.models import SentimentResponse
from app.llm import generate_structured_json


def analyze_sentiment_agent(text: str) -> dict:
    prompt = f'''
    text : {text}
    You are a sentiment analysis expert. Analyze the following text and return ONLY valid JSON.
    '''

    result =  generate_structured_json(
        prompt,
        SentimentResponse,
        model_name=settings.GEMINI_MODEL)
    
    return result
