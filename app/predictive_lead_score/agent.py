#app/predictive_lead_score/agent.py
from app.llm import analyze_audio_file
from app.predictive_lead_score.utils import parse_gemini_generate_content_response

async def lead_score_agent(file_path: str, mime_type: str) -> dict:
    prompt = """
    You are an AI sales qualification expert. Analyze this audio and score it as a B2B lead from 0 to 1.

    Score based on:
    - Interest in product/service
    - Pain points or challenges
    - Urgency or timeline
    - Budget or investment signals
    - Decision-maker presence
    - Objections or concerns
    - Next steps or follow-up

    Even if it's not a sales call (e.g. interview, personal story), score based on *lead potential*.

    Respond with ONLY valid JSON:
    {
      "transcription": "<full exact text>",
      "score": <float 0 to 1>,
      "explanation": "<1-2 sentences>"
    }
    """
    response = analyze_audio_file(file_path, mime_type, prompt)

    response = parse_gemini_generate_content_response(response)
    return response