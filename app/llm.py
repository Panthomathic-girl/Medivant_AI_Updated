# app/llm.py
import json
from pathlib import Path
from google import generativeai as genai
from google.generativeai import protos
from config import settings
from typing import Any, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)

def _upload_and_wait(file_path: str, mime_type: str, display_name: str = None):
    """Upload file and wait until fully processed (critical for video/audio)"""
    uploaded = genai.upload_file(
        path=file_path,
        mime_type=mime_type,
        display_name=display_name or Path(file_path).name
    )
    logger.info(f"Uploaded {file_path} → {uploaded.name}")

    # Wait until ACTIVE (videos can take 10–30 seconds)
    print(f"Processing your file... (this can take 10–60 seconds for video)")
    for _ in range(60):  # Max 60 seconds wait
        time.sleep(3)
        file = genai.get_file(uploaded.name)
        if file.state.name == "ACTIVE":
            logger.info("File processing complete.")
            return uploaded
        if file.state.name == "FAILED":
            raise RuntimeError("Gemini failed to process the file.")
        print(f"   → Still processing... ({file.state.name})")
    
    raise TimeoutError("File took too long to process. Try a shorter clip.")

def analyze_audio_file(filepath: str, mime_type: str, prompt: str) -> Any:
    uploaded = _upload_and_wait(filepath, mime_type)
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content([prompt, uploaded])
        return response
    finally:
        try:
            genai.delete_file(uploaded.name)
        except:
            pass

def analyze_video_file(filepath: str, mime_type: str, prompt: str) -> Any:
    uploaded = _upload_and_wait(filepath, mime_type)
    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content([prompt, uploaded])
        return response
    finally:
        try:
            genai.delete_file(uploaded.name)
        except:
            pass

        
def generate_text_response(prompt: str) -> str:

    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_structured_json(
    prompt: str,
    response_schema: Any,
    model_name: Optional[str] = None,
    temperature: float = 0.2
) -> Dict[str, Any]:

    model_name = model_name or settings.GEMINI_MODEL

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            # safety_settings=RELAXED_SAFETY
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
            ),
        )

        # Native JSON mode success
        if response.text:
            return json.loads(response.text)
        
        return response
            
    except Exception as e:
        print(f"JSON mode failed ({e}), falling back to text mode...")