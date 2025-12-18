# app/transcribe_audio.py
from pathlib import Path
import google.generativeai as genai
# from config import settings
import logging

genai.configure(api_key="AIzaSyCTUBl9gmT0toV6x7hqVxPjylkZ2Fa-fWE")
logger = logging.getLogger(__name__)

def transcribe_audio_file(
    filepath: str,
    mime_type: str = None,
    model_name: str = "gemini-2.5-flash"
) -> str:
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    if mime_type is None:
        mime_map = {
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".m4a": "audio/m4a",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
            ".flac": "audio/flac",
        }
        mime_type = mime_map.get(file_path.suffix.lower(), "audio/mpeg")

    print(f"Transcribing: {file_path.name}")
    print(f"MIME type: {mime_type}\n")

    try:
        # Upload + transcribe
        audio_file = genai.upload_file(path=str(file_path), mime_type=mime_type)

        model = genai.GenerativeModel(model_name)

        prompt = """
        Transcribe this sales/planning call VERBATIM.
        Use speaker labels:
        [Rep]: ...
        [Prospect]: ...
        [Participant 1]: ...

        Include every word, "um", laughter, pauses if audible.
        NEVER summarize. NEVER skip anything.
        """

        response = model.generate_content(
            [prompt, audio_file],
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=8192
            )
        )

        transcript = response.text.strip()

        # Print + save
        print("="*80)
        print("FULL TRANSCRIPT")
        print("="*80)
        print(transcript)
        print("="*80)

        # Save to .txt file automatically
        output_file = file_path.with_suffix(".transcript.txt")
        output_file.write_text(transcript, encoding="utf-8")
        print(f"\nTranscript saved to: {output_file}")

        return transcript

    except Exception as e:
        error_msg = f"Transcription failed: {e}"
        print(error_msg)
        logger.error(error_msg)
        return error_msg
    finally:
        try:
            genai.delete_file(audio_file.name)
        except:
            pass


# ──────────────────────────────────────────────────────────────
# RUN DIRECTLY LIKE THIS (NO command-line args needed anymore)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change this line to your file — or paste full path
    AUDIO_FILE = Path("./audio_data/2025-11-20 Planning Meeting Audio.mp3")

    if not AUDIO_FILE.exists():
        print(f"File not found: {AUDIO_FILE}")
        print("Update the AUDIO_FILE path in this script!")
    else:
        transcribe_audio_file(str(AUDIO_FILE))