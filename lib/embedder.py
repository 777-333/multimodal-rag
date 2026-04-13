import os
import time
import numpy as np
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-embedding-2-preview"

# Delay between every embedding call (seconds) to stay under rate limits
_CALL_DELAY = 0.3

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _normalize(vec: list[float]) -> list[float]:
    a = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm > 0:
        a = a / norm
    return a.tolist()


def _embed_with_retry(contents, config, max_retries: int = 8) -> list[float]:
    """Call embed_content with exponential backoff on 429 errors."""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            time.sleep(_CALL_DELAY)
            response = get_client().models.embed_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            return _normalize(response.embeddings[0].values)
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise


def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    return _embed_with_retry(
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )


def embed_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return _embed_with_retry(
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )


def embed_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/mp3",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    return _embed_with_retry(
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )


def embed_video(
    video_bytes: bytes,
    mime_type: str = "video/mp4",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
    return _embed_with_retry(
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )


def embed_pdf_page_bytes(
    pdf_bytes: bytes,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    return _embed_with_retry(
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )


def embed_query(text: str) -> list[float]:
    return embed_text(text, task_type="RETRIEVAL_QUERY")
