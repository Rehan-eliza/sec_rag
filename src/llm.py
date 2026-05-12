"""
llm.py
------
Wraps the single Ollama API call.

Design constraint: the answer to any user question must come from exactly
one call to this function. No chaining, no follow-up calls.

Supports both:
  - Blocking call  →  call_ollama(prompt)         returns str
  - Streaming call →  stream_ollama(prompt)        yields str chunks
"""

from __future__ import annotations

import json
from typing import Generator

import requests

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def ollama_is_available() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def model_is_available() -> bool:
    """Return True if the configured model is pulled and ready."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        # Match on model name prefix (handles tags like "gemma4:e4b")
        return any(OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models)
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Blocking call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str) -> str:
    """
    Send prompt to Ollama and return the full response text.
    Raises RuntimeError on connection or API failure.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":  OLLAMA_TEMPERATURE,
            "num_predict":  OLLAMA_NUM_PREDICT,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 s.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")


# ---------------------------------------------------------------------------
# Streaming call  (used by Streamlit's st.write_stream)
# ---------------------------------------------------------------------------

def stream_ollama(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from Ollama. Yields string chunks as they arrive.
    Raises RuntimeError on connection failure.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }

    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 s.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")
