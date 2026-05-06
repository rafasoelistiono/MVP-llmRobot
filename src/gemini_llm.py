from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_if_present(env_path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries without overriding the real environment."""
    path = env_path or ROOT / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class GeminiLLM:
    def __init__(self) -> None:
        load_dotenv_if_present()
        try:
            self.api_key = os.environ["GEMINI_API_KEY"]
        except KeyError as exc:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to the environment or .env.") from exc

        self.model = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
        if not self.model.strip():
            raise RuntimeError("GEMINI_MODEL is set but empty.")
        print(f"[LLM] Gemini model selected: {self.model}")

    def generate(self, prompt: str) -> str:
        print(f"[LLM] Calling Gemini model: {self.model}")
        errors: list[str] = []

        for generator in (self._generate_with_google_genai, self._generate_with_google_generativeai, self._generate_with_rest):
            try:
                text = generator(prompt)
            except Exception as exc:  # noqa: BLE001 - report all SDK/API failures clearly.
                errors.append(f"{generator.__name__}: {exc}")
                continue
            if not text or not text.strip():
                errors.append(f"{generator.__name__}: empty response")
                continue
            return text

        raise RuntimeError("Gemini request failed. " + " | ".join(errors))

    def _generate_with_google_genai(self, prompt: str) -> str:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-genai package is not installed") from exc

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(response, "text", None)
        if text:
            return str(text)
        return _extract_text_from_response(response)

    def _generate_with_google_generativeai(self, prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-generativeai package is not installed") from exc

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text:
            return str(text)
        return _extract_text_from_response(response)

    def _generate_with_rest(self, prompt: str) -> str:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        encoded_model = urllib.parse.quote(model_path, safe="/")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"{encoded_model}:generateContent?key={urllib.parse.quote(self.api_key)}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc}") from exc

        candidates = response_payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"no candidates in API response: {response_payload}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
        if not text.strip():
            raise RuntimeError(f"candidate contained no text: {response_payload}")
        return text


def _extract_text_from_response(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)
