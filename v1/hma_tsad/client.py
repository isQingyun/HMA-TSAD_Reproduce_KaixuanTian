from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any

import requests

from .config import ModelConfig


@dataclass(frozen=True)
class ModelResponse:
    content: dict[str, Any]
    usage: dict[str, Any]
    request_id: str | None
    model: str | None


class DashScopeClient:
    def __init__(self, config: ModelConfig, project_root: str | Path = ".") -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.api_key = self._load_api_key()
        self.endpoint = config.api_base.rstrip("/") + "/chat/completions"

    def _load_api_key(self) -> str:
        from_environment = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if from_environment:
            return from_environment
        key_path = Path(self.config.api_key_file)
        if not key_path.is_absolute():
            key_path = self.project_root / key_path
        if not key_path.exists():
            raise RuntimeError(
                "DASHSCOPE_API_KEY is unset and the configured key file does not exist: "
                f"{key_path}"
            )
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"API key file is empty: {key_path}")
        return key

    @staticmethod
    def _image_data_url(path: str | Path) -> str:
        image_path = Path(path)
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            left = text.find("{")
            right = text.rfind("}")
            if left < 0 or right <= left:
                raise
            parsed = json.loads(text[left : right + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Model response must be a JSON object")
        return parsed

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str | Path] | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for path in image_paths or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(path)},
                    "min_pixels": 65536,
                    "max_pixels": 2621440,
                }
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
            "enable_thinking": False,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"Retryable API status {response.status_code}: {response.text[:300]}")
                response.raise_for_status()
                raw = response.json()
                message = raw["choices"][0]["message"]
                parsed = self._parse_json(message.get("content", ""))
                return ModelResponse(
                    content=parsed,
                    usage=raw.get("usage", {}),
                    request_id=response.headers.get("x-request-id") or raw.get("request_id"),
                    model=raw.get("model"),
                )
            except (requests.RequestException, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"DashScope request failed after {self.config.max_retries} attempts: {last_error}")

