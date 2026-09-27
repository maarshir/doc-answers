"""Вызов модели Anthropic. Синхронный: вопрос один, параллелить нечего.

Логика повторов та же, что в promptdiff: повторяем только то, что может
пройти со второго раза (перегрузка, сбой сервера, обрыв сети), с растущей
паузой и разбросом. 400 или 401 повторять бессмысленно.
"""

import os
import random
import time
from dataclasses import dataclass, field

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
RETRY_CODES = {429, 500, 502, 503, 529}
MAX_TOKENS = 700
DEFAULT_MODEL = "claude-haiku-4-5"


class ModelError(RuntimeError):
    """Модель не ответила."""


@dataclass
class Reply:
    text: str
    model: str
    usage: dict = field(default_factory=dict)   # как пришло от API, для token-counter
    attempts: int = 1
    seconds: float = 0.0


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 20.0) -> float:
    return min(cap, base * 2 ** attempt) * (0.5 + random.random() / 2)


def ask(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    *,
    key: str | None = None,
    http: httpx.Client | None = None,
    max_attempts: int = 4,
    sleep=time.sleep,
) -> Reply:
    """Один запрос. http и sleep можно подменить в тестах."""
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ModelError("Не задана переменная окружения ANTHROPIC_API_KEY")

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # temperature не задаём: у части новых моделей параметр ограничен,
        # и запрос мог бы упасть с 400 на ровном месте
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}

    own = http is None
    http = http or httpx.Client(timeout=60)
    started = time.monotonic()
    last = ""
    try:
        for attempt in range(max_attempts):
            try:
                response = http.post(API_URL, json=payload, headers=headers)
            except httpx.RequestError as err:
                last = f"сеть: {err}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    text = "".join(
                        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
                    )
                    return Reply(
                        text=text.strip(),
                        model=data.get("model", model),
                        usage=data.get("usage", {}),
                        attempts=attempt + 1,
                        seconds=round(time.monotonic() - started, 3),
                    )
                if response.status_code not in RETRY_CODES:
                    raise ModelError(f"{response.status_code}: {response.text[:200]}")
                last = f"{response.status_code}: {response.text[:200]}"
            if attempt < max_attempts - 1:
                sleep(backoff_delay(attempt))
    finally:
        if own:
            http.close()
    raise ModelError(f"Не удалось за {max_attempts} попыток. Последнее: {last}")
