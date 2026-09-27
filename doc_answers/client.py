"""Вызов модели: Anthropic или Groq. Синхронный: вопрос один, параллелить нечего.

Логика повторов та же, что в promptdiff: повторяем только то, что может
пройти со второго раза (перегрузка, лимит частоты, сбой сервера, обрыв сети),
с растущей паузой и разбросом. Если сервер сам написал в retry-after, сколько
ждать (так делает Groq на бесплатном уровне), ждём столько. 400 или 401
повторять бессмысленно.
"""

import os
import random
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import httpx

from . import providers
from .providers import ANTHROPIC

RETRY_CODES = {429, 500, 502, 503, 529}
# Старые имена оставлены для совместимости: это значения для Anthropic
MAX_TOKENS = ANTHROPIC.max_tokens
DEFAULT_MODEL = ANTHROPIC.default_model
# Дольше этого не ждём по retry-after за один раз, даже если сервер просит больше
MAX_RETRY_AFTER = 90.0


class ModelError(RuntimeError):
    """Модель не ответила."""


@dataclass
class Reply:
    text: str
    model: str
    usage: dict = field(default_factory=dict)   # как пришло от API, разбирает token-counter
    attempts: int = 1
    seconds: float = 0.0
    provider: str = "anthropic"


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 20.0) -> float:
    return min(cap, base * 2 ** attempt) * (0.5 + random.random() / 2)


def retry_after(response: httpx.Response) -> float | None:
    """Сколько секунд просит подождать сервер. Число секунд или дата; иначе None."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    if seconds != seconds or seconds < 0:  # NaN или прошлое
        return 0.0
    return min(seconds, MAX_RETRY_AFTER)


def ask(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    *,
    provider=None,
    key: str | None = None,
    http: httpx.Client | None = None,
    max_attempts: int | None = None,
    sleep=time.sleep,
) -> Reply:
    """Один запрос. Поставщик по умолчанию по имени модели. http и sleep подменяются в тестах."""
    provider = provider or providers.for_model(model)
    key = key or os.environ.get(provider.key_env)
    if not key:
        raise ModelError(f"Не задана переменная окружения {provider.key_env}")
    max_attempts = max_attempts or provider.max_attempts

    headers, payload = provider.request(key, system, user, model)

    own = http is None
    http = http or httpx.Client(timeout=60)
    started = time.monotonic()
    last = ""
    try:
        for attempt in range(max_attempts):
            wait = None
            try:
                response = http.post(provider.url, json=payload, headers=headers)
            except httpx.RequestError as err:
                last = f"сеть: {err}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    try:
                        parsed = provider.parse(data, model)
                    except providers.ProviderError as err:
                        raise ModelError(f"непонятный ответ {provider.name}: {err}") from None
                    if parsed.truncated_empty:
                        # Повтор дал бы то же самое: потолок длины ушёл на рассуждение
                        raise ModelError(
                            f"пустой ответ: модель потратила все {provider.max_tokens} токенов "
                            "на рассуждение и не успела ответить")
                    return Reply(
                        text=parsed.text,
                        model=parsed.model,
                        usage=data.get("usage") or {},
                        attempts=attempt + 1,
                        seconds=round(time.monotonic() - started, 3),
                        provider=provider.name,
                    )
                if response.status_code not in RETRY_CODES:
                    raise ModelError(f"{response.status_code}: {response.text[:200]}")
                last = f"{response.status_code}: {response.text[:200]}"
                wait = retry_after(response)
            if attempt < max_attempts - 1:
                sleep(backoff_delay(attempt) if wait is None else wait)
    finally:
        if own:
            http.close()
    raise ModelError(f"Не удалось за {max_attempts} попыток. Последнее: {last}")
