"""Поставщики моделей: Anthropic и Groq.

Тот же подход, что в promptdiff (core/providers.py): всё, чем поставщики
отличаются (адрес, заголовок с ключом, где системный промпт, где текст ответа,
как посчитаны токены), собрано здесь. Повторы и паузы общие, они в client.py.
Groq отвечает в формате OpenAI Chat Completions, поэтому OpenAICompatible
подойдёт и для других совместимых API.

Модуль не берётся из promptdiff зависимостью нарочно: это разные проекты,
и doc-answers должен ставиться и работать без него.
"""

from dataclasses import dataclass, field

from token_counter import Usage


class ProviderError(ValueError):
    """Неизвестный поставщик или ответ не того формата."""


@dataclass(frozen=True)
class Parsed:
    """Разобранный ответ поставщика."""
    text: str
    model: str
    usage: Usage
    # Ответ оборван потолком длины и текста нет: модель всё потратила на рассуждение
    truncated_empty: bool = False


@dataclass(frozen=True)
class Anthropic:
    name: str = "anthropic"
    key_env: str = "ANTHROPIC_API_KEY"
    url: str = "https://api.anthropic.com/v1/messages"
    version: str = "2023-06-01"
    default_model: str = "claude-haiku-4-5"
    # Ответ по четырём кускам со ссылками укладывается с запасом
    max_tokens: int = 700
    max_attempts: int = 4

    def request(self, key: str, system: str, user: str, model: str) -> tuple[dict, dict]:
        headers = {"x-api-key": key, "anthropic-version": self.version, "content-type": "application/json"}
        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            # temperature не задаём: у части новых моделей параметр ограничен,
            # и запрос мог бы упасть с 400 на ровном месте
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        return headers, payload

    def parse(self, data: dict, model: str) -> Parsed:
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return Parsed(text.strip(), data.get("model", model), Usage.from_api(data.get("usage") or {}))

    def usage(self, raw: dict) -> Usage:
        return Usage.from_api(raw or {})


@dataclass(frozen=True)
class OpenAICompatible:
    name: str
    key_env: str
    url: str
    default_model: str
    # Больше, чем у Anthropic: у моделей с рассуждением (gpt-oss) рассуждение
    # входит в тот же потолок, и при 700 ответ со ссылками может не успеть закончиться
    max_tokens: int = 1024
    # Бесплатный уровень часто отвечает 429, повторов нужно больше
    max_attempts: int = 8
    # Дополнительные поля запроса для моделей, имя которых начинается с ключа
    extras: dict = field(default_factory=dict)

    def extra(self, model: str) -> dict:
        for prefix, fields in self.extras.items():
            if model.startswith(prefix):
                return dict(fields)
        return {}

    def request(self, key: str, system: str, user: str, model: str) -> tuple[dict, dict]:
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        payload = {
            "model": model,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.extra(model),
        }
        return headers, payload

    def parse(self, data: dict, model: str) -> Parsed:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("в ответе нет choices")
        choice = choices[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        return Parsed(
            text=text,
            model=data.get("model", model),
            usage=self.usage(data.get("usage") or {}),
            truncated_empty=not text and choice.get("finish_reason") == "length",
        )

    def usage(self, raw: dict) -> Usage:
        # В формате OpenAI токены из кэша промпта уже входят в prompt_tokens,
        # token-counter их вычитает, чтобы не посчитать дважды
        return Usage.from_openai(raw or {})


ANTHROPIC = Anthropic()

GROQ = OpenAICompatible(
    name="groq",
    key_env="GROQ_API_KEY",
    url="https://api.groq.com/openai/v1/chat/completions",
    default_model="openai/gpt-oss-120b",
    extras={
        # gpt-oss рассуждает перед ответом. Пересказать найденные куски со ссылками
        # можно и с низким усилием, а токены рассуждения идут в лимит токенов
        # в минуту бесплатного уровня. Сам текст рассуждения не нужен.
        "openai/gpt-oss": {"reasoning_effort": "low", "include_reasoning": False},
    },
)

PROVIDERS = {p.name: p for p in (ANTHROPIC, GROQ)}


def get(name: str):
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ProviderError(f"Неизвестный поставщик {name!r}, есть: {', '.join(PROVIDERS)}") from None


def for_model(model: str, name: str | None = None):
    """Поставщик по флагу или по имени модели.

    У моделей на Groq в имени есть косая черта (openai/gpt-oss-120b,
    meta-llama/...), у Claude нет. Для имён без черты на Groq
    (например llama-3.3-70b-versatile) нужен явный --provider groq.
    """
    if name:
        return get(name)
    return GROQ if "/" in model else ANTHROPIC


def choose(model: str | None, provider: str | None, env: dict) -> tuple[str, object]:
    """Модель и поставщик для запуска: флаги, потом .env, потом ключи.

    model и provider это значения флагов (None, если не заданы).
    Если ничего не сказано и есть только ключ Groq, берём Groq: так бесплатный
    путь работает без лишних флагов. Если заданы оба ключа, по умолчанию Anthropic.
    """
    provider = provider or env.get("DOC_ANSWERS_PROVIDER") or None
    model = model or env.get("DOC_ANSWERS_MODEL") or None
    if model:
        return model, for_model(model, provider)
    if provider:
        chosen = get(provider)
    elif not env.get(ANTHROPIC.key_env) and env.get(GROQ.key_env):
        chosen = GROQ
    else:
        chosen = ANTHROPIC
    return chosen.default_model, chosen
