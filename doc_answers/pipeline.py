"""Весь путь от вопроса до ответа: поиск, промпт, модель, проверка ссылок, цена."""

from dataclasses import dataclass, field
from pathlib import Path

from token_counter import PriceError, Usage, cost, estimate_tokens, find_price, format_usd, load_prices

from . import client, providers
from .chunker import DEFAULT_OVERLAP, DEFAULT_SIZE, chunk_sections
from .loader import load_folder
from .prompt import NO_ANSWER, SYSTEM, Checked, build_user_message, check_answer
from .search import Hit, Index

DEFAULT_K = 4


@dataclass
class Result:
    question: str
    hits: list[Hit]
    mode: str                      # "no_hits", "demo" или "model"
    user_message: str = ""
    reply: client.Reply | None = None
    checked: Checked | None = None
    notes: list[str] = field(default_factory=list)   # цена, предупреждения

    @property
    def text(self) -> str:
        if self.mode == "no_hits":
            return NO_ANSWER
        if self.checked:
            return self.checked.text
        return ""


def build_index(folder: str | Path, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> tuple[Index, list[str]]:
    sections, warnings = load_folder(folder)
    return Index(chunk_sections(sections, size, overlap)), warnings


def spent_line(model: str, usage: dict, prices_path=None, provider=None) -> str:
    """Сколько стоил настоящий запрос. Цены и арифметика из token-counter.

    usage в том виде, как пришёл от API: у Anthropic и у Groq поля разные,
    разбирает их поставщик.
    """
    provider = provider or providers.for_model(model)
    tokens = provider.usage(usage)
    head = f"Токены: {tokens.input_tokens} на входе, {tokens.output_tokens} на выходе."
    if tokens.cache_read_tokens:
        head += f" Из кэша промпта {tokens.cache_read_tokens}."
    try:
        price = find_price(model, load_prices(prices_path))
    except PriceError as err:
        return f"{head} Цена неизвестна: {err}"
    return f"{head} Цена {format_usd(cost(tokens, price).total)} (цены от {price.checked})."


def estimate_line(model: str, user_message: str, prices_path=None, provider=None) -> str:
    """Прикидка сверху без сети: сколько стоил бы запрос с ключом.

    Выход считаем по потолку длины у поставщика: настоящий ответ почти всегда
    короче, но для прикидки сверху это честная граница.
    """
    provider = provider or providers.for_model(model)
    limit = provider.max_tokens
    tokens_in = estimate_tokens(SYSTEM).tokens + estimate_tokens(user_message).tokens
    head = f"≈{tokens_in} токенов на входе (прикидка без токенизатора), не больше {limit} на выходе."
    try:
        price = find_price(model, load_prices(prices_path))
    except PriceError as err:
        return f"{head} Цена неизвестна: {err}"
    upper = cost(Usage(input_tokens=tokens_in, output_tokens=limit), price).total
    return f"{head} С ключом запрос к {model} стоил бы не больше ≈{format_usd(upper)}."


def answer(
    question: str,
    index: Index,
    *,
    k: int = DEFAULT_K,
    model: str = client.DEFAULT_MODEL,
    use_model: bool = True,
    provider=None,
    key: str | None = None,
    ask=None,
    prices_path=None,
) -> Result:
    """Ответ на вопрос.

    Если поиск ничего не нашёл, модель не вызывается вовсе: ей не на что
    опереться, и любой ответ был бы выдумкой за наши деньги.
    Без ключа (или с use_model=False) возвращаются найденные куски и прикидка цены.
    """
    if not question.strip():
        raise ValueError("Пустой вопрос")
    provider = provider or providers.for_model(model)

    hits = index.search(question, k)
    if not hits:
        return Result(question, hits, "no_hits",
                      notes=["Ни в одном документе нет слов из вопроса, модель не вызывалась."])

    chunks = [h.chunk for h in hits]
    message = build_user_message(question, chunks)

    if not use_model:
        return Result(question, hits, "demo", message,
                      notes=[estimate_line(model, message, prices_path, provider)])

    ask = ask or client.ask   # берём в момент вызова, чтобы тесты могли подменить client.ask
    reply = ask(SYSTEM, message, model, key=key, provider=provider)
    checked = check_answer(reply.text, chunks)
    notes = [spent_line(reply.model, reply.usage, prices_path, provider)]
    notes.extend(f"Внимание: {p}." for p in checked.problems)
    return Result(question, hits, "model", message, reply, checked, notes)
