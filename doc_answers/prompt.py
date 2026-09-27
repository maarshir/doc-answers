"""Промпт для модели и разбор её ответа.

Модель получает только найденные куски, каждый со своим номером вида
[otpusk.md#2], и обязана ссылаться на эти номера. После ответа мы сами
проверяем ссылки: существуют ли такие куски и есть ли ссылка вообще.
Верить модели на слово, что она «опиралась на документы», нельзя.
"""

import re
from dataclasses import dataclass

from .chunker import Chunk

# Точная фраза для случая «ответа нет». Одна и та же в промпте, в разборе ответа
# и в проверках набора вопросов, поэтому живёт в одной константе.
NO_ANSWER = "В документах нет ответа на этот вопрос."

SYSTEM = f"""Ты отвечаешь на вопросы строго по фрагментам документов, которые даны в сообщении.

Правила:
- Используй только сведения из фрагментов. Не добавляй ничего из общих знаний, даже если уверен.
- После каждого утверждения ставь ссылку на фрагмент в квадратных скобках точно так, как он подписан, например [otpusk.md#2]. Если утверждение опирается на два фрагмента, поставь обе ссылки.
- Если во фрагментах нет ответа на вопрос, ответь ровно одной фразой: {NO_ANSWER}
- Если ответ есть только частично, ответь на ту часть, что есть, и прямо скажи, чего во фрагментах нет.
- Отвечай коротко и по-русски, без вступлений вроде «Согласно документам»."""

CITATION = re.compile(r"\[([^\[\]\s]+#\d+)\]")


def build_user_message(question: str, chunks: list[Chunk]) -> str:
    """Сообщение пользователя: фрагменты, потом вопрос.

    Вопрос стоит в конце, после фрагментов: так модель читает материал,
    уже зная, что в нём искать, и вопрос не теряется за длинным контекстом.
    """
    parts = ["Фрагменты документов:"]
    for chunk in chunks:
        title = f" ({chunk.title})" if chunk.title else ""
        parts.append(f"[{chunk.id}]{title}\n{chunk.text}")
    parts.append(f"Вопрос: {question.strip()}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class Checked:
    text: str
    no_answer: bool                 # модель сказала, что ответа нет
    cited: tuple[Chunk, ...]        # куски, на которые есть ссылки, в порядке первого упоминания
    unknown: tuple[str, ...]        # ссылки на куски, которых модели не давали
    problems: tuple[str, ...]       # что не так с ответом, для вывода человеку


def is_no_answer(text: str) -> bool:
    """Модель сказала «ответа нет». Сравниваем без учёта регистра и точки в конце,
    но только целую фразу: «ответа нет» внутри длинного ответа это другое."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".!")
    return norm(text) == norm(NO_ANSWER)


def check_answer(text: str, given: list[Chunk]) -> Checked:
    """Проверяет ссылки в ответе модели."""
    by_id = {c.id: c for c in given}
    cited, unknown = [], []
    for ref in dict.fromkeys(CITATION.findall(text)):
        if ref in by_id:
            cited.append(by_id[ref])
        else:
            unknown.append(ref)

    no_answer = is_no_answer(text)
    problems = []
    if unknown:
        problems.append("ссылки на фрагменты, которых модели не давали: " + ", ".join(unknown))
    if not no_answer and not cited:
        problems.append("в ответе нет ни одной ссылки на источник, проверьте его вручную")
    return Checked(text.strip(), no_answer, tuple(cited), tuple(unknown), tuple(problems))
