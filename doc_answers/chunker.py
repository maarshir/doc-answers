"""Нарезка разделов на куски для поиска.

Кусок должен быть достаточно коротким, чтобы в промпт влезло несколько
найденных кусков, и достаточно длинным, чтобы мысль в нём не обрывалась.
Поэтому режем по абзацам, длинный абзац по предложениям и только очень
длинное предложение по словам. Соседние куски немного перекрываются,
чтобы факт на границе не потерялся.
"""

import re
from dataclasses import dataclass

from .loader import Section

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

DEFAULT_SIZE = 600
DEFAULT_OVERLAP = 120


@dataclass(frozen=True)
class Chunk:
    id: str       # «otpusk.md#2»: по нему модель ссылается на источник
    source: str
    title: str
    text: str

    @property
    def where(self) -> str:
        """Источник для человека: файл и раздел."""
        return f"{self.source}, {self.title}" if self.title else self.source


def _split_long(text: str, size: int) -> list[str]:
    """Предложение длиннее size режем по словам. Слово длиннее size не режем."""
    parts, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def _units(text: str, size: int) -> list[tuple[int, str]]:
    """Текст раздела в список кусочков (номер абзаца, текст), каждый не длиннее size."""
    units = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for number, paragraph in enumerate(paragraphs):
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        if len(paragraph) <= size:
            units.append((number, paragraph))
            continue
        for sentence in SENTENCE_END.split(paragraph):
            if len(sentence) <= size:
                units.append((number, sentence))
            else:
                units.extend((number, piece) for piece in _split_long(sentence, size))
    return units


def _join(units: list[tuple[int, str]]) -> str:
    """Кусочки из одного абзаца через пробел, из разных через пустую строку."""
    text = ""
    for i, (number, piece) in enumerate(units):
        if i == 0:
            text = piece
        elif units[i - 1][0] == number:
            text += " " + piece
        else:
            text += "\n\n" + piece
    return text


def _length(units: list[tuple[int, str]]) -> int:
    return len(_join(units))


def chunk_section(section: Section, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Тексты кусков одного раздела."""
    if size < 50:
        raise ValueError("Размер куска меньше 50 символов не имеет смысла")
    if not 0 <= overlap < size:
        raise ValueError("Перекрытие должно быть от 0 и меньше размера куска")

    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for unit in _units(section.text, size):
        if current and _length(current + [unit]) > size:
            chunks.append(current)
            # Перекрытие: берём с конца готового куска целые кусочки, пока влезают.
            # Половину предложения не берём: оборванная фраза в начале куска
            # сбивает и поиск, и модель.
            tail: list[tuple[int, str]] = []
            for prev in reversed(current):
                if _length([prev] + tail) > overlap:
                    break
                tail.insert(0, prev)
            current = tail if _length(tail + [unit]) <= size else []
        current.append(unit)
    if current:
        chunks.append(current)
    return [_join(c) for c in chunks]


def chunk_sections(sections: list[Section], size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """Все куски всех разделов. Номер куска сквозной внутри файла: otpusk.md#1, #2, ..."""
    chunks = []
    counters: dict[str, int] = {}
    for section in sections:
        for text in chunk_section(section, size, overlap):
            counters[section.source] = counters.get(section.source, 0) + 1
            chunk_id = f"{section.source}#{counters[section.source]}"
            chunks.append(Chunk(chunk_id, section.source, section.title, text))
    return chunks
