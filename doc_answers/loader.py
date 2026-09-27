"""Чтение папки с документами.

Документ сразу делится на разделы: у маркдауна это заголовки, у PDF страницы,
у простого текста один раздел на весь файл. Раздел нужен для ссылки на источник:
«otpusk.md, Как оформить» полезнее, чем просто «otpusk.md».
"""

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class LoadError(ValueError):
    """Папку или файл прочитать нельзя."""


@dataclass(frozen=True)
class Section:
    source: str   # путь к файлу относительно папки с документами
    title: str    # заголовок раздела, «стр. 3» для PDF, пусто для простого текста
    text: str


def split_markdown(text: str, source: str) -> list[Section]:
    """Режет маркдаун по заголовкам. Заголовок раздела собирается из цепочки:
    «Отпуск / Как оформить», чтобы было видно, откуда кусок.

    Строка с # внутри блока кода заголовком не считается.
    """
    sections: list[Section] = []
    path: list[str] = []          # текущая цепочка заголовков
    lines: list[str] = []
    in_code = False

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section(source, " / ".join(path), body))
        lines.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            del path[level - 1:]
            # если уровни пропущены (# сразу ###), цепочка просто короче
            path.append(match.group(2))
        else:
            lines.append(line)
    flush()
    return sections


def read_pdf(path: Path, source: str) -> list[Section]:
    """Текст PDF по страницам. Сканы без текстового слоя дают пустоту, это честно
    видно в выводе как «пустой файл», распознавание текста здесь не делается."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf есть в requirements.txt
        raise LoadError("Для PDF нужен пакет pypdf: pip install pypdf") from None

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as err:  # pypdf бросает много разных ошибок на битых файлах
        raise LoadError(f"{source}: не удалось прочитать PDF ({err})") from None

    return [
        Section(source, f"стр. {number}", text.strip())
        for number, text in enumerate(pages, 1)
        if text.strip()
    ]


def load_file(path: Path, source: str) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return read_pdf(path, source)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise LoadError(f"{source}: файл не в UTF-8") from None
    if suffix in {".md", ".markdown"}:
        return split_markdown(text, source)
    text = text.strip()
    return [Section(source, "", text)] if text else []


def load_folder(folder: str | Path) -> tuple[list[Section], list[str]]:
    """Все подходящие файлы папки и подпапок.

    Возвращает разделы и список предупреждений: какие файлы пропущены и почему.
    Один битый файл не должен ронять всю папку, но и молча пропадать не должен.
    """
    root = Path(folder)
    if not root.is_dir():
        raise LoadError(f"Нет такой папки: {root}")

    sections: list[Section] = []
    warnings: list[str] = []
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    for path in files:
        source = path.relative_to(root).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES | PDF_SUFFIXES:
            warnings.append(f"{source}: пропущен, формат {path.suffix or 'без расширения'} не поддерживается")
            continue
        try:
            found = load_file(path, source)
        except LoadError as err:
            warnings.append(str(err))
            continue
        if not found:
            warnings.append(f"{source}: пустой файл, текста не найдено")
        sections.extend(found)

    if not sections:
        raise LoadError(f"В папке {root} нет ни одного документа с текстом (.md, .txt, .pdf)")
    return sections, warnings
