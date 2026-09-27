"""Набор вопросов для проверки: поиск без ключа и выгрузка в формат promptdiff.

Качество ответа по документам складывается из двух частей, и проверять их
лучше отдельно. Первая: нашёл ли поиск нужный документ. Это проверяется
без модели, бесплатно и точно. Вторая: ответила ли модель по найденному,
со ссылкой и без выдумки. Это проверяет promptdiff на тех же вопросах.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .prompt import NO_ANSWER, SYSTEM, build_user_message
from .search import Index


class QuestionError(ValueError):
    """Файл с вопросами составлен неверно."""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool = True
    source: str = ""                       # файл, где лежит ответ
    facts: tuple[str, ...] = field(default_factory=tuple)   # что обязано быть в ответе
    note: str = ""


def _text_list(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise QuestionError(f"{where}: нужен список строк")
    items = []
    for item in value:
        # bool проверяем раньше int: в Python True тоже int
        if isinstance(item, bool) or item is None or not isinstance(item, (str, int, float)):
            raise QuestionError(f"{where}: значение {item!r} возьмите в кавычки")
        items.append(str(item))
    return tuple(items)


def load_questions(path: str | Path) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise QuestionError("Файл должен содержать непустой список вопросов")
    questions, seen = [], set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise QuestionError(f"Вопрос {i}: ожидался набор полей")
        qid = item.get("id")
        if not qid or qid in seen:
            raise QuestionError(f"Вопрос {i}: нет id или такой id уже есть")
        seen.add(qid)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise QuestionError(f"Вопрос {qid}: нет текста вопроса")
        answerable = item.get("answerable", True)
        if not isinstance(answerable, bool):
            raise QuestionError(f"Вопрос {qid}: answerable это true или false")
        source = item.get("source", "")
        facts = _text_list(item.get("facts", []), f"Вопрос {qid}, facts")
        if answerable and not (source and facts):
            raise QuestionError(f"Вопрос {qid}: у вопроса с ответом нужны source и facts")
        if not answerable and (source or facts):
            raise QuestionError(f"Вопрос {qid}: у вопроса без ответа не может быть source и facts")
        questions.append(Question(qid, item["question"].strip(), answerable, source, facts, item.get("note", "")))
    return questions


@dataclass(frozen=True)
class SearchRow:
    question: Question
    rank: int | None      # место куска с ответом в выдаче, начиная с 1; None, если не нашёлся
    found: tuple[str, ...]  # id найденных кусков

    @property
    def ok(self) -> bool:
        """Для вопроса с ответом: кусок с ответом в выдаче. Для вопроса без ответа
        поиск ничего не обещает: пустая выдача хорошо, непустая тоже допустима,
        тогда отказ это работа модели."""
        return self.rank is not None if self.question.answerable else True


def has_answer(chunk, q: Question) -> bool:
    """Кусок из нужного файла, и в нём есть все факты. Мало найти правильный файл:
    если в промпт попал соседний раздел без ответа, модель честно откажется,
    и виноват будет поиск, а не модель."""
    text = chunk.text.lower().replace("ё", "е")
    return chunk.source == q.source and all(f.lower().replace("ё", "е") in text for f in q.facts)


def search_report(index: Index, questions: list[Question], k: int) -> list[SearchRow]:
    rows = []
    for q in questions:
        hits = index.search(q.question, k)
        rank = None
        if q.answerable:
            rank = next((i for i, h in enumerate(hits, 1) if has_answer(h.chunk, q)), None)
        rows.append(SearchRow(q, rank, tuple(h.chunk.id for h in hits)))
    return rows


def format_search_report(rows: list[SearchRow], k: int) -> str:
    answerable = [r for r in rows if r.question.answerable]
    top1 = sum(1 for r in answerable if r.rank == 1)
    topk = sum(1 for r in answerable if r.rank is not None)
    lines = [f"{'вопрос':<22} {'где ответ':<18} место  найдено"]
    for r in rows:
        if r.question.answerable:
            place = str(r.rank) if r.rank else "нет"
            want = r.question.source
        else:
            place = "-"
            want = "(ответа нет)"
        found = ", ".join(r.found) if r.found else "ничего, модель не вызывается"
        lines.append(f"{r.question.id:<22} {want:<18} {place:>5}  {found}")
    lines.append("")
    lines.append(f"Кусок с ответом на первом месте: {top1} из {len(answerable)}, среди первых {k}: {topk} из {len(answerable)}.")
    no_answer = [r for r in rows if not r.question.answerable]
    if no_answer:
        empty = sum(1 for r in no_answer if not r.found)
        lines.append(
            f"Вопросы без ответа: {len(no_answer)}, из них поиск сам отсеял {empty}; "
            "остальные должна отсеять модель (это проверяет promptdiff)."
        )
    return "\n".join(lines)


# ---- выгрузка в promptdiff ----

PLAIN = f"""Ответь на вопрос по фрагментам документов из сообщения. Укажи, из какого фрагмента взят ответ.
Если ответа во фрагментах нет, напиши: {NO_ANSWER}"""


class _Literal(str):
    """Многострочная строка в YAML блоком |, чтобы файл читался глазами."""


def _literal(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Literal, _literal)


def _dump(data, header: str) -> str:
    body = yaml.dump(data, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=1000)
    return header + body


def promptdiff_prompts() -> str:
    variants = [
        {"id": "strict", "note": "промпт doc-answers: только фрагменты, ссылка после каждого утверждения, точная фраза отказа",
         "prompt": _Literal(SYSTEM + "\n")},
        {"id": "plain", "note": "тот же договор об ответе, но без правил: для сравнения",
         "prompt": _Literal(PLAIN + "\n")},
    ]
    header = ("# Создано командой: python -m doc_answers export-promptdiff\n"
              "# Не править руками: файл пересоздаётся из doc_answers/prompt.py.\n\n")
    return _dump(variants, header)


def promptdiff_cases(index: Index, questions: list[Question], k: int, settings: str = "") -> str:
    """Задачи для promptdiff: вход это то самое сообщение, которое doc-answers
    отправил бы модели, с найденными кусками. Вопросы, на которые поиск ничего
    не нашёл, не выгружаются: модель на них в doc-answers не вызывается."""
    cases = []
    for q in questions:
        hits = index.search(q.question, k)
        if not hits:
            continue
        if q.answerable:
            # факт и имя файла из ссылки обязаны быть в ответе, отказа быть не должно
            expect = {"contains_all": [*q.facts, q.source], "contains_none": ["нет ответа"]}
        else:
            expect = {"contains_all": ["нет ответа на этот вопрос"]}
        case = {"id": q.id, "input": _Literal(build_user_message(q.question, [h.chunk for h in hits]) + "\n"),
                "expect": expect}
        if q.note:
            case["note"] = q.note
        cases.append(case)
    header = (f"# Создано командой: python -m doc_answers export-promptdiff ({settings or f'k={k}'})\n"
              "# Не править руками: вопросы живут в eval/questions.yaml, куски берутся из examples/docs.\n\n")
    return _dump(cases, header)
