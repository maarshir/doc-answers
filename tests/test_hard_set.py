"""Трудный набор вопросов: файл читается и сходится с документами.

Сам итог поиска на этом наборе тест не проверяет: промахи там ожидаемы,
и правка поиска должна менять число в README, а не ронять тесты.
"""

from conftest import ROOT
from doc_answers.evalset import load_questions

HARD_DOCS = ROOT / "examples" / "hard" / "docs"
HARD_QUESTIONS = ROOT / "eval" / "hard.yaml"


def test_hard_questions_load_and_have_notes():
    questions = load_questions(HARD_QUESTIONS)
    assert 8 <= len(questions) <= 10
    assert all(q.note for q in questions)
    assert any(not q.answerable for q in questions)


def test_every_source_exists_and_contains_its_facts():
    for q in load_questions(HARD_QUESTIONS):
        if not q.answerable:
            continue
        path = HARD_DOCS / q.source
        assert path.is_file(), f"{q.id}: нет файла {q.source}"
        text = path.read_text(encoding="utf-8")
        for fact in q.facts:
            assert fact in text, f"{q.id}: в {q.source} нет «{fact}»"
