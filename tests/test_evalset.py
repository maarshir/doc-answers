import pytest
import yaml

from conftest import ROOT
from doc_answers.cli import DEFAULT_DOCS, DEFAULT_QUESTIONS
from doc_answers.evalset import (
    QuestionError, promptdiff_cases, promptdiff_prompts, load_questions, search_report,
)
from doc_answers.pipeline import DEFAULT_K, build_index
from doc_answers.prompt import SYSTEM

QUESTIONS = ROOT / DEFAULT_QUESTIONS


@pytest.fixture
def index():
    return build_index(ROOT / DEFAULT_DOCS)[0]


def write(tmp_path, text):
    path = tmp_path / "q.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("text,match", [
    ("- id: a\n  question: вопрос\n", "source и facts"),
    ("- id: a\n  question: вопрос\n  answerable: false\n  source: a.md\n", "не может быть"),
    ("- id: a\n  question: вопрос\n  source: a.md\n  facts: [yes]\n", "в кавычки"),
    ("- id: a\n  question: вопрос\n  source: a.md\n  facts: \"28\"\n", "список"),
    ("- id: a\n  question: в\n  source: a.md\n  facts: [\"1\"]\n- id: a\n  question: в\n  source: a.md\n  facts: [\"1\"]\n", "уже есть"),
    ("- id: a\n  question: вопрос\n  answerable: нет\n", "true или false"),
    ("[]", "непустой"),
])
def test_bad_question_files(tmp_path, text, match):
    with pytest.raises(QuestionError, match=match):
        load_questions(write(tmp_path, text))


def test_numbers_in_facts_become_text(tmp_path):
    q = load_questions(write(tmp_path, "- id: a\n  question: вопрос\n  source: a.md\n  facts: [28]\n"))
    assert q[0].facts == ("28",)


def test_every_answer_is_found_by_search(index):
    """Поиск на примерах находит кусок с ответом для каждого вопроса. Если правка
    нарезки или поиска это сломает, тест покажет, какой вопрос пострадал."""
    rows = search_report(index, load_questions(QUESTIONS), DEFAULT_K)
    missed = [r.question.id for r in rows if not r.ok]
    assert missed == []


def test_promptdiff_cases_input_is_the_real_message(index):
    """Вход задачи в promptdiff совпадает с тем, что doc-answers отправил бы модели."""
    from doc_answers.prompt import build_user_message
    questions = load_questions(QUESTIONS)
    cases = {c["id"]: c for c in yaml.safe_load(promptdiff_cases(index, questions, 2))}
    q = next(q for q in questions if q.id == "per_diem")
    expected = build_user_message(q.question, [h.chunk for h in index.search(q.question, 2)])
    assert cases["per_diem"]["input"] == expected + "\n"
    assert cases["per_diem"]["expect"] == {"contains_all": ["1500", "komandirovki.md"], "contains_none": ["нет ответа"]}
    assert cases["sick_leave"]["expect"] == {"contains_all": ["нет ответа на этот вопрос"]}


def test_unrelated_question_is_filtered_by_search(index):
    rows = {r.question.id: r for r in search_report(index, load_questions(QUESTIONS), DEFAULT_K)}
    assert rows["capital"].found == ()
    assert rows["sick_leave"].found != ()   # этот должна отсеять модель


def test_promptdiff_files_follow_promptdiff_format(index):
    """Те же требования, что у загрузчика promptdiff (core/cases.py и core/variants.py).
    Настоящим загрузчиком promptdiff эти файлы проверяются в CI (tests.yml)."""
    cases = yaml.safe_load(promptdiff_cases(index, load_questions(QUESTIONS), DEFAULT_K))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert set(case) <= {"id", "input", "expect", "note"}
        assert case["input"].strip()
        assert set(case["expect"]) <= {"exact", "contains_all", "contains_none"}
        for key in ("contains_all", "contains_none"):
            for item in case["expect"].get(key, []):
                assert isinstance(item, str)
    assert "capital" not in ids and "sick_leave" in ids

    variants = yaml.safe_load(promptdiff_prompts())
    assert [v["id"] for v in variants] == ["strict", "plain"]
    assert variants[0]["prompt"].strip() == SYSTEM.strip()
