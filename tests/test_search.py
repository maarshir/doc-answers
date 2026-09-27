import pytest

from doc_answers.chunker import Chunk
from doc_answers.search import Index, tokenize


def chunk(i, text, title=""):
    return Chunk(f"d.md#{i}", "d.md", title, text)


def test_tokenize_stems_and_drops_stop_words():
    assert tokenize("Сколько дней отпуска?") == tokenize("дней отпуском")
    # Известное ограничение стеммера: беглые гласные он не сводит, «дней» и «дни» разные
    assert tokenize("дней") != tokenize("дни")
    assert "сколько" not in tokenize("сколько")
    assert tokenize("") == []


def test_tokenize_yo_and_numbers():
    assert tokenize("её ещё") == []
    assert tokenize("Ёлка") == tokenize("елка")
    assert tokenize("28 дней")[0] == "28"


def test_right_chunk_first():
    index = Index([
        chunk(1, "Суточные 1500 рублей за каждый день поездки."),
        chunk(2, "Отпуск 28 календарных дней в год."),
        chunk(3, "Ноутбук выдают в первый день."),
    ])
    hits = index.search("сколько дней отпуска", k=3)
    assert hits[0].chunk.id == "d.md#2"
    assert "отпуск" in hits[0].matched


def test_title_counts_in_search():
    index = Index([chunk(1, "Пишут заявку в канал.", "Отпуск / Как оформить"), chunk(2, "Выдают в первый день.", "Ноутбук")])
    assert index.search("как оформить отпуск")[0].chunk.id == "d.md#1"


def test_unrelated_question_finds_nothing():
    index = Index([chunk(1, "Отпуск 28 дней."), chunk(2, "Суточные 1500 рублей.")])
    assert index.search("Какая столица Австралии?") == []
    assert index.search("и в на") == []   # одни служебные слова


def test_rare_word_weighs_more():
    index = Index([
        chunk(1, "день день день работы"),
        chunk(2, "день дежурства"),
        chunk(3, "день отдыха"),
    ])
    # «дежурство» есть в одном куске, «день» во всех: редкое слово решает
    assert index.search("день дежурства")[0].chunk.id == "d.md#2"


def test_repeated_question_word_does_not_double():
    index = Index([chunk(1, "отпуск"), chunk(2, "ноутбук")])
    assert index.search("отпуск отпуск отпуск")[0].score == pytest.approx(index.search("отпуск")[0].score)


def test_k_limits_and_ties_keep_document_order():
    index = Index([chunk(i, "отпуск") for i in range(1, 6)])
    hits = index.search("отпуск", k=3)
    assert [h.chunk.id for h in hits] == ["d.md#1", "d.md#2", "d.md#3"]
    with pytest.raises(ValueError):
        index.search("отпуск", k=0)


def test_empty_index_is_error():
    with pytest.raises(ValueError):
        Index([])
