import re

import pytest

from doc_answers.chunker import chunk_section, chunk_sections
from doc_answers.loader import Section, load_folder


def sec(text, source="a.md", title="Т"):
    return Section(source, title, text)


def words(text):
    return re.findall(r"\w+", text)


def test_short_section_is_one_chunk():
    assert chunk_section(sec("Один абзац.\n\nВторой абзац."), size=100, overlap=0) == ["Один абзац.\n\nВторой абзац."]


def test_paragraphs_are_packed_up_to_size():
    text = "\n\n".join(["абзац номер " + str(i) + " " + "x" * 30 for i in range(6)])
    chunks = chunk_section(sec(text), size=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # без перекрытия каждый абзац ровно в одном куске
    assert sum(c.count("абзац номер") for c in chunks) == 6


def test_long_paragraph_is_split_by_sentences():
    sentences = [f"Предложение {i} про отпуск и командировки." for i in range(10)]
    chunks = chunk_section(sec(" ".join(sentences)), size=120, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 120
        assert c.endswith(".")   # предложение не обрывается посередине


def test_very_long_sentence_is_split_by_words():
    text = " ".join(["слово"] * 100)
    chunks = chunk_section(sec(text), size=60, overlap=0)
    assert all(len(c) <= 60 for c in chunks)
    assert sum(len(words(c)) for c in chunks) == 100


def test_overlap_repeats_whole_previous_sentence():
    text = "Первое предложение тут. Второе предложение тут. Третье предложение тут. Четвёртое предложение тут."
    chunks = chunk_section(sec(text), size=55, overlap=30)
    assert len(chunks) >= 2
    # конец первого куска целиком повторяется в начале второго
    last = chunks[0].split(". ")[-1]
    assert chunks[1].startswith(last.rstrip("."))


def test_no_words_lost(examples):
    sections, _ = load_folder(examples)
    for section in sections:
        chunks = chunk_section(section, size=150, overlap=0)
        assert words(" ".join(chunks)) == words(re.sub(r"\s+", " ", section.text))


def test_examples_chunks_fit_size(examples):
    sections, _ = load_folder(examples)
    for size in (150, 300, 600):
        for chunk in chunk_sections(sections, size, size // 5):
            assert len(chunk.text) <= size


def test_ids_are_per_file_and_sequential():
    sections = [sec("а", "x.md", "1"), sec("б", "x.md", "2"), sec("в", "y.txt", "")]
    chunks = chunk_sections(sections, 100, 0)
    assert [c.id for c in chunks] == ["x.md#1", "x.md#2", "y.txt#1"]
    assert chunks[0].where == "x.md, 1"
    assert chunks[2].where == "y.txt"


@pytest.mark.parametrize("size,overlap", [(10, 0), (100, 100), (100, -1)])
def test_bad_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_section(sec("текст"), size=size, overlap=overlap)
