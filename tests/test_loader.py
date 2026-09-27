import pytest

from conftest import make_pdf
from doc_answers.loader import LoadError, load_folder, split_markdown


def test_markdown_sections_keep_heading_chain():
    text = "# Отпуск\n\nВступление.\n\n## Сколько дней\n\n28 дней.\n\n## Как оформить\n\nЗаявка.\n"
    sections = split_markdown(text, "o.md")
    assert [(s.title, s.text) for s in sections] == [
        ("Отпуск", "Вступление."),
        ("Отпуск / Сколько дней", "28 дней."),
        ("Отпуск / Как оформить", "Заявка."),
    ]


def test_markdown_heading_levels_go_back_up():
    text = "# А\n## Б\nтекст б\n# В\nтекст в\n"
    titles = [s.title for s in split_markdown(text, "x.md")]
    assert titles == ["А / Б", "В"]


def test_markdown_hash_inside_code_block_is_not_heading():
    text = "# Раздел\n\n```bash\n# это комментарий\necho 1\n```\n"
    sections = split_markdown(text, "x.md")
    assert len(sections) == 1
    assert "# это комментарий" in sections[0].text


def test_markdown_empty_sections_skipped():
    assert [s.title for s in split_markdown("# А\n\n# Б\nтекст\n", "x.md")] == ["Б"]


def test_text_file_is_one_section(tmp_path):
    (tmp_path / "a.txt").write_text("Первая строка.\n\nВторая.", encoding="utf-8")
    sections, warnings = load_folder(tmp_path)
    assert len(sections) == 1 and sections[0].title == "" and sections[0].source == "a.txt"
    assert warnings == []


def test_subfolders_and_relative_names(tmp_path):
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "b.md").write_text("# Б\nтекст", encoding="utf-8")
    sections, _ = load_folder(tmp_path)
    assert sections[0].source == "hr/b.md"


def test_bad_files_are_skipped_with_warning(tmp_path):
    (tmp_path / "ok.md").write_text("# Раздел\nтекст", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "cp1251.txt").write_bytes("привет".encode("cp1251"))
    (tmp_path / ".hidden.md").write_text("секрет", encoding="utf-8")
    sections, warnings = load_folder(tmp_path)
    assert [s.source for s in sections] == ["ok.md"]
    text = "\n".join(warnings)
    assert "pic.png" in text and "empty.txt" in text and "cp1251.txt" in text
    assert "hidden" not in text   # скрытые файлы даже не упоминаются


def test_missing_or_empty_folder(tmp_path):
    with pytest.raises(LoadError, match="Нет такой папки"):
        load_folder(tmp_path / "nope")
    with pytest.raises(LoadError, match="нет ни одного документа"):
        load_folder(tmp_path)


def test_pdf_pages(tmp_path):
    (tmp_path / "memo.pdf").write_bytes(make_pdf("Laptop replacement every 3 years"))
    sections, warnings = load_folder(tmp_path)
    assert warnings == []
    assert sections[0].title == "стр. 1"
    assert "Laptop replacement every 3 years" in sections[0].text


def test_broken_pdf_is_warning_not_crash(tmp_path):
    (tmp_path / "ok.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    sections, warnings = load_folder(tmp_path)
    assert [s.source for s in sections] == ["ok.txt"]
    assert any("broken.pdf" in w for w in warnings)


def test_examples_load_without_warnings(examples):
    sections, warnings = load_folder(examples)
    assert warnings == []
    assert {s.source for s in sections} == {
        "otpusk.md", "komandirovki.md", "oborudovanie.md", "dezhurstva.md", "novichkam.txt",
    }
