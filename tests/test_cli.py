import io

import pytest

from conftest import EXAMPLES
from doc_answers import client
from doc_answers.cli import main


@pytest.fixture(autouse=True)
def no_key(monkeypatch, tmp_path):
    # чтобы .env разработчика не влиял на тесты
    monkeypatch.chdir(tmp_path)
    # setenv перед delenv, чтобы monkeypatch запомнил переменную и убрал её после теста,
    # даже если тест прочитал .env и выставил её заново
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY")


def run(*argv):
    out = io.StringIO()
    code = main(list(argv), out)
    return code, out.getvalue()


def test_ask_without_key_is_demo():
    code, text = run("ask", "Сколько дней отпуска?", "--docs", str(EXAMPLES), "--k", "2")
    assert code == 0
    assert "Демонстрационный режим (нет ANTHROPIC_API_KEY)" in text
    assert "1. [otpusk.md#1] otpusk.md, Отпуск / Сколько дней" in text
    assert "≈" in text


def test_ask_show_prompt():
    code, text = run("ask", "Сколько дней отпуска?", "--docs", str(EXAMPLES), "--show-prompt")
    assert "--- сообщение модели ---" in text and "Вопрос: Сколько дней отпуска?" in text


def test_ask_nothing_found():
    code, text = run("ask", "Какая столица Австралии?", "--docs", str(EXAMPLES))
    assert code == 0
    assert "В документах нет ответа" in text and "модель не вызывалась" in text


def test_env_file_is_read(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    code, text = run("ask", "Сколько дней отпуска?", "--docs", str(EXAMPLES), "--no-model")
    assert "флаг --no-model" in text


def test_missing_folder_is_clear_error(capsys):
    code, _ = run("ask", "вопрос", "--docs", "nope")
    assert code == 2
    assert "Нет такой папки" in capsys.readouterr().err


def test_search_eval_passes_on_examples():
    code, text = run("search-eval", "--docs", str(EXAMPLES), "--questions", str(EXAMPLES.parent.parent / "eval" / "questions.yaml"))
    assert code == 0
    assert "Кусок с ответом на первом месте" in text


def test_export_arena(tmp_path):
    code, _ = run("export-arena", "--docs", str(EXAMPLES),
                  "--questions", str(EXAMPLES.parent.parent / "eval" / "questions.yaml"), "--out", str(tmp_path / "arena"))
    assert code == 0
    assert (tmp_path / "arena" / "cases.yaml").exists() and (tmp_path / "arena" / "prompts.yaml").exists()


def test_ask_with_model_prints_sources(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    def fake(system, user, model, key=None):
        return client.Reply("28 календарных дней [otpusk.md#1].", "claude-haiku-4-5", {"input_tokens": 500, "output_tokens": 20})

    monkeypatch.setattr(client, "ask", fake)
    code, text = run("ask", "Сколько дней отпуска?", "--docs", str(EXAMPLES))
    assert code == 0
    assert text.startswith("28 календарных дней [otpusk.md#1].")
    assert "Источники:\n  [otpusk.md#1] otpusk.md, Отпуск / Сколько дней" in text
    assert "Токены: 500 на входе, 20 на выходе." in text


def test_model_error_is_clear(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    def broken(*a, **kw):
        raise client.ModelError("401: invalid x-api-key")

    monkeypatch.setattr(client, "ask", broken)
    code, _ = run("ask", "Сколько дней отпуска?", "--docs", str(EXAMPLES))
    assert code == 3
    assert "Модель не ответила: 401" in capsys.readouterr().err
