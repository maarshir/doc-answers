import json

import pytest

from doc_answers import client
from doc_answers.pipeline import answer, build_index
from doc_answers.prompt import NO_ANSWER, SYSTEM


@pytest.fixture
def index(examples):
    return build_index(examples)[0]


def never_called(*args, **kwargs):
    raise AssertionError("модель не должна вызываться")


def fake_ask(text, model="claude-haiku-4-5-20251001", usage=None):
    calls = []

    def ask(system, user, model_name, key=None, provider=None):
        calls.append((system, user, model_name))
        ask.providers.append(provider)
        return client.Reply(text, model, usage or {"input_tokens": 1000, "output_tokens": 100})

    ask.calls = calls
    ask.providers = []
    return ask


def test_nothing_found_means_no_model_call(index):
    result = answer("Какая столица Австралии?", index, ask=never_called)
    assert result.mode == "no_hits"
    assert result.text == NO_ANSWER


def test_demo_mode_shows_chunks_and_estimate(index):
    result = answer("Сколько дней отпуска?", index, use_model=False, ask=never_called)
    assert result.mode == "demo"
    assert result.hits[0].chunk.source == "otpusk.md"
    assert "Вопрос: Сколько дней отпуска?" in result.user_message
    assert "≈" in result.notes[0] and "claude-haiku-4-5" in result.notes[0]


def test_model_answer_with_citation_and_cost(index):
    ask = fake_ask("28 календарных дней [otpusk.md#1].")
    result = answer("Сколько дней отпуска?", index, k=2, ask=ask)
    system, user, _ = ask.calls[0]
    assert system == SYSTEM
    assert user.count("[") >= 2   # ушли два куска с подписями
    assert result.mode == "model"
    assert [c.id for c in result.checked.cited] == ["otpusk.md#1"]
    # haiku-4-5: $1 за миллион входа и $5 за миллион выхода: 1000 и 100 токенов это $0.0015
    assert "$0.0015" in result.notes[0]
    assert len(result.notes) == 1   # без предупреждений


def test_made_up_citation_is_reported(index):
    result = answer("Сколько дней отпуска?", index, ask=fake_ask("28 дней [otpusk.md#99]."))
    assert any("otpusk.md#99" in n for n in result.notes)
    assert any("нет ни одной ссылки" in n for n in result.notes)


def test_unknown_model_price_does_not_break_answer(index, tmp_path):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"models": {"other": {
        "provider": "x", "input": "1", "output": "1", "checked": "2026-09-27", "source": "тест"}}}), encoding="utf-8")
    result = answer("Сколько дней отпуска?", index, ask=fake_ask("28 [otpusk.md#1]", model="my-model"),
                    prices_path=prices)
    assert result.text == "28 [otpusk.md#1]"
    assert "Цена неизвестна" in result.notes[0]


def test_empty_question(index):
    with pytest.raises(ValueError):
        answer("   ", index, ask=never_called)


def test_groq_answer_cost_from_openai_usage(index):
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 400}}
    ask = fake_ask("28 календарных дней [otpusk.md#1].", model="openai/gpt-oss-120b", usage=usage)
    result = answer("Сколько дней отпуска?", index, model="openai/gpt-oss-120b", ask=ask)
    assert ask.providers[0].name == "groq"
    assert "Токены: 600 на входе, 100 на выходе. Из кэша промпта 400." in result.notes[0]
    # gpt-oss-120b на Groq: 600 × $0.15 + 400 × $0.075 + 100 × $0.60 за миллион = $0.00018
    assert "$0.00018" in result.notes[0]


def test_demo_estimate_uses_provider_limit(index):
    result = answer("Сколько дней отпуска?", index, model="openai/gpt-oss-120b", use_model=False, ask=never_called)
    assert "не больше 1024 на выходе" in result.notes[0]
    assert "openai/gpt-oss-120b" in result.notes[0]
