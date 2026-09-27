import pytest

from doc_answers import providers
from doc_answers.providers import ANTHROPIC, GROQ, ProviderError, choose, for_model


def test_provider_by_model_name():
    assert for_model("claude-haiku-4-5") is ANTHROPIC
    assert for_model("openai/gpt-oss-120b") is GROQ
    # имя без косой черты на Groq только с явным поставщиком
    assert for_model("llama-3.3-70b-versatile", "groq") is GROQ


def test_unknown_provider():
    with pytest.raises(ProviderError, match="есть: anthropic, groq"):
        providers.get("openai")


@pytest.mark.parametrize("env, expected", [
    ({}, ("claude-haiku-4-5", "anthropic")),
    ({"ANTHROPIC_API_KEY": "a"}, ("claude-haiku-4-5", "anthropic")),
    # только ключ Groq: бесплатный путь без флагов
    ({"GROQ_API_KEY": "g"}, ("openai/gpt-oss-120b", "groq")),
    # оба ключа: по умолчанию Anthropic, как в promptdiff
    ({"ANTHROPIC_API_KEY": "a", "GROQ_API_KEY": "g"}, ("claude-haiku-4-5", "anthropic")),
    ({"DOC_ANSWERS_MODEL": "openai/gpt-oss-20b"}, ("openai/gpt-oss-20b", "groq")),
    ({"DOC_ANSWERS_PROVIDER": "groq", "ANTHROPIC_API_KEY": "a"}, ("openai/gpt-oss-120b", "groq")),
])
def test_choose_from_env(env, expected):
    model, provider = choose(None, None, env)
    assert (model, provider.name) == expected


def test_flags_win_over_env():
    env = {"DOC_ANSWERS_MODEL": "claude-sonnet-4-6", "GROQ_API_KEY": "g"}
    model, provider = choose("openai/gpt-oss-20b", None, env)
    assert (model, provider.name) == ("openai/gpt-oss-20b", "groq")
    model, provider = choose(None, "groq", {"GROQ_API_KEY": "g"})
    assert (model, provider.name) == ("openai/gpt-oss-120b", "groq")


def test_anthropic_request_shape():
    headers, payload = ANTHROPIC.request("k", "система", "вопрос", "claude-haiku-4-5")
    assert headers["x-api-key"] == "k"
    assert payload["system"] == "система"
    assert payload["messages"] == [{"role": "user", "content": "вопрос"}]
    assert payload["max_tokens"] == 700


def test_groq_request_shape():
    headers, payload = GROQ.request("k", "система", "вопрос", "openai/gpt-oss-120b")
    assert headers["authorization"] == "Bearer k"
    assert payload["messages"][0] == {"role": "system", "content": "система"}
    assert payload["messages"][1] == {"role": "user", "content": "вопрос"}
    assert payload["max_completion_tokens"] == 1024
    assert payload["reasoning_effort"] == "low" and payload["include_reasoning"] is False
    # для других моделей Groq этих полей нет: llama их не знает
    _, other = GROQ.request("k", "s", "u", "meta-llama/llama-4-scout-17b-16e-instruct")
    assert "reasoning_effort" not in other


def test_groq_parse_and_cache_tokens():
    data = {
        "model": "openai/gpt-oss-120b",
        "choices": [{"message": {"content": " Ответ [a.md#1] "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 80, "prompt_tokens_details": {"cached_tokens": 300}},
    }
    parsed = GROQ.parse(data, "openai/gpt-oss-120b")
    assert parsed.text == "Ответ [a.md#1]"
    # токены из кэша вычтены из обычного входа, иначе посчитались бы дважды
    assert (parsed.usage.input_tokens, parsed.usage.cache_read_tokens, parsed.usage.output_tokens) == (600, 300, 80)
    assert not parsed.truncated_empty


def test_groq_empty_answer_cut_by_limit():
    data = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}}
    assert GROQ.parse(data, "m").truncated_empty


def test_groq_answer_without_choices():
    with pytest.raises(ProviderError, match="choices"):
        GROQ.parse({"error": "?"}, "m")
