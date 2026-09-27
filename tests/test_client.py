import httpx
import pytest

from doc_answers import client


def fake_http(responses, seen=None):
    """httpx.Client, который вместо сети отдаёт заранее заданные ответы по очереди."""
    queue = list(responses)

    def handler(request):
        if seen is not None:
            seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler))


def ok(text="Ответ [a.md#1]"):
    return httpx.Response(200, json={
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    })


def test_success_and_request_shape():
    seen = []
    reply = client.ask("система", "вопрос", "claude-haiku-4-5", key="k", http=fake_http([ok()], seen))
    assert reply.text == "Ответ [a.md#1]"
    assert reply.usage == {"input_tokens": 100, "output_tokens": 20}
    assert reply.attempts == 1
    body = seen[0].read().decode()
    assert '"system"' in body and seen[0].headers["x-api-key"] == "k"


def test_retries_overload_then_succeeds():
    pauses = []
    http = fake_http([httpx.Response(529, text="overloaded"), httpx.ConnectError("нет сети"), ok()])
    reply = client.ask("s", "u", key="k", http=http, sleep=pauses.append)
    assert reply.attempts == 3
    assert len(pauses) == 2


def test_bad_request_is_not_retried():
    seen = []
    with pytest.raises(client.ModelError, match="400"):
        client.ask("s", "u", key="k", http=fake_http([httpx.Response(400, text="bad"), ok()], seen), sleep=lambda _: None)
    assert len(seen) == 1


def test_gives_up_after_max_attempts():
    http = fake_http([httpx.Response(503)] * 3)
    with pytest.raises(client.ModelError, match="3 попыток"):
        client.ask("s", "u", key="k", http=http, max_attempts=3, sleep=lambda _: None)


def test_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(client.ModelError, match="ANTHROPIC_API_KEY"):
        client.ask("s", "u")


def test_backoff_grows_and_has_cap():
    assert 0.5 <= client.backoff_delay(0) <= 1
    assert 4 <= client.backoff_delay(3) <= 8
    assert client.backoff_delay(20) <= 20
