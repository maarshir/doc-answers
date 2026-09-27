from doc_answers.chunker import Chunk
from doc_answers.prompt import NO_ANSWER, SYSTEM, build_user_message, check_answer, is_no_answer

A = Chunk("otpusk.md#1", "otpusk.md", "Отпуск / Сколько дней", "28 дней в год.")
B = Chunk("novichkam.txt#1", "novichkam.txt", "", "Испытательный срок 3 месяца.")


def test_message_labels_every_chunk_and_ends_with_question():
    message = build_user_message("  Сколько дней отпуска?  ", [A, B])
    assert "[otpusk.md#1] (Отпуск / Сколько дней)\n28 дней в год." in message
    assert "[novichkam.txt#1]\nИспытательный срок" in message   # без заголовка нет пустых скобок
    assert message.index("[otpusk.md#1]") < message.index("[novichkam.txt#1]")
    assert message.endswith("Вопрос: Сколько дней отпуска?")


def test_system_prompt_contains_exact_refusal_phrase():
    assert NO_ANSWER in SYSTEM


def test_citations_are_checked():
    checked = check_answer("28 дней [otpusk.md#1], срок 3 месяца [novichkam.txt#1] [otpusk.md#1]", [A, B])
    assert [c.id for c in checked.cited] == ["otpusk.md#1", "novichkam.txt#1"]   # без повторов
    assert checked.unknown == ()
    assert checked.problems == ()
    assert not checked.no_answer


def test_unknown_citation_is_a_problem():
    checked = check_answer("28 дней [otpusk.md#1] и 14 дней [otpusk.md#9]", [A])
    assert checked.unknown == ("otpusk.md#9",)
    assert "otpusk.md#9" in checked.problems[0]


def test_answer_without_citation_is_a_problem():
    checked = check_answer("Отпуск 28 дней.", [A])
    assert checked.cited == ()
    assert "нет ни одной ссылки" in checked.problems[0]


def test_refusal_needs_no_citation():
    checked = check_answer(NO_ANSWER, [A])
    assert checked.no_answer and checked.problems == ()


def test_refusal_detection_is_whole_phrase():
    assert is_no_answer("в документах нет ответа на этот вопрос")
    assert is_no_answer("  В документах нет ответа на этот вопрос!  ")
    assert not is_no_answer("Про больничный в документах нет ответа на этот вопрос, но отпуск 28 дней [otpusk.md#1].")
