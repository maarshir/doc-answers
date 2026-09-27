"""Командная строка: python -m doc_answers ask|search-eval|export-promptdiff.

export-arena оставлена как второе имя export-promptdiff: так проект назывался раньше
(prompt-arena), и старые команды из заметок должны работать."""

import argparse
import os
import sys
import textwrap
from pathlib import Path

from . import client, providers
from .chunker import DEFAULT_OVERLAP, DEFAULT_SIZE
from .evalset import promptdiff_cases, promptdiff_prompts, format_search_report, load_questions, search_report
from .loader import LoadError
from .pipeline import DEFAULT_K, answer, build_index

DEFAULT_DOCS = "examples/docs"
DEFAULT_QUESTIONS = "eval/questions.yaml"


def load_env(path: str | Path = ".env") -> None:
    """Читает простой файл KEY=VALUE. Уже заданные переменные не трогает."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--docs", default=DEFAULT_DOCS, help=f"папка с документами (по умолчанию {DEFAULT_DOCS})")
    p.add_argument("--k", type=int, default=DEFAULT_K, help=f"сколько кусков отдавать модели (по умолчанию {DEFAULT_K})")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"размер куска в символах (по умолчанию {DEFAULT_SIZE})")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                   help=f"перекрытие соседних кусков в символах (по умолчанию {DEFAULT_OVERLAP})")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m doc_answers", description="Ответы на вопросы по своим документам со ссылкой на источник")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ответить на вопрос")
    ask.add_argument("question", help="вопрос в кавычках")
    _common(ask)
    ask.add_argument("--model", default=None,
                     help=f"модель (по умолчанию DOC_ANSWERS_MODEL, иначе {providers.ANTHROPIC.default_model}, "
                          f"а если задан только GROQ_API_KEY, {providers.GROQ.default_model})")
    ask.add_argument("--provider", choices=sorted(providers.PROVIDERS), default=None,
                     help="поставщик (по умолчанию DOC_ANSWERS_PROVIDER или по имени модели: косая черта значит groq)")
    ask.add_argument("--no-model", action="store_true", help="не вызывать модель, только показать найденное (так же без ключа)")
    ask.add_argument("--show-prompt", action="store_true", help="показать сообщение, которое уходит модели")
    ask.add_argument("--prices", default=None, help="свой файл цен для token-counter")

    ev = sub.add_parser("search-eval", help="проверить поиск на наборе вопросов, без модели и без ключа")
    _common(ev)
    ev.add_argument("--questions", default=DEFAULT_QUESTIONS)

    ex = sub.add_parser("export-promptdiff", aliases=["export-arena"], help="выгрузить набор вопросов в формат promptdiff")
    _common(ex)
    ex.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ex.add_argument("--out", default="eval/promptdiff", help="папка для prompts.yaml и cases.yaml")
    return p.parse_args(argv)


def _wrap(text: str, indent: str = "   ") -> str:
    return "\n".join(textwrap.fill(line, 96, initial_indent=indent, subsequent_indent=indent) if line else ""
                     for line in text.splitlines())


def run_ask(args, out) -> int:
    index, warnings = build_index(args.docs, args.size, args.overlap)
    for w in warnings:
        print(f"Пропущено: {w}", file=sys.stderr)

    model, provider = providers.choose(args.model, args.provider, os.environ)
    has_key = bool(os.environ.get(provider.key_env))
    use_model = has_key and not args.no_model

    result = answer(args.question, index, k=args.k, model=model, provider=provider,
                    use_model=use_model, prices_path=args.prices)

    if result.mode == "model":
        print(result.text, file=out)
        if result.checked.cited:
            print("\nИсточники:", file=out)
            for chunk in result.checked.cited:
                print(f"  [{chunk.id}] {chunk.where}", file=out)
    elif result.mode == "no_hits":
        print(result.text, file=out)
    else:
        if has_key:
            reason = "флаг --no-model"
        elif args.model or args.provider or os.environ.get("DOC_ANSWERS_PROVIDER") or os.environ.get("DOC_ANSWERS_MODEL"):
            reason = f"нет {provider.key_env}"
        else:
            # Поставщик не выбран: подсказываем оба ключа, у Groq есть бесплатный уровень
            reason = f"нет {providers.ANTHROPIC.key_env} или {providers.GROQ.key_env}"
        print(f"Демонстрационный режим ({reason}): модель не вызывалась.", file=out)
        print("Вот куски, которые получила бы модель, лучшие сверху:\n", file=out)
        for i, hit in enumerate(result.hits, 1):
            print(f"{i}. [{hit.chunk.id}] {hit.chunk.where}  (счёт {hit.score:.2f}, совпали: {', '.join(hit.matched)})", file=out)
            print(_wrap(hit.chunk.text), file=out)
            print(file=out)

    if args.show_prompt and result.user_message:
        print("\n--- сообщение модели ---\n" + result.user_message, file=out)

    for note in result.notes:
        print(f"\n{note}" if note == result.notes[0] else note, file=out)
    return 0


def run_search_eval(args, out) -> int:
    index, _ = build_index(args.docs, args.size, args.overlap)
    rows = search_report(index, load_questions(args.questions), args.k)
    print(f"Поиск: k={args.k}, куски по {args.size} символов, перекрытие {args.overlap}, всего кусков {len(index.chunks)}\n", file=out)
    print(format_search_report(rows, args.k), file=out)
    # Код возврата 1, если хоть для одного вопроса кусок с ответом не нашёлся: так проверку можно ставить в CI
    return 0 if all(r.ok for r in rows) else 1


def run_export(args, out) -> int:
    index, _ = build_index(args.docs, args.size, args.overlap)
    questions = load_questions(args.questions)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    settings = f"k={args.k}, size={args.size}, overlap={args.overlap}"
    (target / "prompts.yaml").write_text(promptdiff_prompts(), encoding="utf-8")
    (target / "cases.yaml").write_text(promptdiff_cases(index, questions, args.k, settings), encoding="utf-8")
    print(f"Записано: {target / 'prompts.yaml'}, {target / 'cases.yaml'}", file=out)
    return 0


def main(argv=None, out=None) -> int:
    out = out or sys.stdout
    load_env()
    args = parse_args(argv)
    try:
        if args.command == "ask":
            return run_ask(args, out)
        if args.command == "search-eval":
            return run_search_eval(args, out)
        return run_export(args, out)
    except (LoadError, ValueError) as err:
        # ProviderError и QuestionError тоже ValueError. Понятная строка вместо трассировки.
        print(f"Ошибка: {err}", file=sys.stderr)
        return 2
    except client.ModelError as err:
        print(f"Модель не ответила: {err}", file=sys.stderr)
        return 3
