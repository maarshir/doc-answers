"""Поиск кусков по вопросу: BM25 по основам слов.

Почему не векторный поиск: он требует либо ключа к API эмбеддингов, либо
скачивания модели на сотни мегабайт, и результат трудно объяснить. BM25
работает без сети за миллисекунды, и по каждому найденному куску видно,
какие слова совпали. Слабое место известно заранее: синонимы
(«ноутбук» и «компьютер») он не свяжет. Для небольшой папки с документами
на одном языке это разумная цена.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import snowballstemmer

from .chunker import Chunk

WORD = re.compile(r"[0-9a-zа-яё]+")

# Служебные слова, которые есть почти в каждом вопросе и ничего не говорят о теме.
# «сколько», «когда», «кто» тоже здесь: они про вид ответа, а не про тему.
STOP_WORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее её мне было вот от меня еще ещё нет о об из ему теперь когда даже ну ли если уже
или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может
они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже
себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
почти мой тем чтобы нее неё сейчас были куда зачем всех никогда можно при наконец два
другой хоть после над больше тот через эти нас про всего них какая много разве три
эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более
всегда конечно всю между сколько какие каких каком какую нужно ли мне меня нам это
делать сделать
the a an and or of to in on for is are be how what when who which do does
""".split())

_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Текст в список основ слов без служебных слов.

    «отпуска», «отпуск» и «отпуском» дают одну основу «отпуск»,
    иначе вопрос «сколько дней отпуска» не нашёл бы раздел «Отпуск».
    Числа остаются как есть: «28» в вопросе должно находить «28» в тексте.
    """
    words = WORD.findall(text.lower().replace("ё", "е"))
    result = []
    for word in words:
        if word in STOP_WORDS:
            continue
        if word.isdigit():
            result.append(word)
        elif re.search("[а-я]", word):
            result.append(_ru.stemWord(word))
        else:
            result.append(_en.stemWord(word))
    return result


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    matched: tuple[str, ...]   # какие основы из вопроса нашлись в куске


class Index:
    """BM25 по кускам. Строится в памяти за один проход, для папки
    из сотен файлов этого достаточно, хранить индекс на диске пока незачем."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Нечего искать: нет ни одного куска")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # Заголовок раздела тоже участвует в поиске: вопрос «как оформить отпуск»
        # должен находить кусок из раздела «Отпуск / Как оформить», даже если
        # в самом тексте куска слова «отпуск» нет.
        self.docs = [Counter(tokenize(f"{c.title}\n{c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths) or 1.0
        df = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        # Вариант idf из Lucene: всегда положительный, даже для слова, которое есть почти везде
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def score(self, terms: list[str], i: int) -> tuple[float, tuple[str, ...]]:
        doc, length = self.docs[i], self.lengths[i]
        total, matched = 0.0, []
        for term in dict.fromkeys(terms):   # повтор слова в вопросе не удваивает вес
            tf = doc.get(term, 0)
            if not tf:
                continue
            matched.append(term)
            norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
            total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total, tuple(matched)

    def search(self, question: str, k: int = 4) -> list[Hit]:
        """Лучшие k кусков. Куски без единого совпавшего слова не возвращаются вовсе:
        пустой результат значит «в документах про это ничего нет», и модель
        тогда не вызывается."""
        if k < 1:
            raise ValueError("k должно быть не меньше 1")
        terms = tokenize(question)
        hits = []
        for i, chunk in enumerate(self.chunks):
            score, matched = self.score(terms, i)
            if score > 0:
                hits.append(Hit(chunk, score, matched))
        # При равном счёте порядок как в документах: результат не зависит от случая
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
