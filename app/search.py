"""Search helpers that forgive mistakes.

The meaning search (embeddings) is good at "what is this about" but weak at short
questions like "git" and at spelling mistakes. This module adds the missing half:

  * correct_question(): fix probable typos using the words the document really uses
  * keyword scores:     exact and close-spelling word matches, rarer words count more
  * fuse():             blend the meaning ranking and the keyword ranking into one
  * snippet():          cut a short excerpt centred on the matching word

Everything here is plain Python (no models), so it is fast and easy to test.
"""

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Words that carry no topic ("what is the ..."), so they are ignored when matching.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being both but by
    can could did do does doing down during each few for from further get give had has have having he her
    here hers him his how i if in into is it its just like list me more most my no nor not of off on once
    only or other our out over own please same she should so some such tell than that the their them then
    there these they this those through to too under until up very want was we were what when where which
    while who whom whose why will with would you your yours
    document documents doc docs file files pdf text show
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_FUZZY_CUTOFF = 0.82  # how similar two words must be to count as "the same word, misspelled"
_FUZZY_WEIGHT = 0.7  # a close match counts a bit less than an exact one


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers: "Git-Flow v2" -> ["git", "flow", "v2"]."""
    return _WORD_RE.findall(text.lower())


def keyword_terms(question: str) -> list[str]:
    """The topic words of a question, in order, without duplicates or stop words."""
    terms: list[str] = []
    for word in tokenize(question):
        if word not in STOPWORDS and len(word) > 1 and word not in terms:
            terms.append(word)
    return terms


@dataclass
class KeywordMatch:
    """How well each chunk matches the question's keywords."""

    scores: list[float]  # one score per chunk (0 = no keyword found in it)
    matched_terms: list[str] = field(default_factory=list)  # question words found somewhere in the text
    words: list[str] = field(default_factory=list)  # the document's own spellings of them


class TextIndex:
    """Word statistics of all chunks, built once per question (fast, in memory)."""

    def __init__(self, texts: list[str]) -> None:
        self.counts = [Counter(tokenize(text)) for text in texts]
        self.doc_freq: Counter[str] = Counter(word for counts in self.counts for word in counts)
        self.vocabulary = set(self.doc_freq)
        # Only longer words are compared for spelling: short ones differ too easily ("git" ~ "get").
        self._fuzzy_pool = sorted(word for word in self.vocabulary if len(word) >= 4)

    def correct(self, question: str) -> str:
        """Replace a word the document never uses by a very similar word it does use.

        "what is the documenbt about" -> "what is the document about"
        """

        def fix(match: re.Match[str]) -> str:
            word = match.group(0)
            low = word.lower()
            if len(low) < 4 or low in STOPWORDS or low in self.vocabulary:
                return word
            close = difflib.get_close_matches(low, self._fuzzy_pool, n=1, cutoff=_FUZZY_CUTOFF)
            return close[0] if close else word

        return re.sub(r"[A-Za-z0-9]+", fix, question)

    def _variants(self, term: str) -> dict[str, float]:
        """The spellings of `term` that occur in the document, with a weight each."""
        found: dict[str, float] = {}
        if term in self.vocabulary:
            found[term] = 1.0
        if len(term) >= 4:
            for close in difflib.get_close_matches(term, self._fuzzy_pool, n=4, cutoff=_FUZZY_CUTOFF):
                found.setdefault(close, _FUZZY_WEIGHT)
        return found

    def match(self, question: str) -> KeywordMatch:
        """Score every chunk by the question's keywords (rarer words score higher)."""
        total = len(self.counts)
        scores = [0.0] * total
        matched_terms: list[str] = []
        words: list[str] = []

        for term in keyword_terms(question):
            variants = self._variants(term)
            if not variants:
                continue
            matched_terms.append(term)
            words.extend(word for word in variants if word not in words)
            for position, counts in enumerate(self.counts):
                best = 0.0
                for word, weight in variants.items():
                    frequency = counts.get(word, 0)
                    if frequency:
                        rarity = math.log(1 + total / (1 + self.doc_freq[word]))
                        # Repeats help, but with diminishing returns (a word 20 times is not 20x better).
                        best = max(best, weight * rarity * frequency * 2.2 / (frequency + 1.2))
                scores[position] += best
        return KeywordMatch(scores=scores, matched_terms=matched_terms, words=words)


def fuse_scores(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Blend several best-first rankings into one (reciprocal rank fusion), with scores.

    An item near the top of any list, and especially of several, gets a higher score.
    Returns (position, score) pairs, best first; equal scores keep the lower position first.
    """
    totals: Counter[int] = Counter()
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            totals[position] += 1.0 / (k + rank)
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """The order part of fuse_scores(): chunk positions, best first."""
    return [position for position, _ in fuse_scores(rankings, k)]


def snippet(text: str, words: list[str], width: int = 320) -> str:
    """A short excerpt of `text`, centred on the first matching word when there is one."""
    text = text.strip()
    if len(text) <= width:
        return text

    lowered = text.lower()
    hits = [m.start() for word in words if (m := re.search(rf"\b{re.escape(word)}", lowered))]
    start = max(0, min(hits) - width // 4) if hits else 0
    if start > 0:  # begin at a word boundary, not in the middle of a word
        space = text.rfind(" ", 0, start)
        start = space + 1 if space != -1 else start
    end = start + width
    if end < len(text):  # end at a word boundary too
        space = text.rfind(" ", start, end)
        end = space if space > start else end
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


# --------------------------------------------------------------------------
# Typos in the words that decide HOW a question is answered ("summary", "email", ...).
# --------------------------------------------------------------------------
# A short, safe list. Words that are also real words one letter away ("contact" and
# "contract", "about" and "abort", "whose" and "whole") are deliberately NOT here,
# because fixing them would turn an ordinary question into a different kind of question.
_TYPO_TARGETS = (
    "document", "resume", "summary", "summarize", "summarise", "overview",
    "email", "phone", "mobile", "telephone", "linkedin", "github", "candidate", "applicant",
)  # fmt: skip


def fix_trigger_typos(question: str) -> str:
    """Fix likely typos of the words that pick the kind of answer.

    "what is the documenbt about" -> "what is the document about", "emial" -> "email".
    A word is only changed when it is very close to one of the words above, or has
    exactly the same letters in a different order (a typing slip).
    """

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        low = word.lower()
        if len(low) < 4 or low in _TYPO_TARGETS or low in STOPWORDS:
            return word
        close = difflib.get_close_matches(low, _TYPO_TARGETS, n=1, cutoff=0.75)
        if not close:
            return word
        ratio = difflib.SequenceMatcher(None, low, close[0]).ratio()
        same_letters = sorted(low) == sorted(close[0])
        return close[0] if ratio >= 0.85 or (same_letters and ratio >= 0.75) else word

    return re.sub(r"[A-Za-z]+", fix, question)
