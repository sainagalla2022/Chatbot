"""Tests for the typo-tolerant keyword search helpers (plain Python, no models)."""

from app import search
from app.search import TextIndex


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert search.tokenize("Git-Flow v2, node.js!") == ["git", "flow", "v2", "node", "js"]


def test_keyword_terms_drop_stop_words_and_duplicates():
    assert search.keyword_terms("what is git git") == ["git"]
    assert search.keyword_terms("give all commands in git") == ["commands", "git"]
    assert search.keyword_terms("what is in the document") == []  # nothing left but filler


def test_correct_fixes_typos_using_the_documents_own_words():
    index = TextIndex(["The document explains kubernetes clusters and pods."])

    fixed = index.correct("what is the documenbt about kuberntes")

    assert fixed == "what is the document about kubernetes"


def test_correct_leaves_short_words_stop_words_and_unknown_words_alone():
    index = TextIndex(["The document explains kubernetes."])
    # "gti" is too short to guess, "the" is a stop word, "zebra" resembles nothing in the text.
    assert index.correct("the gti zebra") == "the gti zebra"


def test_correct_keeps_the_original_letter_case_when_nothing_needs_fixing():
    index = TextIndex(["Kubernetes runs pods."])
    assert index.correct("What is Kubernetes") == "What is Kubernetes"


def test_match_scores_chunks_that_contain_the_keyword():
    index = TextIndex(["git commit and git push", "docker run image", "nothing relevant here"])

    match = index.match("git")

    assert match.scores[0] > 0
    assert match.scores[1] == 0 and match.scores[2] == 0
    assert match.matched_terms == ["git"]
    assert match.words == ["git"]


def test_match_forgives_plurals_and_misspellings():
    index = TextIndex(["run the command now", "unrelated words only"])

    plural = index.match("commands")  # the document says "command"
    assert plural.scores[0] > 0 and plural.scores[1] == 0
    assert plural.words == ["command"]

    typo = index.match("comand")  # missing an "m"
    assert typo.scores[0] > 0


def test_exact_matches_outscore_close_spellings():
    index = TextIndex(["kubernetes is great", "kubernetez is great"])
    scores = index.match("kubernetes").scores
    assert scores[0] > scores[1] > 0


def test_a_rare_word_counts_for_more_than_a_common_one():
    texts = ["alpha beta"] * 6 + ["alpha gamma"]  # "alpha" is everywhere, "gamma" is rare
    index = TextIndex(texts)
    assert index.match("gamma").scores[6] > index.match("alpha").scores[6]


def test_repeating_a_word_helps_but_with_diminishing_returns():
    index = TextIndex(["git", "git git", "git " * 40, "other"])
    scores = index.match("git").scores
    assert scores[0] < scores[1] < scores[2]
    assert scores[2] < scores[0] * 3  # 40 mentions are nowhere near 40 times better


def test_match_ignores_question_words_the_document_never_uses():
    index = TextIndex(["git commit"])
    match = index.match("git zebra")
    assert match.matched_terms == ["git"]


def test_fuse_puts_items_that_both_searches_like_first():
    assert search.fuse([[3, 1, 2], [2, 3]]) == [3, 2, 1]


def test_fuse_is_stable_and_handles_empty_lists():
    assert search.fuse([[], []]) == []
    assert search.fuse([[5], [4]]) == [4, 5]  # equal scores: the lower position first


def test_snippet_returns_short_text_unchanged():
    assert search.snippet("  short text  ", ["short"]) == "short text"


def test_snippet_centres_on_the_matching_word_and_cuts_at_word_boundaries():
    text = ("filler words " * 40) + "the git command is here " + ("more filler " * 40)

    excerpt = search.snippet(text, ["git"], width=120)

    assert "git" in excerpt
    assert excerpt.startswith("…") and excerpt.endswith("…")
    assert len(excerpt) <= 124
    core = excerpt.strip("…")
    assert not core.startswith("iller") and not core.endswith("fill")  # no half words


def test_snippet_starts_at_the_beginning_when_nothing_matches():
    text = "Beginning of the text. " + "x " * 500
    excerpt = search.snippet(text, ["zzz"], width=80)
    assert excerpt.startswith("Beginning of the text.") and excerpt.endswith("…")


def test_fix_trigger_typos_repairs_common_slips():
    assert search.fix_trigger_typos("what is the documenbt about") == "what is the document about"
    assert search.fix_trigger_typos("give me a sumary of my resum") == "give me a summary of my resume"
    assert search.fix_trigger_typos("what is the emial address") == "what is the email address"  # swapped letters
    assert search.fix_trigger_typos("phne number and Linkedn") == "phone number and linkedin"


def test_fix_trigger_typos_never_turns_real_words_into_trigger_words():
    for question in [
        "what is the contract period",  # one letter from "contact"
        "how to abort a build",  # one letter from "about"
        "explain the whole process",  # one letter from "whose"
        "what is a prone patient",  # "prone" is close to "phone"
        "what is the documentary about",
    ]:
        assert search.fix_trigger_typos(question) == question, question
