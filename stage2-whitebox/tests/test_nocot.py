from s2.nocot import score_continuation


def test_first_integer_is_the_answer_and_clean():
    scored = score_continuation("42", 42, [17, 25])
    assert scored["correct"]
    assert not scored["leaked"]


def test_intermediate_operator_and_words_are_leaks():
    assert score_continuation("17 42", 42, [17]).get("leaked")
    assert score_continuation("42 + 1", 42, []).get("leaked")
    assert score_continuation("The answer is 42", 42, []).get("leaked")
    assert not score_continuation("  42", 42, []).get("leaked")
