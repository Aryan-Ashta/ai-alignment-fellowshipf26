from s2.cot_gen import answer_after_think


def test_answer_is_the_last_integer_after_think():
    text = "<think> 17 + 25 = 42 </think>\n\nThe result is 42."
    assert answer_after_think(text) == 42
    assert answer_after_think("no tag 42") is None
