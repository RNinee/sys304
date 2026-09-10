import numpy as np

from infer import _parse_items, _softmax, build_input


def test_build_input_without_keyword():
    assert build_input("  hello  ") == "hello"


def test_build_input_with_keyword():
    assert build_input("the bridge collapsed", "collapse") == (
        "keyword: collapse\nthe bridge collapsed"
    )


def test_build_input_does_not_double_prefix():
    raw = "keyword: flood\nwater in the streets"
    assert build_input(raw, "flood") == raw


def test_parse_items_requires_text():
    items, error = _parse_items([{"text": ""}])
    assert items == []
    assert error == "text is required"


def test_parse_items_accepts_batch():
    items, error = _parse_items(
        [
            {"text": "Forest fire near La Ronge"},
            {"text": "Love skiing", "keyword": "body"},
        ]
    )
    assert error is None
    assert items[0] == ("Forest fire near La Ronge", None)
    assert items[1] == ("Love skiing", "body")


def test_softmax_rows_sum_to_one():
    rows = _softmax(np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64))
    assert np.allclose(rows.sum(axis=-1), 1.0)
    assert rows[1, 0] > 0.99
