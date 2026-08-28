from infer import build_input


def test_build_input_without_keyword():
    assert build_input("  hello  ") == "hello"


def test_build_input_with_keyword():
    assert build_input("the bridge collapsed", "collapse") == (
        "keyword: collapse\nthe bridge collapsed"
    )


def test_build_input_does_not_double_prefix():
    raw = "keyword: flood\nwater in the streets"
    assert build_input(raw, "flood") == raw
