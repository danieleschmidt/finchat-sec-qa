from finchat_sec_qa.multi_company import compare_question_across_filings


def test_success():
    docs = {"a": "alpha beta", "b": "beta gamma"}
    results = compare_question_across_filings("beta", docs)
    assert len(results) == 2
    assert results[0].answer


def test_edge_case_invalid_input():
    docs = {"a": "alpha"}
    try:
        compare_question_across_filings("", docs)
    except ValueError:
        pass
    else:
        assert False
    try:
        compare_question_across_filings("beta", {})
    except ValueError:
        pass
    else:
        assert False
