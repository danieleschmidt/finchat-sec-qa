from finchat_sec_qa.qa_engine import FinancialQAEngine


def test_engine_persistence(tmp_path):
    store = tmp_path / "index.pkl"
    engine = FinancialQAEngine(storage_path=store)
    engine.add_document("doc", "alpha beta")
    assert store.exists()

    engine2 = FinancialQAEngine(storage_path=store)
    answer, _ = engine2.answer_with_citations("alpha")
    assert "alpha beta" in answer
