from finchat_sec_qa.qa_engine import FinancialQAEngine


def test_engine_persistence(tmp_path):
    store = tmp_path / "index.joblib"
    engine = FinancialQAEngine(storage_path=store)
    engine.add_document("doc", "alpha beta")
    assert store.exists()

    engine2 = FinancialQAEngine(storage_path=store)
    answer, _ = engine2.answer_with_citations("alpha")
    assert "alpha beta" in answer


def test_migrate_old_pickle(tmp_path):
    old = tmp_path / "index.pkl"
    engine = FinancialQAEngine(storage_path=old)
    engine.add_document("doc", "text")
    assert old.exists()
    new_store = tmp_path / "index.joblib"
    FinancialQAEngine(storage_path=new_store)
    assert new_store.exists() and not old.exists()
