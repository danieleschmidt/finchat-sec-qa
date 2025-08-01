from pathlib import Path

from fastapi.testclient import TestClient
from finchat_sec_qa.server import app
from finchat_sec_qa.edgar_client import FilingMetadata
from finchat_sec_qa.qa_engine import FinancialQAEngine


class DummyClient:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_recent_filings(self, ticker: str, form_type: str = "10-K", limit: int = 1):
        return [FilingMetadata(cik="1", accession_no="1", form_type=form_type, filing_date=None, document_url=self.path)]

    def download_filing(self, filing: FilingMetadata, destination=None):
        Path(self.path).write_text("alpha")
        return self.path


def test_query_endpoint(tmp_path):
    with TestClient(app) as client:
        app.state.client = DummyClient(tmp_path / "f.html")
        app.state.engine = FinancialQAEngine(storage_path=tmp_path / "i.joblib")
        resp = client.post('/query', json={'question': 'alpha?', 'ticker': 'AAPL'})
        assert resp.status_code == 200
        assert 'alpha' in resp.json()['answer']


def test_query_invalid_payload(tmp_path):
    with TestClient(app) as client:
        app.state.client = DummyClient(tmp_path / 'f.html')
        app.state.engine = FinancialQAEngine(storage_path=tmp_path / 'i.joblib')
        resp = client.post('/query', json={'ticker': 'AAPL'})
        assert resp.status_code == 422


def test_risk_endpoint():
    with TestClient(app) as client:
        resp = client.post('/risk', json={'text': 'lawsuit pending'})
        assert resp.status_code == 200
        assert 'litigation' in resp.json()['flags']


def test_risk_invalid_payload():
    with TestClient(app) as client:
        resp = client.post('/risk', json={'text': ''})
        assert resp.status_code == 422


def test_health_endpoint():
    """Test health check endpoint returns proper status."""
    with TestClient(app) as client:
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'timestamp' in data
        assert 'services' in data
        assert isinstance(data['services'], dict)


def test_health_endpoint_with_services(tmp_path):
    """Test health check includes service status when available."""
    with TestClient(app) as client:
        # Set up app state as if initialized
        app.state.client = DummyClient(tmp_path / "f.html")
        app.state.engine = FinancialQAEngine(storage_path=tmp_path / "i.joblib")
        
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'healthy'
        assert data['services']['edgar_client'] == 'ready'
        assert data['services']['qa_engine'] == 'ready'


def test_engine_saved_on_shutdown(tmp_path):
    idx = tmp_path / 'i.joblib'

    class DummyEngine(FinancialQAEngine):
        def __init__(self, path):
            super().__init__(storage_path=path)
            self.saved = False

        def save(self, path=None):
            super().save(path)
            self.saved = True

    with TestClient(app) as client:
        app.state.client = DummyClient(tmp_path / 'f.html')
        engine = DummyEngine(idx)
        app.state.engine = engine
        client.post('/query', json={'question': 'q', 'ticker': 'AAPL'})
    assert engine.saved and idx.exists()
