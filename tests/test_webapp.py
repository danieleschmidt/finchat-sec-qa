from finchat_sec_qa.webapp import app


def test_auth_required(monkeypatch):
    from finchat_sec_qa import webapp
    monkeypatch.setattr(webapp, 'SECRET_TOKEN', 't')
    with app.test_client() as client:
        resp = client.post('/query', json={'question': 'q', 'ticker': 'AAPL'})
        assert resp.status_code == 401


def test_query(monkeypatch, tmp_path):
    from finchat_sec_qa import webapp
    engine = webapp.engine
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)

    class DummyClient:
        def get_recent_filings(self, ticker, limit=1):
            return [type('F', (), {'accession_no': '1', 'document_url': str(tmp_path / 'f.html')})]

        def download_filing(self, filing):
            path = tmp_path / 'f.html'
            path.write_text('alpha')
            return path

    monkeypatch.setattr(webapp, 'client', DummyClient())
    engine.chunks.clear()
    with app.test_client() as client_app:
        resp = client_app.post('/query', json={'question': 'a?', 'ticker': 'AAPL'})
        assert resp.status_code == 200
        assert 'alpha' in resp.json['answer']


def test_query_invalid(monkeypatch):
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    with app.test_client() as client:
        resp = client.post('/query', json={'ticker': 'AAPL'})
        assert resp.status_code == 400


def test_risk_invalid(monkeypatch):
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    with app.test_client() as client:
        resp = client.post('/risk', json={'text': ''})
        assert resp.status_code == 400

