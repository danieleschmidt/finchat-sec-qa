import logging
from unittest.mock import Mock
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

    # Create mock client with proper Mock objects
    mock_client = Mock()
    mock_filing = Mock()
    mock_filing.accession_no = '1'
    mock_filing.document_url = str(tmp_path / 'f.html')
    
    mock_client.get_recent_filings.return_value = [mock_filing]
    
    def download_side_effect(filing):
        path = tmp_path / 'f.html'
        path.write_text('alpha')
        return path
    
    mock_client.download_filing.side_effect = download_side_effect
    
    monkeypatch.setattr(webapp, 'client', mock_client)
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


def test_query_exception_handling_and_logging(monkeypatch, caplog):
    """Test that query endpoint handles exceptions properly and logs appropriately."""
    from finchat_sec_qa import webapp
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    
    # Test validation error handling
    with caplog.at_level(logging.WARNING):
        with app.test_client() as client:
            # Missing required field should trigger ValidationError
            resp = client.post('/query', json={'ticker': 'AAPL'})  # Missing 'question'
            assert resp.status_code == 400
            assert 'Invalid request data' in resp.get_data(as_text=True)
    
    # Check that validation error was logged
    assert any('Invalid query request data' in record.message for record in caplog.records)


def test_query_filing_not_found_handling(monkeypatch, caplog):
    """Test handling when no filings are found for a ticker."""
    from finchat_sec_qa import webapp
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    
    # Create mock client that returns no filings
    mock_client = Mock()
    mock_client.get_recent_filings.return_value = []  # No filings found
    
    monkeypatch.setattr(webapp, 'client', mock_client)
    
    with caplog.at_level(logging.WARNING):
        with app.test_client() as client:
            resp = client.post('/query', json={'question': 'What is the revenue?', 'ticker': 'INVALID'})
            assert resp.status_code == 404
            assert 'No filings found' in resp.get_data(as_text=True)
    
    # Check that warning was logged
    assert any('No filings found for ticker: INVALID' in record.message for record in caplog.records)


def test_query_file_error_handling(monkeypatch, caplog, tmp_path):
    """Test handling when filing file cannot be read."""
    from finchat_sec_qa import webapp
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    
    # Create mock client that simulates file error
    mock_client = Mock()
    mock_filing = Mock()
    mock_filing.accession_no = '1'
    mock_filing.document_url = 'http://example.com'
    
    mock_client.get_recent_filings.return_value = [mock_filing]
    mock_client.download_filing.return_value = tmp_path / 'nonexistent.html'  # Return path to non-existent file
    
    monkeypatch.setattr(webapp, 'client', mock_client)
    
    with caplog.at_level(logging.ERROR):
        with app.test_client() as client:
            resp = client.post('/query', json={'question': 'What is the revenue?', 'ticker': 'AAPL'})
            assert resp.status_code == 500
            assert 'Error processing filing' in resp.get_data(as_text=True)
    
    # Check that error was logged
    assert any('Filing file not found' in record.message for record in caplog.records)


def test_risk_exception_handling_and_logging(monkeypatch, caplog):
    """Test that risk endpoint handles exceptions properly and logs appropriately."""
    from finchat_sec_qa import webapp
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    
    # Test validation error handling
    with caplog.at_level(logging.WARNING):
        with app.test_client() as client:
            # Empty text should trigger ValidationError
            resp = client.post('/risk', json={'text': ''})
            assert resp.status_code == 400
            assert 'Invalid request data' in resp.get_data(as_text=True)
    
    # Check that validation error was logged
    assert any('Invalid risk request data' in record.message for record in caplog.records)


def test_risk_assessment_error_handling(monkeypatch, caplog):
    """Test handling when risk assessment fails."""
    from finchat_sec_qa import webapp
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    
    class FailingRiskAnalyzer:
        def assess(self, text):
            raise RuntimeError("Risk analysis failed")
    
    monkeypatch.setattr(webapp, 'risk', FailingRiskAnalyzer())
    
    with caplog.at_level(logging.ERROR):
        with app.test_client() as client:
            resp = client.post('/risk', json={'text': 'Some text to analyze'})
            assert resp.status_code == 500
            assert 'Error performing risk assessment' in resp.get_data(as_text=True)
    
    # Check that error was logged
    assert any('Error performing risk assessment' in record.message for record in caplog.records)


def test_successful_operations_logging(monkeypatch, caplog, tmp_path):
    """Test that successful operations are logged with appropriate info messages."""
    from finchat_sec_qa import webapp
    engine = webapp.engine
    monkeypatch.delenv("FINCHAT_TOKEN", raising=False)
    
    # Create mock client for successful operations test
    mock_client = Mock()
    mock_filing = Mock()
    mock_filing.accession_no = '1'
    mock_filing.document_url = str(tmp_path / 'f.html')
    
    mock_client.get_recent_filings.return_value = [mock_filing]
    
    def download_side_effect(filing):
        path = tmp_path / 'f.html'
        path.write_text('revenue increased by 15%')
        return path
    
    mock_client.download_filing.side_effect = download_side_effect
    
    monkeypatch.setattr(webapp, 'client', mock_client)
    engine.chunks.clear()
    
    with caplog.at_level(logging.INFO):
        with app.test_client() as client_app:
            # Test successful query
            resp = client_app.post('/query', json={'question': 'revenue growth?', 'ticker': 'AAPL'})
            assert resp.status_code == 200
            
            # Test successful risk assessment
            resp = client_app.post('/risk', json={'text': 'The company is performing well'})
            assert resp.status_code == 200
    
    # Check that success operations were logged
    log_messages = [record.message for record in caplog.records]
    assert any('Processing query for ticker: AAPL' in msg for msg in log_messages)
    assert any('Query completed successfully' in msg for msg in log_messages)
    assert any('Processing risk analysis for text of' in msg for msg in log_messages)
    assert any('Risk analysis completed with' in msg for msg in log_messages)

